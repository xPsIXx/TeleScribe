"""Core Telegram bot — handles transcription, group history, admin commands."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    Update,
    constants,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from telescribe.config import AppConfig
from telescribe.history import MessageStore
from telescribe.logger import get_logger
from telescribe.transcriber import BaseTranscriber, create_transcriber

logger = get_logger("bot")

# Reload coordination — set by web dashboard, consumed by bot
_reload_requested = False


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors from the bot."""
    logger.error("Unhandled error: %s", context.error, exc_info=context.error)
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ An internal error occurred.")
        except Exception:
            pass


def format_messages_for_summary(messages: list[dict]) -> str:
    """Format messages into a readable block for the LLM summary prompt."""
    lines = []
    for msg in messages:
        name = msg.get("first_name") or msg.get("username") or f"User {msg['user_id']}"
        ts = datetime.fromtimestamp(msg["timestamp"], tz=timezone.utc).strftime("%H:%M")
        text = msg.get("voice_transcription") or msg.get("text") or ""
        if text:
            lines.append(f"[{ts}] {name}: {text}")
    return "\n".join(lines)


class TalkscribeBot:
    """The main bot class."""

    def __init__(self, config: AppConfig, transcriber: BaseTranscriber, store: MessageStore):
        self.config = config
        self.transcriber = transcriber
        self.store = store
        self._app: Optional[Application] = None
        self._llm_client: Optional = None
        logger.info("Bot instance created")

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import AsyncOpenAI

            logger.info("Initializing LLM client: %s (model=%s)", self.config.llm.base_url, self.config.llm.model)
            self._llm_client = AsyncOpenAI(
                api_key=self.config.llm.api_key or "not-needed",
                base_url=self.config.llm.base_url,
            )
        return self._llm_client

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        admins = self.config.bot.admin_user_ids
        return not admins or user_id in admins

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is in the authorized users list.
        Empty list means no one is authorized (except transcription, which is public)."""
        auth_list = self.config.bot.authorized_users
        return user_id in auth_list

    async def _download_audio(self, file_id: str) -> tuple[bytes, str]:
        """Download a file from Telegram, return (data, mime_type)."""
        file = await self._app.bot.get_file(file_id)
        data = await file.download_as_bytearray()
        mime_type = file.file_path.split(".")[-1] if file.file_path else "ogg"
        mime_map = {"oga": "audio/ogg", "ogg": "audio/ogg", "mp3": "audio/mpeg", "mp4": "audio/mp4",
                     "m4a": "audio/mp4", "wav": "audio/wav", "webm": "audio/webm"}
        return bytes(data), mime_map.get(mime_type, "audio/ogg")

    async def _send_private(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
        """Send a message visible only to the requesting user.

        In privacy mode, uses Telegram's ephemeral messages — the message
        appears in the group chat but is visible ONLY to the target user
        and the bot. Everyone else sees nothing.

        Uses Bot API 10.3+ EphemeralMessageParameters format.
        In non-privacy mode or private chats, replies normally in-chat.
        """
        safe_text = self._escape_markdown(text) if text.startswith("❌") or text.startswith("⚠️") else text

        if self.config.bot.privacy_mode and update.effective_chat.type in (
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP
        ):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=safe_text,
                api_kwargs={"ephemeral_message_parameters": {"receiver_user_id": update.effective_user.id}},
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
        else:
            await update.message.reply_text(
                text=safe_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )

    def _escape_markdown(self, text: str) -> str:
        """Escape Telegram markdown special characters in text."""
        special = set('_*[]()~`>#+-=|{}.!')
        bs = chr(92)
        result = []
        for ch in text:
            if ch in special:
                result.append(bs + ch)
            else:
                result.append(ch)
        return ''.join(result)

    # ---- Command Handlers ----

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start — show help."""
        user = update.effective_user
        logger.info("Command /start from user %s (%s)", user.id, user.first_name)
        await self._send_private(update, context,
            "👋 *Welcome to TeleScribe!*\\n\\n"
            "I can transcribe voice messages and help summarize conversations.\\n\\n"
            "*/summarize* — Summarize all unsummarized transcriptions from others\\n"
            "*/reply* — Generate replies to all unreplied transcriptions from others\\n"
            "*/transcribe* — Transcribe past voice messages (backlog)\\n"
            "*/help* — Show this message"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help."""
        user = update.effective_user
        logger.info("Command /help from user %s (%s)", user.id, user.first_name)
        await self.cmd_start(update, context)

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages — transcribe and reply publicly (no button)."""
        if not update.effective_user:
            return

        user_id = update.effective_user.id
        message = update.message
        if not message:
            return

        voice = message.voice or message.audio
        document = message.document
        file_id = None
        mime_type = "audio/ogg"

        if voice:
            file_id = voice.file_id
            mime_type = "audio/ogg"
            logger.info("Voice msg from user %s: duration=%ss, file_id=%s", user_id, getattr(voice, 'duration', '?'), file_id[:20])
        elif document and document.mime_type and document.mime_type.startswith("audio/"):
            file_id = document.file_id
            mime_type = document.mime_type
            logger.info("Audio file from user %s: mime=%s, file_id=%s", user_id, mime_type, file_id[:20])

        if not file_id:
            return

        # Optional "Transcribing..." feedback (ephemeral — temporary status)
        if self.config.bot.show_transcribing_feedback:
            await self._send_private(update, context, "🎙️ Transcribing...")

        try:
            start_t = time.perf_counter()
            audio_data, mime = await self._download_audio(file_id)
            logger.debug("Audio downloaded: %d bytes, type=%s", len(audio_data), mime)

            result = await self.transcriber.transcribe(audio_data, mime)
            elapsed = time.perf_counter() - start_t

            if not result.text or result.text == "(no speech detected)":
                logger.warning("Transcription returned empty text for user %s", user_id)
                if self.config.bot.show_transcribing_feedback:
                    await self._send_private(update, context, "⚠️ Could not transcribe that audio.")
                return

            logger.info("Transcription OK: %d chars, %0.1fs total (lang=%s)", len(result.text), elapsed, result.language)

            # Store in history with transcription (replied=0, summarized=0 by default)
            await self.store.store_message(
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                user_id=user_id,
                username=update.effective_user.username or "",
                first_name=update.effective_user.first_name or "",
                voice_transcription=result.text,
                timestamp=message.date.timestamp(),
            )
            logger.debug("Stored transcription in history for chat %s", update.effective_chat.id)

            # Build reply text with optional header (no button)
            parts = []
            if self.config.bot.show_transcription_header:
                parts.append("📝 *Transcription:*")
            parts.append(result.text)
            reply_text = "\n".join(parts)

            safe_text = self._escape_markdown(reply_text) if reply_text.startswith("❌") or reply_text.startswith("⚠️") else reply_text
            await update.message.reply_text(
                text=safe_text,
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            logger.exception("Transcription failed for user %s", user_id)
            if self.config.bot.show_transcribing_feedback:
                await self._send_private(update, context, f"❌ Transcription failed: {str(e)}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Store text messages in history (all users, no auth required)."""
        if not update.effective_user:
            return
        if not update.message or not update.message.text:
            return

        message = update.message
        logger.debug("Storing text msg from user %s in chat %s: %.50s...",
                     update.effective_user.id, message.chat_id, message.text)
        await self.store.store_message(
            chat_id=message.chat_id,
            message_id=message.message_id,
            user_id=update.effective_user.id,
            username=update.effective_user.username or "",
            first_name=update.effective_user.first_name or "",
            text=message.text,
            reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
            timestamp=message.date.timestamp(),
        )

    async def cmd_transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /transcribe — force-transcribe backlog voice messages not yet transcribed."""
        user = update.effective_user
        if not user:
            return
        if self.config.bot.auth_required_summarize and not self._is_authorized(user.id):
            logger.warning("Unauthorized user %s tried /transcribe", user.id)
            await self._send_private(update, context, "❌ You are not authorized to use this command.")
            return

        chat_id = update.effective_chat.id
        logger.info("Command /transcribe from user %s in chat %s", user.id, chat_id)
        await self._send_private(update, context, "🎙️ Looking for untranscribed voice messages...")

        try:
            # Get voice messages that were stored as text but not transcribed
            conn = await self.store._get_conn()
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND voice_transcription IS NULL
                     AND (text IS NULL OR text = '')
                   ORDER BY timestamp ASC
                   LIMIT 10""",
                (chat_id,),
            )
            rows = await cursor.fetchall()
            untranscribed = [dict(row) for row in rows]

            if not untranscribed:
                await self._send_private(update, context, "✅ No untranscribed voice messages found.")
                return

            await self._send_private(update, context, f"🎙️ Found {len(untranscribed)} untranscribed messages. This feature requires audio file IDs stored in the database — backlog messages may not have downloadable audio. Try using the original voice message in Telegram instead.")

        except Exception as e:
            logger.exception("Transcribe failed for user %s", user.id)
            await self._send_private(update, context, f"❌ Transcribe failed: {str(e)}")

    async def cmd_summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /summarize — summarize all unsummarized voice transcriptions from others."""
        user = update.effective_user
        if not user:
            return

        if self.config.bot.auth_required_summarize and not self._is_authorized(user.id):
            logger.warning("Unauthorized user %s tried /summarize", user.id)
            await self._send_private(update, context, "❌ You are not authorized to use this command.")
            return

        chat_id = update.effective_chat.id
        max_age = self.config.history.summarize_retention_days
        logger.info("Command /summarize from user %s in chat %s (max_age=%d days)", user.id, chat_id, max_age)
        await self._send_private(update, context, "📊 Looking for unsummarized transcriptions from others...")

        try:
            # Exclude the requesting user's own messages — only summarize other people's
            transcriptions = await self.store.get_unsummarized_transcriptions(
                chat_id=chat_id,
                max_age_days=max_age,
                exclude_user_id=user.id,
            )

            if not transcriptions:
                logger.info("No unsummarized transcriptions found in chat %s", chat_id)
                await self._send_private(update, context, "✅ No unsummarized transcriptions to process.")
                return

            logger.debug("Found %d unsummarized transcriptions for summarization", len(transcriptions))
            formatted = format_messages_for_summary(transcriptions)
            prompt = self.config.bot.prompts.summary + "\n\n" + formatted

            start_t = time.perf_counter()
            client = self._get_llm_client()
            response = await client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes chat conversations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            # Log the raw response structure for debugging
            try:
                raw = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                logger.debug("LLM response raw: choices=%d, first_choice_type=%s",
                             len(response.choices),
                             type(response.choices[0].message).__name__ if response.choices else "none")
            except Exception:
                pass

            summary = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    summary = (choice.message.content or "").strip()
                elif hasattr(choice, 'text'):
                    summary = (choice.text or "").strip()

            elapsed = time.perf_counter() - start_t
            msg_count = len(transcriptions)

            if not summary:
                logger.warning("Summary returned empty — LLM response had no content (choices=%d)", len(response.choices))
                await self._send_private(update, context, "⚠️ LLM returned an empty summary. Try again or check the LLM endpoint.")
                return

            logger.info("Summary generated: %d msgs, %d chars, %0.1fs", msg_count, len(summary), elapsed)

            # Mark as summarized
            msg_ids = [m["id"] for m in transcriptions]
            await self.store.mark_as_summarized(msg_ids)

            reply = f"📊 *Summary of {msg_count} transcriptions:*\n\n{summary}"
            await self._send_private(update, context, reply)

        except Exception as e:
            logger.exception("Summary failed for user %s", user.id)
            await self._send_private(update, context, f"❌ Summary failed: {str(e)}")

    async def cmd_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reply — generate replies to all unreplied voice transcriptions from others."""
        user = update.effective_user
        if not user:
            return
        if self.config.bot.auth_required_reply and not self._is_authorized(user.id):
            logger.warning("Unauthorized user %s tried /reply", user.id)
            await self._send_private(update, context, "❌ You are not authorized to use this command.")
            return

        chat_id = update.effective_chat.id
        max_age = self.config.history.reply_retention_days
        logger.info("Command /reply from user %s in chat %s (max_age=%d days)", user.id, chat_id, max_age)
        await self._send_private(update, context, "💬 Looking for unreplied transcriptions from others...")

        try:
            # Exclude the requesting user's own messages — only reply to other people's
            transcriptions = await self.store.get_unreplied_transcriptions(
                chat_id=chat_id,
                max_age_days=max_age,
                exclude_user_id=user.id,
            )

            if not transcriptions:
                logger.info("No unreplied transcriptions found in chat %s", chat_id)
                await self._send_private(update, context, "✅ No unreplied transcriptions to process.")
                return

            # Build a prompt from the unreplied texts
            messages_text = []
            for msg in transcriptions:
                name = msg.get("first_name") or msg.get("username") or f"User {msg['user_id']}"
                ts = datetime.fromtimestamp(msg["timestamp"], tz=timezone.utc).strftime("%H:%M")
                text = msg["voice_transcription"]
                messages_text.append(f"[{ts}] {name}: {text}")

            combined = "\n".join(messages_text)
            user_message = " ".join(context.args) if context.args else "Respond to these messages naturally."
            system_prompt = self.config.bot.prompts.reply
            system_prompt += f"\n\nThe following transcriptions need a reply:\n{combined}"

            start_t = time.perf_counter()
            client = self._get_llm_client()
            response = await client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )

            # Log the raw response structure for debugging
            try:
                logger.debug("LLM reply response: choices=%d", len(response.choices) if response.choices else 0)
            except Exception:
                pass

            reply = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    reply = (choice.message.content or "").strip()
                elif hasattr(choice, 'text'):
                    reply = (choice.text or "").strip()

            elapsed = time.perf_counter() - start_t
            msg_count = len(transcriptions)

            if not reply:
                logger.warning("Reply returned empty — LLM response had no content (choices=%d)", len(response.choices))
                await self._send_private(update, context, "⚠️ LLM returned an empty reply. Try again or check the LLM endpoint.")
                return

            logger.info("Reply generated: %d msgs, %d chars, %0.1fs", msg_count, len(reply), elapsed)

            # Mark as replied
            msg_ids = [m["id"] for m in transcriptions]
            await self.store.mark_as_replied(msg_ids)

            await self._send_private(update, context, reply)

        except Exception as e:
            logger.exception("Reply failed for user %s", user.id)
            await self._send_private(update, context, f"❌ Reply failed: {str(e)}")

    # ---- Lifecycle ----

    async def post_init(self, application: Application) -> None:
        """Set bot commands for private and group chats."""
        self._app = application
        commands = [
            BotCommand("start", "Show welcome message", api_kwargs={"is_ephemeral": True}),
            BotCommand("help", "Show this help", api_kwargs={"is_ephemeral": True}),
            BotCommand("summarize", "Summarize unsummarized transcriptions from others", api_kwargs={"is_ephemeral": True}),
            BotCommand("reply", "Reply to unreplied transcriptions from others", api_kwargs={"is_ephemeral": True}),
            BotCommand("transcribe", "Transcribe backlog voice messages", api_kwargs={"is_ephemeral": True}),
        ]
        await application.bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(commands)
        logger.info("Bot commands registered")

        # Start background task to watch for hot-reload signals
        application.create_task(self._reload_watcher(), "reload-watcher")

    async def _reload_watcher(self) -> None:
        """Periodically check for reload signal and hot-reload transcriber."""
        global _reload_requested
        while True:
            await asyncio.sleep(5)
            if _reload_requested:
                _reload_requested = False
                logger.info("Hot-reload signal received, reloading transcriber...")
                try:
                    # Reload config from disk
                    self.config = AppConfig.load()
                    # Re-create transcriber
                    old = self.transcriber
                    await old.close()
                    self.transcriber = create_transcriber(self.config)
                    # Reset LLM client so it picks up new config
                    self._llm_client = None
                    logger.info("Transcriber hot-reloaded: engine=%s, model=%s",
                                 self.config.transcription.engine, self.config.transcription.model)
                except Exception as e:
                    logger.exception("Hot-reload failed: %s", e)

    def run(self) -> None:
        """Run the bot with polling."""
        app = (
            ApplicationBuilder()
            .token(self.config.telegram_bot_token)
            .post_init(self.post_init)
            .concurrent_updates(True)
            .build()
        )

        # Voice/audio handlers
        app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO | filters.Document.AUDIO |
            filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
            self.handle_voice,
        ))

        # Text message handler (store history)
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_text,
        ))

        # Command handlers
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("summarize", self.cmd_summarize))
        app.add_handler(CommandHandler("reply", self.cmd_reply))
        app.add_handler(CommandHandler("transcribe", self.cmd_transcribe))

        # Error handler
        app.add_error_handler(error_handler)

        # Run
        logger.info("Starting bot polling (Telegram API)...")
        app.run_polling()