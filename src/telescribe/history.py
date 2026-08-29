"""Group message history stored in SQLite — stores all messages for summarization and reply tracking."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from telescribe.logger import get_logger

logger = get_logger("history")


class MessageStore:
    """Async SQLite store for group chat messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        logger.info("Message store initialized: %s", self.db_path)

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            logger.debug("Opening SQLite connection: %s", self.db_path)
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            await self._init_db()
        return self._conn

    async def _init_db(self) -> None:
        conn = await self._get_conn()
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT,
                first_name TEXT,
                text TEXT,
                voice_transcription TEXT,
                reply_to_message_id INTEGER,
                is_topic BOOLEAN DEFAULT 0,
                topic_id INTEGER,
                timestamp REAL NOT NULL,
                replied BOOLEAN DEFAULT 0,
                summarized BOOLEAN DEFAULT 0,
                UNIQUE(chat_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_time
                ON messages(chat_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_chat_user
                ON messages(chat_id, user_id);
        """)
        # Migrate: add columns if they don't exist (for existing databases)
        try:
            await conn.execute("ALTER TABLE messages ADD COLUMN replied BOOLEAN DEFAULT 0")
        except aiosqlite.OperationalError:
            pass  # column already exists
        try:
            await conn.execute("ALTER TABLE messages ADD COLUMN summarized BOOLEAN DEFAULT 0")
        except aiosqlite.OperationalError:
            pass  # column already exists
        await conn.commit()
        logger.debug("Database schema initialized")

    async def store_message(
        self,
        chat_id: int,
        message_id: int,
        user_id: int,
        text: str = "",
        username: str = "",
        first_name: str = "",
        voice_transcription: str = "",
        reply_to_message_id: Optional[int] = None,
        is_topic: bool = False,
        topic_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Store a message in history."""
        conn = await self._get_conn()
        ts = timestamp or time.time()

        has_text = bool(text.strip())
        has_transcription = bool(voice_transcription.strip())
        source = "voice_xcript" if has_transcription else "text" if has_text else "other"

        await conn.execute(
            """INSERT OR REPLACE INTO messages
               (chat_id, message_id, user_id, username, first_name, text,
                voice_transcription, reply_to_message_id, is_topic, topic_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, message_id, user_id, username, first_name, text,
             voice_transcription, reply_to_message_id, is_topic, topic_id, ts),
        )
        await conn.commit()
        logger.debug("Stored %s msg: chat=%s, user=%s, msg=%s (%d chars)",
                      source, chat_id, user_id, message_id, len(text or voice_transcription))

    async def get_messages_since(
        self,
        chat_id: int,
        since: float,
        limit: int = 1000,
        user_id: Optional[int] = None,
    ) -> list[dict]:
        """Get messages in a chat since a timestamp."""
        conn = await self._get_conn()

        if user_id:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ? AND timestamp >= ? AND user_id = ?
                   ORDER BY timestamp ASC LIMIT ?""",
                (chat_id, since, user_id, limit),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ? AND timestamp >= ?
                   ORDER BY timestamp ASC LIMIT ?""",
                (chat_id, since, limit),
            )

        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Retrieved %d messages since ts=%s from chat %s", len(result), since, chat_id)
        return result

    async def get_recent_messages(
        self,
        chat_id: int,
        limit: int = 50,
    ) -> list[dict]:
        """Get most recent messages."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM messages
               WHERE chat_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Retrieved %d recent messages from chat %s", len(result), chat_id)
        return result

    async def get_voice_transcriptions_since(
        self,
        chat_id: int,
        since: float,
        exclude_user_id: Optional[int] = None,
    ) -> list[dict]:
        """Get voice transcriptions since a timestamp, optionally excluding a user."""
        conn = await self._get_conn()
        if exclude_user_id:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND timestamp >= ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                     AND user_id != ?
                   ORDER BY timestamp ASC""",
                (chat_id, since, exclude_user_id),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND timestamp >= ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                   ORDER BY timestamp ASC""",
                (chat_id, since),
            )
        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Retrieved %d voice transcriptions since ts=%s from chat %s", len(result), since, chat_id)
        return result

    async def get_unreplied_transcriptions(
        self,
        chat_id: int,
        max_age_days: int = 30,
        exclude_user_id: Optional[int] = None,
    ) -> list[dict]:
        """Get voice transcriptions that haven't been replied to, within max_age_days."""
        conn = await self._get_conn()
        cutoff = time.time() - (max_age_days * 86400)
        if exclude_user_id:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                     AND replied = 0
                     AND user_id != ?
                     AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (chat_id, exclude_user_id, cutoff),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                     AND replied = 0
                     AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (chat_id, cutoff),
            )
        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Found %d unreplied transcriptions in chat %s", len(result), chat_id)
        return result

    async def get_unsummarized_transcriptions(
        self,
        chat_id: int,
        max_age_days: int = 30,
        exclude_user_id: Optional[int] = None,
    ) -> list[dict]:
        """Get voice transcriptions that haven't been summarized, within max_age_days."""
        conn = await self._get_conn()
        cutoff = time.time() - (max_age_days * 86400)
        if exclude_user_id:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                     AND summarized = 0
                     AND user_id != ?
                     AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (chat_id, exclude_user_id, cutoff),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM messages
                   WHERE chat_id = ?
                     AND voice_transcription IS NOT NULL
                     AND voice_transcription != ''
                     AND summarized = 0
                     AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (chat_id, cutoff),
            )
        rows = await cursor.fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Found %d unsummarized transcriptions in chat %s", len(result), chat_id)
        return result

    async def mark_as_replied(self, message_ids: list[int]) -> None:
        """Mark messages as replied."""
        if not message_ids:
            return
        conn = await self._get_conn()
        placeholders = ",".join("?" for _ in message_ids)
        await conn.execute(
            f"UPDATE messages SET replied = 1 WHERE id IN ({placeholders})",
            message_ids,
        )
        await conn.commit()
        logger.debug("Marked %d messages as replied", len(message_ids))

    async def mark_as_summarized(self, message_ids: list[int]) -> None:
        """Mark messages as summarized."""
        if not message_ids:
            return
        conn = await self._get_conn()
        placeholders = ",".join("?" for _ in message_ids)
        await conn.execute(
            f"UPDATE messages SET summarized = 1 WHERE id IN ({placeholders})",
            message_ids,
        )
        await conn.commit()
        logger.debug("Marked %d messages as summarized", len(message_ids))

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.debug("SQLite connection closed")

    async def prune_old_messages(self, retention_days: int = 90) -> int:
        """Delete messages older than retention_days. Returns count deleted."""
        conn = await self._get_conn()
        cutoff = time.time() - (retention_days * 86400)
        cursor = await conn.execute(
            "DELETE FROM messages WHERE timestamp < ?",
            (cutoff,),
        )
        await conn.commit()
        deleted = cursor.rowcount
        logger.info("Pruned %d messages older than %d days", deleted, retention_days)
        return deleted

    async def get_message_count(self) -> int:
        """Get total number of stored messages across all chats."""
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT COUNT(*) as cnt FROM messages")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0