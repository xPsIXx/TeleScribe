"""Pluggable transcription engine — local (faster-whisper), Moonshine, Parakeet, and remote (OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

from telescribe.config import AppConfig
from telescribe.logger import get_logger, log_failure

logger = get_logger("transcriber")

MOONSHINE_LANGUAGES = {"en", "es", "zh", "ja", "ko", "vi", "ar", "uk"}
MOONSHINE_CHUNK_S = 20.0
MOONSHINE_OVERLAP_S = 1.5
PARAKEET_CHUNK_S = 60.0
PARAKEET_OVERLAP_S = 2.0


class TranscriptionResult:
    """Result of a transcription."""

    def __init__(self, text: str, language: str = "en", duration_seconds: float = 0.0):
        self.text = text
        self.language = language
        self.duration_seconds = duration_seconds


class BaseTranscriber(ABC):
    """Abstract base for all transcription backends.

    `transcribe()` serializes calls (ASR runtimes are not thread-safe) and
    subclasses implement `_transcribe_impl`.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        started = time.perf_counter()
        logger.info(
            "ASR begin class=%s bytes=%d mime=%s",
            type(self).__name__,
            len(audio_data or b""),
            mime_type or "?",
        )
        try:
            async with self._lock:
                result = await self._transcribe_impl(audio_data, mime_type)
        except Exception as e:
            log_failure(
                logger,
                "asr",
                e,
                class_=type(self).__name__,
                mime=mime_type or "?",
                bytes=len(audio_data or b""),
                elapsed_s=round(time.perf_counter() - started, 2),
            )
            raise
        elapsed = time.perf_counter() - started
        empty = not result.text or result.text == "(no speech detected)"
        if empty:
            logger.warning(
                "ASR empty class=%s mime=%s bytes=%d audio_s=%.1f elapsed_s=%.1f",
                type(self).__name__,
                mime_type or "?",
                len(audio_data or b""),
                result.duration_seconds,
                elapsed,
            )
        else:
            logger.info(
                "ASR ok class=%s chars=%d lang=%s audio_s=%.1f elapsed_s=%.1f",
                type(self).__name__,
                len(result.text),
                result.language,
                result.duration_seconds,
                elapsed,
            )
        return result

    @abstractmethod
    async def _transcribe_impl(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        """Transcribe audio bytes to text. Called with the instance lock held."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...


class FasterWhisperTranscriber(BaseTranscriber):
    """Local transcription via faster-whisper (CTranslate2)."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.config.transcription.device
            compute_type = self.config.transcription.compute_type

            # float16 is not supported on CPU — fall back to int8
            if device == "cpu" and compute_type == "float16":
                logger.warning(
                    "float16 compute type not supported on CPU, falling back to int8"
                )
                compute_type = "int8"

            download_root = str(Path(self.config.data_dir) / "models")
            Path(download_root).mkdir(parents=True, exist_ok=True)

            logger.info(
                "Loading faster-whisper model: %s (device=%s, compute=%s, download_root=%s)",
                self.config.transcription.model,
                device,
                compute_type,
                download_root,
            )
            try:
                self._model = WhisperModel(
                    self.config.transcription.model,
                    device=device,
                    compute_type=compute_type,
                    download_root=download_root,
                )
            except Exception as e:
                log_failure(
                    logger,
                    "whisper.load",
                    e,
                    model=self.config.transcription.model,
                    device=device,
                    compute=compute_type,
                    download_root=download_root,
                )
                raise
        return self._model

    async def _transcribe_impl(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_data, mime_type)

    def _transcribe_sync(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        model = self._get_model()
        cfg = self.config.transcription

        suffix = _mime_to_ext(mime_type)
        tmp_path = _write_temp_audio(audio_data, suffix)

        try:
            start_t = time.perf_counter()
            model_name = cfg.model or ""
            is_distil = "distil" in model_name.lower()
            # Distil-whisper degrades with previous-text conditioning on long audio.
            condition = False if is_distil else cfg.condition_on_previous_text
            temperature: float | list[float] = cfg.temperature
            if temperature == 0 or temperature == 0.0:
                temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

            vad_parameters = None
            if cfg.vad_filter:
                vad_parameters = {
                    "min_silence_duration_ms": cfg.vad_min_silence_ms,
                    "speech_pad_ms": cfg.vad_speech_pad_ms,
                    "threshold": cfg.vad_threshold,
                }

            h_thresh = cfg.hallucination_silence_threshold or None
            kwargs = dict(
                beam_size=cfg.beam_size,
                best_of=cfg.best_of,
                patience=cfg.patience,
                length_penalty=cfg.length_penalty,
                repetition_penalty=cfg.repetition_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                temperature=temperature,
                suppress_blank=cfg.suppress_blank,
                condition_on_previous_text=condition,
                vad_filter=cfg.vad_filter,
                vad_parameters=vad_parameters,
                no_speech_threshold=cfg.no_speech_threshold,
                log_prob_threshold=cfg.log_prob_threshold,
                compression_ratio_threshold=cfg.compression_ratio_threshold,
                initial_prompt=cfg.initial_prompt,
                language=cfg.language,
                hallucination_silence_threshold=h_thresh,
                word_timestamps=bool(h_thresh),
            )

            segments, info = model.transcribe(tmp_path, **kwargs)

            text_parts = []
            last_end = 0.0
            for seg in segments:
                text_parts.append(seg.text)
                if getattr(seg, "end", None) is not None:
                    last_end = max(last_end, float(seg.end))

            text = " ".join(text_parts).strip()
            elapsed = time.perf_counter() - start_t
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            coverage = (last_end / duration) if duration > 0 else 1.0
            logger.info(
                "faster-whisper: %0.1fs audio in %0.1fs (lang=%s, last_seg=%.1fs, coverage=%.0f%%, cot=%s, vad=%s)",
                duration,
                elapsed,
                getattr(info, "language", "?"),
                last_end,
                coverage * 100,
                condition,
                cfg.vad_filter,
            )
            if duration > 5 and coverage < 0.85:
                logger.warning(
                    "Transcription coverage low (%.0f%%) — audio may have been cut off at silence",
                    coverage * 100,
                )

            return TranscriptionResult(
                text=text,
                language=getattr(info, "language", "en") or "en",
                duration_seconds=duration,
            )
        finally:
            _cleanup_temp(tmp_path)

    async def close(self) -> None:
        self._model = None


class MoonshineTranscriber(BaseTranscriber):
    """Transcription via Moonshine — edge-optimized ASR, runs on CPU.

    Moonshine is trained for short utterances (~30s). Long Telegram voice
    notes are chunked with overlap so pauses don't drop the rest of the clip.
    Language comes from the dashboard model card (en/es/...) falling back to
    the language field.
    """

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._transcriber = None
        self._model_path = None
        self._model_arch = None
        self._loaded_lang = None

    def _language(self) -> str:
        model = (self.config.transcription.model or "").strip().lower()
        lang = (self.config.transcription.language or "").strip().lower()
        if model in MOONSHINE_LANGUAGES:
            return model
        if lang in MOONSHINE_LANGUAGES:
            return lang
        return "en"

    def _load(self):
        lang = self._language()
        cfg = self.config.transcription
        vad_key = (
            lang,
            bool(cfg.moonshine_vad),
            float(cfg.moonshine_vad_threshold),
            float(cfg.moonshine_vad_max_segment),
        )
        if self._transcriber is not None and getattr(self, "_loaded_vad_key", None) == vad_key:
            return

        from moonshine_voice import Transcriber, get_model_for_language

        if cfg.moonshine_vad:
            options = {
                "vad_threshold": str(cfg.moonshine_vad_threshold),
                "vad_look_behind_sample_count": "8192",
                "vad_max_segment_duration": str(cfg.moonshine_vad_max_segment),
            }
        else:
            # Treat the whole clip as speech so pauses are not dropped.
            options = {
                "vad_threshold": "0.0",
                "vad_look_behind_sample_count": "8192",
                "vad_max_segment_duration": "3600.0",
            }

        logger.info(
            "Loading Moonshine language=%s vad=%s threshold=%s max_seg=%s",
            lang, cfg.moonshine_vad, options["vad_threshold"], options["vad_max_segment_duration"],
        )
        try:
            self._model_path, self._model_arch = get_model_for_language(lang)
            self._transcriber = Transcriber(
                model_path=self._model_path,
                model_arch=self._model_arch,
                options=options,
            )
        except Exception as e:
            log_failure(logger, "moonshine.load", e, language=lang, vad=cfg.moonshine_vad)
            raise
        self._loaded_lang = lang
        self._loaded_vad_key = vad_key
        logger.info("Moonshine model loaded: path=%s, arch=%s", self._model_path, self._model_arch)

    async def _transcribe_impl(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_data, mime_type)

    def _transcribe_sync(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        self._load()

        suffix = _mime_to_ext(mime_type)
        tmp_audio = _write_temp_audio(audio_data, suffix)
        tmp_wav = tmp_audio + ".wav"

        try:
            _ffmpeg_to_wav(tmp_audio, tmp_wav)

            from moonshine_voice import load_wav_file
            audio_data_f, sample_rate = load_wav_file(tmp_wav)
            duration = len(audio_data_f) / sample_rate

            start_t = time.perf_counter()
            chunk_texts: list[str] = []
            for i, chunk in enumerate(_iter_audio_chunks(audio_data_f, sample_rate, MOONSHINE_CHUNK_S, MOONSHINE_OVERLAP_S)):
                try:
                    transcript = self._transcriber.transcribe_without_streaming(
                        chunk, sample_rate=sample_rate, flags=0
                    )
                except Exception as e:
                    logger.warning(
                        "Moonshine chunk %d failed (samples=%d): %s",
                        i, len(chunk), e, exc_info=True,
                    )
                    continue
                lines = []
                for line in transcript.lines:
                    if line.text and line.text.strip():
                        lines.append(line.text.strip())
                if lines:
                    chunk_texts.append(" ".join(lines))

            if not chunk_texts:
                logger.warning("Moonshine produced no text from %d samples (%.1fs)", len(audio_data_f), duration)

            text = _join_chunk_texts(chunk_texts)
            elapsed = time.perf_counter() - start_t
            logger.info(
                "Moonshine: %0.1fs audio in %0.1fs (rtf=%.2f), %d chunks, lang=%s",
                duration,
                elapsed,
                elapsed / max(duration, 0.01),
                len(chunk_texts),
                self._loaded_lang,
            )
            return TranscriptionResult(
                text=text or "(no speech detected)",
                language=self._loaded_lang or "en",
                duration_seconds=duration,
            )
        finally:
            _cleanup_temp(tmp_audio)
            _cleanup_temp(tmp_wav)

    async def close(self) -> None:
        self._transcriber = None
        self._loaded_lang = None


class ParakeetTranscriber(BaseTranscriber):
    """NVIDIA Parakeet TDT 0.6B v2 through sherpa-onnx (nemo_transducer).

    This is the real ~0.6B TDT model (not the 110M CTC distill). INT8 ONNX,
    CPU-friendly. Long audio is decoded in overlapping windows so the tail
    is not dropped.
    """

    MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._recognizer = None
        self._model_dir: Optional[Path] = None

    def _get_model_dir(self) -> Path:
        """Ensure model is downloaded and return its directory."""
        import tarfile
        import urllib.request

        cache_dir = Path(self.config.data_dir) / "models" / self.MODEL_NAME
        if cache_dir.exists() and (cache_dir / "tokens.txt").exists() and _parakeet_encoder(cache_dir):
            return cache_dir

        logger.info("Downloading Parakeet TDT 0.6B v2 to %s...", cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Path.with_suffix would turn "...0.6b-v2-int8" into "...0.tar.bz2"
        archive = cache_dir.parent / f"{self.MODEL_NAME}.tar.bz2"

        url = f"{self.MODEL_URL}/{self.MODEL_NAME}.tar.bz2"
        try:
            urllib.request.urlretrieve(url, archive)

            with tarfile.open(archive, "r:bz2") as tar:
                try:
                    tar.extractall(path=cache_dir.parent, filter="data")
                except TypeError:
                    tar.extractall(path=cache_dir.parent)
        except Exception as e:
            log_failure(logger, "parakeet.download", e, url=url, dest=str(cache_dir))
            raise
        finally:
            archive.unlink(missing_ok=True)
        logger.info("Parakeet model downloaded")
        return cache_dir

    def _load(self):
        if self._recognizer is not None:
            return

        import sherpa_onnx

        model_dir = self._get_model_dir()
        encoder_path = _parakeet_encoder(model_dir)
        if encoder_path is None:
            raise FileNotFoundError(
                f"Parakeet encoder not found in {model_dir}. Re-download the TDT 0.6B v2 model."
            )
        encoder = str(encoder_path)
        decoder = str(_first_existing(model_dir, ["decoder.int8.onnx", "decoder.onnx"]))
        joiner = str(_first_existing(model_dir, ["joiner.int8.onnx", "joiner.onnx"]))
        tokens = str(model_dir / "tokens.txt")

        logger.info(
            "Loading Parakeet TDT from %s (encoder=%s)",
            model_dir,
            encoder,
        )
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                model_type="nemo_transducer",
                debug=False,
                num_threads=int(os.getenv("OMP_NUM_THREADS", "2")),
                sample_rate=16000,
                feature_dim=80,
            )
        except Exception as e:
            log_failure(
                logger,
                "parakeet.load",
                e,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
            )
            raise
        self._model_dir = model_dir
        logger.info("Parakeet TDT 0.6B v2 loaded")

    async def _transcribe_impl(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_data, mime_type)

    def _transcribe_sync(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        import soundfile as sf

        self._load()

        suffix = _mime_to_ext(mime_type)
        tmp_audio = _write_temp_audio(audio_data, suffix)
        tmp_wav = tmp_audio + ".wav"

        try:
            _ffmpeg_to_wav(tmp_audio, tmp_wav)

            audio, sample_rate = sf.read(tmp_wav, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            duration = audio.shape[-1] / sample_rate

            start_t = time.perf_counter()
            chunk_texts: list[str] = []
            for i, chunk in enumerate(_iter_audio_chunks(audio, sample_rate, PARAKEET_CHUNK_S, PARAKEET_OVERLAP_S)):
                try:
                    stream = self._recognizer.create_stream()
                    stream.accept_waveform(sample_rate, chunk)
                    self._recognizer.decode_stream(stream)
                    piece = (stream.result.text or "").strip()
                except Exception as e:
                    logger.warning(
                        "Parakeet chunk %d failed (samples=%d): %s",
                        i, len(chunk), e, exc_info=True,
                    )
                    continue
                if piece:
                    chunk_texts.append(piece)

            if not chunk_texts:
                logger.warning("Parakeet produced no text from %.1fs audio", duration)

            result = _join_chunk_texts(chunk_texts)
            elapsed = time.perf_counter() - start_t
            logger.info(
                "Parakeet: %0.1fs audio in %0.1fs (rtf=%.2f), %d chunks",
                duration,
                elapsed,
                elapsed / max(duration, 0.01),
                len(chunk_texts),
            )
            return TranscriptionResult(
                text=result or "(no speech detected)",
                duration_seconds=duration,
            )
        finally:
            _cleanup_temp(tmp_audio)
            _cleanup_temp(tmp_wav)

    async def close(self) -> None:
        self._recognizer = None


class OpenAITranscriber(BaseTranscriber):
    """Transcription via OpenAI-compatible API (OpenAI, Groq, custom endpoint)."""

    def __init__(self, config: AppConfig, api_key: str, base_url: str, model: str = "whisper-1"):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def _transcribe_impl(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        client = self._get_client()
        suffix = _mime_to_ext(mime_type)
        tmp_path = _write_temp_audio(audio_data, suffix)

        try:
            with open(tmp_path, "rb") as f:
                transcript = await client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    response_format="text",
                )
            return TranscriptionResult(text=str(transcript).strip())
        finally:
            _cleanup_temp(tmp_path)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class GroqTranscriber(OpenAITranscriber):
    """Groq-optimized transcription using whisper-large-v3-turbo."""

    def __init__(self, config: AppConfig, api_key: str):
        super().__init__(
            config=config,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model="whisper-large-v3-turbo",
        )


# ---- Helpers ----

_MIME_EXT_MAP = {
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def _mime_to_ext(mime_type: str) -> str:
    return _MIME_EXT_MAP.get(mime_type, ".ogg")


def _write_temp_audio(data: bytes, suffix: str) -> str:
    """Write audio bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


def _cleanup_temp(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _ffmpeg_to_wav(src: str, dst: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-800:]
        log_failure(logger, "ffmpeg", RuntimeError(err), src=src, dst=dst, rc=result.returncode)
        raise RuntimeError(f"ffmpeg failed converting {src}: {err}")


def _iter_audio_chunks(audio, sample_rate: int, chunk_s: float, overlap_s: float):
    """Yield overlapping windows of a 1-D audio array."""
    n = len(audio)
    chunk = max(int(chunk_s * sample_rate), 1)
    hop = max(int((chunk_s - overlap_s) * sample_rate), 1)
    if n <= chunk:
        yield audio
        return
    start = 0
    while start < n:
        end = min(start + chunk, n)
        yield audio[start:end]
        if end >= n:
            break
        start += hop


def _join_chunk_texts(parts: Iterable[str]) -> str:
    """Join window transcripts, dropping duplicated overlap words."""
    out = ""
    for part in parts:
        p = (part or "").strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        prev_words = out.split()
        new_words = p.split()
        max_k = min(16, len(prev_words), len(new_words))
        k = 0
        for n in range(max_k, 0, -1):
            if prev_words[-n:] == new_words[:n]:
                k = n
                break
        rest = " ".join(new_words[k:])
        if rest:
            out = f"{out} {rest}"
    return out.strip()


def _first_existing(directory: Path, names: list[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return directory / names[0]


def _parakeet_encoder(model_dir: Path) -> Optional[Path]:
    for name in ("encoder.int8.onnx", "encoder.onnx"):
        path = model_dir / name
        if path.exists():
            return path
    return None


# ---- Factory ----

def create_transcriber(config: AppConfig) -> BaseTranscriber:
    """Factory: create the right transcriber based on config."""
    engine = config.transcription.engine
    model = config.transcription.model or ""

    if engine == "local":
        if model in MOONSHINE_LANGUAGES or model.startswith("sherpa-onnx"):
            logger.warning(
                "Model %r is not a Whisper checkpoint; using distil-large-v3",
                model,
            )
            config.transcription.model = "distil-large-v3"
        return FasterWhisperTranscriber(config)
    elif engine == "moonshine":
        return MoonshineTranscriber(config)
    elif engine == "parakeet":
        return ParakeetTranscriber(config)
    elif engine == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        return OpenAITranscriber(config, api_key=api_key, base_url="https://api.openai.com/v1")
    elif engine == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        return GroqTranscriber(config, api_key=api_key)
    elif engine == "custom":
        api_key = os.getenv("ASR_API_KEY", "")
        base_url = os.getenv("ASR_BASE_URL", "")
        return OpenAITranscriber(
            config,
            api_key=api_key,
            base_url=base_url,
            model=config.transcription.model,
        )
    else:
        raise ValueError(f"Unknown ASR engine: {engine}")
