"""Pluggable transcription engine — local (faster-whisper), Moonshine, Parakeet, and remote (OpenAI-compatible)."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from telescribe.config import AppConfig
from telescribe.logger import get_logger

logger = get_logger("transcriber")


class TranscriptionResult:
    """Result of a transcription."""

    def __init__(self, text: str, language: str = "en", duration_seconds: float = 0.0):
        self.text = text
        self.language = language
        self.duration_seconds = duration_seconds


class BaseTranscriber(ABC):
    """Abstract base for all transcription backends."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...


class FasterWhisperTranscriber(BaseTranscriber):
    """Local transcription via faster-whisper (CTranslate2)."""

    def __init__(self, config: AppConfig):
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

            logger.info(
                "Loading faster-whisper model: %s (device=%s, compute=%s)",
                self.config.transcription.model,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                self.config.transcription.model,
                device=device,
                compute_type=compute_type,
            )
        return self._model

    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        """Transcribe using local faster-whisper model."""
        model = self._get_model()

        suffix = _mime_to_ext(mime_type)
        tmp_path = _write_temp_audio(audio_data, suffix)

        try:
            start_t = time.perf_counter()
            segments, info = model.transcribe(
                tmp_path,
                beam_size=self.config.transcription.beam_size,
                best_of=self.config.transcription.best_of,
                patience=self.config.transcription.patience,
                length_penalty=self.config.transcription.length_penalty,
                repetition_penalty=self.config.transcription.repetition_penalty,
                no_repeat_ngram_size=self.config.transcription.no_repeat_ngram_size,
                temperature=self.config.transcription.temperature,
                suppress_blank=self.config.transcription.suppress_blank,
                condition_on_previous_text=self.config.transcription.condition_on_previous_text,
                vad_filter=self.config.transcription.vad_filter,
                no_speech_threshold=self.config.transcription.no_speech_threshold,
                log_prob_threshold=self.config.transcription.log_prob_threshold,
                compression_ratio_threshold=self.config.transcription.compression_ratio_threshold,
                language=self.config.transcription.language,
            )

            text_parts = []
            for seg in segments:
                text_parts.append(seg.text)

            text = " ".join(text_parts).strip()
            elapsed = time.perf_counter() - start_t
            logger.info("faster-whisper: %0.1fs audio in %0.1fs (lang=%s)", info.duration, elapsed, info.language)

            return TranscriptionResult(text=text, language=info.language, duration_seconds=info.duration)
        finally:
            _cleanup_temp(tmp_path)

    async def close(self) -> None:
        self._model = None


class MoonshineTranscriber(BaseTranscriber):
    """Transcription via Moonshine — edge-optimized ASR, runs on CPU.

    Uses the Python API directly: get_model_for_language() to download and
    cache models, Transcriber.transcribe_without_streaming() for offline
    transcription. Extremely low latency (~100ms).
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._transcriber = None
        self._model_path = None
        self._model_arch = None

    def _load(self):
        if self._transcriber is not None:
            return

        from moonshine_voice import Transcriber, get_model_for_language

        lang = self.config.transcription.language or "en"
        logger.info("Loading Moonshine model for language=%s", lang)
        self._model_path, self._model_arch = get_model_for_language(lang)
        self._transcriber = Transcriber(
            model_path=self._model_path,
            model_arch=self._model_arch,
        )
        logger.info("Moonshine model loaded: path=%s, arch=%s", self._model_path, self._model_arch)

    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        import numpy as np
        import soundfile as sf

        self._load()

        suffix = _mime_to_ext(mime_type)
        tmp_audio = _write_temp_audio(audio_data, suffix)
        tmp_wav = tmp_audio + ".wav"

        try:
            # Convert to 16kHz mono WAV
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_audio, "-ar", "16000", "-ac", "1",
                 "-sample_fmt", "s16", tmp_wav],
                capture_output=True, check=True,
            )

            # Load with Moonshine's load_wav_file
            from moonshine_voice import load_wav_file
            audio_data_f, sample_rate = load_wav_file(tmp_wav)
            duration = len(audio_data_f) / sample_rate

            start_t = time.perf_counter()

            # Use offline transcription (non-streaming) for best accuracy
            transcript = self._transcriber.transcribe_without_streaming(
                audio_data_f, sample_rate=sample_rate, flags=0
            )

            elapsed = time.perf_counter() - start_t

            # Collect all lines
            lines = []
            for line in transcript.lines:
                if line.text and line.text.strip():
                    lines.append(line.text.strip())
            text = "\n".join(lines)

            logger.info("Moonshine: %0.1fs audio in %0.1fs (rtf=%.2f), %d segments",
                         duration, elapsed, elapsed / max(duration, 0.01), len(lines))

            return TranscriptionResult(text=text or "(no speech detected)", duration_seconds=duration)

        finally:
            _cleanup_temp(tmp_audio)
            _cleanup_temp(tmp_wav)

    async def close(self) -> None:
        self._transcriber = None


class ParakeetTranscriber(BaseTranscriber):
    """Transcription via NVIDIA Parakeet TDT model through sherpa-onnx.

    Parakeet is one of the most accurate English ASR models (~6.3% WER).
    Uses offline ONNX runtime — no GPU needed, extremely fast on CPU.
    Auto-downloads model files on first use.
    """

    MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    MODEL_NAME = "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8"

    def __init__(self, config: AppConfig):
        self.config = config
        self._recognizer = None
        self._model_dir: Optional[Path] = None

    def _get_model_dir(self) -> Path:
        """Ensure model is downloaded and return its directory."""
        import urllib.request
        import tarfile
        import shutil

        cache_dir = Path(self.config.data_dir) / "models" / self.MODEL_NAME
        if cache_dir.exists() and (cache_dir / "tokens.txt").exists():
            return cache_dir

        logger.info("Downloading Parakeet model to %s...", cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir.with_suffix(".tar.bz2")

        url = f"{self.MODEL_URL}/{self.MODEL_NAME}.tar.bz2"
        urllib.request.urlretrieve(url, archive)

        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache_dir.parent)

        archive.unlink()
        logger.info("Parakeet model downloaded")
        return cache_dir

    def _load(self):
        if self._recognizer is not None:
            return

        import sherpa_onnx

        model_dir = self._get_model_dir()
        # Find the model file - could be model.onnx, model.int8.onnx, etc.
        model_files = list(model_dir.glob("model*.onnx"))
        model_path = str(model_files[0]) if model_files else str(model_dir / "model.int8.onnx")
        tokens = str(model_dir / "tokens.txt")

        logger.info("Loading Parakeet model from %s (model=%s)", model_dir, model_path)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model_path,
            tokens=tokens,
            debug=False,
            num_threads=int(os.getenv("OMP_NUM_THREADS", "2")),
        )
        self._model_dir = model_dir
        logger.info("Parakeet model loaded")

    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
        import numpy as np
        import soundfile as sf

        self._load()

        suffix = _mime_to_ext(mime_type)
        tmp_audio = _write_temp_audio(audio_data, suffix)
        tmp_wav = tmp_audio + ".wav"

        try:
            # Convert to 16kHz mono WAV
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_audio, "-ar", "16000", "-ac", "1",
                 "-sample_fmt", "s16", tmp_wav],
                capture_output=True, check=True,
            )

            audio, sample_rate = sf.read(tmp_wav, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            duration = audio.shape[-1] / sample_rate

            start_t = time.perf_counter()
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            self._recognizer.decode_stream(stream)
            result = stream.result.text.strip()
            elapsed = time.perf_counter() - start_t

            logger.info("Parakeet: %0.1fs audio in %0.1fs (rtf=%.2f)", duration, elapsed, elapsed / max(duration, 0.01))
            return TranscriptionResult(text=result or "(no speech detected)", duration_seconds=duration)

        finally:
            _cleanup_temp(tmp_audio)
            _cleanup_temp(tmp_wav)

    async def close(self) -> None:
        self._recognizer = None


class OpenAITranscriber(BaseTranscriber):
    """Transcription via OpenAI-compatible API (OpenAI, Groq, custom endpoint)."""

    def __init__(self, config: AppConfig, api_key: str, base_url: str, model: str = "whisper-1"):
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

    async def transcribe(self, audio_data: bytes, mime_type: str = "") -> TranscriptionResult:
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
            return TranscriptionResult(text=transcript.strip())
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


# ---- Factory ----

def create_transcriber(config: AppConfig) -> BaseTranscriber:
    """Factory: create the right transcriber based on config."""
    engine = config.transcription.engine

    if engine == "local":
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