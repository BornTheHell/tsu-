# tts.py — Модуль "Голос": Silero TTS (основной) + edge-tts (fallback)
from __future__ import annotations

import io
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import config
from logger import get_logger

log = get_logger("TTS")


# ════════════════════════════════════════════════════════════════════════════════
#  Базовый интерфейс
# ════════════════════════════════════════════════════════════════════════════════

class BaseTTS(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Произносит текст (блокирующий вызов)."""

    @abstractmethod
    def speak_async(self, text: str) -> threading.Thread:
        """Произносит текст в отдельном потоке."""


# ════════════════════════════════════════════════════════════════════════════════
#  Silero TTS (локальный, offline, быстрый)
# ════════════════════════════════════════════════════════════════════════════════

class SileroTTS(BaseTTS):
    """
    Использует Silero v3 через torch.hub.
    Модели скачиваются автоматически при первом запуске (~50 MB).

    Установка: pip install torch torchaudio
    """

    def __init__(self):
        import torch

        log.info("Загружаю Silero TTS (%s)…", config.TTS_VOICE)
        device = torch.device(config.TTS_DEVICE)

        self._model, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=config.TTS_LANGUAGE,
            speaker=config.TTS_VOICE,
            trust_repo=True,
        )
        self._model.to(device)
        self._device  = device
        self._speaker = config.TTS_SPEAKER
        self._rate    = config.TTS_RATE
        self._lock    = threading.Lock()
        log.info("Silero TTS готов. Диктор: %s", self._speaker)

    def _synthesize(self, text: str) -> "torch.Tensor":
        import torch
        with self._lock:
            audio = self._model.apply_tts(
                text=text,
                speaker=self._speaker,
                sample_rate=24000,
                put_accent=True,
                put_yo=True,
            )
        return audio

    def _play(self, audio_tensor) -> None:
        """Воспроизводит тензор через sounddevice."""
        import sounddevice as sd
        import numpy as np

        audio_np = audio_tensor.numpy()
        sd.play(audio_np, samplerate=24000)
        sd.wait()

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        log.info("TTS: «%s»", text[:60])
        try:
            audio = self._synthesize(text)
            self._play(audio)
        except Exception as exc:
            log.error("Silero ошибка: %s", exc)

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ════════════════════════════════════════════════════════════════════════════════
#  edge-tts (через Microsoft Edge TTS, нужен интернет)
# ════════════════════════════════════════════════════════════════════════════════

class EdgeTTS(BaseTTS):
    """
    Использует edge-tts (бесплатный, но требует интернет).
    pip install edge-tts

    Голоса для русского:
      ru-RU-SvetlanaNeural  — женский
      ru-RU-DmitryNeural    — мужской
    """

    VOICE = "ru-RU-DmitryNeural"

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        import asyncio
        import edge_tts
        import tempfile
        import sounddevice as sd
        import soundfile as sf

        log.info("Edge TTS: «%s»", text[:60])

        async def _run():
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            communicate = edge_tts.Communicate(text, self.VOICE)
            await communicate.save(tmp)
            return tmp

        try:
            tmp_file = asyncio.run(_run())
            data, sr = sf.read(tmp_file)
            sd.play(data, sr)
            sd.wait()
            os.unlink(tmp_file)
        except Exception as exc:
            log.error("Edge TTS ошибка: %s", exc)

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ════════════════════════════════════════════════════════════════════════════════
#  Фабрика
# ════════════════════════════════════════════════════════════════════════════════

def create_tts(backend: str = config.TTS_BACKEND) -> BaseTTS:
    """
    Фабричная функция.
    При ошибке загрузки Silero — автоматически переключается на edge-tts.
    """
    if backend == "silero":
        try:
            return SileroTTS()
        except Exception as exc:
            log.warning("Silero недоступен (%s), переключаюсь на edge-tts", exc)
            return EdgeTTS()
    elif backend == "edge":
        return EdgeTTS()
    else:
        raise ValueError(f"Неизвестный TTS backend: {backend}")


# ════════════════════════════════════════════════════════════════════════════════
#  TextToSpeech — обёртка с очередью (не блокирует основной поток)
# ════════════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    """
    Обёртка над BaseTTS с очередью.
    Можно вызывать say() из любого потока — TTS воспроизводится последовательно.
    """

    def __init__(self):
        self._engine = create_tts()
        self._queue: "queue.Queue[str]" = __import__("queue").Queue()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.5)
                self._engine.speak(text)
                self._queue.task_done()
            except __import__("queue").Empty:
                pass

    def say(self, text: str):
        """Добавляет текст в очередь воспроизведения."""
        if text:
            self._queue.put(text)

    def say_now(self, text: str):
        """Очищает очередь и произносит сразу (приоритет)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        self.say(text)

    def shutdown(self):
        self._stop.set()
