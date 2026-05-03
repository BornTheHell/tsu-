# audio.py — STT + VAD + TTS  (GPU edition: Whisper на CUDA)
from __future__ import annotations

import io
import logging
import queue
import re
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

import config

log = logging.getLogger("audio")

# ══════════════════════════════════════════════════════════════════════════════
#  Пост-обработка STT
# ══════════════════════════════════════════════════════════════════════════════

_STT_CORRECTIONS: dict[str, str] = {
    r"\bстемистый\b":   "стим",
    r"\bстема\b":       "стим",
    r"\bстемом\b":      "стим",
    r"\bстеме\b":       "стим",
    r"\bстима\b":       "стим",
    r"\bчистим\b":      "стим",
    r"\bдискор\b":      "дискорд",
    r"\bдискот\b":      "дискорд",
    r"\bтелег[уо]\b":   "телеграм",
    r"\bютуб\b":        "youtube",
    r"\bзапустил\b":    "запусти",
    r"\bвключил\b":     "включи",
    r"\bоткрыл\b":      "открой",
}


def _correct_stt(text: str) -> str:
    result = text
    for pattern, replacement in _STT_CORRECTIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    if result != text:
        log.debug("STT correction: «%s» → «%s»", text, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Silero VAD
# ══════════════════════════════════════════════════════════════════════════════

class SileroVAD:
    def __init__(self) -> None:
        log.info("Загружаю Silero VAD…")
        # VAD всегда на CPU — он лёгкий, не нужен GPU
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()
        self._chunk_size = self._detect_chunk_size()
        log.info("Silero VAD готов. Чанк: %d сэмплов (%d мс).",
                 self._chunk_size, self._chunk_size * 1000 // config.VAD_SAMPLE_RATE)
        self.reset()

    def _detect_chunk_size(self) -> int:
        for size in (config.VAD_CHUNK_SAMPLES, 512, 256):
            try:
                test = torch.zeros(size)
                with torch.no_grad():
                    self._model(test, config.VAD_SAMPLE_RATE)
                return size
            except Exception:
                continue
        raise RuntimeError(
            "Silero VAD: не найден рабочий размер чанка. "
            r"Удали кэш: %USERPROFILE%\.cache\torch\hub\snakers4_silero-vad_master"
        )

    def reset(self) -> None:
        self._model.reset_states()

    def is_speech(self, chunk_int16: np.ndarray) -> float:
        audio_f32 = chunk_int16.astype(np.float32) / 32768.0
        n = len(audio_f32)
        if n < self._chunk_size:
            padded = np.zeros(self._chunk_size, dtype=np.float32)
            padded[:n] = audio_f32
            audio_f32 = padded
        elif n > self._chunk_size:
            audio_f32 = audio_f32[:self._chunk_size]

        with torch.no_grad():
            return self._model(torch.from_numpy(audio_f32), config.VAD_SAMPLE_RATE).item()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size


# ══════════════════════════════════════════════════════════════════════════════
#  Recorder
# ══════════════════════════════════════════════════════════════════════════════

class Recorder:
    def __init__(self, vad: SileroVAD) -> None:
        self._vad    = vad
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("sd status: %s", status)
        self._audio_q.put(indata[:, 0].copy())

    def record_phrase(self) -> Optional[np.ndarray]:
        self._vad.reset()
        self._audio_q.queue.clear()

        collected: list[np.ndarray] = []
        speech_n  = 0
        silence_n = 0
        recording = False
        chunk_size = self._vad.chunk_size

        log.info("Слушаю…")

        try:
            stream = sd.InputStream(
                samplerate=config.VAD_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
                callback=self._sd_callback,
            )
            stream.start()
        except Exception as exc:
            log.error("Не могу открыть микрофон: %s", exc)
            return None

        try:
            while True:
                try:
                    chunk = self._audio_q.get(timeout=3.0)
                except queue.Empty:
                    log.warning("Микрофон молчит 3 сек")
                    continue

                prob = self._vad.is_speech(chunk)

                if prob >= config.VAD_THRESHOLD:
                    speech_n  += 1
                    silence_n  = 0
                else:
                    silence_n += 1
                    speech_n   = 0

                if not recording and speech_n >= config.VAD_MIN_SPEECH_CHUNKS:
                    recording = True
                    log.debug("▶ Речь (prob=%.2f)", prob)

                if recording:
                    collected.append(chunk)
                    if silence_n >= config.VAD_SILENCE_CHUNKS:
                        log.debug("■ Конец фразы: %d чанков = %.1f сек",
                                  len(collected),
                                  len(collected) * chunk_size / config.VAD_SAMPLE_RATE)
                        break
        finally:
            stream.stop()
            stream.close()

        if len(collected) < config.VAD_MIN_SPEECH_CHUNKS:
            return None
        return np.concatenate(collected)


# ══════════════════════════════════════════════════════════════════════════════
#  STT — faster-whisper на GPU (если доступен)
# ══════════════════════════════════════════════════════════════════════════════

class SpeechRecognizer:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        device       = config.WHISPER_DEVICE        # "cuda" или "cpu"
        compute_type = config.WHISPER_COMPUTE_TYPE   # "float16" или "int8"

        log.info("Загружаю Whisper '%s' на '%s' [%s]…",
                 config.WHISPER_MODEL_SIZE, device, compute_type)

        self._model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
            download_root=config.MODELS_DIR,
        )
        log.info("Whisper готов (device=%s).", device)

    def transcribe(self, audio_int16: np.ndarray) -> str:
        import scipy.io.wavfile as wav

        buf = io.BytesIO()
        wav.write(buf, config.VAD_SAMPLE_RATE, audio_int16)
        buf.seek(0)

        try:
            segments, info = self._model.transcribe(
                buf,
                language=config.WHISPER_LANGUAGE,
                beam_size=5,
                best_of=5,
                initial_prompt="Джарвис, запусти стим. Открой браузер. Привет.",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=200,
                    threshold=0.45,
                    min_speech_duration_ms=100,
                ),
                suppress_tokens=[-1],
                word_timestamps=False,
                temperature=0.0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            raw_text = " ".join(s.text.strip() for s in segments).strip()
            text = _correct_stt(raw_text)

            log.info("STT: «%s» (lang=%s, prob=%.2f)",
                     text, info.language, info.language_probability)
            return text
        except Exception as exc:
            log.error("Ошибка Whisper: %s", exc)
            return ""


# ══════════════════════════════════════════════════════════════════════════════
#  TTS — Silero v3.1 (CPU)
# ══════════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    def __init__(self) -> None:
        log.info("Загружаю Silero TTS ('%s') на CPU…", config.TTS_SPEAKER)

        loaded = torch.hub.load(
            repo_or_dir=config.TTS_MODEL_REPO,
            model=config.TTS_MODEL_FILE,
            language=config.TTS_LANGUAGE,
            speaker=config.TTS_MODEL_ID,
            trust_repo=True,
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.to(torch.device(config.TTS_DEVICE))  # CPU

        self._model = model
        self._lock  = threading.Lock()
        self._q: queue.Queue[str] = queue.Queue()
        self._stop  = threading.Event()
        threading.Thread(target=self._worker, daemon=True, name="tts").start()

        log.info("Silero TTS готов (speaker=%s).", config.TTS_SPEAKER)

    def say(self, text: str) -> None:
        if text and text.strip():
            self._q.put(text.strip())

    def say_sync(self, text: str) -> None:
        if text and text.strip():
            self._play(text.strip())

    def shutdown(self) -> None:
        self._stop.set()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
                self._play(text)
                self._q.task_done()
            except queue.Empty:
                pass
            except Exception as exc:
                log.error("TTS worker: %s", exc)

    def _play(self, text: str) -> None:
        if len(text) > 300:
            text = text[:297] + "..."
        with self._lock:
            try:
                audio = self._model.apply_tts(
                    text=text,
                    speaker=config.TTS_SPEAKER,
                    sample_rate=config.TTS_SAMPLE_RATE,
                    put_accent=True,
                    put_yo=True,
                )
                sd.play(audio.numpy(), samplerate=config.TTS_SAMPLE_RATE)
                sd.wait()
            except Exception as exc:
                log.error("TTS._play: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
#  AudioPipeline
# ══════════════════════════════════════════════════════════════════════════════

class AudioPipeline:
    def __init__(self) -> None:
        self._vad = SileroVAD()
        self._rec = Recorder(self._vad)
        self._stt = SpeechRecognizer()
        self.tts  = TextToSpeech()

    def listen(self) -> str:
        audio = self._rec.record_phrase()
        if audio is None:
            return ""
        return self._stt.transcribe(audio)

    def say(self, text: str) -> None:
        self.tts.say(text)

    def say_sync(self, text: str) -> None:
        self.tts.say_sync(text)
