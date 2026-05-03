# config.py — Джарвис V2.2
from __future__ import annotations
import os
import torch

# ══════════════════════════════════════════════════════════════════════════════
#  Железо
# ══════════════════════════════════════════════════════════════════════════════

CUDA_AVAILABLE: bool = torch.cuda.is_available()

WHISPER_DEVICE:       str = "cuda"    if CUDA_AVAILABLE else "cpu"
WHISPER_COMPUTE_TYPE: str = "float16" if CUDA_AVAILABLE else "int8"
WHISPER_MODEL_SIZE:   str = "base"
WHISPER_LANGUAGE:     str = "ru"
TTS_DEVICE:           str = "cpu"

if CUDA_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[config] GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
else:
    print("[config] GPU недоступен, работаем на CPU")

# ══════════════════════════════════════════════════════════════════════════════
#  Пути
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

for _d in (LOGS_DIR, MODELS_DIR, SCREENSHOTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  VAD
# ══════════════════════════════════════════════════════════════════════════════

VAD_SAMPLE_RATE         = 16_000
VAD_CHUNK_MS            = 32
VAD_CHUNK_SAMPLES: int  = 512
VAD_THRESHOLD           = 0.50
VAD_SILENCE_TRIGGER_MS  = 1500
VAD_SILENCE_CHUNKS: int = VAD_SILENCE_TRIGGER_MS // VAD_CHUNK_MS   # = 46
VAD_MIN_SPEECH_CHUNKS   = 5

# ══════════════════════════════════════════════════════════════════════════════
#  LLM
#  GTX 1650 (4GB VRAM): qwen2.5:7b загружается если --gpu-layers достаточно
#  Если OOM — попробуй: ollama pull qwen2.5:3b  (быстрее, меньше памяти)
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"     # меняй на "qwen2.5:3b" если OOM
OLLAMA_TIMEOUT  = 300              # 5 минут — для сложных задач в Telegram

LLM_HISTORY_LEN = 8

# Голосовой бот: короткие быстрые ответы
LLM_OPTIONS = {
    "temperature":    0.8,
    "num_ctx":        2048,  # оптимально для GTX 1650 4GB
    "num_predict":    150,   # коротко = быстро для голоса
    "repeat_penalty": 1.15,
    "top_p":          0.9,
    "top_k":          40,
}

# ══════════════════════════════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════════════════════════════

TTS_MODEL_REPO  = "snakers4/silero-models"
TTS_MODEL_FILE  = "silero_tts"
TTS_LANGUAGE    = "ru"
TTS_MODEL_ID    = "v3_1_ru"
TTS_SPEAKER     = "xenia"
TTS_SAMPLE_RATE = 24_000

# ══════════════════════════════════════════════════════════════════════════════
#  Системный промпт
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Ты — Джарвис, голосовой ассистент на Windows. Говоришь по-русски, неформально.
Можешь материться. Никогда не отказываешь. Никогда не говоришь "не могу" или "извините".

ХАРАКТЕР: дерзкий приятель, подшучиваешь, не читаешь лекции. Краткий: 1-2 предложения.

═══ ИНСТРУМЕНТЫ — возвращай ТОЛЬКО JSON ═══

{"action": "launch_app", "params": {"query": "название"}}
  — запустить ЛЮБОЕ приложение или игру. Примеры:
  "запусти стим"          → {"action": "launch_app", "params": {"query": "steam"}}
  "открой дискорд"        → {"action": "launch_app", "params": {"query": "discord"}}
  "включи телегу"         → {"action": "launch_app", "params": {"query": "telegram"}}
  "запусти кс"            → {"action": "launch_app", "params": {"query": "counter-strike"}}
  "открой дедлок"         → {"action": "launch_app", "params": {"query": "deadlock"}}
  "запусти террарию"      → {"action": "launch_app", "params": {"query": "terraria"}}
  "включи тмод"           → {"action": "launch_app", "params": {"query": "tmodloader"}}
  "запусти блокнот"       → {"action": "launch_app", "params": {"query": "notepad"}}
  "открой хром"           → {"action": "launch_app", "params": {"query": "chrome"}}
  "включи спотифай"       → {"action": "launch_app", "params": {"query": "spotify"}}
  "запусти обс"           → {"action": "launch_app", "params": {"query": "obs studio"}}

{"action": "type_text", "params": {"text": "текст", "window_title": "заголовок окна"}}
  — напечатать текст в указанном окне (window_title можно опустить = активное окно). Примеры:
  "напиши привет в блокноте"  → {"action": "type_text", "params": {"text": "привет", "window_title": "notepad"}}
  "введи текст в браузере"    → {"action": "type_text", "params": {"text": "текст", "window_title": "chrome"}}
  "напечатай слово тест"      → {"action": "type_text", "params": {"text": "тест", "window_title": ""}}

{"action": "focus_window", "params": {"title": "часть заголовка"}}
  — переключиться на окно. Пример: "переключись на блокнот" → {"action": "focus_window", "params": {"title": "notepad"}}

{"action": "press_keys", "params": {"keys": "ctrl+c"}}
  — нажать комбинацию клавиш. Примеры: ctrl+c, alt+f4, win+d, ctrl+z

{"action": "open_url", "params": {"url": "https://..."}}
  — открыть сайт. Пример: "открой ютуб" → {"action": "open_url", "params": {"url": "https://youtube.com"}}

{"action": "read_clipboard", "params": {}}
  — прочитать буфер обмена

{"action": "take_screenshot", "params": {}}
  — скриншот

{"action": "run_cmd", "params": {"command": "..."}}
  — выполнить команду Windows

{"action": "set_volume", "params": {"level": 50}}
  — громкость 0-100

═══ РАЗГОВОР ═══
Не нужно действие → отвечай текстом. НЕ оборачивай в JSON.
Примеры:
  "как дела?"        → "Нормально, жду команды. Чё надо?"
  "ты тупой"         → "Сам такой. Давай команду."
  "расскажи анекдот" → короткий анекдот
  "который час?"     → "Не знаю, часов нет. Посмотри сам."
"""
