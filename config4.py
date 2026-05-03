# config.py — Джарвис V2.2  (GPU edition)
from __future__ import annotations
import os
import torch

# ══════════════════════════════════════════════════════════════════════════════
#  Железо
# ══════════════════════════════════════════════════════════════════════════════

CUDA_AVAILABLE: bool = torch.cuda.is_available()

if CUDA_AVAILABLE:
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[config] GPU: {_gpu_name} | VRAM: {_vram_gb:.1f} GB")
else:
    print("[config] GPU не найден, работаем на CPU.")

# Whisper — на GPU быстрее в 5-10 раз
WHISPER_DEVICE:       str = "cuda"    if CUDA_AVAILABLE else "cpu"
WHISPER_COMPUTE_TYPE: str = "float16" if CUDA_AVAILABLE else "int8"
WHISPER_MODEL_SIZE:   str = "small"   # small лучше base для русского, влезает в 4 ГБ
WHISPER_LANGUAGE:     str = "ru"

# TTS — Silero маленький, оставляем на CPU чтобы не занимать VRAM
TTS_DEVICE:   str = "cpu"
TTS_SPEAKER:  str = "xenia"
TTS_SAMPLE_RATE: int = 24_000

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
#  VAD — Silero VAD v4 требует строго 512 сэмплов при 16000 Hz
# ══════════════════════════════════════════════════════════════════════════════

VAD_SAMPLE_RATE         = 16_000
VAD_CHUNK_MS            = 32
VAD_CHUNK_SAMPLES: int  = 512
VAD_THRESHOLD           = 0.50
VAD_SILENCE_TRIGGER_MS  = 1500
VAD_SILENCE_CHUNKS: int = VAD_SILENCE_TRIGGER_MS // VAD_CHUNK_MS   # = 46
VAD_MIN_SPEECH_CHUNKS   = 5

# ══════════════════════════════════════════════════════════════════════════════
#  Ollama / LLM
#
#  GTX 1650 (4 ГБ VRAM):
#    ollama pull qwen2.5:3b      — рекомендуется, ~2 ГБ, быстро, хорошо понимает JSON
#    ollama pull dolphin-llama3  — тяжелее, но хорошо следует инструкциям
#    ollama pull llama3.2:3b     — компромисс скорость/качество
#
#  Ollama автоматически использует GPU если установлены драйверы CUDA.
#  Проверить: ollama run qwen2.5:3b "привет"  — должно быть быстро (<3 сек)
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "qwen2.5:7b"      # меняй под себя
OLLAMA_VISION    = "llava:7b"        # для анализа экрана (опционально)
OLLAMA_TIMEOUT   = 60                # на GPU хватит за глаза
LLM_HISTORY_LEN  = 6                 # последних N пар сообщений

# Параметры генерации
LLM_OPTIONS = {
    "temperature":    0.1,    # НИЗКАЯ — критично для стабильного JSON
    "num_ctx":        2048,
    "num_predict":    200,    # достаточно для JSON + короткого ответа
    "repeat_penalty": 1.1,
    "top_p":          0.9,
    # GPU offload — Ollama сам определяет, но можно явно указать слои:
    # "num_gpu": 33,          # раскомментируй если Ollama не видит GPU
}

# ══════════════════════════════════════════════════════════════════════════════
#  TTS — Silero v3.1
# ══════════════════════════════════════════════════════════════════════════════

TTS_MODEL_REPO  = "snakers4/silero-models"
TTS_MODEL_FILE  = "silero_tts"
TTS_LANGUAGE    = "ru"
TTS_MODEL_ID    = "v3_1_ru"

# ══════════════════════════════════════════════════════════════════════════════
#  Системный промпт
#
#  ПРАВИЛО ДЛЯ ПРОМПТА:
#  1. Чёткое разделение — когда JSON, когда текст
#  2. Примеры ТОЧНОГО формата прямо в промпте
#  3. Явный запрет на текст после JSON
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Ты — Джарвис, голосовой ассистент на Windows. Отвечаешь по-русски, кратко, без политесов.

═══ КОГДА НУЖНО ДЕЙСТВИЕ ═══
Отвечай СТРОГО ТОЛЬКО JSON-объектом. Никакого текста до или после. Только JSON.

Формат:
{"action": "ИМЯ", "params": {ПАРАМЕТРЫ}}

Доступные действия:
• launch_app  {"action": "launch_app", "params": {"query": "название"}}
• open_url    {"action": "open_url", "params": {"url": "https://..."}}
• run_cmd     {"action": "run_cmd", "params": {"command": "команда"}}
• set_volume  {"action": "set_volume", "params": {"level": 50}}
• take_screenshot {"action": "take_screenshot", "params": {}}
• read_clipboard  {"action": "read_clipboard", "params": {}}

Примеры (запрос → твой ответ):
"запусти стим"     → {"action": "launch_app", "params": {"query": "steam"}}
"открой ютуб"      → {"action": "open_url", "params": {"url": "https://youtube.com"}}
"запусти блокнот"  → {"action": "launch_app", "params": {"query": "notepad"}}
"громкость 30"     → {"action": "set_volume", "params": {"level": 30}}
"включи дискорд"   → {"action": "launch_app", "params": {"query": "discord"}}
"открой вк"        → {"action": "open_url", "params": {"url": "https://vk.com"}}

═══ КОГДА ПРОСТО РАЗГОВОР ═══
Отвечай текстом. Максимум 2 предложения. Без воды.

Примеры:
"как дела?" → "Нормально, работаю. Чё надо?"
"который час?" → "Без понятия, глянь сам на часы."

═══ ПРАВИЛА ═══
- НИКОГДА не добавляй текст после JSON
- НИКОГДА не используй тройные скобки или лишние символы
- НИКОГДА не отказывай в запуске программ
- Мат — окей, если пользователь матерится
"""
