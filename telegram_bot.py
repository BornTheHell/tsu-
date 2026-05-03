# telegram_bot.py — Джарвис TG Bot v3
# pip install python-telegram-bot==20.7 ddgs easyocr httpx
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import subprocess
import sys
import tempfile

import requests
from telegram import Update, Message
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
    MessageHandler, filters,
)
from telegram.request import HTTPXRequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Brain
import config

# ══════════════════════════════════════════════════════════════════════════════
#  Настройки
# ══════════════════════════════════════════════════════════════════════════════

BOT_TOKEN = os.environ.get("JARVIS_TG_TOKEN", "")
PROXY_URL = ""
TG_MAX    = 4000

# Параметры LLM для Telegram
TG_LLM_OPTIONS = {
    "temperature":    0.5,
    "num_ctx":        4096,
    "num_predict":    800,
    "repeat_penalty": 1.1,
}

# Системный промпт специально для Telegram — без tool-call, только русский
TG_SYSTEM_PROMPT = """\
Ты — Джарвис, умный ассистент. Отвечаешь ТОЛЬКО на русском языке.
НИКОГДА не используй китайский, английский или другие языки.
Если вопрос на русском — отвечай по-русски.
Если нужно решить задачу — решай пошагово, показывай все вычисления.
Отвечай полно и развёрнуто.
"""

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _n in ("httpx", "httpcore", "easyocr"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("tg_bot")

_brains: dict[int, Brain] = {}

def get_brain(uid: int) -> Brain:
    if uid not in _brains:
        _brains[uid] = Brain()
    return _brains[uid]


# ══════════════════════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════════════════════

def _chunks(text: str, size: int = TG_MAX) -> list[str]:
    if not text:
        return ["..."]
    return [text[i:i+size] for i in range(0, len(text), size)]


async def _reply(msg: Message, text: str) -> None:
    for part in _chunks(text):
        try:
            await msg.reply_text(part)
        except Exception as exc:
            log.error("send error: %s", exc)


def _strip_tool_json(text: str) -> str:
    """Убирает JSON tool-call из текста."""
    text = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
    text = re.sub(r'\{"action":\s*"[^"]+".+?\}', "", text, flags=re.DOTALL)
    return text.strip()


def _is_search(text: str) -> bool:
    triggers = [
        "найди", "поищи", "загугли", "что такое", "кто такой",
        "последние новости", "актуальный курс", "сколько стоит",
        "погода в", "новости о",
    ]
    return any(t in text.lower() for t in triggers)


# ══════════════════════════════════════════════════════════════════════════════
#  Ollama
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_chat(messages: list, options: dict = None, timeout: int = 120) -> str:
    """Запрос к qwen2.5 для решения задач и обычного диалога."""
    payload = {
        "model":    config.OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  options or TG_LLM_OPTIONS,
    }
    try:
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload, timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except requests.Timeout:
        return "Модель думает слишком долго. Попробуй ещё раз."
    except requests.ConnectionError:
        return "Ollama не запущена. Запусти `ollama serve`."
    except Exception as exc:
        log.error("ollama error: %s", exc)
        return f"Ошибка: {exc}"


def _ollama_unload(model: str) -> None:
    try:
        requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
    except Exception:
        pass


def _ollama_list_models() -> list[str]:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  OCR — распознавание текста с изображения
# ══════════════════════════════════════════════════════════════════════════════

_ocr_reader = None

def _get_ocr_reader():
    """Инициализирует EasyOCR с русским и английским языками."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            log.info("Загружаю EasyOCR (ru+en)…")
            # gpu=False чтобы не конкурировать с Whisper/Ollama за VRAM
            _ocr_reader = easyocr.Reader(["ru", "en"], gpu=config.CUDA_AVAILABLE)
            log.info("EasyOCR готов.")
        except ImportError:
            log.error("easyocr не установлен: pip install easyocr")
            return None
    return _ocr_reader


def _ocr_extract_text(image_bytes: bytes) -> str:
    """
    Извлекает текст с изображения через EasyOCR.
    Возвращает строку с распознанным текстом или пустую строку при ошибке.
    """
    reader = _get_ocr_reader()
    if reader is None:
        return ""

    try:
        import numpy as np
        from PIL import Image
        import io

        # Конвертируем bytes → PIL → numpy для EasyOCR
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Масштабируем если слишком маленькое (улучшает OCR)
        w, h = img.size
        if max(w, h) < 1000:
            scale = 1000 / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        img_np = np.array(img)

        log.info("Запускаю OCR на изображении %dx%d…", img.width, img.height)
        results = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(results).strip()
        log.info("OCR извлёк %d символов.", len(text))
        return text

    except Exception as exc:
        log.error("OCR ошибка: %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  Анализ изображений: OCR → qwen2.5 (основной путь)
#  llava → qwen2.5 (резервный путь если OCR не сработал)
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_image(image_bytes: bytes, question: str) -> str:
    """
    Основной алгоритм анализа изображения с задачей:
    
    1. OCR (EasyOCR) → извлекаем текст задачи
    2. Если текст найден → отдаём в qwen2.5 для решения
    3. Если OCR не дал результата → пробуем llava напрямую
    """

    # ── Шаг 1: OCR ────────────────────────────────────────────────────────────
    log.info("Шаг 1: OCR…")
    ocr_text = _ocr_extract_text(image_bytes)

    if ocr_text and len(ocr_text.strip()) > 20:
        # ── Шаг 2: qwen2.5 решает задачу по OCR-тексту ───────────────────────
        log.info("OCR успешен (%d симв.), решаю через qwen2.5…", len(ocr_text))

        user_prompt = (
            f"На изображении найден следующий текст (распознан через OCR):\n\n"
            f"---\n{ocr_text}\n---\n\n"
            f"Запрос пользователя: {question}\n\n"
            f"Реши все задачи пошагово. Покажи все вычисления и формулы. "
            f"Если несколько задач — реши каждую отдельно с номером."
        )

        answer = _ollama_chat(
            messages=[
                {"role": "system", "content": TG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": 0.1,   # детерминизм для математики
                "num_ctx":     4096,
                "num_predict": 2000,
            },
            timeout=180,
        )

        if answer and len(answer) > 30:
            # Добавляем распознанный текст для прозрачности
            return (
                f"📝 Распознанный текст:\n{ocr_text[:500]}{'...' if len(ocr_text)>500 else ''}\n\n"
                f"📐 Решение:\n{answer}"
            )

    # ── Шаг 3: Резервный путь — llava ─────────────────────────────────────────
    log.info("OCR дал мало текста, пробую llava…")

    available = _ollama_list_models()
    vision_model = next(
        (m for vm in ("llava", "llava-llama3", "moondream")
         for m in available if vm in m.lower()),
        None
    )

    if not vision_model:
        if not ocr_text:
            return (
                "❌ Не удалось распознать текст с изображения.\n\n"
                "Попробуй:\n"
                "• Прислать более чёткое фото\n"
                "• Переписать задачу текстом\n\n"
                "Для лучшего распознавания установи: `ollama pull llava`"
            )
        # OCR дал немного текста — попробуем решить хотя бы это
        return _ollama_chat(
            messages=[
                {"role": "system", "content": TG_SYSTEM_PROMPT},
                {"role": "user",   "content":
                    f"Частично распознанный текст задачи:\n{ocr_text}\n\n"
                    f"Попробуй решить что можно."},
            ],
            timeout=120,
        )

    # Запускаем llava
    _ollama_unload(config.OLLAMA_MODEL)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a math/physics tutor. "
        "Read ALL text from this image carefully. "
        "Then solve every problem step by step. "
        "Write your response in Russian language. "
        f"User question: {question}"
    )

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  vision_model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1500,
                    "num_ctx":     4096,
                },
            },
            timeout=240,
        )

        if resp.status_code == 200:
            content = resp.json().get("response", "").strip()
            if content and len(content) > 30:
                log.info("llava ответил (%d симв.)", len(content))
                # Если llava ответил нормально — переводим/дополняем через qwen
                if ocr_text:
                    return (
                        f"📝 Распознанный текст:\n{ocr_text[:400]}\n\n"
                        f"📐 Решение:\n{content}"
                    )
                return f"📐 Решение:\n{content}"

        log.warning("llava вернул плохой ответ, пробую через OCR-текст…")

    except Exception as exc:
        log.error("llava error: %s", exc)

    # Последний резерв — если есть хоть что-то от OCR
    if ocr_text:
        return _ollama_chat(
            messages=[
                {"role": "system", "content": TG_SYSTEM_PROMPT},
                {"role": "user",   "content":
                    f"Текст с изображения (OCR):\n{ocr_text}\n\nРеши задачу."},
            ],
            timeout=120,
        )

    return (
        "Не удалось прочитать текст с изображения.\n\n"
        "Попробуй:\n"
        "• Прислать более чёткое фото\n"
        "• Написать задачу текстом — решу сразу"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Whisper (голосовые)
# ══════════════════════════════════════════════════════════════════════════════

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("Загружаю Whisper…")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def _transcribe_bytes(ogg_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        ogg = os.path.join(tmp, "v.ogg")
        wav = os.path.join(tmp, "v.wav")
        with open(ogg, "wb") as f:
            f.write(ogg_bytes)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", ogg, "-ar", "16000", "-ac", "1", wav],
                capture_output=True, check=True, timeout=30,
            )
        except FileNotFoundError:
            return "[ffmpeg не найден — скачай с ffmpeg.org]"
        except subprocess.CalledProcessError:
            return "[Ошибка конвертации аудио]"

        segs, _ = _get_whisper().transcribe(
            wav, language=config.WHISPER_LANGUAGE,
            beam_size=5, vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segs).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Поиск
# ══════════════════════════════════════════════════════════════════════════════

def _web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Установи: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5, region="ru-ru"))
    except Exception as exc:
        return f"Ошибка поиска: {exc}"

    if not results:
        return "Ничего не нашёл."

    context = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('body','')}"
        for i, r in enumerate(results)
    )

    answer = _ollama_chat(
        messages=[
            {"role": "system", "content": TG_SYSTEM_PROMPT},
            {"role": "user",   "content":
                f"Вопрос: {query}\n\nДанные поиска:\n{context}\n\nОтветь кратко по-русски."},
        ],
        options={"temperature": 0.2, "num_predict": 500, "num_ctx": 4096},
    )

    sources = "\n".join(
        f"• {r.get('title','?')}: {r.get('href','')}"
        for r in results[:3]
    )
    return f"{answer}\n\nИсточники:\n{sources}"


# ══════════════════════════════════════════════════════════════════════════════
#  Telegram обработчики
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update.message,
        f"Привет, {update.effective_user.first_name}! Я Джарвис.\n\n"
        "📷 Решаю задачи по фото (матан, физика, алгебра)\n"
        "🎤 Понимаю голосовые сообщения\n"
        "💬 Пишу код и объясняю темы\n"
        "🔍 Ищу информацию в интернете\n\n"
        "Пиши, говори или кидай фото задачи."
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    get_brain(update.effective_user.id).clear_history()
    await _reply(update.message, "История очищена.")


async def cmd_models(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    models = _ollama_list_models()
    text = ("Модели Ollama:\n" + "\n".join(f"• {m}" for m in models)
            if models else "Ollama недоступна.")
    await _reply(update.message, text)


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text  = update.message.text.strip()
    uid   = update.effective_user.id

    await update.message.chat.send_action(ChatAction.TYPING)

    if _is_search(text):
        await _reply(update.message, "Ищу…")
        result = await asyncio.get_event_loop().run_in_executor(
            None, _web_search, text
        )
        await _reply(update.message, result)
        return

    # Прямой запрос к Ollama с русским промптом (минуя brain с его tool-call)
    answer = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _ollama_chat(
            messages=[
                {"role": "system", "content": TG_SYSTEM_PROMPT},
                *get_brain(uid)._history.as_list(),
                {"role": "user",   "content": text},
            ],
        )
    )

    # Обновляем историю
    get_brain(uid)._history.add_user(text)
    get_brain(uid)._history.add_assistant(answer)

    answer = _strip_tool_json(answer)
    await _reply(update.message, answer or "Не понял.")


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    vbytes = bytes(await (await update.message.voice.get_file()).download_as_bytearray())
    text   = await asyncio.get_event_loop().run_in_executor(None, _transcribe_bytes, vbytes)

    if not text or text.startswith("["):
        await _reply(update.message, text or "Не удалось распознать.")
        return

    await _reply(update.message, f"Распознано: «{text}»")

    if _is_search(text):
        result = await asyncio.get_event_loop().run_in_executor(None, _web_search, text)
        await _reply(update.message, result)
        return

    answer = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _ollama_chat(
            messages=[
                {"role": "system", "content": TG_SYSTEM_PROMPT},
                *get_brain(uid)._history.as_list(),
                {"role": "user",   "content": text},
            ],
        )
    )
    get_brain(uid)._history.add_user(text)
    get_brain(uid)._history.add_assistant(answer)
    await _reply(update.message, _strip_tool_json(answer) or "Не понял.")


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    caption  = (update.message.caption or "").strip()
    question = caption or "Реши все задачи пошагово."

    await _reply(update.message, "Смотрю на задачу…")
    await update.message.chat.send_action(ChatAction.TYPING)

    pbytes = bytes(await (await update.message.photo[-1].get_file()).download_as_bytearray())
    result = await asyncio.get_event_loop().run_in_executor(
        None, _analyze_image, pbytes, question
    )
    await _reply(update.message, result)


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    doc  = update.message.document
    mime = doc.mime_type or ""

    if mime.startswith("image/"):
        await _reply(update.message, "Смотрю на изображение…")
        fbytes = bytes(await (await doc.get_file()).download_as_bytearray())
        caption = (update.message.caption or "").strip()
        result  = await asyncio.get_event_loop().run_in_executor(
            None, _analyze_image, fbytes, caption or "Реши задачу пошагово."
        )
        await _reply(update.message, result)
    else:
        await _reply(update.message, "Присылай фото задачи или текст.")


async def handle_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("TG error: %s", ctx.error)


# ══════════════════════════════════════════════════════════════════════════════
#  Запуск
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not BOT_TOKEN:
        print("Задай токен: set JARVIS_TG_TOKEN=твой_токен")
        return

    # Прогрев OCR при старте (чтобы первый запрос не тормозил)
    log.info("Инициализирую OCR…")
    _get_ocr_reader()

    models = _ollama_list_models()
    log.info("Ollama модели: %s", models)

    rq = dict(connect_timeout=30.0, read_timeout=30.0,
              write_timeout=30.0, pool_timeout=30.0)
    if PROXY_URL:
        rq["proxy"] = PROXY_URL

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(HTTPXRequest(**rq))
        .get_updates_request(HTTPXRequest(connect_timeout=30.0, read_timeout=30.0))
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("reset",  cmd_reset))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("help",   cmd_start))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE,                   handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL,            handle_document))
    app.add_error_handler(handle_error)

    log.info("Бот запущен.")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
