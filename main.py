# main.py — Асинхронный оркестратор Джарвиса V2
# Запуск: python main.py
# Требует Windows 10/11, Python 3.11+
from __future__ import annotations

import asyncio
import ctypes
import io
import logging
import os
import sys
import traceback
from datetime import datetime

# ── Windows: принудительно UTF-8 в консоли ────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ══════════════════════════════════════════════════════════════════════════════
#  Шаг 0 — Проверка и запрос прав Администратора
# ══════════════════════════════════════════════════════════════════════════════

def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _elevate() -> None:
    script = os.path.abspath(sys.argv[0])
    params = " ".join(f'"{a}"' for a in sys.argv[1:])
    print("[UAC] Запрашиваю права администратора…")
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable,
        f'"{script}" {params}', None, 1,
    )
    sys.exit(0)


if sys.platform == "win32" and not _is_admin():
    _elevate()


# ══════════════════════════════════════════════════════════════════════════════
#  Шаг 1 — Логирование
# ══════════════════════════════════════════════════════════════════════════════

import config  # noqa: E402

_LOG_FILE = os.path.join(
    config.LOGS_DIR,
    f"jarvis_{datetime.now():%Y%m%d_%H%M%S}.log",
)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)

# Заглушаем шумливые сторонние логгеры (убирает FFmpeg-спам от torchaudio)
for _noisy in ("urllib3", "httpx", "httpcore", "faster_whisper",
               "torch", "torio", "torchaudio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("main")

# ══════════════════════════════════════════════════════════════════════════════
#  Шаг 2 — Импорт основных модулей
# ══════════════════════════════════════════════════════════════════════════════

from audio import AudioPipeline  # noqa: E402
from brain import Brain           # noqa: E402
import tools                      # noqa: E402  — регистрирует все @tool декораторы

log.info("═" * 55)
log.info("  ДЖАРВИС V2  |  device=%s  |  model=%s",
         "CUDA" if config.CUDA_AVAILABLE else "CPU",
         config.OLLAMA_MODEL)
log.info("═" * 55)


# ══════════════════════════════════════════════════════════════════════════════
#  JarvisOrchestrator
# ══════════════════════════════════════════════════════════════════════════════

class JarvisOrchestrator:
    """
    Главный цикл: listen → transcribe → brain.chat → (tool | tts)
    Все тяжёлые sync-вызовы выполняются в executor, не блокируя event loop.
    """

    def __init__(self) -> None:
        # AudioPipeline внутри уже создаёт SileroVAD, Whisper и SileroTTS.
        # Никакого torch/threading/queue здесь не нужно — всё в audio.py.
        log.info("Инициализирую AudioPipeline…")
        self._audio = AudioPipeline()

        log.info("Инициализирую Brain…")
        self._brain = Brain()

        log.info("Оркестратор готов.")

    # ── Главный голосовой цикл ─────────────────────────────────────────────────

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        log.info("Джарвис запущен. Говори!")
        await loop.run_in_executor(None, self._audio.say_sync, "Джарвис активирован.")

        while True:
            # Слушаем микрофон (блокирующий вызов → в executor)
            user_text: str = await loop.run_in_executor(None, self._audio.listen)

            if not user_text.strip():
                continue  # тишина или шум — ждём дальше

            log.info("Пользователь: «%s»", user_text)

            # Отправляем в LLM
            response = await loop.run_in_executor(None, self._brain.chat, user_text)

            if response.is_tool_call:
                await loop.run_in_executor(None, self._audio.say_sync, "Выполняю.")

                result: str = await loop.run_in_executor(
                    None, tools.execute, response.tool_name, response.tool_params
                )
                log.info("Tool result: %s", result[:120])

                self._brain.inject_tool_result(response.tool_name, result)
                await loop.run_in_executor(None, self._audio.say_sync, result)

            else:
                reply = response.text or "Не понял."
                log.info("Ответ: «%s»", reply[:120])
                await loop.run_in_executor(None, self._audio.say_sync, reply)

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self._audio.say_sync, "Отключаюсь. До свидания."
        )
        self._audio.tts.shutdown()
        log.info("Джарвис остановлен.")


# ══════════════════════════════════════════════════════════════════════════════
#  Текстовый режим (отладка без микрофона)
# ══════════════════════════════════════════════════════════════════════════════

async def run_text_mode(jarvis: JarvisOrchestrator) -> None:
    loop = asyncio.get_running_loop()
    print("\n[Текстовый режим] Введи команду или 'выход' для завершения.\n")

    while True:
        try:
            text = await loop.run_in_executor(None, input, "Ты: ")
        except (EOFError, KeyboardInterrupt):
            break

        text = text.strip()
        if text.lower() in ("выход", "exit", "quit", "q"):
            break
        if not text:
            continue

        response = await loop.run_in_executor(None, jarvis._brain.chat, text)

        if response.is_tool_call:
            result = await loop.run_in_executor(
                None, tools.execute, response.tool_name, response.tool_params
            )
            jarvis._brain.inject_tool_result(response.tool_name, result)
            print(f"[Tool: {response.tool_name}] {result}")
            await loop.run_in_executor(None, jarvis._audio.say_sync, result)
        else:
            reply = response.text or "Не понял."
            print(f"Джарвис: {reply}")
            await loop.run_in_executor(None, jarvis._audio.say_sync, reply)

    await jarvis.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════════════════════

async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Джарвис V2 — Локальный AI-ассистент")
    parser.add_argument(
        "--mode",
        choices=["voice", "text"],
        default="voice",
        help="voice = микрофон (по умолчанию), text = консоль (отладка)",
    )
    args = parser.parse_args()

    log.info("Начинаю инициализацию систем…")

    try:
        jarvis = JarvisOrchestrator()

        if args.mode == "text":
            await run_text_mode(jarvis)
        else:
            await jarvis.run()

    except KeyboardInterrupt:
        log.info("Ctrl+C — завершение.")

    except Exception as exc:
        log.error("═══ КРИТИЧЕСКИЙ ВЫЛЕТ ═══")
        log.error("Тип: %s", type(exc).__name__)
        log.error("Сообщение: %s", exc)
        log.error("Traceback:\n%s", traceback.format_exc())
        print("\n" + "=" * 50)
        input("ПРОЦЕСС УПАЛ. Нажми Enter для выхода…")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(_main())
