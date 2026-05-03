# logger.py — Настройка логирования для всего проекта
import logging
import sys
import os
from datetime import datetime

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"jarvis_{datetime.now():%Y%m%d_%H%M%S}.log")


def get_logger(name: str) -> logging.Logger:
    """Возвращает настроенный логгер для модуля."""
    logger = logging.getLogger(name)
    if logger.handlers:          # не дублировать хэндлеры при повторном вызове
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Форматтер ──────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Хэндлер → консоль (INFO+) ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # ── Хэндлер → файл (DEBUG+) ────────────────────────────────────────────────
    file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_h)
    return logger
