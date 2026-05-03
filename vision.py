# vision.py — Модуль "Зрение": скриншот + мультимодальный анализ
from __future__ import annotations

import base64
import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import pyautogui
from PIL import Image   # pip install Pillow

import config
from logger import get_logger

log = get_logger("Vision")


class VisionEngine:
    """
    Захватывает экран и отправляет изображение в Ollama (LLaVA / Qwen-VL).
    Зависит от LLMEngine для выполнения запроса.
    """

    def __init__(self, llm_engine=None):
        self._llm = llm_engine

    def set_llm(self, llm_engine):
        self._llm = llm_engine

    # ── Захват экрана ───────────────────────────────────────────────────────────

    def capture_screen(
        self,
        region: Optional[tuple] = None,   # (x, y, width, height)
        save: bool = True,
        max_size: int = 1280,
    ) -> str:
        """
        Делает скриншот и возвращает base64-строку PNG.

        Args:
            region:   Область захвата. None = весь экран.
            save:     Сохранять ли файл на диск.
            max_size: Максимальная сторона изображения (ресайз для LLM).
        """
        if region:
            img: Image.Image = pyautogui.screenshot(region=region)
        else:
            img: Image.Image = pyautogui.screenshot()

        # Ресайз, чтобы не перегружать LLM
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            log.debug("Изображение уменьшено до %s", new_size)

        if save:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.SCREENSHOTS_DIR, f"vision_{ts}.png")
            img.save(path)
            log.info("Скриншот сохранён: %s", path)

        # Конвертация в base64
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64

    # ── Анализ экрана ───────────────────────────────────────────────────────────

    def analyze_screen(self, question: str = "Что изображено на экране?") -> str:
        """
        Делает скриншот и спрашивает мультимодальную модель.

        Требует: ollama pull llava (или qwen2-vl)
        """
        if self._llm is None:
            return "VisionEngine: LLMEngine не установлен."

        log.info("Анализирую экран: «%s»", question)
        b64 = self.capture_screen()

        response = self._llm.chat(
            user_message=question,
            image_base64=b64,
            use_vision=True,
        )
        answer = response.reply_text or response.raw_text
        log.info("Vision ответ: %s", answer[:100])
        return answer

    def describe_screen(self) -> str:
        """Краткое описание текущего состояния экрана."""
        return self.analyze_screen("Кратко опиши, что сейчас на экране. На русском языке.")

    def find_element(self, description: str) -> str:
        """Ищет UI-элемент на экране."""
        return self.analyze_screen(
            f"Найди на экране элемент: '{description}'. "
            f"Укажи его примерные координаты и цвет."
        )
