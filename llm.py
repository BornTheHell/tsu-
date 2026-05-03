# llm.py — Модуль "Мозг": Ollama + структурированный вывод (Function Calling)
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from logger import get_logger

log = get_logger("LLM")


@dataclass
class LLMResponse:
    """Структурированный ответ от LLM."""
    raw_text: str                        # сырой текст от модели
    is_action: bool = False              # True если это команда, а не текст
    action: Optional[str] = None        # название действия
    params: Dict[str, Any] = None       # параметры действия
    reply_text: Optional[str] = None    # текстовый ответ (если не команда)

    def __post_init__(self):
        if self.params is None:
            self.params = {}


# ════════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    Движок для работы с Ollama.
    Поддерживает:
    - Обычный диалог
    - Структурированный вывод (JSON-действия)
    - Мультимодальный ввод (изображения для LLaVA / Qwen-VL)
    - Историю диалога
    """

    def __init__(self, tools_description: str = ""):
        self._base_url       = config.OLLAMA_BASE_URL
        self._model          = config.OLLAMA_MODEL
        self._vision_model   = config.OLLAMA_VISION
        self._timeout        = config.OLLAMA_TIMEOUT
        self._tools_desc     = tools_description
        self._history: List[Dict[str, str]] = []
        log.info("LLMEngine инициализирован. Модель: %s", self._model)

    # ── Системный промпт ────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return config.SYSTEM_PROMPT.format(tools_list=self._tools_desc)

    # ── Парсинг ответа ──────────────────────────────────────────────────────────

    @staticmethod
    def _try_parse_action(text: str) -> Optional[Tuple[str, dict]]:
        """
        Пытается извлечь JSON-команду из текста модели.
        Поддерживает:
          - чистый JSON
          - JSON внутри ```json … ``` блоков
        """
        # Ищем JSON-блок
        patterns = [
            r"```json\s*([\s\S]+?)\s*```",   # ```json ... ```
            r"```\s*([\s\S]+?)\s*```",        # ``` ... ```
            r"(\{[\s\S]+\})",                  # просто фигурные скобки
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                candidate = m.group(1).strip()
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "action" in data:
                        return data["action"], data.get("params", {})
                except json.JSONDecodeError:
                    continue
        return None

    # ── Основной запрос ─────────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        image_base64: Optional[str] = None,
        use_vision: bool = False,
    ) -> LLMResponse:
        """
        Отправляет сообщение в Ollama, возвращает LLMResponse.

        Args:
            user_message:  Текст запроса пользователя
            image_base64:  Изображение в base64 (для мультимодального запроса)
            use_vision:    Использовать мультимодальную модель
        """
        model = self._vision_model if (image_base64 or use_vision) else self._model

        # Формируем контент сообщения
        user_content: Any
        if image_base64:
            user_content = [
                {"type": "text",  "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]
        else:
            user_content = user_message

        # Добавляем в историю
        self._history.append({"role": "user", "content": user_content})

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *self._history,
        ]

        payload = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options": {
                "temperature": 0.3,     # детерминизм для команд
                "num_ctx":     4096,
            },
        }

        log.debug("Запрос к Ollama [%s]: %s…", model, str(user_message)[:80])

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            log.error("Ollama не ответила за %d сек", self._timeout)
            return LLMResponse(
                raw_text="",
                reply_text="Извини, модель долго думает. Попробуй ещё раз.",
            )
        except requests.RequestException as exc:
            log.error("Ошибка соединения с Ollama: %s", exc)
            return LLMResponse(
                raw_text="",
                reply_text="Не могу подключиться к Ollama. Убедись, что она запущена.",
            )

        data    = resp.json()
        raw_msg = data.get("message", {})
        raw_text = raw_msg.get("content", "").strip()

        log.debug("Ответ Ollama: %s…", raw_text[:120])

        # Сохраняем ответ ассистента в историю
        self._history.append({"role": "assistant", "content": raw_text})
        self._trim_history()

        # Пытаемся разобрать как команду
        parsed = self._try_parse_action(raw_text)
        if parsed:
            action, params = parsed
            log.info("Распознана команда: %s(%s)", action, params)
            return LLMResponse(
                raw_text=raw_text,
                is_action=True,
                action=action,
                params=params,
            )

        return LLMResponse(raw_text=raw_text, reply_text=raw_text)

    # ── Прямой текстовый запрос (без истории) ──────────────────────────────────

    def query(self, prompt: str, system: str = "") -> str:
        """
        Одноразовый запрос без истории диалога.
        Используется для поиска, суммаризации и т.д.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":    self._model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": 0.2, "num_ctx": 4096},
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as exc:
            log.error("query() ошибка: %s", exc)
            return ""

    # ── Управление историей ─────────────────────────────────────────────────────

    def _trim_history(self, max_turns: int = 10):
        """Обрезаем историю, чтобы не переполнять контекст."""
        if len(self._history) > max_turns * 2:
            self._history = self._history[-(max_turns * 2):]

    def clear_history(self):
        self._history.clear()
        log.info("История диалога очищена.")

    def update_tools(self, tools_description: str):
        self._tools_desc = tools_description
