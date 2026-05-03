# brain.py — Ollama API + история + фикс парсера JSON
from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import requests

import config

log = logging.getLogger("brain")

# ══════════════════════════════════════════════════════════════════════════════
#  Фразы-признаки отказа
# ══════════════════════════════════════════════════════════════════════════════

_REFUSAL_MARKERS = (
    "не могу позволить",
    "противоречит правилам",
    "политике системы",
    "не могу выполнить эту",
    "извините, но я не могу",
)


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


# ══════════════════════════════════════════════════════════════════════════════
#  Структуры ответа
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    raw: str
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  Парсер JSON — устойчив к }}} и мусору после JSON
# ══════════════════════════════════════════════════════════════════════════════

def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Извлекает первый валидный JSON-объект из текста.
    Считает глубину скобок — останавливается ровно когда объект закрыт.
    Игнорирует лишние }} и текст после JSON.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]  # первый корректно закрытый объект

    return None


def _try_parse_tool(raw: str) -> Optional[Tuple[str, dict]]:
    """Пытается извлечь action+params из ответа модели."""
    # Убираем markdown-блоки
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    candidate = _extract_balanced_json(cleaned)
    if not candidate:
        return None

    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "action" in data:
            return str(data["action"]), dict(data.get("params", {}))
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ChatHistory
# ══════════════════════════════════════════════════════════════════════════════

class ChatHistory:
    def __init__(self, maxlen: int = config.LLM_HISTORY_LEN) -> None:
        self._buf: Deque[Dict[str, str]] = deque(maxlen=maxlen * 2)

    def add_user(self, text: str) -> None:
        self._buf.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._buf.append({"role": "assistant", "content": text})

    def as_list(self) -> list[Dict[str, str]]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def purge_refusals(self) -> int:
        original = list(self._buf)
        cleaned = [
            m for m in original
            if not (m["role"] == "assistant" and _is_refusal(m["content"]))
        ]
        removed = len(original) - len(cleaned)
        if removed:
            self._buf.clear()
            self._buf.extend(cleaned)
            log.info("Purge: удалено %d отказов из истории.", removed)
        return removed


# ══════════════════════════════════════════════════════════════════════════════
#  Brain
# ══════════════════════════════════════════════════════════════════════════════

class Brain:
    def __init__(self) -> None:
        self._history        = ChatHistory()
        self._url            = f"{config.OLLAMA_BASE_URL}/api/chat"
        self._refusal_streak = 0
        log.info("Brain инициализирован. Модель: %s", config.OLLAMA_MODEL)

    # ── Публичный API ──────────────────────────────────────────────────────────

    def chat(self, user_text: str, options: dict = None) -> LLMResponse:
        self._history.add_user(user_text)

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            *self._history.as_list(),
        ]

        payload = {
            "model":    config.OLLAMA_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  options if options is not None else config.LLM_OPTIONS,
        }

        log.debug("→ Ollama [%d msg]: «%s»", len(self._history.as_list()), user_text[:80])

        raw_text = self._call_ollama(payload)
        log.debug("← Ollama raw: «%s»", raw_text[:200])

        # ── Детектируем отказ ──────────────────────────────────────────────────
        if _is_refusal(raw_text):
            self._refusal_streak += 1
            log.warning("Отказ #%d. Чищу историю.", self._refusal_streak)
            self._history.purge_refusals()
            if self._refusal_streak >= 2:
                self._history.clear()
                self._refusal_streak = 0
                log.warning("Два отказа подряд — история сброшена.")
            raw_text = "Окей."
        else:
            self._refusal_streak = 0
            self._history.add_assistant(raw_text)

        return self._build_response(raw_text)

    def inject_tool_result(self, tool_name: str, result: str) -> None:
        self._history.add_assistant(f"[{tool_name}: {result}]")

    def clear_history(self) -> None:
        self._history.clear()
        self._refusal_streak = 0
        log.info("История очищена.")

    # ── HTTP ───────────────────────────────────────────────────────────────────

    def _call_ollama(self, payload: dict) -> str:
        try:
            resp = requests.post(
                self._url,
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()

        except requests.Timeout:
            log.error("Ollama: таймаут (%d сек)", config.OLLAMA_TIMEOUT)
            return "Ollama думает слишком долго, попробуй ещё раз."

        except requests.ConnectionError:
            log.error("Ollama: нет соединения")
            return "Ollama не запущена. Открой терминал и запусти: ollama serve"

        except Exception as exc:
            log.exception("Ollama: неизвестная ошибка")
            return f"Ошибка: {exc}"

    # ── Парсинг ответа ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_response(raw: str) -> LLMResponse:
        parsed = _try_parse_tool(raw)
        if parsed:
            action, params = parsed
            log.info("Tool call: %s(%s)", action, params)
            return LLMResponse(
                raw=raw,
                is_tool_call=True,
                tool_name=action,
                tool_params=params,
            )
        log.info("Текстовый ответ: «%s»", raw[:80])
        return LLMResponse(raw=raw, text=raw)
