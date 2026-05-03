# tools_registry.py — Реестр функций (Tools/Actions)
# Добавляй новые команды, просто используя декоратор @registry.register(...)

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from logger import get_logger

log = get_logger("Registry")


@dataclass
class ToolDefinition:
    """Описание одного инструмента для LLM."""
    name:        str
    description: str
    params_schema: Dict[str, Any]          # JSON Schema для параметров
    handler:     Callable[..., str]        # функция-исполнитель


class ToolsRegistry:
    """
    Реестр всех доступных инструментов.
    Позволяет регистрировать новые команды через декоратор или явный вызов.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    # ── Регистрация ─────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        description: str,
        params_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Декоратор для регистрации функции как инструмента.

        Пример использования:
            @registry.register(
                name="open_browser",
                description="Открыть браузер по URL",
                params_schema={"url": {"type": "string", "description": "URL для открытия"}}
            )
            def open_browser(url: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            tool = ToolDefinition(
                name=name,
                description=description,
                params_schema=params_schema or {},
                handler=fn,
            )
            self._tools[name] = tool
            log.debug("Зарегистрирован инструмент: %s", name)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def add(self, tool: ToolDefinition):
        """Явная регистрация объекта ToolDefinition."""
        self._tools[tool.name] = tool
        log.debug("Добавлен инструмент: %s", tool.name)

    # ── Выполнение ──────────────────────────────────────────────────────────────

    def execute(self, action: str, params: Dict[str, Any]) -> str:
        """
        Выполняет инструмент по имени.
        Возвращает строку-результат (для ответа пользователю).
        """
        tool = self._tools.get(action)
        if not tool:
            msg = f"Неизвестное действие: «{action}»"
            log.warning(msg)
            return msg

        log.info("Выполняю: %s(%s)", action, params)
        try:
            result = tool.handler(**params)
            return str(result) if result is not None else "Готово."
        except TypeError as exc:
            msg = f"Неверные параметры для {action}: {exc}"
            log.error(msg)
            return msg
        except Exception as exc:
            msg = f"Ошибка выполнения {action}: {exc}"
            log.exception(msg)
            return msg

    # ── Интроспекция ────────────────────────────────────────────────────────────

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def tools_description(self) -> str:
        """Строка для подстановки в системный промпт LLM."""
        lines = []
        for t in self._tools.values():
            params_str = ", ".join(
                f"{k}: {v.get('type','any')} — {v.get('description','')}"
                for k, v in t.params_schema.items()
            )
            lines.append(f"  • {t.name}({params_str}): {t.description}")
        return "\n".join(lines) if lines else "  (нет зарегистрированных инструментов)"

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Глобальный реестр — импортируй его в actions.py и других модулях
registry = ToolsRegistry()
