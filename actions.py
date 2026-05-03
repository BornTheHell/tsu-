# actions.py — Модуль "Руки": управление Windows
from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyautogui

import config
from logger import get_logger
from tools_registry import registry

log = get_logger("Actions")

# ════════════════════════════════════════════════════════════════════════════════
#  Регистрация встроенных инструментов
# ════════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="open_browser",
    description="Открыть веб-браузер по указанному URL",
    params_schema={
        "url": {"type": "string", "description": "Полный URL, например https://google.com"},
    },
)
def open_browser(url: str = "https://google.com") -> str:
    log.info("Открываю браузер: %s", url)
    webbrowser.open(url)
    return f"Браузер открыт: {url}"


@registry.register(
    name="open_app",
    description="Открыть приложение по имени исполняемого файла или пути",
    params_schema={
        "app": {"type": "string", "description": "Имя exe или полный путь (notepad, calc, explorer…)"},
    },
)
def open_app(app: str) -> str:
    log.info("Запускаю: %s", app)
    subprocess.Popen(app, shell=True)
    return f"Запущено: {app}"


@registry.register(
    name="close_app",
    description="Закрыть процесс по имени (например, notepad.exe)",
    params_schema={
        "process_name": {"type": "string", "description": "Имя процесса без пути"},
    },
)
def close_app(process_name: str) -> str:
    log.info("Завершаю процесс: %s", process_name)
    result = subprocess.run(
        ["taskkill", "/F", "/IM", process_name],
        capture_output=True, text=True, encoding="cp866"
    )
    if result.returncode == 0:
        return f"Процесс {process_name} завершён."
    return f"Не удалось завершить {process_name}: {result.stderr.strip()}"


@registry.register(
    name="take_screenshot",
    description="Сделать скриншот экрана и сохранить файл",
    params_schema={},
)
def take_screenshot() -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SCREENSHOTS_DIR, f"screenshot_{ts}.png")
    img  = pyautogui.screenshot()
    img.save(path)
    log.info("Скриншот сохранён: %s", path)
    return f"Скриншот сохранён: {path}"


@registry.register(
    name="create_file",
    description="Создать текстовый файл с указанным содержимым",
    params_schema={
        "path":    {"type": "string", "description": "Путь к файлу (полный или относительный)"},
        "content": {"type": "string", "description": "Текстовое содержимое файла"},
    },
)
def create_file(path: str, content: str = "") -> str:
    full = Path(path).expanduser().resolve()
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    log.info("Файл создан: %s", full)
    return f"Файл создан: {full}"


@registry.register(
    name="read_file",
    description="Прочитать текстовый файл и вернуть его содержимое",
    params_schema={
        "path": {"type": "string", "description": "Путь к файлу"},
    },
)
def read_file(path: str) -> str:
    full = Path(path).expanduser().resolve()
    if not full.exists():
        return f"Файл не найден: {full}"
    text = full.read_text(encoding="utf-8", errors="replace")
    log.info("Прочитан файл: %s (%d симв.)", full, len(text))
    return text[:4000] if len(text) > 4000 else text   # ограничение для TTS


@registry.register(
    name="set_volume",
    description="Установить громкость системы (0–100)",
    params_schema={
        "level": {"type": "integer", "description": "Уровень громкости от 0 до 100"},
    },
)
def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    # nircmd.exe — рекомендую скачать отдельно; fallback через PowerShell
    ps_cmd = (
        f"$obj = New-Object -ComObject WScript.Shell; "
        f"$obj.SendKeys([char]174 * 50); "   # mute первым нажатием
    )
    # Более надёжный способ через PowerShell + WMI
    ps_set = (
        f"(Get-WmiObject -Namespace root/cimv2 -Class Win32_PnPEntity | "
        f"Where-Object {{$_.Name -like '*Audio*'}}) | ForEach-Object {{}} ; "
        f"[audio]::Volume = {level / 100}"
    )
    # Простейший вариант через nircmd (если установлен)
    nircmd = rf"C:\tools\nircmd.exe setsysvolume {int(level / 100 * 65535)}"
    try:
        subprocess.run(nircmd, shell=True, check=True, capture_output=True)
        return f"Громкость установлена: {level}%"
    except Exception:
        log.warning("nircmd не найден, использую PowerShell")
        # PowerShell через PInvoke — работает без nircmd
        ps = (
            "Add-Type -TypeDefinition '"
            "using System.Runtime.InteropServices; "
            "[Guid(\"5CDF2C82-841E-4546-9722-0CF74078229A\"), "
            "InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] "
            "interface IAudioEndpointVolume { }';"
        )
        # Fallback: просто сообщаем (реализацию через pycaw можно добавить)
        return f"Установка громкости: {level}% (nircmd не найден — добавьте pycaw)"


@registry.register(
    name="type_text",
    description="Напечатать текст в активном окне через клавиатуру",
    params_schema={
        "text": {"type": "string", "description": "Текст для ввода"},
    },
)
def type_text(text: str) -> str:
    pyautogui.write(text, interval=0.03)
    return f"Напечатан текст: {text[:40]}…"


@registry.register(
    name="press_keys",
    description="Нажать комбинацию клавиш (например, ctrl+c, alt+f4)",
    params_schema={
        "keys": {"type": "string", "description": "Комбинация через +, например 'ctrl+c'"},
    },
)
def press_keys(keys: str) -> str:
    key_list = [k.strip() for k in keys.lower().split("+")]
    pyautogui.hotkey(*key_list)
    return f"Нажата комбинация: {keys}"


@registry.register(
    name="run_command",
    description="Выполнить произвольную команду cmd/PowerShell",
    params_schema={
        "cmd": {"type": "string", "description": "Команда для выполнения в cmd"},
    },
)
def run_command(cmd: str) -> str:
    log.info("Выполняю команду: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, encoding="cp866", timeout=15
        )
        out = (result.stdout + result.stderr).strip()
        return out[:2000] if out else "Команда выполнена без вывода."
    except subprocess.TimeoutExpired:
        return "Команда превысила таймаут (15 сек)"
    except Exception as exc:
        return f"Ошибка: {exc}"


# ════════════════════════════════════════════════════════════════════════════════
#  ActionEngine — обёртка для вызова из JarvisCore
# ════════════════════════════════════════════════════════════════════════════════

class ActionEngine:
    """
    Тонкая обёртка над глобальным реестром.
    JarvisCore работает именно с этим объектом.
    """

    def execute(self, action: str, params: dict) -> str:
        return registry.execute(action, params)

    def tools_description(self) -> str:
        return registry.tools_description()

    def has_action(self, action: str) -> bool:
        return action in registry
