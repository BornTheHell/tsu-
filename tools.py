# tools.py — OS-инструменты v4: умный поиск программ + реальное управление ОС
from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import webbrowser
import winreg
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyperclip
import pyautogui
import pygetwindow as gw   # pip install pygetwindow
from thefuzz import fuzz, process

import config

log = logging.getLogger("tools")

_REGISTRY: Dict[str, Callable[..., str]] = {}


def tool(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def execute(tool_name: str, params: Dict[str, Any]) -> str:
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return f"Неизвестный инструмент: «{tool_name}»"
    log.info("Выполняю инструмент: %s(%s)", tool_name, params)
    try:
        result = fn(**params)
        return str(result) if result is not None else "Готово."
    except TypeError as exc:
        log.error("Неверные параметры для %s: %s", tool_name, exc)
        return f"Неверные параметры: {exc}"
    except Exception as exc:
        log.exception("Ошибка %s", tool_name)
        return f"Ошибка: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
#  Словари алиасов
# ══════════════════════════════════════════════════════════════════════════════

_RU_ALIASES: Dict[str, str] = {
    # ── Браузеры ──────────────────────────────────────────────────────────────
    "браузер":           "chrome",
    "хром":              "chrome",
    "гугл хром":         "chrome",
    "гугл":              "chrome",
    "файерфокс":         "firefox",
    "опера":             "opera",
    "яндекс браузер":    "yandex browser",
    "яндекс":            "yandex browser",
    "edge":              "microsoft edge",
    "майкрософт эдж":    "microsoft edge",
    # ── Игровые платформы ─────────────────────────────────────────────────────
    "стим":              "steam",
    "стима":             "steam",
    "стеме":             "steam",
    "стемистый":         "steam",
    "эпик":              "epic games launcher",
    "эпик геймс":        "epic games launcher",
    "epic":              "epic games launcher",
    "гог":               "gog galaxy",
    "баттлнет":          "battle.net",
    "battle net":        "battle.net",
    "ориджин":           "origin",
    "юплей":             "ubisoft connect",
    "убисофт":           "ubisoft connect",
    "uplay":             "ubisoft connect",
    # ── Игры ──────────────────────────────────────────────────────────────────
    "кс":                "counter-strike",
    "ксго":              "counter-strike",
    "кс2":               "counter-strike 2",
    "контр страйк":      "counter-strike",
    "counter strike":    "counter-strike",
    "cs2":               "counter-strike 2",
    "дедлок":            "deadlock",
    "deadlock":          "deadlock",
    "террария":          "terraria",
    "terraria":          "terraria",
    "тмод":              "tmodloader",
    "tmod":              "tmodloader",
    "майнкрафт":         "minecraft",
    "minecraft":         "minecraft",
    "доту":              "dota 2",
    "дота":              "dota 2",
    "dota":              "dota 2",
    "гта":               "grand theft auto",
    "гта 5":             "grand theft auto v",
    "gta":               "grand theft auto",
    "вот":               "world of tanks",
    "танки":             "world of tanks",
    "варфрейм":          "warframe",
    "warframe":          "warframe",
    "апекс":             "apex legends",
    "apex":              "apex legends",
    "овервотч":          "overwatch",
    "overwatch":         "overwatch",
    "вэлорант":          "valorant",
    "valorant":          "valorant",
    "раст":              "rust",
    "rust":              "rust",
    "pubg":              "playerunknown",
    "пабг":              "playerunknown",
    "дивинити":          "divinity",
    "рдр":               "red dead",
    "ведьмак":           "witcher",
    "киберпанк":         "cyberpunk",
    "элден ринг":        "elden ring",
    "хогвартс":          "hogwarts legacy",
    # ── Мессенджеры ───────────────────────────────────────────────────────────
    "дискорд":           "discord",
    "дискор":            "discord",
    "телеграм":          "telegram",
    "телега":            "telegram",
    "вацап":             "whatsapp",
    "ватсап":            "whatsapp",
    "скайп":             "skype",
    "вайбер":            "viber",
    # ── Офис ──────────────────────────────────────────────────────────────────
    "блокнот":           "notepad",
    "ворд":              "microsoft word",
    "word":              "microsoft word",
    "эксель":            "microsoft excel",
    "excel":             "microsoft excel",
    "пауэрпоинт":        "microsoft powerpoint",
    # ── Медиа ─────────────────────────────────────────────────────────────────
    "спотифай":          "spotify",
    "влс":               "vlc",
    "медиаплеер":        "vlc",
    "потокс":            "potplayer",
    # ── Разработка ────────────────────────────────────────────────────────────
    "вс код":            "visual studio code",
    "vs code":           "visual studio code",
    "студия":            "visual studio",
    "идея":              "intellij idea",
    "питон":             "python",
    # ── Системное ─────────────────────────────────────────────────────────────
    "проводник":         "explorer",
    "калькулятор":       "calculator",
    "диспетчер задач":   "task manager",
    "диспетчер":         "task manager",
    "командная строка":  "cmd",
    "реестр":            "regedit",
    # ── Прочее ────────────────────────────────────────────────────────────────
    "зум":               "zoom",
    "обс":               "obs studio",
    "obs":               "obs studio",
    "фотошоп":           "photoshop",
    "ворон":             "voron",
    "ардуино":           "arduino",
}

# Системные exe которые всегда доступны через PATH
_SYSTEM_APPS: Dict[str, str] = {
    "notepad":     "notepad.exe",
    "calc":        "calc.exe",
    "calculator":  "calc.exe",
    "mspaint":     "mspaint.exe",
    "explorer":    "explorer.exe",
    "taskmgr":     "taskmgr.exe",
    "task manager": "taskmgr.exe",
    "cmd":         "cmd.exe",
    "powershell":  "powershell.exe",
    "regedit":     "regedit.exe",
    "mstsc":       "mstsc.exe",
    "control":     "control.exe",
    "snippingtool": "SnippingTool.exe",
    "notepad++":   "notepad++",   # если в PATH
}

# ══════════════════════════════════════════════════════════════════════════════
#  Сканер установленных программ
# ══════════════════════════════════════════════════════════════════════════════

# Паттерны имён файлов которые НЕ надо запускать (установщики, деинсталляторы)
_SKIP_PATTERNS = re.compile(
    r"(uninstall|uninst|setup|install|update|updater|crash|helper|"
    r"redist|vcredist|directx|runtime|dxsetup|vc_redist|"
    r"dotnet|framework|prerequisit)",
    re.IGNORECASE,
)


def _is_launcher_exe(path: str) -> bool:
    """True если файл — явно установщик/деинсталлятор, а не приложение."""
    name = Path(path).stem.lower()
    return bool(_SKIP_PATTERNS.search(name))


def _collect_candidates() -> Dict[str, str]:
    """
    Возвращает словарь {нижний_регистр_названия: путь_к_exe_или_lnk}.

    Источники (приоритет по порядку):
    1. Start Menu .lnk — самый надёжный источник (там уже человекочитаемые имена)
    2. Реестр Uninstall (DisplayName → DisplayIcon / InstallLocation)
    3. Системные exe
    """
    candidates: Dict[str, str] = {}

    # ── 1. Start Menu ярлыки (.lnk) ───────────────────────────────────────────
    start_dirs = [
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
    ]
    for sdir in start_dirs:
        for lnk in glob.glob(os.path.join(sdir, "**", "*.lnk"), recursive=True):
            stem = Path(lnk).stem.lower()
            # Пропускаем деинсталляторы в ярлыках
            if _SKIP_PATTERNS.search(stem):
                continue
            # .lnk умеет запускаться через os.startfile — это надёжнее subprocess
            candidates[stem] = lnk

    # ── 2. Реестр Uninstall ───────────────────────────────────────────────────
    reg_paths = [
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER,
         r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, path in reg_paths:
        try:
            with winreg.OpenKey(hive, path) as key:
                idx = 0
                while True:
                    try:
                        sub_name = winreg.EnumKey(key, idx)
                        idx += 1
                        with winreg.OpenKey(key, sub_name) as sub:
                            _reg_add_entry(sub, candidates)
                    except OSError:
                        break
        except (FileNotFoundError, PermissionError):
            pass

    # ── 3. Системные ──────────────────────────────────────────────────────────
    candidates.update(_SYSTEM_APPS)

    log.debug("Кандидатов найдено: %d", len(candidates))
    return candidates


def _reg_add_entry(sub_key, candidates: dict) -> None:
    """Читает один ключ реестра и добавляет exe в candidates."""
    try:
        display_name, _ = winreg.QueryValueEx(sub_key, "DisplayName")
    except (FileNotFoundError, OSError):
        return

    dn = str(display_name).strip()
    dn_lower = dn.lower()

    # Пропускаем записи типа "Microsoft Visual C++ 2019 Redistributable"
    if _SKIP_PATTERNS.search(dn_lower):
        return

    exe_path: Optional[str] = None

    # Пробуем DisplayIcon
    try:
        icon_raw, _ = winreg.QueryValueEx(sub_key, "DisplayIcon")
        icon_exe = icon_raw.split(",")[0].strip().strip('"')
        if (icon_exe.lower().endswith(".exe")
                and os.path.isfile(icon_exe)
                and not _is_launcher_exe(icon_exe)):
            exe_path = icon_exe
    except (FileNotFoundError, OSError):
        pass

    # Пробуем InstallLocation — ищем main exe там
    if not exe_path:
        try:
            loc, _ = winreg.QueryValueEx(sub_key, "InstallLocation")
            loc = loc.strip().strip('"')
            if loc and os.path.isdir(loc):
                exe_path = _find_main_exe(loc, dn)
        except (FileNotFoundError, OSError):
            pass

    if exe_path:
        candidates[dn_lower] = exe_path
        stem = Path(exe_path).stem.lower()
        candidates.setdefault(stem, exe_path)
    elif dn_lower not in candidates:
        # Нет exe, но есть имя — добавим на случай нечёткого поиска
        candidates[dn_lower] = dn_lower


def _find_main_exe(folder: str, app_name: str) -> Optional[str]:
    """
    Ищет главный exe в папке приложения.
    Алгоритм:
      1. Exe с именем похожим на app_name (нечёткий поиск)
      2. Самый тяжёлый exe (скорее всего главный)
    """
    try:
        exes = [
            f for f in glob.glob(os.path.join(folder, "*.exe"))
            if not _is_launcher_exe(f)
        ]
        if not exes:
            # Ищем глубже (один уровень)
            exes = [
                f for f in glob.glob(os.path.join(folder, "*", "*.exe"))
                if not _is_launcher_exe(f)
            ]
        if not exes:
            return None

        # Нечёткое сопоставление имени exe с именем приложения
        stems = [Path(e).stem.lower() for e in exes]
        best, score = process.extractOne(
            app_name.lower(), stems, scorer=fuzz.partial_ratio
        )
        if score >= 60:
            idx = stems.index(best)
            return exes[idx]

        # Фоллбэк: берём самый тяжёлый файл
        return max(exes, key=os.path.getsize)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  launch_app
# ══════════════════════════════════════════════════════════════════════════════

@tool("launch_app")
def launch_app(query: str) -> str:
    """Запускает приложение или игру по названию."""
    q = query.lower().strip()

    # ── 1. Алиас ──────────────────────────────────────────────────────────────
    resolved = q
    for ru_key, en_val in _RU_ALIASES.items():
        if ru_key in q:
            resolved = en_val
            log.debug("Alias: «%s» → «%s»", q, resolved)
            break

    # ── 2. Кандидаты ──────────────────────────────────────────────────────────
    candidates = _collect_candidates()
    names = list(candidates.keys())

    # ── 3. Нечёткий поиск (token_set_ratio лучше для частичных совпадений) ────
    best_match, score = process.extractOne(
        resolved, names, scorer=fuzz.token_set_ratio
    )
    log.info("Поиск «%s»: лучший «%s» (score=%d)", resolved, best_match, score)

    if score < 40:
        return f"«{query}» не найдена. Проверь название."

    target = candidates[best_match]
    return _launch_target(best_match, target)


def _launch_target(name: str, target: str) -> str:
    """Запускает файл двумя способами. Логирует результат."""
    log.info("Запуск: %s → %s", name, target)

    # Способ 1: os.startfile — лучший для .lnk ярлыков
    try:
        os.startfile(target)
        return f"Запускаю {name}."
    except Exception as e1:
        log.warning("os.startfile не сработал (%s): %s", target, e1)

    # Способ 2: subprocess.Popen — для exe напрямую
    try:
        subprocess.Popen(
            f'"{target}"',
            shell=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        return f"Запускаю {name}."
    except Exception as e2:
        log.error("subprocess.Popen не сработал (%s): %s", target, e2)

    return f"Не удалось запустить {name}."


# ══════════════════════════════════════════════════════════════════════════════
#  type_in_window — написать текст в активном окне (РЕАЛЬНЫЕ РУКИ)
# ══════════════════════════════════════════════════════════════════════════════

@tool("type_text")
def type_text(text: str, window_title: str = "") -> str:
    """
    Печатает текст в указанном окне (или в активном, если окно не задано).
    Использует pyautogui.write() через клипборд чтобы корректно работал Unicode.
    """
    import time

    if window_title:
        # Ищем окно по подстроке заголовка
        matches = [w for w in gw.getAllWindows()
                   if window_title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return f"Окно «{window_title}» не найдено."
        win = matches[0]
        win.activate()
        time.sleep(0.3)   # даём окну активироваться

    # Вставляем через буфер обмена — единственный надёжный способ для кириллицы
    old_clipboard = ""
    try:
        old_clipboard = pyperclip.paste()
    except Exception:
        pass

    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")

    # Восстанавливаем буфер обмена через секунду
    import threading
    def _restore():
        import time as t
        t.sleep(1.0)
        try:
            pyperclip.copy(old_clipboard)
        except Exception:
            pass
    threading.Thread(target=_restore, daemon=True).start()

    log.info("type_text: «%s» → окно «%s»", text[:50], window_title or "активное")
    return f"Написал: {text[:50]}"


@tool("focus_window")
def focus_window(title: str) -> str:
    """Переключается на окно по части заголовка."""
    import time
    matches = [w for w in gw.getAllWindows()
               if title.lower() in w.title.lower() and w.title.strip()]
    if not matches:
        return f"Окно «{title}» не найдено."
    win = matches[0]
    win.activate()
    time.sleep(0.2)
    log.info("Активировано окно: %s", win.title)
    return f"Переключился на «{win.title}»."


@tool("press_keys")
def press_keys(keys: str) -> str:
    """
    Нажимает комбинацию клавиш.
    Формат: 'ctrl+c', 'alt+f4', 'win+d', 'enter', 'escape' и т.д.
    """
    key_list = [k.strip().lower() for k in keys.split("+")]
    pyautogui.hotkey(*key_list)
    log.info("Нажата комбинация: %s", keys)
    return f"Нажал {keys}."


# ══════════════════════════════════════════════════════════════════════════════
#  Остальные инструменты
# ══════════════════════════════════════════════════════════════════════════════

@tool("read_clipboard")
def read_clipboard() -> str:
    try:
        text = pyperclip.paste()
        return f"В буфере ({len(text)} симв.): {text[:500]}" if text else "Буфер пуст."
    except Exception as exc:
        return f"Не могу прочитать буфер: {exc}"


@tool("take_screenshot")
def take_screenshot() -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SCREENSHOTS_DIR, f"screenshot_{ts}.png")
    try:
        pyautogui.screenshot().save(path)
        return f"Скриншот: {path}"
    except Exception as exc:
        return f"Не удалось: {exc}"


@tool("open_url")
def open_url(url: str) -> str:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    webbrowser.open(url)
    return f"Открываю {url}"


@tool("run_cmd")
def run_cmd(command: str) -> str:
    log.info("CMD: %s", command)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, encoding="cp866", errors="replace", timeout=20,
        )
        out = (result.stdout + result.stderr).strip()
        return out[:600] if out else "Выполнено."
    except subprocess.TimeoutExpired:
        return "Таймаут (20 сек)."
    except Exception as exc:
        return f"Ошибка: {exc}"


@tool("set_volume")
def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        devices = AudioUtilities.GetSpeakers()
        iface   = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume  = cast(iface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"Громкость: {level}%."
    except ImportError:
        pass
    for nircmd in (r"C:\tools\nircmd.exe", r"C:\nircmd\nircmd.exe"):
        if os.path.exists(nircmd):
            subprocess.run([nircmd, "setsysvolume", str(int(level/100*65535))])
            return f"Громкость: {level}%."
    return "Установи pycaw: pip install pycaw"
