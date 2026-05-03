# search.py — Модуль "Поиск": DuckDuckGo + суммаризация через LLM
from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass

from logger import get_logger

log = get_logger("Search")


@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str


class WebSearchEngine:
    """
    Локальный веб-поиск через DuckDuckGo (без API-ключей).
    pip install duckduckgo-search

    Опционально суммаризирует результаты через LLM.
    """

    def __init__(self, llm_engine=None, max_results: int = 5):
        self._llm         = llm_engine
        self._max_results = max_results

    def set_llm(self, llm_engine):
        self._llm = llm_engine

    # ── Поиск ───────────────────────────────────────────────────────────────────

    def search(self, query: str) -> List[SearchResult]:
        """Возвращает список результатов поиска."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            log.error("duckduckgo-search не установлен. pip install duckduckgo-search")
            return []

        log.info("Поиск: «%s»", query)
        results: List[SearchResult] = []

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self._max_results, region="ru-ru"):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    ))
        except Exception as exc:
            log.error("DuckDuckGo ошибка: %s", exc)

        log.info("Найдено %d результатов", len(results))
        return results

    # ── Суммаризация ─────────────────────────────────────────────────────────────

    def search_and_summarize(self, query: str) -> str:
        """
        Ищет в интернете и возвращает суммаризированный ответ через LLM.
        Если LLM недоступна — возвращает первый snippet.
        """
        results = self.search(query)
        if not results:
            return "Поиск не дал результатов."

        if self._llm is None:
            # Без LLM — просто возвращаем сниппеты
            parts = [f"• {r.title}: {r.snippet}" for r in results[:3]]
            return "\n".join(parts)

        # Формируем контекст для LLM
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {r.title}\n{r.snippet}\nИсточник: {r.url}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            f"Вопрос пользователя: {query}\n\n"
            f"Результаты поиска:\n{context}\n\n"
            f"На основе этих данных дай краткий ответ на русском языке. "
            f"Не копируй текст дословно — сделай выжимку."
        )

        system = (
            "Ты — умный ассистент. Используй только предоставленный контекст "
            "для ответа. Будь лаконичен."
        )

        log.info("Суммаризирую результаты через LLM…")
        answer = self._llm.query(prompt, system=system)
        return answer if answer else results[0].snippet

    # ── Быстрый факт-чек ────────────────────────────────────────────────────────

    def quick_fact(self, query: str) -> str:
        """
        Быстрый ответ на фактический вопрос.
        Использует DuckDuckGo instant answers API.
        """
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.answers(query))
                if results:
                    return results[0].get("text", "")
        except Exception as exc:
            log.warning("quick_fact ошибка: %s", exc)
        return ""
