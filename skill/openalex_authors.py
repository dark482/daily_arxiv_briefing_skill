"""OpenAlex: look up each arXiv author by display name → cited_by_count (search API)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

OPENALEX_API = "https://api.openalex.org"

# https://docs.openalex.org/how-to-use-api/rate-limits-and-authentication
DEFAULT_MAILTO = "you@example.com"


def _norm_name_key(name: str) -> str:
    """Cache key: lowercase, collapse whitespace."""
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return s


class OpenAlexAuthorRanker:
    """
    For each distinct author string from arXiv, query OpenAlex ``/authors?search=...``,
    take the **first** hit's ``cited_by_count``, and sum over authors on the paper.

    Name collisions (common names → wrong profile) are inherent; see README.
    """

    def __init__(
        self,
        mailto: str = DEFAULT_MAILTO,
        author_cache_path: Path | None = None,
        request_sleep_s: float = 0.11,
    ):
        self.mailto = (mailto or DEFAULT_MAILTO).strip()
        self.author_cache_path = author_cache_path
        self.request_sleep_s = max(0.0, float(request_sleep_s))
        self._session = requests.Session()
        self._author_cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.author_cache_path and self.author_cache_path.is_file():
            raw = json.loads(self.author_cache_path.read_text(encoding="utf-8"))
            self._author_cache = dict(raw)

    def save_caches(self) -> None:
        if not self.author_cache_path:
            return
        self.author_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.author_cache_path.write_text(
            json.dumps(self._author_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.request_sleep_s:
            time.sleep(self.request_sleep_s)
        q = dict(params)
        q["mailto"] = self.mailto
        r = self._session.get(
            url,
            params=q,
            timeout=50,
            headers={
                "User-Agent": f"daily_arxiv_briefing_openalex/1.0 (mailto:{self.mailto})",
            },
        )
        r.raise_for_status()
        return r.json()

    def cited_by_for_name(self, display_name: str) -> tuple[int, bool]:
        """
        Search OpenAlex authors by name; use top result's cited_by_count.

        Returns:
            (cited_by_count, found) — found False if search returned no results.
        """
        raw = (display_name or "").strip()
        if not raw:
            return 0, False
        key = _norm_name_key(raw)
        if key in self._author_cache:
            ent = self._author_cache[key]
            return int(ent.get("cited_by_count", 0)), bool(ent.get("found"))

        data = self._get(
            f"{OPENALEX_API}/authors",
            {"search": raw, "per_page": 5},
        )
        results = data.get("results") or []
        if not results:
            self._author_cache[key] = {"cited_by_count": 0, "found": False}
            return 0, False

        top = results[0]
        c = int(top.get("cited_by_count", 0) or 0)
        self._author_cache[key] = {
            "cited_by_count": c,
            "found": True,
            "openalex_author_id": top.get("id"),
            "matched_display_name": top.get("display_name"),
        }
        return c, True

    def paper_author_citations_total(
        self, author_names: list[str]
    ) -> tuple[int, int, int]:
        """
        Sum cited_by_count (top search hit per distinct name).

        Returns:
            (total_citations, num_distinct_names_used, num_names_with_openalex_hit)
        """
        seen: set[str] = set()
        total = 0
        hits = 0
        for raw in author_names:
            name = (raw or "").strip()
            key = _norm_name_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            c, found = self.cited_by_for_name(name)
            total += c
            if found:
                hits += 1
        return total, len(seen), hits
