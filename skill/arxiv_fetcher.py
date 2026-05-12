"""Module A: fetch recent arXiv metadata via API (title + abstract only, no PDF)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    category: str
    published_date: str
    link: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "category": self.category,
            "published_date": self.published_date,
            "link": self.link,
        }


def _parse_date_range(date_range: str) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    dr = (date_range or "last_7_days").strip().lower()
    if dr == "last_7_days":
        start = end - timedelta(days=7)
    elif dr == "last_30_days":
        start = end - timedelta(days=30)
    elif dr == "last_14_days":
        start = end - timedelta(days=14)
    else:
        start = end - timedelta(days=7)
    return start, end


def _build_search_query(user_query: str, date_range: str) -> str:
    parts = [p.strip() for p in user_query.split(",") if p.strip()]
    if not parts:
        parts = ["machine learning"]
    or_clause = " OR ".join(f'all:"{p.replace(chr(34), "")}"' for p in parts)
    start, end = _parse_date_range(date_range)
    fmt = "%Y%m%d%H%M"
    submitted = f"submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"
    return f"({or_clause}) AND ({submitted})"


def _id_to_arxiv_id(entry_id: str) -> str:
    # http://arxiv.org/abs/2401.12345v2 -> 2401.12345v2
    m = re.search(r"arxiv\.org/abs/([^?#]+)", entry_id, re.I)
    return m.group(1) if m else entry_id.rsplit("/", 1)[-1]


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def fetch_papers(
    query: str,
    date_range: str = "last_7_days",
    max_papers: int = 20,
    timeout: int = 60,
) -> list[PaperRecord]:
    """
    Query arXiv API; returns up to max_papers entries, newest first.
    """
    search_query = _build_search_query(query, date_range)
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(max(1, max_papers), 200),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    papers: list[PaperRecord] = []
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id", default="")
        title = _clean_text(entry.findtext(f"{ATOM}title", default=""))
        summary = _clean_text(entry.findtext(f"{ATOM}summary", default=""))
        published = entry.findtext(f"{ATOM}published", default="")[:10]
        authors = [
            _clean_text(a.findtext(f"{ATOM}name", default=""))
            for a in entry.findall(f"{ATOM}author")
        ]
        authors = [a for a in authors if a]
        cat_el = entry.find(f"{ARXIV_NS}primary_category")
        category = cat_el.get("term", "") if cat_el is not None else ""
        aid = _id_to_arxiv_id(eid)
        link = f"https://arxiv.org/abs/{aid}"
        papers.append(
            PaperRecord(
                arxiv_id=aid,
                title=title,
                abstract=summary,
                authors=authors,
                category=category,
                published_date=published,
                link=link,
            )
        )
    return papers[:max_papers]
