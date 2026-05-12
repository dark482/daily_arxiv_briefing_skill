"""Orchestration: arXiv fetch → OpenAlex author citations (rank) → embedding graph only → report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skill.arxiv_fetcher import fetch_papers
from skill.extractor import extract_fields
from skill.graph_builder import build_graph_bundle
from skill.openalex_authors import OpenAlexAuthorRanker
from skill.ranker import embed_texts
from skill.report_generator import build_markdown_report_openalex, render_similarity_graph, write_csv


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


class DailyArxivOpenAlexSkill:
    """
    Ranking: arXiv author names → OpenAlex ``/authors?search=`` (top hit) ``cited_by_count``, summed.
    Cosine embedding similarity is used only inside ``build_graph_bundle`` (edges + centrality).
    """

    def __init__(
        self,
        outputs_dir: Path | None = None,
        similarity_edge_threshold: float = 0.55,
        model_name: str | None = None,
        force_redownload: bool = False,
        openalex_mailto: str | None = None,
        openalex_author_cache: Path | None = None,
    ):
        root = _package_root()
        self.outputs_dir = outputs_dir or (root / "outputs")
        self.similarity_edge_threshold = similarity_edge_threshold
        self.model_name = model_name
        self.force_redownload = force_redownload
        self.openalex_mailto = openalex_mailto
        self.openalex_author_cache = openalex_author_cache or (
            root / "data" / "openalex_author_name_cache.json"
        )

    def run(self, inp: dict[str, Any]) -> dict[str, Any]:
        query = str(inp.get("query", "")).strip()
        date_range = str(inp.get("date_range", "last_7_days"))
        max_papers = int(inp.get("max_papers", 20))
        top_k = int(inp.get("top_k", 5))
        force_hf = bool(inp.get("force_redownload", self.force_redownload))
        mailto = str(
            inp.get("openalex_mailto") or self.openalex_mailto or "user@example.com"
        ).strip()

        records = fetch_papers(query, date_range=date_range, max_papers=max_papers)
        if not records:
            empty = {
                "summary_table": [],
                "highlight_papers": [],
                "keyword_network": "{}",
                "daily_report": "# Briefing\n\nNo papers found for this query/date window.",
            }
            return empty

        texts = [f"{r.title} {r.abstract}" for r in records]
        emb = embed_texts(texts, model_name=self.model_name, force_redownload=force_hf)

        oa = OpenAlexAuthorRanker(
            mailto=mailto,
            author_cache_path=self.openalex_author_cache,
        )

        rows_dict: list[dict[str, Any]] = []
        for r in records:
            ex = extract_fields(r.title, r.abstract, query)
            cites, n_names, n_resolved = oa.paper_author_citations_total(r.authors)
            row = {
                **r.to_dict(),
                **ex,
                "author_citations_total": int(cites),
                "openalex_matched": n_resolved > 0,
                "openalex_author_count": int(n_names),
                "openalex_authors_resolved": int(n_resolved),
            }
            rows_dict.append(row)
        oa.save_caches()

        gb = build_graph_bundle(
            rows_dict, emb, similarity_threshold=self.similarity_edge_threshold
        )
        for row in rows_dict:
            aid = row["arxiv_id"]
            row["centrality_score"] = float(gb.centrality.get(aid, 0.0))
            row["community_id"] = int(gb.communities.get(aid, 0))

        max_cites = max((r["author_citations_total"] for r in rows_dict), default=0)
        denom = float(max_cites) if max_cites > 0 else 1.0
        for row in rows_dict:
            row["final_score"] = float(row["author_citations_total"]) / denom

        rows_dict.sort(
            key=lambda x: (x["author_citations_total"], x["centrality_score"]),
            reverse=True,
        )
        final_scores = {r["arxiv_id"]: r["final_score"] for r in rows_dict}
        communities = {r["arxiv_id"]: r["community_id"] for r in rows_dict}

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        table_path = self.outputs_dir / "summary_table.csv"
        csv_fields = [
            "arxiv_id",
            "title",
            "published_date",
            "author_citations_total",
            "openalex_matched",
            "openalex_author_count",
            "openalex_authors_resolved",
            "centrality_score",
            "final_score",
            "community_id",
            "key_methods",
        ]
        csv_rows = []
        for r in rows_dict:
            cr = {k: r.get(k) for k in csv_fields}
            cr["key_methods"] = ";".join(r.get("key_methods", []))
            csv_rows.append(cr)
        write_csv(csv_rows, table_path)

        fig_path = self.outputs_dir / "paper_similarity_graph.png"
        render_similarity_graph(
            gb.similarity_graph,
            rows_dict,
            fig_path,
            final_scores=final_scores,
            communities=communities,
            similarity_threshold=self.similarity_edge_threshold,
        )

        top = rows_dict[:top_k]
        md = build_markdown_report_openalex(
            query=query,
            date_range=date_range,
            rows=rows_dict,
            highlight=top,
            keyword_network_summary=gb.keyword_network_json,
        )
        (self.outputs_dir / "daily_report.md").write_text(md, encoding="utf-8")

        summary_table = [
            {
                "title": r["title"],
                "authors": r["authors"],
                "arxiv_id": r["arxiv_id"],
                "published_date": r["published_date"],
                "author_citations_total": r["author_citations_total"],
                "openalex_matched": r["openalex_matched"],
                "openalex_author_count": r["openalex_author_count"],
                "openalex_authors_resolved": r["openalex_authors_resolved"],
                "centrality_score": r["centrality_score"],
                "final_score": r["final_score"],
                "brief_summary": r["brief_summary"],
                "key_methods": r["key_methods"],
                "why_relevant": r["why_relevant"],
                "link": r["link"],
            }
            for r in rows_dict
        ]
        highlights = [
            {
                **{
                    k: r[k]
                    for k in (
                        "title",
                        "authors",
                        "arxiv_id",
                        "link",
                        "problem",
                        "method_line",
                        "contribution",
                        "why_relevant",
                    )
                },
                "key_methods": r["key_methods"],
                "author_citations_total": r["author_citations_total"],
                "openalex_matched": r["openalex_matched"],
                "openalex_author_count": r["openalex_author_count"],
                "openalex_authors_resolved": r["openalex_authors_resolved"],
            }
            for r in top
        ]

        out = {
            "summary_table": summary_table,
            "highlight_papers": highlights,
            "keyword_network": gb.keyword_network_json,
            "daily_report": md,
            "artifacts": {
                "daily_report_md": str(self.outputs_dir / "daily_report.md"),
                "summary_table_csv": str(table_path),
                "paper_similarity_graph_png": str(fig_path),
            },
        }
        return out


def run_skill(inp: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    skill = DailyArxivOpenAlexSkill(**kwargs)
    return skill.run(inp)


def run_from_json_file(path: Path, **kwargs: Any) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return run_skill(data, **kwargs)
