"""Module E: Markdown report, CSV table, similarity graph PNG."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _short_arxiv_id(aid: str) -> str:
    s = str(aid).strip()
    if "v" in s and s.rsplit("v", 1)[-1].isdigit():
        s = s.rsplit("v", 1)[0]
    return s[-14:] if len(s) > 14 else s


def _title_prefix_words(title: str, max_words: int = 5, max_chars: int = 40) -> str:
    """Use leading words of the title for compact node labels."""
    if not title or not str(title).strip():
        return ""
    words = str(title).split()
    if not words:
        return ""
    chunk = " ".join(words[:max_words])
    if len(words) > max_words:
        chunk += "…"
    if len(chunk) > max_chars:
        chunk = chunk[: max_chars - 1].rstrip() + "…"
    return chunk


def render_similarity_graph(
    G: nx.Graph,
    papers: list[dict[str, Any]],
    out_path: Path,
    final_scores: dict[str, float],
    communities: dict[str, int],
    similarity_threshold: float = 0.55,
    title_label_max_words: int = 5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))
    if G.number_of_nodes() == 0:
        plt.text(0.5, 0.5, "No nodes", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    n = G.number_of_nodes()
    # 更疏的布局，减轻边与点的堆叠感
    k_spread = max(1.2, 3.8 / max(1.0, float(n) ** 0.5))
    pos = nx.spring_layout(G, seed=42, k=k_spread, iterations=120)

    scores = [final_scores.get(node, 0.0) for node in G.nodes()]
    comms = [communities.get(node, 0) for node in G.nodes()]
    # 缩小节点大小区间，减少大圆盖住边和字
    s_min, s_max = min(scores) if scores else 0.0, max(scores) if scores else 1.0
    span = max(s_max - s_min, 1e-6)
    norm = [(float(s) - s_min) / span for s in scores]
    sizes = [220 + 2800 * t for t in norm]

    try:
        cmap = plt.colormaps["tab10"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(c % 10) for c in comms]

    # 边：红色、略提高对比度；透明度/线宽仍随相似度变化
    edge_alphas: list[float] = []
    edge_widths: list[float] = []
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", similarity_threshold))
        t = (w - similarity_threshold) / max(1e-6, 1.0 - similarity_threshold)
        t = float(np.clip(t, 0.0, 1.0))
        edge_alphas.append(0.28 + 0.42 * t)
        edge_widths.append(0.55 + 1.1 * t)

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=edge_alphas,
        edge_color="#c62828",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colors,
        alpha=0.92,
        linewidths=0.6,
        edgecolors="#222222",
    )

    # 每个节点都标注：标题前若干个词（无标题时用 arXiv id）
    aid_to_title = {str(p["arxiv_id"]): str(p.get("title") or "") for p in papers}
    labels: dict[Any, str] = {}
    for node in G.nodes():
        tid = aid_to_title.get(str(node), "")
        prefix = _title_prefix_words(tid, max_words=title_label_max_words)
        labels[node] = prefix if prefix else _short_arxiv_id(str(node))

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=7,
        font_color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="#e0e0e0",
            linewidth=0.6,
            alpha=0.88,
        ),
    )

    plt.title(
        f"Paper similarity network (edge if cosine sim ≥ {similarity_threshold:g})\n"
        f"Node size ∝ normalized rank score; color = community. "
        f"Red edges = embedding cosine similarity only. "
        f"Labels = first {title_label_max_words} title words (or arXiv id)."
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_markdown_report(
    query: str,
    date_range: str,
    rows: list[dict[str, Any]],
    highlight: list[dict[str, Any]],
    keyword_network_summary: str,
) -> str:
    lines = [
        "# Daily arXiv Research Briefing",
        "",
        f"**Topic / query:** {query}",
        "",
        f"**Date range:** {date_range}",
        "",
        "## Summary table",
        "",
        "| Rank | Title | Relevance | Novelty | Centrality | Final | Key methods |",
        "| --- | --- | ---:| ---:| ---:| ---:| --- |",
    ]
    for i, r in enumerate(rows, start=1):
        title = r["title"].replace("|", "\\|")[:80]
        km = ", ".join(r.get("key_methods", [])[:4])
        lines.append(
            f"| {i} | {title} | {r['relevance_score']:.3f} | {r['novelty_score']:.3f} | "
            f"{r['centrality_score']:.3f} | {r['final_score']:.3f} | {km} |"
        )
    lines.extend(["", "## Top highlights", ""])
    for i, h in enumerate(highlight, start=1):
        lines.extend(
            [
                f"### {i}. {h['title']}",
                "",
                f"- **arXiv:** [{h['arxiv_id']}]({h.get('link', '')})",
                f"- **Authors:** {', '.join(h.get('authors', [])[:6])}{'…' if len(h.get('authors', [])) > 6 else ''}",
                f"- **Problem:** {h.get('problem', '')}",
                f"- **Method:** {h.get('method_line', '')}",
                f"- **Contribution:** {h.get('contribution', '')}",
                f"- **Why relevant:** {h.get('why_relevant', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Keyword / paper network (bipartite summary)",
            "",
            "Paper–keyword edges connect each paper to extracted method/theme terms.",
            "",
            "```json",
            keyword_network_summary[:8000],
            "```",
            "",
            "*See `outputs/paper_similarity_graph.png` for the semantic similarity graph.*",
            "",
        ]
    )
    return "\n".join(lines)


def build_markdown_report_openalex(
    query: str,
    date_range: str,
    rows: list[dict[str, Any]],
    highlight: list[dict[str, Any]],
    keyword_network_summary: str,
) -> str:
    """Report when ranking uses OpenAlex Σ author cited_by_count (not query similarity)."""
    lines = [
        "# Daily arXiv Briefing (OpenAlex author citations)",
        "",
        f"**Topic / query (arXiv search only):** {query}",
        "",
        f"**Date range:** {date_range}",
        "",
        "**Ranking:** for each **arXiv author name**, OpenAlex ``/authors?search=…`` returns candidates; we take the **first** hit's "
        "lifetime ``cited_by_count`` and **sum** over distinct names on the paper. "
        "Common names may map to the wrong profile. "
        "Embedding cosine similarity is **only** for the similarity graph (edges + centrality).",
        "",
        "## Summary table",
        "",
        "| Rank | Title | Σ author cites | OA hits | Centrality | Norm | Key methods |",
        "| --- | --- | ---:| --- | ---:| ---:| --- |",
    ]
    for i, r in enumerate(rows, start=1):
        title = r["title"].replace("|", "\\|")[:72]
        km = ", ".join(r.get("key_methods", [])[:4])
        na = int(r.get("openalex_author_count", 0))
        nr = int(r.get("openalex_authors_resolved", 0))
        oa = f"{nr}/{na}" if na else "—"
        lines.append(
            f"| {i} | {title} | {int(r['author_citations_total'])} | {oa} | "
            f"{r['centrality_score']:.3f} | {r['final_score']:.3f} | {km} |"
        )
    lines.extend(["", "## Top highlights", ""])
    for i, h in enumerate(highlight, start=1):
        lines.extend(
            [
                f"### {i}. {h['title']}",
                "",
                f"- **arXiv:** [{h['arxiv_id']}]({h.get('link', '')})",
                f"- **Σ author citations (OpenAlex, by name search):** {int(h.get('author_citations_total', 0))} "
                f"({int(h.get('openalex_authors_resolved', 0))}/{int(h.get('openalex_author_count', 0))} "
                f"authors with at least one search hit)",
                f"- **Authors:** {', '.join(h.get('authors', [])[:6])}{'…' if len(h.get('authors', [])) > 6 else ''}",
                f"- **Problem:** {h.get('problem', '')}",
                f"- **Method:** {h.get('method_line', '')}",
                f"- **Contribution:** {h.get('contribution', '')}",
                f"- **Query overlap (heuristic):** {h.get('why_relevant', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Keyword / paper network (bipartite summary)",
            "",
            "Paper–keyword edges connect each paper to extracted method/theme terms.",
            "",
            "```json",
            keyword_network_summary[:8000],
            "```",
            "",
            "*See `outputs/paper_similarity_graph.png` for the embedding similarity graph (not used for citation ranking).*",
            "",
        ]
    )
    return "\n".join(lines)
