"""Paper similarity graph + bipartite paper-keyword graph for SNA-style analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GraphBundle:
    similarity_graph: nx.Graph
    centrality: dict[str, float]  # arxiv_id -> normalized degree centrality
    communities: dict[str, int]  # arxiv_id -> community id
    keyword_network_json: str  # serialized bipartite summary


def _keyword_nodes_from_papers(papers: list[dict[str, Any]]) -> tuple[list, list, list]:
    """Returns (paper_nodes, keyword_nodes, edges) for JSON export."""
    kw_set: set[str] = set()
    edges: list[dict[str, str]] = []
    for p in papers:
        pid = p["arxiv_id"]
        for kw in p.get("key_methods", []):
            kid = f"kw:{kw}"
            kw_set.add(kw)
            edges.append({"source": pid, "target": kid, "type": "paper_keyword"})
    keywords = sorted(kw_set)
    return [p["arxiv_id"] for p in papers], [f"kw:{k}" for k in keywords], edges


def build_similarity_graph(
    arxiv_ids: list[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.55,
) -> nx.Graph:
    G = nx.Graph()
    for aid in arxiv_ids:
        G.add_node(aid)
    if len(arxiv_ids) < 2:
        return G
    sim = cosine_similarity(embeddings)
    n = len(arxiv_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                G.add_edge(arxiv_ids[i], arxiv_ids[j], weight=float(sim[i, j]))
    return G


def degree_centrality_scores(G: nx.Graph, arxiv_ids: list[str]) -> dict[str, float]:
    if not arxiv_ids:
        return {}
    if G.number_of_nodes() == 0:
        return {aid: 0.0 for aid in arxiv_ids}
    dc = nx.degree_centrality(G)
    return {aid: float(dc.get(aid, 0.0)) for aid in arxiv_ids}


def community_labels(G: nx.Graph, arxiv_ids: list[str]) -> dict[str, int]:
    labels = {aid: 0 for aid in arxiv_ids}
    if G.number_of_edges() == 0:
        return labels
    try:
        from networkx.algorithms import community

        comms = list(community.greedy_modularity_communities(G, weight="weight"))
        for idx, comm in enumerate(comms):
            for node in comm:
                if node in labels:
                    labels[node] = idx
    except Exception:
        pass
    return labels


def build_graph_bundle(
    papers: list[dict[str, Any]],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.55,
) -> GraphBundle:
    ids = [p["arxiv_id"] for p in papers]
    G = build_similarity_graph(ids, embeddings, similarity_threshold=similarity_threshold)
    cent = degree_centrality_scores(G, ids)
    comm = community_labels(G, ids)
    pn, kn, edges = _keyword_nodes_from_papers(papers)
    bipartite = {
        "type": "paper_keyword_bipartite",
        "paper_nodes": pn,
        "keyword_nodes": kn,
        "edges": edges[:500],
        "num_keyword_nodes": len(kn),
    }
    return GraphBundle(
        similarity_graph=G,
        centrality=cent,
        communities=comm,
        keyword_network_json=json.dumps(bipartite, ensure_ascii=False),
    )
