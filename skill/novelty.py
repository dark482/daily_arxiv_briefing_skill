"""Module C: novelty ~ 1 - max cosine similarity to historical paper embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("papers", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_cache(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def novelty_scores(
    embeddings: np.ndarray,
    arxiv_ids: list[str],
    cache_path: Path,
    current_batch_only: bool = False,
) -> np.ndarray:
    """
    For each paper i: novelty_i = 1 - max_j cos(emb_i, emb_j) where j runs over
    historical cache (excluding same arxiv_id) and optionally other papers in batch.
    If no history and no peers, novelty defaults to 1.0.
    """
    n = len(arxiv_ids)
    if n == 0:
        return np.array([])
    history = load_cache(cache_path)
    hist_emb: list[np.ndarray] = []
    hist_ids: list[str] = []
    for h in history:
        hid = h.get("arxiv_id")
        vec = h.get("embedding")
        if hid and isinstance(vec, list) and len(vec) > 0:
            hist_emb.append(np.asarray(vec, dtype=np.float64))
            hist_ids.append(str(hid))
    H = np.stack(hist_emb) if hist_emb else np.zeros((0, embeddings.shape[1]))
    scores = np.ones(n, dtype=np.float64)
    for i in range(n):
        peers: list[np.ndarray] = []
        if H.shape[0]:
            mask = np.array([hist_ids[j] != arxiv_ids[i] for j in range(len(hist_ids))])
            if mask.any():
                peers.append(H[mask])
        if not current_batch_only and n > 1:
            others = np.delete(embeddings, i, axis=0)
            peers.append(others)
        if not peers:
            scores[i] = 1.0
            continue
        P = np.vstack(peers)
        sims = cosine_similarity(embeddings[i : i + 1], P)[0]
        max_sim = float(np.max(sims)) if sims.size else 0.0
        scores[i] = float(np.clip(1.0 - max_sim, 0.0, 1.0))
    return scores


def append_to_cache(
    cache_path: Path,
    arxiv_ids: list[str],
    embeddings: np.ndarray,
    max_entries: int = 500,
) -> None:
    history = load_cache(cache_path)
    seen = {h.get("arxiv_id") for h in history}
    for i, aid in enumerate(arxiv_ids):
        if aid in seen:
            continue
        history.append(
            {
                "arxiv_id": aid,
                "embedding": embeddings[i].tolist(),
            }
        )
        seen.add(aid)
    history = history[-max_entries:]
    save_cache(cache_path, history)
