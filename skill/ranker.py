"""Module B: semantic relevance via sentence-transformers cosine similarity."""

from __future__ import annotations

import os

# Hugging Face Hub 模型下载走镜像（huggingface_hub 读取 HF_ENDPOINT）
# 若需官方源：export HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_model_name: str | None = None


def _is_local_model_path(name: str) -> bool:
    return os.path.isfile(name) or os.path.isdir(name)


def _get_model(model_name: str | None = None, force_redownload: bool = False):
    global _model, _model_name
    name = model_name or _DEFAULT_MODEL
    cached_ok = _model is not None and _model_name == name
    if cached_ok and not force_redownload:
        return _model

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    if force_redownload and not _is_local_model_path(name):
        local_dir = snapshot_download(repo_id=name, force_download=True)
        _model = SentenceTransformer(local_dir)
    else:
        _model = SentenceTransformer(name)
    _model_name = name
    return _model


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    force_redownload: bool = False,
) -> np.ndarray:
    model = _get_model(model_name, force_redownload=force_redownload)
    return np.asarray(model.encode(texts, show_progress_bar=False, convert_to_numpy=True))


def relevance_scores(
    query: str,
    paper_embeddings: np.ndarray,
    model_name: str | None = None,
    force_redownload: bool = False,
) -> np.ndarray:
    """Cosine similarity between query embedding and each precomputed paper embedding row."""
    if paper_embeddings.size == 0:
        return np.array([])
    q = embed_texts([query], model_name=model_name, force_redownload=force_redownload)
    sims = cosine_similarity(q, paper_embeddings)[0]
    sims = np.clip(sims, 0.0, 1.0)
    return sims.astype(np.float64)
