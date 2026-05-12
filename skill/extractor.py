"""Module D: lightweight extraction from title+abstract (rules + keyword lexicon, no LLM)."""

from __future__ import annotations

import re
from typing import Any

_GRAPH_NETWORK = [
    "graph neural network",
    "graph convolutional",
    "message passing",
    "heterogeneous graph",
    "temporal graph",
    "knowledge graph",
    "link prediction",
    "node classification",
    "community detection",
    "network embedding",
    "social network",
    "influence maximization",
    "random walk",
    "pagerank",
    "spectral graph",
    "gnn",
]
_NLP_LLM = [
    "large language model",
    "instruction tuning",
    "parameter-efficient fine-tuning",
    "low-rank adaptation",
    "retrieval-augmented generation",
    "chain-of-thought",
    "named entity recognition",
    "machine translation",
    "question answering",
    "sentiment analysis",
    "speech recognition",
    "text summarization",
    "language modeling",
    "pre-trained language model",
    "transformer",
    "attention mechanism",
    "bert",
    "seq2seq",
    "llm",
]
_CV = [
    "object detection",
    "instance segmentation",
    "semantic segmentation",
    "image classification",
    "pose estimation",
    "optical flow",
    "3d reconstruction",
    "neural radiance field",
    "vision transformer",
    "self-supervised visual",
    "few-shot learning",
    "zero-shot learning",
    "gan",
    "generative adversarial network",
]
_GENERATIVE = [
    "diffusion model",
    "score-based generative",
    "variational autoencoder",
    "vae",
    "normalizing flow",
    "energy-based model",
    "flow matching",
]
_RL = [
    "reinforcement learning",
    "deep reinforcement learning",
    "policy gradient",
    "actor-critic",
    "q-learning",
    "markov decision process",
    "multi-agent reinforcement learning",
    "inverse reinforcement learning",
    "offline reinforcement learning",
    "model-based reinforcement learning",
]
_CLASSIC_ML = [
    "support vector machine",
    "random forest",
    "gradient boosting",
    "logistic regression",
    "k-means",
    "gaussian mixture",
    "expectation-maximization",
    "principal component analysis",
    "kernel method",
    "naive bayes",
    "decision tree",
    "clustering",
    "dimensionality reduction",
]
_OPTIMIZATION_TRAINING = [
    "stochastic gradient descent",
    "adam optimizer",
    "contrastive learning",
    "self-supervised learning",
    "meta-learning",
    "continual learning",
    "domain adaptation",
    "transfer learning",
    "multi-task learning",
    "neural architecture search",
    "knowledge distillation",
    "regularization",
    "generalization bound",
    "optimization",
]
_SYSTEMS_EFFICIENCY = [
    "federated learning",
    "distributed training",
    "model compression",
    "quantization",
    "pruning",
    "efficient inference",
    "edge computing",
    "hardware acceleration",
]
_THEORY_CAUSAL_PROB = [
    "causal inference",
    "causal discovery",
    "bayesian inference",
    "variational inference",
    "markov chain monte carlo",
    "probabilistic graphical model",
    "information theory",
]
_TABULAR_TIME_REC = [
    "time series forecasting",
    "time series",
    "sequential modeling",
    "recommendation system",
    "collaborative filtering",
    "tabular data",
    "survival analysis",
]
_SCIENCE_OTHER = [
    "molecular",
    "protein structure",
    "drug discovery",
    "physics-informed",
    "scientific machine learning",
    "neural ode",
    "partial differential equation",
]

METHOD_KEYWORDS: list[str] = []
for _group in (
    _GRAPH_NETWORK,
    _NLP_LLM,
    _CV,
    _GENERATIVE,
    _RL,
    _CLASSIC_ML,
    _OPTIMIZATION_TRAINING,
    _SYSTEMS_EFFICIENCY,
    _THEORY_CAUSAL_PROB,
    _TABULAR_TIME_REC,
    _SCIENCE_OTHER,
):
    METHOD_KEYWORDS.extend(_group)


def _first_sentences(text: str, max_chars: int = 400) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last > 80:
        return cut[: last + 1].strip()
    return cut.rsplit(" ", 1)[0] + "…"


def _keyword_hits(blob: str, kw: str) -> bool:
    """Match phrase or whole word (short acronyms use word boundaries to cut false positives)."""
    k = kw.lower()
    if " " in k or len(k) > 5:
        return k in blob
    return re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", blob) is not None


def extract_key_methods(title: str, abstract: str) -> list[str]:
    blob = f"{title} {abstract}".lower()
    found: list[str] = []
    for kw in METHOD_KEYWORDS:
        if _keyword_hits(blob, kw):
            found.append(kw.title() if kw.islower() else kw)
    out: list[str] = []
    for f in found:
        if f not in out:
            out.append(f)
    return out[:12]


def extract_fields(title: str, abstract: str, query: str) -> dict[str, Any]:
    """Heuristic problem / method / contribution / task lines from abstract only."""
    methods = extract_key_methods(title, abstract)
    problem = _first_sentences(abstract, 280)
    method = "; ".join(methods[:5]) if methods else "See abstract for methodology."
    start = min(len(abstract) // 3, 200)
    contribution = (
        _first_sentences(abstract[start:], 280) if len(abstract) > 120 else problem
    )
    task = "General" if not methods else methods[0]
    qparts = [p.strip().lower() for p in query.split(",") if p.strip()]
    overlap = [p for p in qparts if p in abstract.lower() or p in title.lower()]
    why = (
        f"Overlaps with your query themes: {', '.join(overlap)}."
        if overlap
        else f"Semantically aligned with query: {query[:120]}."
    )
    return {
        "problem": problem,
        "method_line": method,
        "contribution": contribution,
        "dataset_task": task,
        "key_methods": methods,
        "why_relevant": why,
        "brief_summary": _first_sentences(abstract, 450),
    }
