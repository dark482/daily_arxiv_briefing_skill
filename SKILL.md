---
name: daily-arxiv-briefing-openalex
description: "Fetch arXiv metadata, rank by OpenAlex cited_by_count via author-name search (top hit per name); embeddings only for graph; Markdown + CSV."
author: dark482
version: 1.0.0
tags:
  - arxiv
  - semantic-search
  - briefing
  - graph-analysis
  - education
metadata:
  openclaw:
    requires:
      bins:
        - python3
        - pip
---

# Daily arXiv Briefing (OpenAlex)

## When to use

The user needs recent arXiv papers for a **research topic / keywords** and a **time window**, using **titles and abstracts only** (no PDF download), and wants:

- A ranked shortlist by **Σ OpenAlex `cited_by_count`** (per arXiv author name → `/authors?search`, first hit)  
- **Cosine embedding similarity only** for the similarity graph (edges + centrality), not for ranking  
- A Markdown briefing, `summary_table.csv`, a similarity-network PNG, and structured JSON  

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. By default `HF_ENDPOINT=https://hf-mirror.com` is set in `run_skill.py`. For the official Hub: `export HF_ENDPOINT=https://huggingface.co`.

## Run

```bash
python run_skill.py --query "graph neural networks, social network analysis" --date_range last_7_days --max_papers 20 --top_k 5
```

Or with a JSON input file:

```bash
python run_skill.py --input examples/example_input.json
```

Use `--force-redownload` if the model cache is corrupted.

## Inputs / outputs

- **Input (JSON)**: `query`, `date_range` (`last_7_days` | `last_14_days` | `last_30_days`), `max_papers`, `top_k`; optional `force_redownload`.  
- **Output directory**: default `outputs/` — `daily_report.md`, `summary_table.csv`, `paper_similarity_graph.png` (if produced), `skill_output.json`.

## Modules

- `skill/arxiv_fetcher.py` — arXiv API  
- `skill/ranker.py` — semantic relevance  
- `skill/novelty.py` — novelty and local embedding cache (`data/paper_cache.json`)  
- `skill/graph_builder.py` — similarity graph and paper–keyword bipartite summary  
- `skill/report_generator.py` — reports and plots  
- `skill/pipeline.py` — orchestration  

## Compliance

Follow the [arXiv API terms of use](https://info.arxiv.org/help/api/index.html). This skill is intended for teaching and non-commercial demos.

## Before publishing to StudyClawHub

Set `author` in the frontmatter to your **GitHub username** (must match `sch-submit` checks). After pushing the repo to GitHub, use `/sch-submit` in Claude Code / OpenClaw, or open a registration issue on `Trust-App-AI-Lab/StudyClawHub` per StudyClawHub docs.
