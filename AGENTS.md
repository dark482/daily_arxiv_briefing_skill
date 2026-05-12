---
name: daily-arxiv-briefing-openalex
description: "Fetch arXiv metadata, rank by OpenAlex cited_by_count via author-name search; embeddings only for graph; Markdown + CSV."
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

# Daily arXiv Briefing (agent root)

This repository is a **StudyClawHub / ClawHub–compatible** project. The runnable skill is defined in **`SKILL.md`** at the repo root; orchestration lives under `skill/` and the CLI entrypoint is `run_skill.py`.

For hub indexing, this file provides **agent-level metadata** on the default branch. See `SKILL.md` for full usage, inputs/outputs, and environment setup.
