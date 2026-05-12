# Daily arXiv briefing — OpenAlex citations by author name

Variant of `daily_arxiv_briefing_skill`:

- **arXiv**: metadata (title + abstract + **author names**) via API.
- **Ranking**: for each **distinct** author string from arXiv, call OpenAlex  
  `GET https://api.openalex.org/authors?search=<name>`  
  and use the **first** result’s **`cited_by_count`** (lifetime). Sum these values for all authors on the paper. **No** work/arXiv-URL lookup in OpenAlex.
- **Embeddings / cosine similarity**: **only** for the similarity **graph** (edges + degree centrality in the table).

## Caveats (important)

- **同名作者**：常见英文名会搜到「第一个」OpenAlex 档案，**不一定是** arXiv 上那个人。
- **搜不到**：某位作者在 OpenAlex 无结果时，该作者贡献 **0**；列 `openalex_authors_resolved / openalex_author_count` 表示有多少位作者在搜索里至少有一条命中。

## OpenAlex polite pool

Use a real email: [OpenAlex docs](https://docs.openalex.org/how-to-use-api/rate-limits-and-authentication).

```bash
python run_skill.py --query "..." --mailto "you@university.edu"
```

## Setup

```bash
cd daily_arxiv_briefing_openalex
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_skill.py \
  --query "graph neural networks" \
  --date_range last_7_days \
  --max_papers 20 \
  --mailto "you@university.edu"
```

**Cache:** `data/openalex_author_name_cache.json` (per normalized author name).

Outputs: `outputs/daily_report.md`, `summary_table.csv`, `paper_similarity_graph.png`, `skill_output.json`.
