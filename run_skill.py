#!/usr/bin/env python3
"""CLI: arXiv fetch + OpenAlex author-citation ranking + embedding graph only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill.pipeline import DailyArxivOpenAlexSkill, run_skill


def main() -> None:
    p = argparse.ArgumentParser(
        description="arXiv + OpenAlex: rank by Σ author cited_by_count; embeddings only for the graph",
    )
    p.add_argument("--input", type=Path, help="JSON with query, date_range, max_papers, top_k, openalex_mailto")
    p.add_argument("--query", type=str, help="Keywords for arXiv API (comma-separated)")
    p.add_argument("--date_range", type=str, default="last_7_days")
    p.add_argument("--max_papers", type=int, default=20)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.55, help="Cosine edge threshold (embedding graph only)")
    p.add_argument(
        "--mailto",
        type=str,
        default=None,
        help="Your email for OpenAlex polite pool (https://docs.openalex.org/)",
    )
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--dump-json", type=Path, help="Write skill output JSON (default: outputs/skill_output.json)")
    args = p.parse_args()

    skill_kw = dict(
        similarity_edge_threshold=args.threshold,
        force_redownload=args.force_redownload,
        openalex_mailto=args.mailto,
    )

    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if args.mailto:
            data["openalex_mailto"] = args.mailto
        out = run_skill(data, **skill_kw)
    else:
        if not args.query:
            p.error("Provide --query or --input")
        inp = {
            "query": args.query,
            "date_range": args.date_range,
            "max_papers": args.max_papers,
            "top_k": args.top_k,
        }
        if args.mailto:
            inp["openalex_mailto"] = args.mailto
        skill = DailyArxivOpenAlexSkill(**skill_kw)
        out = skill.run(inp)

    dump = args.dump_json or (ROOT / "outputs" / "skill_output.json")
    dump.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in out.items() if k != "daily_report"}
    serializable["daily_report_preview"] = (out.get("daily_report") or "")[:2000]

    def _json_default(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(type(o))

    dump.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    art = out.get("artifacts", {})
    print("Done.")
    for k, v in art.items():
        print(f"  {k}: {v}")
    print(f"  skill_output.json: {dump}")


if __name__ == "__main__":
    main()
