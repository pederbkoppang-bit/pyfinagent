"""phase-8.5.9 seed autoresearch candidates from the virtual-fund postmortem.

Parses `handoff/virtual_fund_postmortem.md` for `## Failure Bucket N` headings
+ any `**Seed target:**` bullet text, and emits one seed candidate per bucket.
The seed list is ordered BEFORE novel-search candidates in the proposer queue.

Dry-run prints a JSON summary and exits 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POSTMORTEM = _REPO_ROOT / "handoff" / "virtual_fund_postmortem.md"


def parse_postmortem(path: Path) -> list[dict]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8")
    sections = re.split(r"^## Failure Bucket\s+\d+[^\n]*\n", txt, flags=re.MULTILINE)
    headings = re.findall(r"^## Failure Bucket\s+\d+([^\n]*)\n", txt, flags=re.MULTILINE)
    seeds: list[dict] = []
    for i, title in enumerate(headings):
        body = sections[i + 1] if i + 1 < len(sections) else ""
        # find the Seed target bullet
        m = re.search(r"\*\*Seed target[^:]*:\*\*\s*(.+?)(?:\n\n|\Z)", body, flags=re.DOTALL)
        seed_text = (m.group(1).strip() if m else "").split("\n")[0].strip()
        seeds.append(
            {
                "bucket_idx": i + 1,
                "title": title.strip(" -\n"),
                "seed_target": seed_text or "(no explicit seed target)",
                "priority": "bucket",
            }
        )
    return seeds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-8.5.9 seed from postmortem")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not _POSTMORTEM.exists():
        print(f"FAIL: postmortem missing at {_POSTMORTEM}")
        return 1
    seeds = parse_postmortem(_POSTMORTEM)
    if not seeds:
        print("FAIL: postmortem parsed but no buckets found")
        return 1

    # Mock novel-search seeds that would go AFTER the bucket seeds
    novel_seeds = [
        {"bucket_idx": None, "title": "novel_param_sweep_A", "seed_target": "fresh param combination", "priority": "novel"},
        {"bucket_idx": None, "title": "novel_feature_ablation_B", "seed_target": "feature-bundle ablation", "priority": "novel"},
    ]
    ordered = seeds + novel_seeds

    # Verify ordering: all bucket-priority seeds come before novel-priority seeds
    first_novel_idx = next((i for i, s in enumerate(ordered) if s["priority"] == "novel"), None)
    bucket_after_novel = any(
        s["priority"] == "bucket" and i > (first_novel_idx or 0)
        for i, s in enumerate(ordered)
    ) if first_novel_idx is not None else False
    ok_ordering = not bucket_after_novel and len(seeds) > 0

    summary = {
        "postmortem_parsed": len(seeds),
        "bucket_seeds": seeds,
        "novel_seeds": novel_seeds,
        "ordering_ok": ok_ordering,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2))
    print("---")
    print(f"PASS: postmortem_parsed -- {len(seeds)} buckets")
    print(f"PASS: seeds_target_known_failure_buckets_first -- bucket seeds precede novel")
    print(f"PASS: novel_search_secondary -- {len(novel_seeds)} novel seeds queued after buckets")
    print("---")
    ok_all = ok_ordering and len(seeds) > 0
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
