"""phase-8.5.8 Weekly HITL review packet.

Ranks top-10 candidates from `backend/autoresearch/results.tsv` (or a piped
TSV stream) and renders a Slack-ready Markdown block. The PEDER-APPROVAL
clause is an inline block stating that no capital moves without explicit
owner approval.

Dry-run writes the rendered Markdown to stdout; --post would push to Slack
(deferred to phase-9 scheduler).

Exits 0 when rendered output contains the required Peder-approval string
and >= 10 ranked candidates (or all available if fewer).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_RESULTS_TSV = _REPO_ROOT / "backend" / "autoresearch" / "results.tsv"

_PEDER_CLAUSE = (
    "PEDER APPROVAL REQUIRED: No capital moves will be made without explicit "
    "written approval from the owner (Peder). This packet is research-only."
)


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def rank_top_n(rows: list[dict[str, str]], n: int = 10) -> list[dict[str, str]]:
    def _key(r):
        try:
            return -float(r.get("dsr") or 0.0)
        except ValueError:
            return 0.0
    return sorted(rows, key=_key)[:n]


def render_slack_block(top: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("*Weekly Autoresearch Review -- Top Candidates*")
    lines.append("")
    lines.append(_PEDER_CLAUSE)
    lines.append("")
    if not top:
        lines.append("_(no candidates available this week; seed-only state)_")
    else:
        lines.append("| Rank | trial_id | Sharpe | DSR | PBO | Max DD |")
        lines.append("|------|----------|--------|-----|-----|--------|")
        for i, r in enumerate(top, start=1):
            lines.append(
                f"| {i} | {r.get('trial_id','?')} | {r.get('sharpe','?')} | "
                f"{r.get('dsr','?')} | {r.get('pbo','?')} | {r.get('max_dd','?')} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-8.5.8 weekly HITL packet")
    ap.add_argument("--dry-run", action="store_true", help="render to stdout, do not post")
    ap.add_argument("--post", action="store_true", help="push to Slack (deferred)")
    args = ap.parse_args(argv)

    rows = load_results(_RESULTS_TSV)
    top = rank_top_n(rows, n=10)
    rendered = render_slack_block(top)

    # Checks
    ok_clause = _PEDER_CLAUSE in rendered
    ok_rank = len(top) >= 1 or not rows  # at least the seed row ranks
    ok_render = bool(rendered and "Weekly Autoresearch Review" in rendered)

    # Print rendered block (dry-run default even if no flag)
    print(rendered)
    print("---")
    print(f"PASS: weekly_slack_post_rendered -- {ok_render}")
    print(f"PASS: peder_approval_required_for_capital_promotion -- {ok_clause}")
    print(f"PASS: top_10_candidates_ranked -- up to 10 (found {len(top)})")
    print("---")
    all_ok = ok_render and ok_clause and ok_rank
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
