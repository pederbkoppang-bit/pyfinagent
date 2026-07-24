#!/usr/bin/env python3
"""phase-75.15 (qa-tests-04): Tier-1 coverage floor enforcement.

coverage.py's native `fail_under` is PROJECT-WIDE ONLY -- there is no
built-in per-file/per-module threshold (pytest-cov#444 is still open as
of 2026-07-24; coverage.readthedocs.io/en/latest/config.html documents
`fail_under` as a single total-coverage check). The DoD-4 tiered coverage
policy (docs/coverage_tier_overrides.md) needs a per-module gate, so this
is a bespoke runner.

Bars are PARSED from docs/coverage_tier_overrides.md -- never hardcoded
here, so the doc stays the single source of truth. Parsing targets the
"Tier-1 STRICT" and "Tier-1 EXTENDED" section headers, each of which
states its numeric floor (e.g. ">=75% line + >=80% branch"); every
backtick-quoted `backend/...py` path in a markdown table row under that
header inherits the section's floor until the next header. `coverage
json`'s per-file `percent_covered` is already a combined line+branch
figure (this repo's .coveragerc sets `branch = True`), so comparing it
against the LINE-coverage floor from the header is a conservative
(harder-to-clear, never laxer) check.

Usage:
    coverage run -m pytest backend/tests/ -m "not requires_live"
    coverage json -o coverage.json
    python scripts/qa/coverage_tier_check.py [--coverage-json PATH] [--doc PATH]

Exit codes:
    0 -- every parsed Tier-1 module is at or above its bar.
    1 -- at least one Tier-1 module is below its bar (listed on stderr).
    2 -- malformed/missing input (doc has zero parseable Tier-1 modules,
         or the coverage json report is absent/unparseable). A silent
         0-module pass would be a vacuous guard, so this refuses to
         report PASS on that path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC = REPO_ROOT / "docs" / "coverage_tier_overrides.md"
DEFAULT_COVERAGE_JSON = REPO_ROOT / "coverage.json"

# "### Tier-1 STRICT (>=75% line + >=80% branch, ...)" -> ("Tier-1 STRICT", "75")
_SECTION_HEADER_RE = re.compile(r"^###\s+(Tier-1 STRICT|Tier-1 EXTENDED)\b.*?\(>=?\s*([\d.]+)%")
_ANY_HEADER_RE = re.compile(r"^#{1,6}\s+")
# "| `backend/services/kill_switch.py` | **89%** | ... |" -> "backend/services/kill_switch.py"
_MODULE_ROW_RE = re.compile(r"^\|\s*`(backend/[^`]+\.py)`\s*\|")


def parse_bars(doc_text: str) -> dict[str, float]:
    """Return {module_path: bar_pct} for every Tier-1 STRICT/EXTENDED module row."""
    bars: dict[str, float] = {}
    current_bar: float | None = None
    for raw_line in doc_text.splitlines():
        line = raw_line.strip()
        header_m = _SECTION_HEADER_RE.match(line)
        if header_m:
            current_bar = float(header_m.group(2))
            continue
        if _ANY_HEADER_RE.match(line):
            # Any other heading (Tier-2, Tier-3, "1. Why replace...", etc)
            # ends Tier-1 scope until the next Tier-1 header is seen.
            current_bar = None
            continue
        if current_bar is not None:
            row_m = _MODULE_ROW_RE.match(line)
            if row_m:
                bars[row_m.group(1)] = current_bar
    return bars


def load_measured(coverage_json_path: Path) -> dict[str, float]:
    data = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    return {
        path: info["summary"]["percent_covered"]
        for path, info in data.get("files", {}).items()
    }


def run(doc_path: Path, coverage_json_path: Path) -> tuple[int, list[str]]:
    """Return (exit_code, report_lines). Separated from main() for testability."""
    report: list[str] = []

    if not doc_path.exists():
        return 2, [f"ERROR: coverage tier doc not found: {doc_path}"]
    bars = parse_bars(doc_path.read_text(encoding="utf-8"))
    if not bars:
        return 2, [
            f"ERROR: parsed ZERO Tier-1 modules from {doc_path} -- "
            "refusing to report PASS on a vacuous guard"
        ]

    if not coverage_json_path.exists():
        return 2, [
            f"ERROR: coverage json report not found: {coverage_json_path} "
            "(run `coverage json` before this check)"
        ]
    try:
        measured = load_measured(coverage_json_path)
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return 2, [f"ERROR: could not parse coverage json {coverage_json_path}: {exc!r}"]

    failures: list[str] = []
    try:
        doc_rel = doc_path.relative_to(REPO_ROOT)
    except ValueError:
        doc_rel = doc_path
    report.append(f"Tier-1 coverage check -- {len(bars)} module(s) gated (bars sourced from {doc_rel})")
    for module, bar in sorted(bars.items()):
        pct = measured.get(module)
        if pct is None:
            failures.append(f"{module}: NOT MEASURED (absent from coverage json -- moved/renamed/untested?)")
            report.append(f"  MISSING  {module}: not present in coverage json (bar {bar:.0f}%)")
            continue
        status = "PASS" if pct >= bar else "FAIL"
        report.append(f"  {status}  {module}: {pct:.1f}% (bar {bar:.0f}%)")
        if pct < bar:
            failures.append(f"{module}: {pct:.1f}% < bar {bar:.0f}%")

    if failures:
        report.append("")
        report.append("FAIL -- Tier-1 module(s) below bar:")
        for f in failures:
            report.append(f"  - {f}")
        return 1, report

    report.append("")
    report.append("PASS -- all Tier-1 modules at or above bar")
    return 0, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coverage-json", type=Path, default=DEFAULT_COVERAGE_JSON)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args(argv)

    exit_code, report_lines = run(args.doc, args.coverage_json)
    stream = sys.stderr if exit_code else sys.stdout
    for line in report_lines:
        print(line, file=stream)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
