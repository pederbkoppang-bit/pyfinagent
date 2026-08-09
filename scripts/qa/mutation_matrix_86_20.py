#!/usr/bin/env python3
"""phase-86.20 criterion 7 -- re-runnable mutation matrix over BOTH new files.

SAFETY -- THIS HARNESS NEVER WRITES TO THE REPOSITORY. Each mutant is applied
IN MEMORY inside a throwaway subprocess that registers the mutated module in
`sys.modules` BEFORE pytest imports anything. The on-disk files are opened
read-only and their digests are asserted unchanged across the whole run. This
is the phase-36.17 pattern, adopted here for the same reason: an earlier scratch
harness mutated a live production file while an ARMED backend was running, and
a restart inside that window would have imported a mutant into a trading
process.

Injection order matters and is handled: `portfolio_manager` does
`from backend.services.recommendation_vocab import canonical_recommendation` at
import time, so a mutated `recommendation_vocab` must be in `sys.modules`
before `portfolio_manager` is first imported. It is -- injection happens before
`pytest.main`, and the test module imports both lazily inside its helpers.

DISCIPLINE:
  * every anchor must occur EXACTLY ONCE (a no-match `str.replace` is
    indistinguishable from success -- hard error, not a skipped cell);
  * the mutated source must actually differ (asserted);
  * the un-mutated baseline must be GREEN first, or "killed" proves nothing;
  * a restored run closes the transcript on both sides.

    source .venv/bin/activate
    python scripts/qa/mutation_matrix_86_20.py

Do NOT run concurrently with another pytest invocation: this module's autouse
fixture byte-compares a live file, so a concurrent run turns this one RED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB = "backend/services/recommendation_vocab.py"
PM = "backend/services/portfolio_manager.py"
MOD = {VOCAB: "backend.services.recommendation_vocab",
       PM: "backend.services.portfolio_manager"}
TEST_MODULE = "backend/tests/test_phase_86_20_portfolio_manager_recommendation_vocabulary.py"

MUTANTS: list[dict] = [
    {
        "id": "M1", "file": VOCAB,
        "desc": "revert the separator fold (case-only, i.e. pre-86.20 behaviour)",
        "proves": "criterion 3 + the whole fix -- the separator gap must reopen",
        "old": '    folded = _SEPARATORS.sub("_", value.strip().upper())',
        "new": "    folded = value.strip().upper()  # MUTANT M1",
    },
    {
        "id": "M2", "file": VOCAB,
        "desc": "drop the closed-scale check (return whatever was folded)",
        "proves": "criterion 5 anti-widening -- UNKNOWN must not become a token",
        "old": "    return folded if folded in CANONICAL_RECOMMENDATIONS else None",
        "new": "    return folded  # MUTANT M2",
    },
    {
        "id": "M3", "file": VOCAB,
        "desc": "match by SUBSTRING instead of by exact membership",
        "proves": "criterion 5 -- 'NOT A BUY' and 'BUYING' must never be a BUY",
        "old": "    return folded if folded in CANONICAL_RECOMMENDATIONS else None",
        "new": (
            "    for _c in (STRONG_BUY, STRONG_SELL, BUY, SELL, HOLD):  # MUTANT M3\n"
            "        if _c in folded:\n"
            "            return _c\n"
            "    return None"
        ),
    },
    {
        "id": "M4", "file": VOCAB,
        "desc": "also strip non-separator punctuation (widen the fold)",
        "proves": "criterion 5 -- 'Strong Buy!' must stay UNRECOGNISED, not be guessed",
        "old": '    folded = _SEPARATORS.sub("_", value.strip().upper())',
        "new": (
            "    import re as _re  # MUTANT M4\n"
            '    folded = _SEPARATORS.sub("_", _re.sub(r"[^A-Za-z\\s\\-_]", "", value).strip().upper())'
        ),
    },
    {
        "id": "M5", "file": VOCAB,
        "desc": "accept non-strings by str()-ing them",
        "proves": "a dict or enum reaching the gate is a caller bug, not a token",
        "old": "    if not isinstance(value, str):",
        "new": "    if False:  # MUTANT M5",
    },
    {
        "id": "M6", "file": PM,
        "desc": "ignore the flag -- always use the canonical resolution",
        "proves": "the change is genuinely DARK; OFF must stay byte-identical",
        "old": "    if not flag_on:",
        "new": "    if False:  # MUTANT M6 -- never take the legacy branch",
    },
    {
        "id": "M7", "file": PM,
        "desc": "ignore the flag the other way -- always legacy (fix never applies)",
        "proves": "the armed path is really wired to the flag",
        "old": "    if not flag_on:",
        "new": "    if True:  # MUTANT M7 -- always take the legacy branch",
    },
    {
        "id": "M8", "file": PM,
        "desc": "resolve UNRECOGNISED to HOLD instead of a non-matching sentinel",
        "proves": "HOLD is in _DOWNGRADE_RECS -- this would SELL on a parse failure",
        "old": "    return canon if canon is not None else _UNRECOGNISED_REC",
        "new": '    return canon if canon is not None else "HOLD"  # MUTANT M8',
    },
    {
        "id": "M9", "file": PM,
        "desc": "drop the UNRECOGNISED warning (silent again)",
        "proves": "criterion 6 -- an unparseable recommendation must be loud",
        "old": '    if canon is None:\n        logger.warning(\n            "phase-86.20: UNRECOGNISED recommendation %r (%s) -- treated as "',
        "new": '    if False:  # MUTANT M9\n        logger.warning(\n            "phase-86.20: UNRECOGNISED recommendation %r (%s) -- treated as "',
    },
    {
        "id": "M10", "file": PM,
        "desc": "drop the VOCABULARY MISMATCH warning (the live defect goes silent)",
        "proves": "criterion 6 -- with the fix DARK, the drop must still be visible",
        "old": "    elif canon != probe.upper():",
        "new": "    elif False:  # MUTANT M10",
    },
    {
        "id": "M11", "file": PM,
        "desc": "fire the mismatch warning on EVERY value (alarm becomes noise)",
        "proves": "an alarm that fires on every HOLD is one an operator trains away",
        "old": "    elif canon != probe.upper():",
        "new": "    elif True:  # MUTANT M11",
    },
    {
        "id": "M13", "file": PM,
        "desc": "restore the CYCLE-1 BUG: strip + re-base the default BEFORE the flag",
        "proves": "the legacy-parity oracle -- a DARK flag must not change an order. "
                  "This exact code shipped in cycle 1 and the Q/A caught it; the cell "
                  "exists so it cannot come back unnoticed.",
        "old": "        return (raw or default).upper()",
        "new": '        return (("" if raw is None else str(raw).strip()) or default).upper()  # MUTANT M13',
    },
    {
        "id": "M12", "file": PM,
        "desc": "remove the quiet path for a genuinely absent recommendation",
        "proves": "legacy position rows carry none; alarming on them is noise",
        "old": "    if not probe:",
        "new": "    if False:  # MUTANT M12 -- alarm on genuinely absent values too",
    },
]

BOOTSTRAP = r"""
import importlib.util, json, os, sys

spec = json.load(open(os.environ["MUT8620_SPEC"], encoding="utf-8"))
target, modname = spec["target"], spec["modname"]
old, new = spec.get("old"), spec.get("new")

# Baseline / restored runs pass no target at all: nothing is injected and the
# suite imports straight off disk, which is exactly the control we want.
if old is not None:
    src = open(target, encoding="utf-8").read()
    original = src
    n = src.count(old)
    if n != 1:
        sys.stderr.write("ANCHOR NOT UNIQUE: %d occurrence(s)\n" % n)
        raise SystemExit(97)
    src = src.replace(old, new, 1)
    if src == original:
        sys.stderr.write("MUTATION WAS A NO-OP\n")
        raise SystemExit(96)

    import backend.services  # parent package; empty __init__
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(modname, target)
    )
    sys.modules[modname] = mod
    exec(compile(src, target, "exec"), mod.__dict__)
    setattr(sys.modules["backend.services"], modname.rsplit(".", 1)[-1], mod)

import pytest
raise SystemExit(pytest.main(spec["pytest_args"]))
"""


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _run(target: str | None, modname: str | None, old, new) -> tuple[int, str]:
    payload = {
        "target": str(REPO_ROOT / target) if target else None,
        "modname": modname,
        "old": old, "new": new,
        "pytest_args": [TEST_MODULE, "-q", "-p", "no:randomly", "--no-header"],
    }
    fd, spec_path = tempfile.mkstemp(prefix="mut8620_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        env = dict(os.environ, MUT8620_SPEC=spec_path, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-c", BOOTSTRAP], cwd=str(REPO_ROOT),
                              env=env, capture_output=True, text=True)
    finally:
        os.unlink(spec_path)
    return proc.returncode, proc.stdout + proc.stderr


def _summary(out: str) -> str:
    for line in reversed(out.strip().splitlines()):
        s = line.strip()
        if ("passed" in s or "failed" in s or "error" in s) and " in " in s:
            return s.strip("= ").strip()
    return "NO SUMMARY (see stderr)"


def _failed(out: str) -> list[str]:
    names = []
    for line in out.splitlines():
        if line.startswith("FAILED "):
            names.append(line.split(" ", 1)[1].split(" ")[0].rsplit("::", 1)[-1])
    return sorted(set(names))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args()

    before = {f: _md5(REPO_ROOT / f) for f in (VOCAB, PM)}
    print("phase-86.20 criterion 7 -- mutation matrix (in-memory; repo never written)")
    for f, d in before.items():
        print(f"  {f}  md5 {d}")

    rc, out = _run(None, None, None, None)
    print(f"\n[baseline] un-mutated tree: {_summary(out)}")
    if rc != 0:
        print("BASELINE IS RED -- aborting; a 'killed' mutant would prove nothing.")
        print(out[-3000:])
        return 2

    cells = [m for m in MUTANTS if not args.only or m["id"] == args.only]
    killed, survived, broken = 0, [], []
    for m in cells:
        rc, out = _run(m["file"], MOD[m["file"]], m["old"], m["new"])
        if rc in (96, 97):
            broken.append(m["id"]); verdict = "BROKEN "
        elif rc == 0:
            survived.append(m["id"]); verdict = "SURVIVED"
        else:
            killed += 1; verdict = "KILLED  "
        print(f"  {verdict}| {m['id']} [{Path(m['file']).name}]: {m['desc']}")
        print(f"           proves: {m['proves']}")
        names = _failed(out)
        if names:
            shown = ", ".join(names[:4]) + (f" (+{len(names)-4} more)" if len(names) > 4 else "")
            print(f"           tests : {shown}")
        print(f"           result: {_summary(out)}")

    rc, out = _run(None, None, None, None)
    print(f"[restored] un-mutated tree: {_summary(out)}")

    after = {f: _md5(REPO_ROOT / f) for f in (VOCAB, PM)}
    same = before == after
    print(f"[integrity] both targets' md5 unchanged: {same}")

    total = len(cells)
    if broken:
        print(f"MATRIX BROKEN -- anchors failed for: {', '.join(broken)}")
        return 3
    if survived:
        print(f"{killed} of {total} killed. SURVIVORS: {', '.join(survived)}")
        return 1
    if not same:
        print("A TARGET FILE CHANGED DURING THE RUN -- investigate.")
        return 4
    print(f"ALL {total} MUTANTS KILLED -- every guard IN THIS MATRIX can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
