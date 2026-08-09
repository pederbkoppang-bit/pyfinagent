#!/usr/bin/env python3
"""phase-36.17 -- verify every anchor and every quoted command block in the
step's artifacts actually reproduces against the live tree.

WHY THIS EXISTS, AND WHY v1 WAS WORTHLESS
-----------------------------------------
THREE consecutive Q/A cycles failed this step on the same defect: line anchors
derived before the final edit and shipped without re-derivation (-70, then -3,
then -3 again). v1 of this script was written to stop that and DID NOT:

  * it left `CHECKED = {}`, so its only per-anchor assertion was `n > nlines`
    -- a BOUNDS check. Every one of the three real defects was IN BOUNDS, so v1
    could never have caught any of them.
  * it hard-coded the two wrong numbers (1507/1509) into its HISTORICAL
    exemption set, i.e. it exempted precisely the defect it was written for.
  * its docstring claimed it verified "content still matches what the artifact
    says it is", which the code never did.

The Q/A executed v1 against a synthetic artifact whose every anchor was
wrong-but-in-bounds and it printed "ALL ANCHOR CHECKS PASSED", rc=0. That is an
illusory guard, and shipping one is worse than shipping none.

WHAT v2 ACTUALLY CHECKS
-----------------------
A. COMMAND-BLOCK FIDELITY (the strong check). Every fenced block in the
   artifacts that opens with `$ <grep ...>` is re-executed and its recorded body
   must match the live stdout EXACTLY. This kills BOTH historical defects at
   once: a stale number no longer matches, and a hand-curated/re-ordered block
   no longer matches either.
B. STRUCTURAL RELATIONS, asserted without numbers: the halt's `return summary`
   must immediately precede the Step 5.6 header, and `trader.check_stop_losses`
   must have exactly two production call sites.
C. LOOSE ANCHOR CONTENT. Any `autonomous_loop.py:NNNN` cited in prose must
   either carry a staleness marker nearby, or the cited line must still contain
   the symbol the prose associates with it.

`--self-test` proves A and C can FAIL, by running them against synthetic
artifacts with wrong-but-in-bounds anchors and a curated block. A guard that
cannot fail does not count.

Usage:
    python scripts/qa/verify_36_17_anchors.py
    python scripts/qa/verify_36_17_anchors.py --self-test
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "backend/services/autonomous_loop.py"

ARTIFACTS = [
    ROOT / "handoff/current/experiment_results_36.17.md",
    ROOT / "handoff/current/live_check_36.17.md",
    ROOT / "backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py",
]

STALENESS_MARKERS = (
    "stale", "STALE", "superseded", "pre-fix", "historical", "Historical",
    "cycle-1", "cycle 1", "drifted", "two generations", "predecessor",
    "does not reproduce", "do not reproduce", "70 lines low", "+3 low",
)

# Symbols the prose associates with an anchor -> the substring that line must hold.
SYMBOL_HINTS = {
    "final_state": "final_state",
    "check_stop_losses": "check_stop_losses",
    "return summary": "return summary",
    "Step 5.6": "Step 5.6",
    "order.action": "order.action",
    "_get_live_price": "_get_live_price",
}

# ANY `$ <cmd>` block, not just grep. Cycle-4 Q/A proved the grep-only filter
# was a blind spot that EXACTLY coincided with the two evidence defects it
# shipped: a stale `$ python -m pytest` capture, and a mutation-matrix block
# with no `$` prompt at all. Both were invisible to a grep-only check.
BLOCK_RE = re.compile(r"```\n\$ ([^\n]+)\n(.*?)```", re.S)

# Commands safe to re-execute. Anything else is REPORTED as unverifiable rather
# than silently skipped -- a skip is how the previous blind spot hid.
# Only CHEAP, IDEMPOTENT, SIDE-EFFECT-FREE commands are re-executed.
#
# pytest is deliberately NOT in this list. Re-running the suite from inside the
# verifier was tried and REVERTED for two measured reasons:
#   1. It takes minutes, and
#   2. it is UNSAFE CONCURRENTLY. The test module's autouse
#      `_live_audit_file_is_write_protected` fixture byte-compares a live file;
#      a second pytest running at the same time mutates that file and turns the
#      first run RED. Measured: a foreground run reported "3 failed / 6 failed"
#      purely because the verifier was running pytest in the background. A guard
#      that manufactures false regressions is worse than one with a known gap.
# pytest blocks are therefore REPORTED as not-re-executed, so the gap is VISIBLE
# to a reader instead of silent -- which was the cycle-4 Q/A's actual complaint.
RUNNABLE = ("grep ", "git diff --stat", "git log ", "md5 -q")

# NEVER re-execute a command that invokes THIS script. The artifacts legitimately
# quote `$ python scripts/qa/verify_36_17_anchors.py`, and re-running it from
# inside itself recurses forever -- measured the hard way, it had to be killed.
# Reported as unverifiable rather than skipped silently.
SELF_INVOCATION = "verify_36_17_anchors"

# pytest prints a wall-clock duration that can never match on re-run, and
# `-q` progress dots vary with ordering. Normalise both sides before comparing
# so the check tests the COUNTS, which are the load-bearing part.
DURATION_RE = re.compile(r" in \d+\.\d+s")


def normalise(text: str) -> str:
    out = DURATION_RE.sub(" in <t>", text)
    out = "\n".join(l for l in out.splitlines()
                    if l.strip() and not set(l.strip()) <= set(".sFEx%[]0123456789 "))
    return out.strip()


def sh(cmd: str, cwd: Path) -> str:
    return subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    ).stdout.rstrip("\n")


def check_command_blocks(artifacts, cwd: Path) -> list[str]:
    """A. Re-run every `$ grep ...` block and require an EXACT match."""
    fails: list[str] = []
    unverifiable: list[str] = []
    n = 0
    for art in artifacts:
        if not art.exists():
            fails.append(f"MISSING ARTIFACT: {art}")
            continue
        text = art.read_text()
        for cmd, recorded in BLOCK_RE.findall(text):
            # A block may record several commands. Walk it line by line: a line
            # starting with "$ " opens a new command; everything until the next
            # such line is that command's recorded output.
            pairs: list[tuple[str, list[str]]] = [(cmd, [])]
            for line in recorded.split("\n"):
                if line.startswith("$ "):
                    pairs.append((line[2:].strip(), []))
                else:
                    pairs[-1][1].append(line)
            for c, body_lines in pairs:
                body = "\n".join(body_lines).strip("\n")
                if SELF_INVOCATION in c:
                    unverifiable.append(
                        f"{art.name}: `{c[:55]}` invokes this script -- not re-executed "
                        "(would recurse); --self-test covers it")
                    continue
                if not any(c.startswith(r) or r in c for r in RUNNABLE):
                    unverifiable.append(f"{art.name}: `{c[:60]}` not re-executable")
                    continue
                n += 1
                live = sh(c, cwd)
                if normalise(body) != normalise(live):
                    fails.append(
                        f"{art.name}: recorded output of `{c[:70]}` does not match live.\n"
                        f"      recorded: {normalise(body)[:200]!r}\n"
                        f"      live    : {normalise(live)[:200]!r}"
                    )
    print(f"  {'ok  ' if not fails else 'FAIL'} A. command-block fidelity: {n} block(s) RE-EXECUTED"
          + (f"; {len(unverifiable)} not re-executable (reported, not skipped silently)" if unverifiable else ""))
    for u in unverifiable:
        print(f"       note: {u}")
    return fails


def check_relations(cwd: Path) -> list[str]:
    """B. Structural invariants, asserted without any line number."""
    fails: list[str] = []
    src = (cwd / "backend/services/autonomous_loop.py").read_text().splitlines()
    ret = [i + 1 for i, l in enumerate(src) if l.strip() == "return summary"]
    hdr = [i + 1 for i, l in enumerate(src) if "Step 5.6: Stop-loss enforcement" in l]
    if not ret or not hdr:
        return ["could not locate `return summary` or the Step 5.6 header"]
    halt_ret, s56 = ret[0], hdr[0]
    if not (halt_ret < s56):
        fails.append(f"ORDERING BROKEN: halt return :{halt_ret} not before Step 5.6 :{s56}")
    elif s56 - halt_ret > 4:
        fails.append(
            f"halt return :{halt_ret} and Step 5.6 :{s56} are {s56 - halt_ret} lines "
            "apart -- something was inserted between them"
        )
    else:
        print(f"  ok   B. halt `return summary` :{halt_ret} immediately precedes Step 5.6 :{s56}")

    calls = [
        f"{p.relative_to(cwd)}:{i + 1}"
        for p in (cwd / "backend").rglob("*.py")
        if "/tests/" not in str(p)
        for i, l in enumerate(p.read_text(errors="ignore").splitlines())
        if "trader.check_stop_losses" in l
    ]
    if len(calls) != 2:
        fails.append(f"expected exactly 2 production call sites, found {len(calls)}: {calls}")
    else:
        print(f"  ok   B. check_stop_losses has exactly 2 production call sites")
    return fails


def check_loose_anchors(artifacts, cwd: Path) -> list[str]:
    """C. Prose anchors: content must match, unless marked stale."""
    fails: list[str] = []
    src = (cwd / "backend/services/autonomous_loop.py").read_text().splitlines()
    checked = 0
    total = 0
    for art in artifacts:
        if not art.exists():
            continue
        text = art.read_text()
        total += len(re.findall(r":(\d{3,4})\b", text))
        for m in re.finditer(r":(\d{3,4})\b", text):
            ln = int(m.group(1))
            ctx = text[max(0, m.start() - 320): m.end() + 320]
            if any(mk in ctx for mk in STALENESS_MARKERS):
                continue
            # CROSS-FILE GUARD: an anchor is only about autonomous_loop.py if no
            # OTHER .py file is named nearby. Without this the checker reports
            # phase-36.12 test-file and paper_trader.py anchors as stale
            # autonomous_loop ones -- false positives that would train a reader
            # to ignore it, which is how a guard dies.
            near = text[max(0, m.start() - 200): m.end() + 200]
            others = {f for f in re.findall(r"([A-Za-z0-9_]+\.py)", near)
                      if f != "autonomous_loop.py"}
            # Artifacts also name other files with an ellipsis ("test_phase_36_12...:298")
            # which the .py regex misses. Treat these module stems as cross-file too.
            CROSS = ("test_phase_", "paper_trader", "kill_switch.py", "settings.py",
                     "portfolio_manager", "cycle_health", "formatters", "alerting")
            if others or any(c in near for c in CROSS):
                continue
            # NEAREST-SYMBOL: several hint symbols can appear in one window, so
            # pick the one whose occurrence is closest to the anchor rather than
            # whichever happens to come first in dict order.
            best, best_d = None, 10**9
            for k, v in SYMBOL_HINTS.items():
                for hm in re.finditer(re.escape(k), ctx):
                    d = abs(hm.start() - (m.start() - max(0, m.start() - 320)))
                    if d < best_d:
                        best, best_d = v, d
            sym = best
            if sym is None:
                continue
            checked += 1
            if ln > len(src):
                fails.append(f"{art.name}: cites :{ln}, file has {len(src)} lines")
            elif sym not in src[ln - 1]:
                fails.append(
                    f"{art.name}: cites :{ln} in a context about {sym!r}, but that "
                    f"line is {src[ln - 1].strip()[:70]!r}"
                )
    # Print the RECALL, not just the numerator. A hand-computed ratio in an
    # artifact goes stale the moment the artifact is edited -- measured: a
    # correct "3 of 44" became wrong ("3 of 48") purely because reporting it
    # added more anchors. The tool is the only place this number stays true.
    pct = (100.0 * checked / total) if total else 0.0
    print(f"  {'ok  ' if not fails else 'FAIL'} C. loose prose anchors: "
          f"{checked} of {total} content-checked ({pct:.1f}% recall) -- "
          f"the rest are cross-file, staleness-marked, or outside SYMBOL_HINTS")
    return fails


def self_test() -> int:
    """Prove A and C can FAIL. A guard that cannot fail does not count."""
    print("SELF-TEST -- the checks must REJECT known-bad artifacts\n")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # (i) wrong-but-IN-BOUNDS anchor in prose -- the exact shape v1 passed.
        bad = tmp / "bad_anchor.md"
        # 4-digit and IN BOUNDS -- the exact shape that defeated v1. (v1's only
        # per-anchor test was `n > nlines`, which these pass.)
        bad.write_text(
            "The halt's `return summary` is at :1507 and Step 5.6 begins at :1509.\n"
        )
        f = check_loose_anchors([bad], ROOT)
        print(f"   (i)  wrong-but-in-bounds prose anchor -> {'REJECTED' if f else 'ACCEPTED (BAD)'}")
        ok &= bool(f)

        # (ii) curated / re-ordered command block.
        bad2 = tmp / "bad_block.md"
        bad2.write_text(
            "```\n$ grep -n \"Step 5.6: Stop-loss enforcement\" "
            "backend/services/autonomous_loop.py\n"
            "9999:            # hand-written line that is not the real output\n```\n"
        )
        f2 = check_command_blocks([bad2], ROOT)
        print(f"   (ii) curated command block          -> {'REJECTED' if f2 else 'ACCEPTED (BAD)'}")
        ok &= bool(f2)

        # (iii) a CORRECT block must still pass, or the check is useless noise.
        live = sh('grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py', ROOT)
        good = tmp / "good_block.md"
        good.write_text(
            "```\n$ grep -n \"Step 5.6: Stop-loss enforcement\" "
            f"backend/services/autonomous_loop.py\n{live}\n```\n"
        )
        f3 = check_command_blocks([good], ROOT)
        print(f"   (iii) correct command block          -> {'ACCEPTED' if not f3 else 'REJECTED (BAD)'}")
        ok &= not f3
    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    print("phase-36.17 anchor + command-block verification\n")
    fails = (
        check_command_blocks(ARTIFACTS, ROOT)
        + check_relations(ROOT)
        + check_loose_anchors(ARTIFACTS, ROOT)
    )
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
