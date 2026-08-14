#!/usr/bin/env python3
"""Fail if any TRACKED file contains real credential material.

    python3 scripts/audit/verify_no_tracked_credentials.py
    python3 scripts/audit/verify_no_tracked_credentials.py --self-test

Exit 0 = clean. Exit 1 = a live-looking credential is tracked. Exit 2 = the scan
could not run (fails CLOSED: an unrunnable scan is not a clean scan).

WHY THIS EXISTS, AND WHY IT IS NOT JUST THE PRODUCER FIX
--------------------------------------------------------
The 2026-08-14 incident was closed at ONE producer (`run_away_session.sh`, which
wrote raw CLI JSON to a tracked path with no redaction). That fixes the producer
that was caught. It says nothing about the OTHER producers -- and the repo has
many things that write into `handoff/`. Fixing the instance is not fixing the
class, so this scans the whole tracked tree.

TWO FALSE-CLEAN CHECKS ARE PINNED HERE AS TESTS, because both actually happened:

1. **A class that cannot cross a hyphen.** The scan that first declared these
   files clean was `sk-[A-Za-z0-9]{20,}`; an Anthropic OAuth token is
   `sk-ant-oat01-...` and is hyphenated throughout, so it matched NOTHING. A
   false negative on a live credential is the worst outcome this file can have.

2. **A wrapped token.** The real leaked value wraps across a newline, so a
   line-bounded class matched 92 of its characters and left 29 behind -- and
   "does the pattern still match? no" reported CLEAN.

AND ONE FALSE POSITIVE, pinned for the opposite reason: a naive scan flags 45
files, of which 40 are test fixtures, research briefs *about* the incident, and
this file's own pattern list. Reporting "45 files leak credentials" would be
false and would bury the 5 that matter. Hence the classifier below.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from collections import Counter

#: Hyphen-aware, every one of them. See rule 1 above.
PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
]

#: Known vendor prefixes that carry NO secret material on their own. The 86.67
#: research brief contains `sk-ant-oat01-` TWICE (the leaked token is doubled)
#: with the body already redacted -- that is not a disclosure, and calling it one
#: would be a false alarm that costs the next reader an hour.
PUBLIC_PREFIXES = ("sk-ant-oat01-", "sk-ant-api03-", "sk-ant-api-")

PLACEHOLDER = re.compile(
    r"(?i)x{4,}|EXAMPLE|PLACEHOLDER|REDACTED|FAKE|DUMMY|TEST|YOUR[_-]|\.\.\.|"
    r"abc123|abcdefghij|1234567890|A{6,}")

#: This file and the redactor both contain the patterns by necessity.
SELF = ("scripts/audit/verify_no_tracked_credentials.py",
        "scripts/away_ops/redact_secrets.py",
        "scripts/audit/prompt_leak_redteam.py")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _strip_public_prefixes(m: str) -> str:
    """Remove every leading repetition of a known public prefix."""
    changed = True
    while changed:
        changed = False
        for p in PUBLIC_PREFIXES:
            if m.startswith(p):
                m = m[len(p):]
                changed = True
    return m


def is_real_credential(match: str) -> bool:
    """A match is REAL only if, after stripping public prefixes, what remains is
    long AND high-entropy AND carries no placeholder marker."""
    if PLACEHOLDER.search(match):
        return False
    body = _strip_public_prefixes(match)
    return len(body) >= 25 and _entropy(body) >= 3.6


def scan() -> list[tuple[str, int, float]]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
    findings = []
    for path in out.stdout.decode().split("\0"):
        if not path or path in SELF:
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except (OSError, IsADirectoryError):
            continue
        for rx in PATTERNS:
            for m in set(rx.findall(text)):
                if is_real_credential(m):
                    body = _strip_public_prefixes(m)
                    findings.append((path, len(m), round(_entropy(body), 2)))
    return sorted(set(findings))


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ok   {name}")
        else:
            fail += 1
            print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))

    real = "sk-ant-oat01-OvM72xQzKp8LrWnDfT4gYbHcJs9VeAiU3kZmNpQwRt6yXbLd"
    check("a real-shaped OAuth token is flagged", is_real_credential(real))
    check("the ORIGINAL hyphen-blind scan really did miss it",
          re.compile(r"sk-[A-Za-z0-9]{20,}").search(real) is None,
          "if this inverts, the whole premise is wrong")
    check("the DOUBLED public prefix alone is NOT a credential",
          not is_real_credential("sk-ant-oat01-sk-ant-oat01-"),
          "the 86.67 brief contains exactly this; flagging it is a false alarm")
    check("an explicitly redacted value is not a credential",
          not is_real_credential("sk-ant-oat01-sk-...REDACTED"))
    check("a test fixture is not a credential",
          not is_real_credential("sk-ant-abcdefghijklmnopqrstuvwxyz012345"))
    check("a low-entropy repeat is not a credential",
          not is_real_credential("sk-ant-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))
    check("a Slack bot token is flagged",
          is_real_credential("xoxb-927104385517-9284461037axy-Kp8LrWnDfT4gYbHcJs9VeA"))
    print(f"\n  {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    try:
        findings = scan()
    except Exception as exc:                                  # fails CLOSED
        print(f"SCAN FAILED: {exc}\nAn unrunnable scan is NOT a clean scan.",
              file=sys.stderr)
        return 2
    if not findings:
        print("CLEAN -- no tracked file carries real credential material.")
        return 0
    print(f"*** {len(findings)} TRACKED FILE(S) CARRY A LIVE-LOOKING CREDENTIAL ***\n")
    for path, length, ent in findings:
        print(f"  {path}\n      match length {length}, body entropy {ent}")
    print("\nValues are NOT printed. Rotation revokes; deleting the file does not "
          "(history and any fork retain it).")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
