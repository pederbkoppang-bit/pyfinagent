"""phase-8.5.3 proposer verification script.

Exercises the three success criteria:
  1. proposer_emits_valid_diff_per_cycle
  2. diff_touches_only_whitelisted_files
  3. reads_results_tsv_and_gitlog

Exits 0 when all three pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.proposer import Proposer, WHITELIST, validate_diff


def case_valid_diff_per_cycle() -> tuple[bool, str]:
    p = Proposer()
    diff = p.propose(
        results_tsv="ticker\tsharpe\nAAPL\t1.2\n",
        git_log=["abc1234 refactor ensemble", "def5678 phase-8.5.2 budget"],
    )
    ok, bad = validate_diff(diff)
    if not ok:
        return False, f"diff invalid: {bad}"
    if not diff.get("trial_id"):
        return False, "diff.trial_id missing"
    return True, "valid diff emitted"


def case_diff_touches_only_whitelisted_files() -> tuple[bool, str]:
    p = Proposer()

    def malicious_llm(*, results_tsv: str, git_log: list[str]):
        return {
            "files": {
                "backend/backtest/experiments/optimizer_best.json": "{}",
                "backend/services/kill_switch.py": "raise Exception",  # not whitelisted
            },
            "rationale": "malicious test",
            "trial_id": "m1",
            "read_results_tsv": True,
            "read_git_log": True,
        }

    diff = p.propose("x", ["y"], llm_call_fn=malicious_llm)
    # Proposer must strip the non-whitelisted path.
    files = diff.get("files") or {}
    if "backend/services/kill_switch.py" in files:
        return False, "non-whitelisted file not stripped"
    for path in files:
        if path not in WHITELIST:
            return False, f"path {path!r} survived but is not in whitelist"
    return True, "whitelist enforced; non-whitelisted path stripped"


def case_reads_results_tsv_and_gitlog() -> tuple[bool, str]:
    p = Proposer()
    results_tsv = "ticker\tsharpe\nAAPL\t1.8\n"
    git_log = ["abc1234 a", "def5678 b"]
    diff = p.propose(results_tsv, git_log)
    if not diff.get("read_results_tsv"):
        return False, "read_results_tsv flag not set"
    if not diff.get("read_git_log"):
        return False, "read_git_log flag not set"
    rationale = diff.get("rationale") or ""
    if str(len(git_log)) not in rationale and str(len(results_tsv)) not in rationale:
        return False, "stub rationale did not reference input sizes"
    return True, "inputs read flags set; rationale references sizes"


def main() -> int:
    cases = [
        ("proposer_emits_valid_diff_per_cycle", case_valid_diff_per_cycle),
        ("diff_touches_only_whitelisted_files", case_diff_touches_only_whitelisted_files),
        ("reads_results_tsv_and_gitlog", case_reads_results_tsv_and_gitlog),
    ]
    all_pass = True
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}: {name} -- {msg}")
        if not ok:
            all_pass = False
    print("---")
    print("PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
