"""
Go-Live drill test: Rollback plan documentation and mechanism (Phase 4.4.6.4).

Standalone, stdlib-only drill. Verifies:
  1. docs/ROLLBACK_PLAN.md exists and covers all required criteria
  2. backend/slack_bot/scheduler.py exports pause_signals() with correct behavior
  3. The stop-signals mechanism accesses the _scheduler global

Run from the repo root:

    python scripts/go_live_drills/rollback_plan_test.py

Exit code 0 on PASS, 1 on any failure.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLBACK_DOC = REPO_ROOT / "docs" / "ROLLBACK_PLAN.md"
SCHEDULER_PATH = REPO_ROOT / "backend" / "slack_bot" / "scheduler.py"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}" + (f" -- {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ""))


def main():
    global passed, failed

    print("=" * 60)
    print("Phase 4.4.6.4 Rollback Plan Drill")
    print("=" * 60)

    # ── Section A: ROLLBACK_PLAN.md document checks ──

    print("\n-- A: Document existence and content --")

    check("S0: ROLLBACK_PLAN.md exists", ROLLBACK_DOC.is_file(),
          str(ROLLBACK_DOC))

    if not ROLLBACK_DOC.is_file():
        print("\nABORT: doc missing, cannot continue content checks")
        print(f"\n{'=' * 60}")
        print(f"Result: {passed} passed, {failed} failed")
        sys.exit(1)

    doc_text = ROLLBACK_DOC.read_text()
    doc_lower = doc_text.lower()

    check("S1: Mentions Sharpe < 0.5 threshold",
          "sharpe" in doc_lower and "0.5" in doc_text,
          "trigger condition documented")

    check("S2: Mentions 14-day window",
          "14-day" in doc_lower or "14 day" in doc_lower or "trailing 14" in doc_lower,
          "monitoring window documented")

    check("S3: Mentions pause_signals command",
          "pause_signals" in doc_text,
          "stop-signals command documented")

    check("S4: Mentions Peder re-approval",
          "peder" in doc_lower and ("re-approv" in doc_lower or "approval" in doc_lower),
          "re-approval gate documented")

    check("S5: Mentions 4.4.6.1 sign-off reference",
          "4.4.6.1" in doc_text,
          "cross-references the go-live approval item")

    check("S6: Mentions investigation checklist",
          "investigation" in doc_lower and "checklist" in doc_lower,
          "post-rollback investigation steps documented")

    check("S7: Mentions rehearsal recipe",
          "rehearsal" in doc_lower and "recipe" in doc_lower,
          "rehearsal procedure documented")

    check("S8: Documents Option A (graceful) and Option B (kill)",
          "option a" in doc_lower and "option b" in doc_lower,
          "multiple rollback methods documented")

    check("S9: Mentions paper trading re-validation",
          "paper" in doc_lower and ("re-validation" in doc_lower or "re-run" in doc_lower or "fresh" in doc_lower),
          "requires fresh paper validation before restart")

    # ── Section B: scheduler.py code checks ──

    print("\n-- B: scheduler.py pause_signals mechanism --")

    check("S10: scheduler.py exists", SCHEDULER_PATH.is_file())

    if not SCHEDULER_PATH.is_file():
        print("\nABORT: scheduler.py missing, cannot continue code checks")
        print(f"\n{'=' * 60}")
        print(f"Result: {passed} passed, {failed} failed")
        sys.exit(1)

    scheduler_src = SCHEDULER_PATH.read_text()
    tree = ast.parse(scheduler_src)

    # Find pause_signals function
    pause_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pause_signals":
            pause_fn = node
            break

    check("S11: pause_signals function defined",
          pause_fn is not None,
          f"line {pause_fn.lineno}" if pause_fn else "not found")

    if pause_fn is not None:
        # Check it references _scheduler
        fn_src = ast.dump(pause_fn)
        check("S12: pause_signals references _scheduler global",
              "_scheduler" in fn_src,
              "accesses the scheduler instance")

        # Check it has a return type annotation or returns bool
        has_return = any(isinstance(n, ast.Return) and n.value is not None
                        for n in ast.walk(pause_fn))
        check("S13: pause_signals has a return value",
              has_return,
              "returns status indicator")

        # Check it calls shutdown
        calls_shutdown = "shutdown" in ast.dump(pause_fn)
        check("S14: pause_signals calls shutdown on scheduler",
              calls_shutdown,
              "actually stops the scheduler")

        # Check it has a logger call
        has_logger = "logger" in ast.dump(pause_fn)
        check("S15: pause_signals logs the action",
              has_logger,
              "audit trail for rollback event")

    # Check _scheduler is a module-level global (may be Assign or AnnAssign)
    has_scheduler_global = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_scheduler":
                    has_scheduler_global = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_scheduler":
                has_scheduler_global = True
    check("S16: _scheduler is a module-level variable",
          has_scheduler_global,
          "pause_signals can access it")

    # ── Summary ──

    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"Result: {passed}/{total} PASS, {failed}/{total} FAIL")

    if failed > 0:
        print("VERDICT: FAIL")
        sys.exit(1)
    else:
        print("VERDICT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
