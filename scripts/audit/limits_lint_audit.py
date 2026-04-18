"""phase-4.9 step 4.9.3 lint audit.

Seven teeth:
1. script_exists: `scripts/governance/lint_limits_usage.py` present.
2. strict_pass_on_clean_tree: running the lint with --strict against
   the current repo exits 0 (no violations).
3. workflow_exists_and_parses: `.github/workflows/governance-lint.yml`
   parses as valid YAML.
4. workflow_invokes_lint: the workflow runs
   `scripts/governance/lint_limits_usage.py --strict`.
5. workflow_triggers: the `on:` block covers BOTH `push` (with
   a paths filter) AND `pull_request`.
6. governance_names_complete: the lint's GOVERNANCE_NAMES tuple
   equals the six fields from limits_schema.RiskLimits exactly.
7. mutation_kills_strict: injecting a
   `MAX_PORTFOLIO_LEVERAGE = 99.0` line into an unapproved file
   makes `--strict` exit 1. Restored on exit (try/finally).

Exit 1 on failure when `--check` is passed; always writes
`handoff/limits_lint_audit.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[2]
LINT = REPO / "scripts" / "governance" / "lint_limits_usage.py"
WORKFLOW = REPO / ".github" / "workflows" / "governance-lint.yml"
OUT = REPO / "handoff" / "limits_lint_audit.json"

# A file in-tree that is NOT on the lint's allowlist -- we inject a
# temporary violation here, then restore.
MUTATION_TARGET = REPO / "backend" / "services" / "kelly_allocator.py"


def _run_lint_strict() -> int:
    proc = subprocess.run(
        [sys.executable, str(LINT), "--strict"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. script exists
    checks["script_exists"] = LINT.exists()
    if not checks["script_exists"]:
        reasons.append(f"lint script missing at {LINT}")

    # 2. strict pass on clean tree
    strict_rc = _run_lint_strict() if LINT.exists() else 99
    checks["strict_pass_on_clean_tree"] = strict_rc == 0
    if strict_rc != 0:
        reasons.append(f"--strict returned {strict_rc} on current tree")

    # 3. workflow exists + parses
    workflow_obj: dict | None = None
    if WORKFLOW.exists():
        try:
            if yaml is None:
                raise RuntimeError("PyYAML not installed")
            workflow_obj = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
            checks["workflow_exists_and_parses"] = isinstance(workflow_obj, dict)
        except Exception as e:
            checks["workflow_exists_and_parses"] = False
            reasons.append(f"workflow YAML parse failed: {e}")
    else:
        checks["workflow_exists_and_parses"] = False
        reasons.append(f"workflow missing at {WORKFLOW}")

    # 4. workflow invokes lint
    invokes = False
    if WORKFLOW.exists():
        src = WORKFLOW.read_text(encoding="utf-8")
        invokes = (
            "scripts/governance/lint_limits_usage.py" in src
            and "--strict" in src
        )
    checks["workflow_invokes_lint"] = invokes
    if not invokes:
        reasons.append("workflow does not invoke lint --strict")

    # 5. triggers: push (with paths) + pull_request
    has_push = has_paths = has_pr = False
    if isinstance(workflow_obj, dict):
        # PyYAML parses the literal `on` key as the boolean True.
        on_block = workflow_obj.get("on") or workflow_obj.get(True)
        if isinstance(on_block, dict):
            push_block = on_block.get("push")
            pr_block = on_block.get("pull_request")
            has_push = isinstance(push_block, dict)
            has_pr = pr_block is not None
            if isinstance(push_block, dict) and push_block.get("paths"):
                has_paths = True
    checks["workflow_has_push_with_paths"] = has_push and has_paths
    checks["workflow_has_pull_request"] = has_pr
    if not (has_push and has_paths):
        reasons.append("workflow missing push: with paths: filter")
    if not has_pr:
        reasons.append("workflow missing pull_request: trigger")

    # 6. GOVERNANCE_NAMES matches RiskLimits fields
    sys.path.insert(0, str(REPO))
    try:
        from backend.governance.limits_schema import RiskLimits
        schema_fields = set(RiskLimits.model_fields.keys())
    except Exception as e:
        schema_fields = set()
        reasons.append(f"could not import RiskLimits: {e}")

    lint_names: set[str] = set()
    if LINT.exists():
        src = LINT.read_text(encoding="utf-8")
        m = re.search(
            r"GOVERNANCE_NAMES:\s*tuple\[str, \.\.\.\]\s*=\s*\((.*?)\)",
            src, re.DOTALL,
        )
        if m:
            lint_names = set(re.findall(r'"([^"]+)"', m.group(1)))
    checks["governance_names_complete"] = (
        bool(schema_fields) and lint_names == schema_fields
    )
    if schema_fields and lint_names != schema_fields:
        reasons.append(
            f"GOVERNANCE_NAMES mismatch: lint={sorted(lint_names)} "
            f"schema={sorted(schema_fields)}"
        )

    # 7. mutation-resistance: inject a violation, confirm --strict exits 1
    mutation_killed = False
    if MUTATION_TARGET.exists() and LINT.exists():
        original = MUTATION_TARGET.read_text(encoding="utf-8")
        try:
            mutated = original + "\n\nMAX_PORTFOLIO_LEVERAGE = 99.0\n"
            MUTATION_TARGET.write_text(mutated, encoding="utf-8")
            rc = _run_lint_strict()
            mutation_killed = rc == 1
            if not mutation_killed:
                reasons.append(
                    f"mutation test: --strict returned {rc}; expected 1"
                )
        finally:
            MUTATION_TARGET.write_text(original, encoding="utf-8")
    else:
        reasons.append(
            f"mutation target missing: {MUTATION_TARGET}"
        )
    checks["mutation_kills_strict"] = mutation_killed

    all_ok = all(checks.values()) and not reasons
    verdict = "PASS" if all_ok else "FAIL"

    result = {
        "step": "4.9.3",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "governance_names_in_lint": sorted(lint_names),
        "governance_names_in_schema": sorted(schema_fields),
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        **{k: v for k, v in checks.items()},
    }, indent=2))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
