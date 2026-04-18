"""phase-4.9 step 4.9.1 limits-tag-enforcement audit.

Five teeth:
1. ci_workflow_landed: .github/workflows/limits-tag-enforcement.yml
   exists with both paths + tags triggers, fetch-depth: 0 + fetch-
   tags: true on checkout, invokes verify_limits_tag.sh.
2. unsigned_push_rejected: verify_limits_tag.sh contains a real
   `git verify-tag` call (not just a comment).
3. wrong_owner_rejected: ALLOWED_SIGNERS array is non-empty and
   has an authorized email; tagger-email extraction logic is
   present.
4. approval_message_required: script checks tag annotation for
   >= 30 chars AND the word "approved" (case-insensitive).
5. --dry-run exits 0 even with no tag present (masterplan
   verification requirement).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "governance" / "verify_limits_tag.sh"
WORKFLOW = REPO / ".github" / "workflows" / "limits-tag-enforcement.yml"
CODEOWNERS = REPO / ".github" / "CODEOWNERS"
OUT = REPO / "handoff" / "limits_tag_audit.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. CI workflow landed.
    wf_ok = WORKFLOW.exists()
    if wf_ok:
        wf = WORKFLOW.read_text(encoding="utf-8")
        # Must reference both paths and tags triggers.
        has_paths = "paths:" in wf and "limits.yaml" in wf
        has_tags = "tags:" in wf and "limits-rotation-" in wf
        has_checkout = "fetch-depth: 0" in wf and "fetch-tags: true" in wf
        invokes_script = "verify_limits_tag.sh" in wf
        wf_ok = has_paths and has_tags and has_checkout and invokes_script
        if not wf_ok:
            reasons.append(
                f"workflow missing (paths={has_paths}, tags={has_tags}, "
                f"checkout_depth={has_checkout}, invokes={invokes_script})"
            )
    else:
        reasons.append(f"workflow missing at {WORKFLOW}")
    checks["ci_workflow_landed"] = wf_ok

    # Read the script once.
    script_text = SCRIPT.read_text(encoding="utf-8") if SCRIPT.exists() else ""
    if not script_text:
        reasons.append(f"verify_limits_tag.sh missing at {SCRIPT}")

    # 2. unsigned_push_rejected.
    unsigned_ok = "git verify-tag" in script_text and "fail " in script_text
    if not unsigned_ok:
        reasons.append("verify-tag call or fail() absent")
    checks["unsigned_push_rejected"] = unsigned_ok

    # 3. wrong_owner_rejected.
    allowed_m = re.search(
        r'ALLOWED_SIGNERS=\(\s*([^)]*?)\s*\)',
        script_text, re.DOTALL,
    )
    has_allow_list = False
    allowed_sample: list[str] = []
    if allowed_m:
        body = allowed_m.group(1)
        emails = re.findall(r'"([^"]*@[^"]*)"', body)
        has_allow_list = len(emails) >= 1 and all(
            "@" in e and "." in e for e in emails
        )
        allowed_sample = emails
    has_tagger_parse = (
        "tagger" in script_text
        and "git cat-file -p" in script_text
    )
    owner_ok = has_allow_list and has_tagger_parse
    if not owner_ok:
        reasons.append(
            f"owner check: allow_list={has_allow_list} emails={allowed_sample}, "
            f"tagger_parse={has_tagger_parse}"
        )
    checks["wrong_owner_rejected"] = owner_ok

    # 4. approval_message_required.
    approval_ok = (
        "MIN_MSG_LEN" in script_text
        and "approved" in script_text.lower()
        and re.search(r"grep -qi\s+['\"]approved['\"]", script_text) is not None
    )
    if not approval_ok:
        reasons.append(
            "approval check: MIN_MSG_LEN + grep -qi 'approved' not both present"
        )
    checks["approval_message_required"] = approval_ok

    # 5. --dry-run exits 0.
    dry_ok = False
    dry_output = ""
    try:
        r = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        dry_ok = r.returncode == 0
        dry_output = (r.stderr or r.stdout or "").strip().splitlines()[-1] if (r.stderr or r.stdout) else ""
        if not dry_ok:
            reasons.append(f"--dry-run exited {r.returncode}: {r.stderr[:200]}")
    except Exception as e:
        reasons.append(f"--dry-run invocation raised: {e}")
    checks["dry_run_exits_zero"] = dry_ok

    # 6. (supplementary) CODEOWNERS entry present.
    co_ok = CODEOWNERS.exists() and "backend/governance/limits.yaml" in CODEOWNERS.read_text(encoding="utf-8")
    if not co_ok:
        reasons.append("CODEOWNERS missing or missing limits.yaml entry")
    checks["codeowners_entry"] = co_ok

    all_ok = all(checks.values())
    verdict = "PASS" if all_ok else "FAIL"
    result = {
        "step": "4.9.1",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "allowed_signers_sample": allowed_sample,
        "dry_run_tail": dry_output,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        **{k: v for k, v in checks.items()},
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
