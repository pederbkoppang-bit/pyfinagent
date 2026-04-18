# Evaluator Critique -- Cycle 89 / phase-4.9 step 4.9.1

Step: 4.9.1 Tag-signed-commit enforcement in CI

## Dual-evaluator run (parallel, anti-rubber-stamp)

## qa-evaluator: CONDITIONAL -> PASS (same agent via SendMessage)

### First pass: CONDITIONAL
"Workflow has duplicate `push:` keys (lines 19 and 23). YAML
spec + GitHub Actions will keep only the last mapping, so
push.paths for limits.yaml is silently dropped -- only tag
pushes trigger CI. Researcher explicitly flagged this as the
'mutually exclusive' gotcha; contract promises Job A + Job B
but only one job exists. Fix: merge into a single 'push:' with
both 'branches+paths' and 'tags'."

Exactly the catch the anti-rubber-stamp rule exists for.

### Fix applied SAME cycle
Merged the two `push:` blocks into a single mapping with
`branches + paths + tags` filters. Verified via yaml.safe_load:

    top-level events: ['push', 'pull_request', 'workflow_dispatch']
    push keys: ['branches', 'paths', 'tags']

### Second pass (same agent via SendMessage): PASS
"YAML duplicate-key bug fixed: all three filters parsed cleanly.
All 9 original review points re-confirmed: verify-tag real,
ALLOWED_SIGNERS real, annotated-tag rejection, approval
message, tag-covers-HEAD, fetch-depth, dry-run behavior,
CODEOWNERS entry, pull_request trigger retained."

## harness-verifier: PASS

8/8 mechanical checks green:
- Immutable verification exits 0.
- Audit clean with 6/6 teeth true.
- Script executable.
- Strict mode (no --dry-run) exits 1 with no tag present
  (negative path proven).
- Workflow YAML has all required keys after fix.
- **Mutation A**: remove ALLOWED_SIGNERS entry -> audit rc=1.
- **Mutation B**: disable `grep -qi 'approved'` -> audit rc=1.
- Files restored verbatim.

## Decision: PASS (evaluator-owned)

5th FAIL/CONDITIONAL arc this session fixed in-cycle via
SendMessage to the same evaluator -- not second-opinion shopped.
Research-gate discipline held for 4th straight cycle.
