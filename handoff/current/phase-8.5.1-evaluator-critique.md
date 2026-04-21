# Q/A Evaluator Critique — phase-8.5 / 8.5.1 Define candidate space

**Q/A run:** `qa_851_remediation_v2`
**Date:** 2026-04-20
**Prior runs:**
- `qa_851_v1` (PASS, 2026-04-19) — superseded; operated on inline-authored brief that did not satisfy the research-gate source-floor in the documented form.
- `qa_851_remediation_v1` (FAIL, 2026-04-20) — blocked by archive-handoff hook race that churned `handoff/current/` into 150+ phantom archive dirs.

This v2 run is NOT verdict-shopping. Evidence is materially new vs v1: (a) a researcher-authored 167-line brief now exists in `handoff/current/` (v1 had a ~60-line closure-style inline brief); (b) the archive-handoff hook is guard-flagged and the three handoff files are stable (the v1-remediation failure mode is fixed at infrastructure level).

---

## 5-item protocol audit

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Research brief substantive + in `handoff/current/` | PASS | `wc -l phase-8.5.1-research-brief.md` = 167 lines. Arithmetic table present (dimension breakdown → 5·4·3·2·5·5·5 = 15,000). Three-variant search section present with named queries: current-year frontier (`…2026`), last-2-year (`…2025`), year-less canonical. 5 sources read-in-full via WebFetch, 11 snippet-only, 16 URLs collected total. Recency scan section present and non-empty. JSON envelope ends `gate_passed: true`. |
| 2 | Contract PRE-commit (ordering research→contract→experiment-results) | PASS | mtime epoch: research-brief=1776697415 → contract=1776697438 → experiment-results=1776697449. Strictly ascending. |
| 3 | Experiment results verbatim | PASS | `phase-8.5.1-experiment-results.md` includes immutable command block with exit 0, independent cartesian recompute (`cartesian product = 15000 == declared = 15000`), and `ls handoff/current/phase-8.5.1-*.md` showing all three files stable after the hook guard. |
| 4 | Log-last discipline | PASS | `handoff/harness_log.md` tail shows last entries are 03:22 UTC 8.5.0 remediation + the NOOP go-live block. No 8.5.1 entry yet — log append correctly deferred until after this Q/A PASS. |
| 5 | No verdict-shopping on unchanged evidence | PASS | v1 evidence was inline brief + hook-churned files. v2 evidence is researcher-authored 167-line brief + guard-flagged stable handoff dir. Material change documented in contract § "Why remediation v2" and experiment-results § "Infrastructure note". Spawning fresh Q/A against updated files is the documented cycle-2 flow (Anthropic harness-design, multi-agent-research-system). |

---

## Deterministic A–G (verbatim)

### A. Immutable verification command
```
$ source .venv/bin/activate && test -f backend/autoresearch/candidate_space.yaml && \
  python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); \
  assert d['estimated_combinations'] >= 10000"
estimated_combinations= 15000
A_PASS
```
Exit 0. 15000 ≥ 10000. **PASS**

### B. Independent cartesian recompute
```
Dims: [('learning_rate', 5), ('max_depth', 4), ('n_estimators', 3), ('rolling_window', 2),
       ('prompts', 5), ('features', 5), ('model_archs', 5)]
Product: 5*4*3*2*5*5*5 = 15000
Declared: 15000
B_PASS
```
Q/A independently multiplied YAML list lengths. Matches declared. **PASS**

### C. transformer_signals membership
YAML dump: `transformer_signals: ['timesfm_forecast_20d', 'chronos_forecast_20d', 'ensemble_blend_median']`.
Both required tokens present. **PASS**

### D. Backing module files exist
```
-rw-r--r--  backend/backtest/ensemble_blend.py   (14050 bytes)
-rw-r--r--  backend/models/chronos_client.py     (7316 bytes)
-rw-r--r--  backend/models/timesfm_client.py     (7734 bytes)
```
All three present and non-trivial. **PASS**

### E. Regression
```
152 passed, 1 skipped, 1 warning in 14.93s
```
Matches the expected 152/1. **PASS**

### F. `.claude/archive-handoff.disabled` exists
```
DISABLED_FLAG_EXISTS
```
**PASS**

### G. Hook early-exit guard in `archive-handoff.sh`
Read lines 1–15 of `.claude/hooks/archive-handoff.sh`:
```bash
# REMEDIATION GUARD (2026-04-20): exit early if the remediation flag
# file exists. Bug: when HEAD masterplan isn't committed, every `done`
# step looks newly-done vs HEAD, so this hook churns the archive dir on
# every masterplan write. `.claude/archive-handoff.disabled` bypasses
# the hook until the operator removes it.
if [ -f "${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/archive-handoff.disabled" ]; then
    exit 0
fi
```
Guard is present, correctly placed BEFORE `set -euo pipefail`, and references the exact flag file that exists on disk. **PASS**

---

## LLM judgment

**Is the research brief a real researcher audit?**
Yes. 167 lines, arithmetic table with per-dimension cardinalities, three named search-variants, 5 in-full sources in the required quality tiers (arXiv preprint, scikit-learn + Hyperopt official docs, Oxford NSR peer-reviewed, MachineLearningMastery practitioner), 11 snippet-only sources in a separate table (satisfies the "10+ URLs" floor with room: 16 total), explicit Recency scan section, Research-Gate Checklist all-checked with the final JSON envelope `gate_passed: true`. This is materially different from the v1 closure-style inline brief.

**Is the infra-fix disclosed honestly?**
Yes. Contract § "Why remediation v2" explicitly names the failure mode ("150+ phantom archive dirs"), cites the source dir (`handoff/archive/phase-8.5.1-v99/`), and documents the fix (flag file + hook guard). Experiment-results § "Infrastructure note" instructs the operator on when to remove the flag. No overclaim — YAML content was already correct; only the handoff-dir instability was in scope for this remediation.

**Is the YAML itself correct?**
Yes. 5·4·3·2·5·5·5 = 15,000 = declared. No inflation. Transformer signals present and backed by three existing modules totaling ~29 KB of code.

**Anti-rubber-stamp check:**
No planted mutation-test was requested for this step; candidate_space.yaml is a pure declaration with arithmetic-verifiable content. The arithmetic recompute IS the mutation-resistance check — if anyone altered a list length without updating `estimated_combinations`, B would fail. B currently passes.

**Contract alignment:**
Immutable criterion in contract line 25 matches masterplan.json. Three success criteria (committed, ≥1e4, transformer_signals) all verified. Research-gate summary in contract cites the 167-line brief by line count and content markers (arithmetic, three-variant, 5 sources). Alignment clean.

**Scope honesty:**
Experiment-results acknowledges "No YAML changes; content was already correct" — the remediation is purely handoff-discipline, not a code change. This is honest scope disclosure.

---

## Violated criteria

`violated_criteria: []`
`violation_details: []`

## Checks run

`checks_run: ["protocol_audit_5item", "immutable_verification_command", "independent_cartesian_recompute", "transformer_signals_membership", "backing_module_existence", "regression_pytest_152_1", "hook_guard_flag_existence", "hook_source_guard_grep", "research_brief_content_grep", "mtime_ordering", "harness_log_tail", "contract_yaml_alignment"]`

---

## Final Decision

**PASS — `qa_851_remediation_v2`**

All 7 deterministic checks (A–G) green. All 5 protocol-audit items green. Research brief is a real, tiered, three-variant audit with honest arithmetic. Infrastructure fix (hook guard) is in place and documented. Evidence is materially new vs v1 (fresh brief + stable handoff dir), so this is not verdict-shopping. No violated criteria.

**Next action for Main:** append the `## Cycle -- 2026-04-20 -- phase=8.5.1 result=PASS` block to `handoff/harness_log.md`, THEN flip masterplan.json phase-8.5.1 status to `done`. Per log-last discipline, the log append must precede the status flip.

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "qa_851_remediation_v2 PASS. All deterministic A-G green (immutable cmd exit 0, cartesian 15000=15000, transformer signals present, 3 backing modules exist, 152/1 regression, archive-handoff guard flag + hook early-exit both in place). 5/5 protocol audit green (substantive 167-line researcher brief, pre-commit mtime order, verbatim verification, log-last deferred, material evidence change vs v1).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5item",
    "immutable_verification_command",
    "independent_cartesian_recompute",
    "transformer_signals_membership",
    "backing_module_existence",
    "regression_pytest_152_1",
    "hook_guard_flag_existence",
    "hook_source_guard_grep",
    "research_brief_content_grep",
    "mtime_ordering",
    "harness_log_tail",
    "contract_yaml_alignment"
  ]
}
```
