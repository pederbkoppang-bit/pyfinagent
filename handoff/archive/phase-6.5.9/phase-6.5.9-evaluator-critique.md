# Q/A Critique — phase-6.5 / step 6.5.9 (E2E smoketest with fixtures)

**Verdict id:** `qa_659_v1`
**Cycle:** 1 (first Q/A on 6.5.9; no prior attempt).
**Date:** 2026-04-19.
**Final Decision:** **PASS**

---

## 5-item protocol audit

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Researcher spawn proof | PASS | `handoff/current/phase-6.5.9-research-brief.md` present (22,179 B). JSON envelope at tail: `external_sources_read_in_full: 7`, `snippet_only_sources: 10`, `urls_collected: 17`, `recency_scan_performed: true`, `internal_files_inspected: 8`, `gate_passed: true`. Three-variant queries explicitly listed (frontier-2026 / last-2-year / year-less canonical). |
| 2 | Contract PRE-commit | PASS | `phase-6.5.9-contract.md` mtime `1776627182` (21:33:02) < `phase-6.5.9-experiment-results.md` mtime `1776627295` (21:34:55). Research brief mtime `1776627129` (21:32:09) < contract. Ordering: research → contract → results. |
| 3 | Experiment results present | PASS | Contains verbatim stdout + EXIT=0, audit-JSONL proof (`wc -l` and `tail -1`), regression `152 passed, 1 skipped`, syntax check, 4-row criterion table, Path-D interpretation block, Known-caveats section with 5 transparency items. |
| 4 | Log-last discipline | PASS | Last block in `handoff/harness_log.md` ends with `**Decision:** PASS. Status flip: pending -> done.` referring to 6.5.7 (22/22 tests). No 6.5.9 block yet. Masterplan `6.5.9.status = "pending"` confirms no premature flip. |
| 5 | No verdict-shopping | PASS | First Q/A on 6.5.9. `phase-6.5.9-evaluator-critique.md` did not exist before this file. |

---

## Deterministic checks (A–K)

### A. Immutable command (re-run)

```
$ source .venv/bin/activate && python scripts/smoketest/intel_e2e.py --fixtures
{
  "ts": "2026-04-19T19:35:50.669117+00:00",
  "overall_ok": true,
  "stages": [
    {"name": "load_registry", "ok": true, "total_sources": 3, "active_count": 2,
     "active_ids": ["stub_http", "stub_rss"]},
    {"name": "scan_sources", "ok": true, "families": ["http", "rss"],
     "per_family_counts": {"http": 1, "rss": 1}, "total_candidates": 2},
    {"name": "score_novelty", "ok": true, "scored_count": 2,
     "scorer": "stub:sha256_tiled_1024"},
    {"name": "enqueue_patches", "ok": true, "enqueued_count": 2,
     "unique_pid_count": 2,
     "sample_patch_ids": ["795578d4f3e2d927", "7fe712540b848c8a"]}
  ]
}
EXIT=0
```

`overall_ok: true`. `per_family_counts` has `http: 1` and `rss: 1` (both ≥ 1). Patch IDs differ from prior run (timestamp-derived), confirming non-cached execution. **PASS.**

### B. Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 13.52s
```

Identical to 6.5.7 baseline. **PASS.**

### C. File existence

- `scripts/smoketest/intel_e2e.py` — present (8728 B, 255 lines).
- `handoff/current/phase-6.5.9-contract.md` — present.
- `handoff/current/phase-6.5.9-experiment-results.md` — present.
- `handoff/current/phase-6.5.9-research-brief.md` — present.
- `handoff/audit/intel_e2e.jsonl` — present, **2 rows** after re-run (1 from contract cycle + 1 from my A re-run; matches expected `+1`).

**PASS.**

### D. Scope

`git status --short` filtered on this step: only
- `?? scripts/smoketest/intel_e2e.py` (new file)
- `?? handoff/audit/intel_e2e.jsonl` (new runtime artifact)
- `?? handoff/current/phase-6.5.9-{contract,experiment-results,research-brief}.md`

No modifications to any phase-6.5 production module (`backend/intel/*`), no test-surface edits, no fixture edits. **PASS.**

### E. Criterion alignment (each of 4)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `overall_ok_true` | PASS | Printed summary line 3: `"overall_ok": true`. |
| 2 | `at_least_one_record_per_extractor_family` | PASS (Path-D interpreted) | `per_family_counts: {"http": 1, "rss": 1}` — both active `source_type` families have ≥1. See adversarial-audit reply below. |
| 3 | `novelty_and_digest_stages_pass` | PASS (Path-D interpreted) | S3 `ok: true` (scored_count=2). S5 `ok: true` evidenced by JSONL row count growing 1→2 on re-run. See adversarial-audit reply below. |
| 4 | `exit_0` | PASS | Shell captured `EXIT=0` on re-run. |

**PASS.**

### F. Path-D interpretation grounding

Contract cites `phase-6.5.path_decision` and `phase-6.5.8.superseded_by = 6.5.9`. Verified on disk:
- `.claude/masterplan.json:2075` — `"path_decision": { "selected": "D", "decided_at": "2026-04-19T18:48:06.575444+00:00", ...}`
- `.claude/masterplan.json:2052` — `"superseded_by": "6.5.9"` on the 6.5.8 step.

Both grounding anchors exist. **PASS.**

### G. Stage ordering

`main()` at `intel_e2e.py:208-244`: appends s1 → s2 → s3 → s4 (with skip-guards) → s5. `overall_ok = all(st["ok"] for st in stages)` is evaluated on line 240 AFTER the s5 append (line 238). Order is correct. **PASS.**

Minor note: the JSON *printed to stdout* is `s5["summary"]` (line 243), which was built with `pre_overall_ok = all(st["ok"] for st in stages[:4])` (line 234) — i.e. the summary's `overall_ok` and `stages` list cover S1-S4 only; S5's own `ok` status is NOT inside the printed JSON. The `post-s5` `overall_ok` on line 240 governs the exit code. This is consistent with the experiment_results caveat #1 (the design choice is deliberate and disclosed) and the failure mode is conservative: if S5 write fails, exit code will be 1 even though the printed JSON would say `overall_ok: true`. Acceptable.

### H. Fail-open-per-stage

All s1..s4 functions are wrapped in `try/except Exception as exc:` and return `{ok: false, error: repr(exc), ...}` (lines 63-69, 88-94, 119-120, 167-173). `_write_audit` (176-183) has a try/except returning bool. Nothing can raise up to `main()` except the final `raise SystemExit(...)` which is explicit. **PASS.**

### I. Audit-write idempotency

`_write_audit` uses `_AUDIT_PATH.open("a", encoding="utf-8")` (line 179) — append mode confirmed. Row count grew from 1 to 2 on re-run. Timestamps differ (`19:33:43` vs `19:35:50`), patch_ids differ. **PASS.**

### J. No live-network or live-BQ at call time

`grep` of `requests\.get|bigquery\.Client|voyageai|google\.genai` in `scripts/smoketest/intel_e2e.py` → **no matches**. `nc._stub_embed` is used for embedding (line 102). `ppq._insert` is monkeypatched via `_captive_insert` (lines 129-145). **PASS.**

### K. ASCII discipline

`file` → `ASCII text`. `open(..., 'rb').read().decode('ascii')` succeeded. **PASS.**

---

## LLM judgment — adversarial Path-D audit

### Counter-argument 1 (criterion 2): "the criterion says *extractor* family, not *source_type*"

An adversarial auditor could argue that the 9-step design reserved distinct extractor modules (6.5.3 institutional, 6.5.4 academic, 6.5.5 AI-frontier, 6.5.6 player-driven) and that "extractor family" is a *behavioral* construct tied to custom parsing logic per type. Under Path D, `BaseScanner` is a single generic scanner — so arguably *no* extractor families exist; there is exactly one extractor, and 2 candidates means 2 records from 1 family.

**Reply.** The Path D `path_decision.summary` on disk says Path D "drops 5 source-specific extractors." The `source_type` field on `SourceRow` and `DocumentCandidate` is exactly the dispatch key that *would* have selected between 6.5.3-6.5.6. Path D compressed the implementations but preserved the type taxonomy. Requiring ≥1 record per `source_type` in the fixture's active set is the smallest still-meaningful reading of criterion 2 that remains executable after Path D. The alternative reading ("≥1 record from each of the 4 dropped extractors") is inexecutable by construction (the extractors don't exist) and would force criterion 2 to be vacuously FAIL or vacuously PASS, neither of which is useful. The contract's interpretation is defensible and disclosed.

**Accepted, not rubber-stamped.** The interpretation would be stronger if the fixture included more than 2 active source_types (the current 2-family check is a minimal-viability bar). Flagging as a non-blocking observation: when phase-7 wires real sources in, a broader family set should be in the fixture to give criterion 2 more teeth.

### Counter-argument 2 (criterion 3): "digest" means Slack digest, period

A strict reading of "novelty_and_digest_stages_pass" from the pre-Path-D era would require a Slack-push stage to return success. S5 emits only JSON + a JSONL row — no Slack wire touched.

**Reply.** `6.5.8.superseded_by = "6.5.9"` is an explicit on-disk pointer saying *the Slack digest step is replaced by the smoketest step*. The masterplan's own change-tracking mechanism resolves the ambiguity: under Path D, the digest artifact is the JSON summary + audit row, not a Slack post. The contract cites this pointer; the experiment_results re-cites it; the smoketest comments cite it (line 189-190).

**Accepted.** This is grounded in the masterplan's change mechanism, not in an ad-hoc redefinition. No criterion was silently weakened; the substitution is declared explicitly.

### Was any criterion silently weakened?

- Criterion 1 `overall_ok_true`: literal, no reinterpretation.
- Criterion 2 `at_least_one_record_per_extractor_family`: reinterpreted, but *declared explicitly* in the contract and experiment_results, with on-disk pointer grounding.
- Criterion 3 `novelty_and_digest_stages_pass`: reinterpreted, declared, pointer-grounded.
- Criterion 4 `exit_0`: literal.

No silent weakening. Both reinterpretations are visible in the handoff files and verifiable against `.claude/masterplan.json`.

### Phase-6.5 closure check

After 6.5.9 flips to `done`:
- 6.5.1 currently `pending` — will be flipped by Main as part of the Path-D closeout. Confirm before phase-status flip.
- 6.5.2 currently `pending` — same.
- 6.5.7 currently `pending` — same.
- 6.5.9 currently `pending` — to be flipped after this Q/A PASS + log-append.
- 6.5.3/4/5/6/8 `dropped`.

**Observation to Main (not a blocker for 6.5.9 itself):** the experiment_results line 96-99 asserts "4/4 kept steps done" but on disk 6.5.1/6.5.2/6.5.7 are still `pending`. This is a stale-state claim at the time of writing but becomes true only after the log-append + status-flip housekeeping. Main should verify the masterplan shows all 4 kept steps as `done` before flipping `phase-6.5` phase-level status.

### Weakest-link (non-blocking)

- No live BQ, no live HTTP, no live Voyage/Gemini. A production operator must run at least one integration-mode cycle against staging BQ before wiring to production data. Disclosed in experiment_results caveat #3.
- S4 monkeypatches `ppq._insert` in-process; production-path BQ insert is covered by 6.5.7 tests but not end-to-end in this smoketest. Disclosed in caveat #4.
- The fixture only has 2 active `source_type` families — criterion 2 is minimally exercised.

All three are disclosed with transparency and appropriate scoping; none blocks PASS.

---

## Violated criteria

**None.**

## Violation details

`[]`

## Checks run

`["syntax", "verification_command_rerun", "regression_pytest", "file_existence", "scope_git_status", "criterion_alignment", "path_d_grounding_masterplan", "stage_ordering", "fail_open_per_stage", "audit_idempotency", "no_live_network", "ascii_discipline", "adversarial_path_d_audit", "silent_weakening_check"]`

---

## Final Decision

**PASS** — `qa_659_v1`.

All four immutable criteria met under a Path-D interpretation that is explicitly grounded in on-disk masterplan pointers (`phase-6.5.path_decision`, `phase-6.5.8.superseded_by = 6.5.9`). Immutable command re-run exit 0. Regression green (152p/1s unchanged). Audit JSONL append is idempotent (1→2 rows). No silent weakening of criteria. No scope creep. ASCII-clean.

**Advisory to Main (non-blocking):**

1. Before flipping `phase-6.5` phase-level status to `done`, verify that 6.5.1, 6.5.2, and 6.5.7 are actually `done` in `.claude/masterplan.json` (currently `pending` on disk per my check F extraction). The experiment_results "4/4 kept steps done" claim will become true only after that housekeeping.
2. The single-source-type-per-family fixture coverage means criterion 2 is at its minimum-viable bar. Non-blocking; revisit when phase-7 wires real sources.
3. A future integration cycle should exercise at least one stage against staging BQ before wiring to prod data.

## Final JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "verdict_id": "qa_659_v1",
  "reason": "All 4 immutable criteria met. Re-run exit 0; overall_ok true; per_family_counts http:1,rss:1; S3+S5 ok; audit JSONL append idempotent (1->2). Path-D interpretations of criteria 2 and 3 grounded in on-disk masterplan.path_decision + 6.5.8.superseded_by=6.5.9. Regression 152p/1s unchanged.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command_rerun", "regression_pytest", "file_existence", "scope_git_status", "criterion_alignment", "path_d_grounding_masterplan", "stage_ordering", "fail_open_per_stage", "audit_idempotency", "no_live_network", "ascii_discipline", "adversarial_path_d_audit", "silent_weakening_check"],
  "advisories": [
    "Main: verify 6.5.1/6.5.2/6.5.7 are actually done on disk before flipping phase-6.5 phase-level status; currently pending.",
    "Criterion 2 is minimally exercised (2 families). Broaden fixture in phase-7.",
    "No live-integration coverage; run staging BQ cycle before prod wiring."
  ]
}
```
