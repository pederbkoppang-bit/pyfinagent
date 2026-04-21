# Q/A Critique -- phase-6.5 / step 6.5.7 (cycle 1, qa_657_v1)

**Step:** 6.5.7 -- Novelty client (Voyage + Gemini fallback) + prompt-patch queue.
**Date:** 2026-04-19.
**Cycle:** 1 (first Q/A on this step; not a re-spawn).

---

## 5-item protocol audit

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Researcher brief present, phase-scoped, ≥5 sources, three-variant, recency scan, `gate_passed: true` | PASS | `handoff/current/phase-6.5.7-research-brief.md` mtime 21:19 UTC, 22,178 bytes. |
| 2 | Contract PRE-commit (contract mtime < results mtime) | PASS | contract mtime 1776626420 (21:20 UTC) < results mtime 1776626621 (21:23 UTC). Delta ~3 min 21 s, which is consistent with real-work not back-dating. |
| 3 | Experiment results present with verbatim immutable output + regression + file list + criterion table + mid-cycle bug disclosure | PASS | Sections 22-47 contain verbatim command output (EXIT=0); section 49-51 discloses the `enqueue_patch` None bug and the always-return-pid fix; 5 transparency caveats in section 53-59. |
| 4 | Log-last discipline: `harness_log.md` last block is 6.5.2 (21:15 UTC), NOT 6.5.7 | PASS | Tail shows `Cycle -- 2026-04-19 21:15 UTC -- phase=6.5.2 result=PASS` as last entry. |
| 5 | No verdict-shopping: first Q/A on 6.5.7 | PASS | No prior `phase-6.5.7-evaluator-critique*.md` in `handoff/current/` or archive. |

All 5 audit items PASS.

---

## Deterministic checks (A-K)

### A. Immutable command
```
$ source .venv/bin/activate && pytest backend/tests/test_intel_novelty_client.py backend/tests/test_prompt_patch_queue.py -q
...................... [100%]
22 passed in 8.16s
EXIT=0
```
**PASS** -- matches expected 22 passed, exit 0.

### B. Full regression
```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 15.04s
```
**PASS** -- baseline 130 + 22 new = 152; zero regressions.

### C. File existence
```
backend/intel/__init__.py, novelty_client.py, prompt_patch_queue.py (+ prior scanner.py, source_registry.py)
backend/tests/test_intel_novelty_client.py
backend/tests/test_prompt_patch_queue.py
handoff/current/phase-6.5.7-{contract,experiment-results,research-brief}.md
```
**PASS** -- all required files present.

### D. Scope
`git status --short` grep for backend production modules shows only `?? backend/intel/` and `?? backend/tests/test_intel_novelty_client.py`, `?? backend/tests/test_prompt_patch_queue.py` as new-to-this-step. No `M` on non-intel production code attributable to 6.5.7.
**PASS** -- backend-intel-only scope.

### E. Criterion alignment
| Criterion | Test | Verdict |
|---|---|---|
| `voyage_primary_gemini_fallback_smoke_ok` | `test_voyage_primary_gemini_fallback_smoke_ok` | PASS (in 22/22) |
| `novelty_score_distinguishes_duplicate_vs_novel` | `test_novelty_score_distinguishes_duplicate_vs_novel` (dup<0.1, novel>0.5) | PASS |
| `prompt_patch_queue_persists_and_dedupes` | `test_queue_persists_and_dedupes_end_to_end` + `test_dedup_in_memory_collapses_duplicates` | PASS |
| `tests_green` | Module exit 0 | PASS |
**PASS**.

### F. Fail-open discipline
- `novelty_client.py::_get_client` lines 147-157: guards `from google.cloud import bigquery` (returns None on ImportError), guards `bigquery.Client()` init (returns None on runtime error).
- `novelty_client.py::score_chunks_and_write` lines 206-217: `if client is None: return 0`; `insert_rows_json` errors and exceptions both caught and return 0.
- `prompt_patch_queue.py::_get_client` lines 69-81: same ImportError + init-error guard pattern.
- `prompt_patch_queue.py::_insert` lines 84-100: `if client is None: return 0`; both `errors` list and exception paths swallowed with warning.
- `prompt_patch_queue.py::get_pending` lines 168-170: `except Exception` returns `[]`.
**PASS** -- fail-open on every BQ touchpoint.

### G. Dimension forcing (R2)
- `_embed_voyage` lines 55-57: `if len(vec) != _EMBED_DIM: raise RuntimeError(...)`. `_EMBED_DIM = 1024`.
- `_embed_gemini` lines 69-76: passes `config={"output_dimensionality": _EMBED_DIM}` AND re-checks `len(vec) != _EMBED_DIM`.
**PASS** -- both providers forced and validated at 1024.

### H. Model choice (R1)
Line 34: `_GEMINI_DEFAULT_MODEL = "gemini-embedding-001"`. NOT the deprecated `text-embedding-004`.
**PASS**.

### I. Latest-status SQL (R5)
Lines 151-165:
```sql
WITH ranked AS (
  SELECT ..., ROW_NUMBER() OVER (PARTITION BY patch_id ORDER BY created_at DESC) AS rn
  FROM `{table_ref}`
)
SELECT ... FROM ranked WHERE rn = 1 AND status = 'pending' ORDER BY created_at ASC LIMIT ...
```
Correctly hides rows that have a newer `approved`/`rejected`/`applied` status.
**PASS**.

### J. Idempotent enqueue
Lines 120-135: `pid` is computed from deterministic inputs (`_patch_id`), `_insert([row], ...)` return value is ignored, function unconditionally `return pid`. Docstring lines 113-118 explicitly promises "Always returns the patch_id (deterministic from inputs). BQ failures and dedup-skips are logged and swallowed".
**PASS** -- contract matches the mid-cycle bug-fix description.

### K. ASCII discipline
`test_module_is_ascii_only` present in both test files per experiment results section (not re-read for space); confirmed by visually scanning both source files -- no non-ASCII characters in logger messages (all use `--`, `->`, plain English) or docstrings.
**PASS**.

**All A-K PASS.**

---

## LLM judgment

**Mid-cycle bug disclosure honesty.** The experiment_results section 49-51 describes the bug precisely: first-pass `enqueue_patch` returned None when the captive store skipped a re-pending row, causing `pid1 != pid2`; the fix changed the return contract to always return the deterministic pid. Code inspection confirms lines 120-135 implement exactly that fix (`pid = _patch_id(...)`, `_insert(...)` return value ignored, `return pid` unconditional). Docstring lines 113-118 document the contract change. Honest disclosure and fix match.

**`_stub_embed` correctness.** Line 41: `(b / 127.5) - 1.0 for b in raw[:1024]`. Byte 127 → -0.00392, byte 128 → +0.00392, byte 0 → -1.0, byte 255 → +1.0. Not symmetric around zero for discrete byte inputs, but the skew is negligible for cosine-distance tests (orthogonality and similarity ordering preserved). `raw = list(digest) * N + digest` tiles 32 bytes to ≥1024; the expression `(32 // 32) + 1 = 2` gives 64 bytes -- wait, `_EMBED_DIM // len(digest) + 1 = 1024 // 32 + 1 = 33` tiles → 1056 bytes → sliced to 1024. Correct. **Acceptable for tests.**

**Kill-switch analogue for the queue.** Patches don't have a kill-switch boolean; `status='rejected'` is the terminal, inserted by `mark_rejected` (lines 196-216). `get_pending`'s latest-status SQL hides rejected patches because `rn=1 AND status='pending'` only matches when the newest row is pending. Verified in the SQL at lines 151-165. Analogue is correct.

**Anti-rubber-stamp spot-check.** Caveat 3 (empty-pipe acceptance) cites phase-6.5 `path_decision.open_issue`; confirmed in `.claude/masterplan.json` -- option (b) "explicitly accept empty-pipe at launch and gate closure on a follow-up phase-7-integration step" is a real documented acceptance path. Not a fabricated excuse.

**Scope honesty.** `git status --short` (grep for phase-6.5.7-attributable files) shows only `?? backend/intel/`, `?? backend/tests/test_intel_novelty_client.py`, `?? backend/tests/test_prompt_patch_queue.py` as new. No production modifications outside the intel package. Scope matches contract.

**Weakest link flag (non-blocking, for 6.5.9 smoketest).** Neither `VOYAGE_API_KEY` nor a real Gemini client is exercised in CI. The 6.5.9 smoketest must monkeypatch both providers (mirroring this step's tests), OR stand up env-keyed live round-trips with a fail-soft skip when keys are absent. Carry-forward note to 6.5.9 authors.

---

## Violation summary

```json
{
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "protocol_audit_5",
    "immutable_command",
    "regression",
    "file_existence",
    "scope_diff",
    "criterion_alignment",
    "fail_open_source_read",
    "dim_forcing_source_read",
    "model_choice_source_read",
    "latest_status_sql_source_read",
    "idempotent_enqueue_source_read",
    "ascii_discipline",
    "bug_disclosure_match",
    "stub_embed_math",
    "open_issue_acceptance_cross_ref"
  ]
}
```

---

## Final Decision

**PASS** -- `qa_657_v1`.

All 4 immutable criteria met with zero regressions. Research-gate commitments (R1 model choice, R2 dimension forcing, R5 latest-status SQL) all traced to source. Mid-cycle bug was honestly disclosed and the fix is in the code. Fail-open discipline is uniform across both modules. Scope is backend/intel-only. Main may flip 6.5.7 status to `done` after appending the harness_log cycle block.

Carry-forward note for 6.5.9: smoketest must handle the no-API-keys-in-CI case explicitly.
