# live_check — phase-86.3

Required shape (from the masterplan): *"live_check_86.3.md with the line count
and sha256 of handoff/kill_switch_audit.jsonl before and after a full
backend/tests run with the backend up, plus the enumeration of
live-host-reaching tests."*

Captured 2026-08-09, live tree (NOT a worktree — a worktree gives a different
failure shape because it lacks gitignored files; compare live-to-live only).

---

## 1. Criterion 1 — full `backend/tests` run, backend up, ZERO rows

Backend confirmed listening first, because the defect only fires when it is:

```
$ curl -s -o /dev/null -w "api/health=%{http_code}\n" -m 5 http://localhost:8000/api/health
api/health=200
```

```
=== BEFORE ===
      62
90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653  handoff/kill_switch_audit.jsonl

$ source .venv/bin/activate && python -m pytest backend/tests -q --timeout=120 --tb=no
12 failed, 3072 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 332.58s (0:05:32)

=== AFTER ===
      62
90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653  handoff/kill_switch_audit.jsonl
```

**Line count 62 → 62. sha256 identical. Delta = 0 rows.**

For contrast, the same suite on 2026-08-08 appended **8 rows** (54 → 62) across
two runs, in 4-row clusters at `22:29:41-43Z` and `22:36:59-22:37:01Z`.

### 1a. The first attempt at this measurement CRASHED — recorded, not hidden

```
backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py ..E
object repr     : ValueError('I/O operation on closed file.')
lost sys.stderr
```

Died at ~13%. **The crash was in my own rewritten file**, which had passed
every isolated run (`4 passed`, `17 passed`, `87 passed`). Cause: I entered
`TestClient` as a context manager, which runs the app **lifespan** mid-suite.
Fixed by constructing the client without `with`, matching
`test_phase_80_2_error_response_contract.py:38`. The numbers above are from the
run **after** that fix. The crashed run is reported because a suite that dies
at 13% cannot evidence "zero rows across the whole suite", and I had labelled
it inconclusive rather than green.

---

## 2. Criterion 5 — status changes vs the measured baseline

**Baseline** (`handoff/current/live_check_85.4.md` §5, live tree, 2026-08-08):
`26 failed, 3017 passed, 12 skipped, 5 xfailed, 1 xpassed`, with all 26 node ids
recorded there.

**Now:** `12 failed, 3072 passed, 12 skipped, 5 xfailed, 1 xpassed`.

### 2a. New failures: ZERO

The current 12 are a strict **subset** of the baseline 26:

```
test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit
test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
test_phase_75_17_verification_paths.py::test_masterplan_diff_touches_only_the_ten_sibling_insertions
test_phase_75_17_verification_paths.py::test_sweep_shape_census_matches_the_corrected_figures
test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token
test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
test_phase_82_39_outcome_rebuild_query.py::test_the_sweeps_recall_limit_is_recorded_not_assumed
test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
```

### 2b. 14 tests went failing → passing. **I claim ONE of them.**

The failure count dropped by 14. **Reporting this as "my change fixed 14 tests"
would be false.** Full accounting, and 11 of the 14 are MEASURED rather than
argued:

| # | Tests | Cause | How established |
|---|---|---|---|
| 11 | `test_64_3_currency_path` ×3, `test_64_4_multi_market_e2e`, `test_dod4_tier1_coverage_investment`, `test_phase_70_3_atomic_swap`, `test_phase_70_4_gate_observability` ×2, `test_price_tolerance_gate` ×3 | **step 36.28's live-pause coupling** — not my change | **Reproduced.** See §2c |
| 1 | `test_book_safety_69.py::test_valid_nav_still_breaches` | fixed by **step 85.5.1** | documented in 86.5's audit basis |
| 1 | `test_phase_23_2_15_...::test_phase_23_2_15_known_pass_scripts_still_pass` | **not root-caused** — see §2d | measured under both conditions |
| 1 | `test_phase_23_2_4_...::test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s` | **this step** | it is this step's own test |

### 2c. The 11: measured by reproducing the baseline's live condition

The 2026-08-08 baseline was taken while the operator's book was **PAUSED**
(ask #21: `paused: True, reason: manual, paused_at 2026-08-08T08:35:16Z`).
Today it is **unpaused** (`paused: false`, measured). Those tests construct
`PaperTrader` without the `kill_switch_state` injection seam, so `execute_buy`
falls back to the module singleton, which replays the real on-disk audit —
**exactly step 36.28.**

Rather than assert that, I forced the singleton back to the baseline's state
(a throwaway pytest plugin; `_AUDIT_PATH` redirected to tmp first, so it wrote
nothing) and re-ran those six files:

```
[paused_book_plugin] kill switch forced paused=True, audit -> /var/folders/.../kill_switch_audit.jsonl
11 failed, 95 passed, 1 warning in 2.98s
```

**All 11 fail again**, node-for-node. They are a live-state artefact of the
baseline, not a regression and not a repair. **This is 36.28 demonstrated live:
suite greenness is coupled to the operator's pause state, and the same tree
gives two different answers depending on it.**

### 2d. The 12th, stated as unexplained

`test_phase_23_2_15_known_pass_scripts_still_pass` passes **in both**
conditions today (`3 passed, 2 xfailed` paused and unpaused). Its 2026-08-08
failure therefore had some other live-system cause — plausibly the anchor
deadlock that 85.6 fixed, since it shells out to verify scripts. **I did not
root-cause it**, and I am not attributing it to this step.

### 2e. Honest reading of criterion 5

Criterion 5 says *"no other test changes status vs a measured baseline"*.
Thirteen other tests **did** change status. **None was caused by this change** —
11 measured by reproduction, 1 attributable to 85.5.1, 1 unexplained but
condition-independent. The baseline is **confounded**: it was captured under a
different live kill-switch state, which is the very coupling 36.28 exists to
remove. Stated plainly rather than reported as a clean pass.

---

## 3. Criterion 4 — enumeration of live-host-reaching tests

**Rule:** an executed call site to a real network client whose host:port
resolves at runtime. Docstrings, strings passed as arguments, and AST-analysis
subjects do **not** count.

| File | Class | Contained now? |
|---|---|---|
| `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | **MUTATING** → live backend | **YES** — rewritten in-process; the guard also refuses it |
| `test_phase_23_2_9_ticker_meta_latency.py` | read-only GET on `:8000` | n/a — GETs deliberately allowed (see below) |
| `test_phase_23_2_13_governance_watcher.py` | read-only GET on `:8000` | n/a |
| `test_phase_23_2_7_red_line_nav_match.py` | read-only GET on `:8000` | n/a |
| `test_phase_76_9_2_max_bridge.py` | POSTs to its **own** `ThreadingHTTPServer` on an ephemeral port | **unaffected by design** — the policy keys on host AND port |

**Excluded after correction** (my first census had them as live-host, wrongly):
`test_phase_36_7_...` (`curl` in a docstring, `:10`), `test_phase_75_17_...`
(`:303` `curl` string passed as an argument, never executed),
`test_phase_75_deploy_surface.py` (AST-inspects another file),
`test_phase_80_2_error_response_contract.py` (in-process `TestClient`).

**GETs are allowed deliberately, not by oversight:** `_backend_is_up()` runs at
**module import** inside a `skipif` and catches only
`(URLError, OSError, TimeoutError)`. Raising on the GET probe would error the
whole module's collection and take `test_phase_23_2_4_audit_log_clean_transitions`
with it — the test criterion 7 requires to keep passing.

### Channels NOT contained — stated, not implied

- **`httpx`** — rides `httpcore`, not urllib3. Not covered.
- **raw `socket`** — not covered.
- **the filesystem channel** — an in-process `kill_switch` write while
  `_AUDIT_PATH` still points at the live file. No network guard can see it.
  **Filed as step 86.6** with the false-negative check written into its
  criteria. Today's exposure is bounded: the one known in-process writer
  (86.1) is inert while `kill_switch_peak_reset_enabled` is `False` (measured).

---

## 4. Criterion 7 — the inherited allowlist is byte-unchanged

```
$ git diff HEAD~1 -- backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py \
      | grep -E '^[-+].*(trigger|sod_snapshot|peak_update|cleanup)'
(no output)
```

`test_phase_23_2_4_audit_log_clean_transitions` is untouched, still reads the
**live** journal, and passes in the full-suite run above.

---

## 5. What this live_check does NOT establish

- It does **not** show the suite is clean — 12 pre-existing failures remain,
  triaged by step 86.5.
- It does **not** cover the `tests/` tree. The root `conftest.py` is loaded for
  it, but no `tests/`-tree run was measured here; criterion 1 is scoped to
  `backend/tests`.
- It does **not** prove the absence of in-process writers to the live journal.
  §8 of `experiment_results_86.3.md` shows my own census produced a **false
  negative** on `test_book_safety_69.py`, which is 86.1's landmine.
- The 12-row and 8-row historical figures are quoted from prior artifacts
  (ask #21, `live_check_85.5.1.md`); only the 62→62 delta above was measured in
  this session.
