# Experiment Results (DRAFT) -- Step 75.13: Python dependency integrity

- **Executor**: Sonnet-tier GENERATE-phase agent (delegated, per contract_75.13.md)
- **Date**: 2026-07-24
- **Scope executed**: contract plan steps 1-9. File writes + tests ONLY -- no pip install/uninstall, no `.env` edit, no touch to `.claude/masterplan.json` / `handoff/harness_log.md`, no git commit/push.

## Boundary proof: zero environment mutation

```
$ .venv/bin/pip freeze | shasum -a 256      # BEFORE first edit
8df19b228e083e35c4c0de62aa3209c38110f2f6227e0008b7e401ae4141e2af  -

$ .venv/bin/pip freeze | shasum -a 256      # AFTER last edit + full mutation matrix
8df19b228e083e35c4c0de62aa3209c38110f2f6227e0008b7e401ae4141e2af  -
```

Identical hash, both 303 lines. No package was installed, uninstalled, or upgraded at any point. Confirmed additionally by:

```
$ bash -n scripts/autoresearch/run_nightly.sh && echo "syntax OK"
syntax OK
$ git diff --stat -- scripts/autoresearch/run_nightly.sh
(empty output -- zero lines changed)
```

The 75.11 paging seam and the phase-51.4 `_embedding_preflight()` soft-skip were read but never edited.

## Files touched

| File | Change | Type |
|---|---|---|
| `backend/requirements.lock` | NEW: 325-line file = 22-line `#`-comment header (regen command, two sync commands, date, pin count) + verbatim 303-line `.venv/bin/pip freeze` output, unmodified | new |
| `backend/requirements.txt` | +6 real declaration lines (`google-cloud-storage`, `numpy`, `exchange-calendars`, `python-dateutil`, `PyYAML`, `pytest`), each with a one-line consumer-anchoring comment; enhanced the existing `xlrd` comment; deleted the `# PDF generation` comment + `fpdf2>=2.7.0` line (zero residue) | modified |
| `.github/workflows/pip-audit.yml` | Added a lock-targeting `pip-audit` run step (`--requirement backend/requirements.lock`), kept the floors-file run step, added `backend/requirements.lock` to both `push`/`pull_request` `paths:` filters, added the lock to the failure-artifact upload list, updated the header comment | modified |
| `scripts/autoresearch/requirements-autoresearch.txt` | NEW: pins `gpt-researcher==0.14.8` + its huggingface-embedding closure (`langchain-huggingface==1.2.1`, `sentence-transformers==5.5.1`), header names the launchd nightly-memo consumer | new |
| `scripts/autoresearch/run_memo.py` | Added `_gpt_researcher_guard()` (find_spec check) + its call site in `main()`, placed BEFORE `_embedding_preflight()` | modified |
| `backend/tests/test_phase_75_deps.py` | NEW: 12 tests covering all 5 criteria groups | new |

```
$ git diff --stat -- backend/requirements.txt .github/workflows/pip-audit.yml scripts/autoresearch/run_memo.py
 .github/workflows/pip-audit.yml  | 23 ++++++++++++++++++++---
 backend/requirements.txt         | 11 +++++++----
 scripts/autoresearch/run_memo.py | 33 +++++++++++++++++++++++++++++++++
 3 files changed, 60 insertions(+), 7 deletions(-)
```

(`backend/requirements.lock`, `scripts/autoresearch/requirements-autoresearch.txt`, `backend/tests/test_phase_75_deps.py` are new/untracked files, so they don't show in `git diff --stat`; see `git status --porcelain` in the live_check evidence.)

**Note on unrelated diffs observed in the working tree**: `git status` also shows modifications to `.claude/masterplan.json`, `handoff/.cycle_heartbeat.json`, `handoff/audit/*.jsonl`, `handoff/current/contract.md`, `handoff/current/research_brief.md`, and `handoff/kill_switch_audit.jsonl`. **None of these are from this executor** -- this session never opened Edit/Write on any of them. They reflect concurrent activity elsewhere in the shared repo (other parallel executors / hooks) during this GENERATE pass. Flagging for the record so Q/A doesn't attribute them to 75.13.

## Immutable verification command -- verbatim, exit 0

```
$ python3 -c "import os; l=open('backend/requirements.lock').read().splitlines(); pins=[x for x in l if '==' in x]; assert len(pins)>=150, 'lock too small to be a real venv snapshot'; assert any(x.lower().startswith('exchange') for x in pins), 'exchange-calendars not locked'; r=open('backend/requirements.txt').read(); assert 'exchange-calendars' in r and 'fpdf2' not in r and 'PyYAML' in r and 'numpy' in r and 'python-dateutil' in r and 'google-cloud-storage' in r and 'xlrd' in r, 'requirements.txt declarations incomplete'; assert os.path.exists('scripts/autoresearch/requirements-autoresearch.txt') and 'gpt-researcher==' in open('scripts/autoresearch/requirements-autoresearch.txt').read(), 'autoresearch manifest missing'; y=open('.github/workflows/pip-audit.yml').read(); assert 'requirements.lock' in y, 'pip-audit not pointed at lock'"
$ echo $?
0
```

## New test suite -- exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q --no-header
............                                                             [100%]
12 passed in 0.04s
```

**Why 12 tests go beyond the immutable command** (research_brief_75.13.md "Vacuous-guard analysis" -- the command is a text-`assert` chain with real weaknesses):
- The command's `len(pins)>=150` counts ANY line containing `==`, including comments. My `test_requirements_lock_real_pin_count_and_exchange_prefix` explicitly excludes `#`-prefixed lines before counting.
- The command's `'<name>' in r` checks are whole-file substrings, satisfiable by a bare comment. My `_parse_requirements()` helper strips comments and requires a real `name==version` line; `test_requirements_txt_six_new_pins_are_real_lines_not_comments` asserts the exact version for all six new declarations, not just presence.
- The command never checks: pytest's declaration (not command-asserted at all -- covered by my parsed-pins test), the run_nightly.sh loud-fail seam (covered behaviorally by the two `test_run_memo_guard_*` tests), the lock's header/regen-command text (covered by `test_requirements_lock_opens_with_header_comment_containing_regen_command`), and the pip-audit.yml run-step/paths-filter shape via real YAML parsing rather than grep (covered by the two `test_pip_audit_yml_*` tests, which use `yaml.safe_load` and defend against the PyYAML YAML-1.1 `on:` -> `True` boolean-key resolver quirk).

## Full backend suite regression check

```
$ .venv/bin/python -m pytest backend/tests/ -q --no-header
10 failed, 1428 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 95.23s
```

The 10 failures are EXACTLY the standing pre-existing baseline named in the executor brief (verified by test-id match, not just count):

| # | Failing test | Baseline name |
|---|---|---|
| 1 | `test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh` | watchdog_no_fire |
| 2 | `test_phase_23_2_15_verify_23_1_smoke.py::test_phase_23_2_15_known_pass_scripts_still_pass` | verify_23_1_smoke |
| 3 | `test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence` | sector_cap_emit |
| 4 | `test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence` | ticker_meta_latency |
| 5-7 | `test_phase_57_1_reject_binding.py` (3 tests) | 57_1 x3 |
| 8 | `test_phase_60_1_deep_pipeline.py::test_60_1_claude_code_rail_declares_latency_profile` | 60_1 latency_profile |
| 9 | `test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off` | 60_3 flag_defaults_off |
| 10 | `test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap` | portfolio_swap zero_buy_gap |

No new failures introduced by this step. `1428 passed` includes all 12 new `test_phase_75_deps.py` tests plus every previously-passing test, unchanged.

## Ruff (F821/F401/F811) over the git-derived scope, re-derived AFTER the last edit

```
$ { git diff --name-only -- '*.py'; git status --porcelain | grep '^??' | grep '\.py$' | sed 's/^?? //'; }
scripts/autoresearch/run_memo.py
backend/tests/test_phase_75_deps.py

$ uvx ruff check --select F821,F401,F811 scripts/autoresearch/run_memo.py backend/tests/test_phase_75_deps.py
All checks passed!
```

## Mutation matrix -- 7/7 killed

Scripted in `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/0a35ec0b-2832-4744-a9ae-fab6b46f19bb/scratchpad/mutation_matrix_75_13.py`, following the exactly-once + byte-restore pattern from the phase-75.9 precedent script. Each mutation asserts its target substring matched exactly once before mutating, and asserts the file is byte-identical to the pre-mutation backup after restoring (verified programmatically inside a `finally` block, not just claimed).

| Mutation | Target | Result |
|---|---|---|
| M1: truncate `requirements.lock` to 100 lines | lock | KILLED -- `test_requirements_lock_real_pin_count_and_exchange_prefix` fails (100 total lines, far under 150 real pins after the 22-line header) |
| M2: unpin `numpy==2.4.4` -> `numpy>=2.0` | requirements.txt | KILLED -- `test_requirements_txt_six_new_pins_are_real_lines_not_comments` fails (operator `>=` != required `==`) |
| M3: restore the `fpdf2>=2.7.0` line (+ its old comment) | requirements.txt | KILLED -- `test_requirements_txt_fpdf2_fully_removed` fails |
| M4: remove the `--requirement backend/requirements.lock` run step from pip-audit.yml | pip-audit.yml | KILLED -- `test_pip_audit_yml_run_step_targets_the_lock` fails |
| M5: remove the `_gpt_researcher_guard()` call site from `run_memo.py::main()` | run_memo.py | KILLED -- `test_run_memo_guard_fails_loud_and_precedes_embedding_preflight` fails (control falls through to the patched `_embedding_preflight` stub, which raises `AssertionError`, since the guard no longer short-circuits) |
| M6 PROBE: move `PyYAML==6.0.3` into a comment | requirements.txt | Documented delta (below) |
| M7 STUB: break the find_spec monkeypatch target inside the test file itself | test_phase_75_deps.py | KILLED -- with the stub's name-match broken, the guard never fires (real `gpt_researcher` IS installed), control reaches the patched `_embedding_preflight` (which raises), so the test fails/errors -- proving the fixture is load-bearing, not vacuous |

### M6 documented command-vs-test delta (the measure-don't-assert proof point)

```json
{
  "mutation": "M6 PROBE: PyYAML==6.0.3 moved into a comment",
  "our_parsed_line_test_fails": true,
  "immutable_command_still_passes": true,
  "immutable_tail": "<clean>"
}
```

With `PyYAML==6.0.3` rewritten as `# PyYAML==6.0.3` (a comment), the immutable verification command's `'PyYAML' in r` whole-file-substring assert **still passes** (exit 0) -- it cannot tell a comment mention from a real declaration. My `test_requirements_txt_pyyaml_exact_case` and `test_requirements_txt_six_new_pins_are_real_lines_not_comments` both **fail** against the identical mutated file. This is the concrete, measured proof that the test suite out-bites the immutable command, per the research brief's explicit call-out that Q/A must verify by reading, not by trusting a green command alone.

Full mutation-matrix script output (all 7 restores verified byte-identical):

```json
[
 {"mutation": "M1 truncate lock to 100 lines", "killed": true, "tail": "1 failed, 11 passed in 0.06s"},
 {"mutation": "M2 unpin numpy== -> numpy>=", "killed": true, "tail": "1 failed, 11 passed in 0.05s"},
 {"mutation": "M3 restore fpdf2 line", "killed": true, "tail": "1 failed, 11 passed in 0.05s"},
 {"mutation": "M4 remove --requirement backend/requirements.lock from pip-audit.yml", "killed": true, "tail": "1 failed, 11 passed in 0.05s"},
 {"mutation": "M5 remove find_spec guard call from run_memo.py main()", "killed": true, "tail": "1 failed, 11 passed in 0.06s"},
 {"mutation": "M6 PROBE: PyYAML==6.0.3 moved into a comment", "killed": true, "tail": "2 failed, 10 passed in 0.06s", "documented_delta": {"our_parsed_line_test_fails": true, "immutable_command_still_passes": true, "immutable_tail": "<clean>", "note": "PROVES our test out-bites the immutable command"}},
 {"mutation": "M7 STUB: break the find_spec monkeypatch target in the test file", "killed": true, "tail": "1 failed, 11 passed in 0.06s", "note": "fixture is load-bearing rather than a vacuous always-pass"}
]

7 mutations run; survivors: NONE
```

Post-mutation-matrix sanity re-check (working tree fully restored):

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q --no-header
12 passed in 0.04s
$ python3 -c "<immutable command>"; echo $?
0
$ .venv/bin/pip freeze | shasum -a 256
8df19b228e083e35c4c0de62aa3209c38110f2f6227e0008b7e401ae4141e2af  -
```

## Documented (NOT executed) fresh-install dry-check

Per the contract's boundary (no installs against the live venv), the following is the exact command sequence an operator would run in a throwaway venv to prove reproducibility. This block is documentation only -- it was never run in this session:

```bash
# Option A: pip
python3 -m venv /tmp/pyfinagent-lock-dry-check
/tmp/pyfinagent-lock-dry-check/bin/pip install --upgrade pip
/tmp/pyfinagent-lock-dry-check/bin/pip install -r backend/requirements.lock
/tmp/pyfinagent-lock-dry-check/bin/pip freeze | wc -l   # expect 303 (matches the lock line count exactly)

# Option B: uv (faster; matches the header's documented sync path)
uv venv /tmp/pyfinagent-lock-dry-check-uv
uv pip sync --python /tmp/pyfinagent-lock-dry-check-uv/bin/python backend/requirements.lock

# Autoresearch closure (separate venv/closure per its own manifest):
pip install -r scripts/autoresearch/requirements-autoresearch.txt
```

## Deviations from the contract / disclosed judgment calls

1. **pip-audit.yml audits BOTH files, not lock-only.** The contract's plan step 3 says "point pip-audit.yml at the lock"; the research brief's own pitfall #1 argues for auditing the deployed graph. I kept the existing floors-file audit step AND added a new lock-audit step (two `pip-audit` invocations) rather than replacing the floors-file step, so CI keeps flagging advisories against the loose-range hypothetical resolution in addition to the exact deployed graph -- strictly additive, no coverage lost. If Q/A wants lock-only, that is a one-line removal of the second run step, not a design change.
2. **Test file is a new file** (`test_phase_75_deps.py`), per the contract's explicit permission to add "a small new file -- executor's call, disclosed" rather than extending `test_phase_75_sre_ops.py` (a different phase-75.11 concern).
3. **`_gpt_researcher_guard()` is a separate named function** (mirroring `_embedding_preflight()`'s existing style) rather than inline code in `main()`, so it is independently unit-testable and self-documenting about the compose-with-51.4 rationale.
4. **Two_embedding_preflight-facing sanity test** (`test_run_memo_guard_passes_through_when_gpt_researcher_present`) was added beyond the contract's explicit ask, to prove the guard does NOT false-positive when gpt_researcher IS importable (real `find_spec` result substituted with a truthy sentinel, and `_embedding_preflight` stub confirmed reached).
5. **google-cloud-storage** was placed in the GCP section (adjacent to `google-cloud-aiplatform`) rather than the "Data & tools" section, for narrative locality; this does not affect any assert (whole-file substring / parsed-line, both location-agnostic).

## Nothing incomplete

All contract plan steps 1-9 executed; all six success criteria addressed (four command-asserted + exit 0, two criteria-only -- pytest declaration and the fresh-install dry-check documentation -- both covered above for Q/A's by-eye check). `live_check_75.13.md` is a separate deliverable outside this executor's scope (Main writes it per the harness protocol); this draft supplies all the verbatim evidence Main needs to compose it.
