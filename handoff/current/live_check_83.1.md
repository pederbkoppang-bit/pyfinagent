# live_check evidence — step 83.1 (design pack + pre-registration)

Captured 2026-08-07.

## Envelope from `handoff/current/research_brief_phase83.md` (transcribed verbatim)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 87,
  "snippet_only_sources": 16,
  "urls_collected": 85,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": true,
    "dry": null,
    "dry_reason": "the 2026-08-04 run persisted no coverage object; asserting dry=true would fabricate a record that does not exist"
  },
  "gate_passed": true
}
```

(`coverage.dry` is null by design — the 2026-08-04 run never persisted a coverage object; the derivation rules for every number are stated in the pack's envelope preamble.)

## SHA-256 — printed alongside the recorded value

```
computed SHA-256: a22cb12ff15d33b874cef8e48150d5f987acbd215892791609e846ad9ad9a5ce
recorded in pack : a22cb12ff15d33b874cef8e48150d5f987acbd215892791609e846ad9ad9a5ce
```

## Ordering — `ls` full timestamps of the pre-registration file and every phase-83 backtest artifact

```
$ ls -laT backend/backtest/experiments/preregistration_phase83_ranking.json
-rw-r--r--  1 ford  staff  3768  7 aug. 12:13:34 2026 backend/backtest/experiments/preregistration_phase83_ranking.json

$ ls -laT backend/backtest/experiments/results/*_phase_83_*.json backend/backtest/experiments/results/phase83*.json
(eval):1: no matches found: backend/backtest/experiments/results/*_phase_83_*.json
```

The phase-83 backtest-artifact population under the PRE-REGISTERED globs is **empty** (no artifact exists, so trivially none predates the ranking file). The population rule itself is pre-registered inside the ranking file because the naive `*83*` glob matches 71 unrelated pre-existing files (measured) whose old mtimes would make the criterion permanently red. Non-vacuity of the guard is proven by the criterion-6 mutation test, which writes a glob-matching backdated artifact, asserts the population SEES it, and asserts the ordering helper then fails (`test_phase_83_1_design_pack.py::test_c6_backdated_phase83_artifact_fails_criterion_5` — green in the suite run).


## Cycle-2 re-capture (2026-08-07, after the append-only amendment + Q/A fixes)

The ranking file gained the content rule + binding naming requirement (amendment
2026-08-07), so its hash CHANGED and was re-recorded in the pack per the
amendment policy. Envelope corrected (urls_collected 85 -> 79 under the stated
rule). Superseding captures, verbatim:

```
computed: 7b34649297c7f21c4f4a67621743cb01b5724f8316d0956d7a94ed7f61b4e5f0
recorded: 7b34649297c7f21c4f4a67621743cb01b5724f8316d0956d7a94ed7f61b4e5f0
```

Corrected envelope (verbatim from the pack):

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 87,
  "snippet_only_sources": 16,
  "urls_collected": 79,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": true,
    "dry": null,
    "dry_reason": "the 2026-08-04 run persisted no coverage object; asserting dry=true would fabricate a record that does not exist"
  },
  "gate_passed": true
}
```

Suite after cycle 2: 8 passed (the new c6b test proves the CONTENT rule catches
a backdated artifact carrying the repo's canonical result_store name -- the
escape shape the Q/A executed -- with an untagged negative control). The
cycle-1 captures above are preserved unedited and superseded by this section.
