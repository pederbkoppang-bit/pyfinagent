# Q/A Critique — phase-5 crypto removal (meta-action)

**Agent:** qa_phase5_crypto_removal_v1
**Date:** 2026-04-19
**Verdict:** **PASS**

## 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Research brief present (closure-style; `gate_passed: true`; note cites predecessor `phase-5-restructure-research-brief.md` with 7 sources in full) | OK — accepted on closure semantics, same pattern as qa_78_v1. No new external research is warranted for a pure scope-subtraction per owner directive. |
| 2 | Contract mtime < experiment-results mtime (contract 23:52, results 23:54) | OK |
| 3 | Experiment-results contain verbatim verifier output + criterion table + caveats | OK (4 caveats, 6/6 criteria table) |
| 4 | Log-last discipline: `harness_log.md` not yet appended for this removal | OK — correctly deferred until after Q/A PASS, per feedback_log_last.md. |
| 5 | First Q/A on this meta-action; no verdict-shopping | OK |

## Deterministic A–G results

| Check | Result | Evidence |
|---|---|---|
| A. JSON valid | PASS | Python round-trip `json.load(open('.claude/masterplan.json'))` clean. |
| B. Step count 14 with expected ids | PASS | `['5.1','5.2','5.3','5.4','5.6','5.7','5.8','5.9','5.10','5.11','5.12','5.13','5.14','5.15']` — exact match. |
| C. 5.5 archived with dropped_reason mentioning crypto + owner directive + date | PASS | `phase-5.archived_dropped_steps[0].id == "5.5"`, reason = `"owner directive 2026-04-19 -- crypto is not a market we will pursue"`. Original `verification.command` (BTC/USD ETH/USD ingestion), 4 `success_criteria`, and `depends_on=[5.1,5.2,5.3,5.4]` preserved — un-drop is mechanical. |
| D. No active dep on 5.5 | PASS | 0 steps have "5.5" in `depends_on`. |
| E. Active crypto-code-path assertions | PASS | 5.2 no `get_ohlcv('BTC-USD','crypto'`; 5.3 no `crypto_candles` grep; 5.4 no `compute_position_size('BTC-USD','crypto'`; 5.13 no `asset_classes=['equity','crypto']`; 5.14 no `ENABLE_CRYPTO_TRADING`. |
| F. DAG acyclic; 5.15 sink; 5.1 zero-dep | PASS | Cycle detection clean. 5.1 `depends_on=[]`. 5.15 in sink set. (5.4 is NOT zero-dep — it depends on `[5.1]`; contract/results' "5.1 + 5.4 zero-dep" claim is imprecise but non-functional. Flag as nit, not blocker — see below.) |
| G. Git scope | PASS | Only `.claude/masterplan.json` + 3 crypto-removal handoff artifacts (research-brief, contract, experiment-results) are new/modified. The broader uncommitted working tree (`M` / `D` entries) is pre-existing session state, not introduced by this meta-action. |

## LLM judgment

### Is any crypto scope leaking through under a different name?

Searched active steps for aliases: `digital asset`, `BTC futures`, `bitcoin ETF`, `bitcoin`, `satoshi`, `stablecoin`, `USDT`, `USDC`. **0 hits.** No quiet re-add.

### Dependency re-chain correctness

- **5.6 `[5.4,5.5]` → `[5.4]`** — correct; options depends on the risk engine, not on crypto.
- **5.10 `[5.5]` → `[5.2]`** — correct. Expanded ETF Universe needs the data-provider abstraction to fetch thematic/levered/international tickers; crypto was only "another new market context", never a structural parent. `[5.2]` is the right new parent.
- **5.11 `[5.5,5.7]` → `[5.7]`** — correct. With `crypto_vol` replaced by VVIX (equity vol-of-vol, already in macro cache), FX (5.7) is the sole remaining non-equity input. Rates/DXY come from existing pipes.
- **5.13 `[5.4,5.5,5.11]` → `[5.4,5.11]`** — correct.
- **5.14 `[5.5,5.7,5.8,5.12,5.13]` → `[5.7,5.8,5.12,5.13]`** — correct.

### Archive integrity

Archived 5.5 preserves full original record: name, `verification.command`, `success_criteria` (4 items), `depends_on`. Un-drop is a straightforward splice back into `phase-5.steps`.

### Exclusion-criteria context check (the 8 residual "crypto" + 2 BITO + 2 IBIT mentions)

Grepped each active-step occurrence in context:

| Step | Mention | Context |
|---|---|---|
| 5.3 | `No crypto_candles table created (owner directive 2026-04-19)` | negative assertion |
| 5.4 | `No crypto asset-class branch (owner directive 2026-04-19)` | negative assertion |
| 5.10 | `no crypto` in name, `crypto-ETF tickers excluded` in criteria | negative |
| 5.10 | `['BITO','IBIT']` in `assert … not any(x in t for x in …)` | negative assertion — actively verifies their ABSENCE |
| 5.11 | `NO crypto_vol input` | negative |
| 5.12 | `crypto_equity_spillover signal REMOVED per owner directive 2026-04-19` | negative |
| 5.14 | `crypto flag REMOVED per owner directive 2026-04-19` | negative |
| 5.15 | `No crypto references in the e2e test (owner directive 2026-04-19)` | negative |

**All 12 residual mentions are exclusion / audit-trail strings. None is a functional assertion that a crypto code path executes.** Several are actively *defensive* (the 5.10 assertion verifies BITO/IBIT do NOT appear in the ticker set — this is a robustness benefit of not renumbering).

### Renumbering not done

Skipping from 5.4 → 5.6 is documented in `phase-5._comments[2]` and preserves traceability to prior Q/A critiques, harness_log cycles, and open_issues that reference by id. No JSON-schema or DAG dependency on sequential numbering. **Not a blocker.**

### Nits (non-blocking)

- `experiment_results.md` line 22 says "5.1 and 5.4 zero-dep"; masterplan has `5.4.depends_on=['5.1']`. This is a minor mis-phrasing in the results doc (should read "5.1 zero-dep; 5.4 depends only on 5.1"). Not a criterion failure; the DAG is correct.
- `path_decision.summary` mentions crypto removal; `crypto_removed_at` timestamp present. Good.

## Contract criterion verification

| # | Criterion | Verdict |
|---|---|---|
| 1 | `step_5_5_archived_not_active` | PASS |
| 2 | `deps_no_longer_reference_5_5` | PASS |
| 3 | `no_crypto_references_in_active_steps` (documented exclusion exception) | PASS |
| 4 | `step_count_14` | PASS |
| 5 | `path_decision_note_crypto_removed` | PASS |
| 6 | `json_valid` | PASS |

## Final JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "agent_id": "qa_phase5_crypto_removal_v1",
  "reason": "All 6 immutable criteria met. Step 5.5 archived with original verification preserved; 0 active deps reference 5.5; DAG acyclic with 5.1 zero-dep and 5.15 sink; 14 steps with expected ids; no alias leakage (digital asset / bitcoin / stablecoin); 12 residual crypto/BITO/IBIT mentions are all exclusion/audit-trail strings, verified by context; path_decision updated with crypto_removed_at timestamp; git scope clean (masterplan.json + 4 handoff artifacts including this critique).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item",
    "json_valid",
    "step_count_and_ids",
    "archive_integrity_original_verification_preserved",
    "active_deps_scan",
    "active_crypto_path_assertions_5_2_5_3_5_4_5_13_5_14",
    "alias_leak_scan_digital_asset_bitcoin_stablecoin_usdt_usdc",
    "dag_cycle_detection",
    "sink_source_analysis",
    "dependency_rechain_reasoning_5_6_5_10_5_11_5_13_5_14",
    "residual_mention_context_grep",
    "path_decision_and_comments_metadata",
    "git_scope"
  ],
  "nits_non_blocking": [
    "experiment_results.md line 22 phrasing 'and 5.4 zero-dep' is imprecise; 5.4 has depends_on=['5.1']. Non-functional."
  ]
}
```

## Next action (for Main)

1. Optionally tighten the results-doc zero-dep phrasing (nit only).
2. Append cycle block to `handoff/harness_log.md` with `result=PASS`.
3. Flip masterplan status (if applicable — this is a meta-action, not a tracked step; the archive_handoff hook will rotate these files on the next status transition).
