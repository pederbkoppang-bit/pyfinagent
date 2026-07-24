# Contract — Step 78.1 (P1: rewire the six signal-overlay services onto the CC rail)

Date: 2026-07-25 | Cycle: 160 | Executor: Main (Opus 5) | Research gate: **PASSED**

## Research-gate summary

`handoff/current/research_brief_78.1.md` (59 KB) — tier `moderate`, envelope
`{"external_sources_read_in_full":6,"urls_collected":30,"recency_scan_performed":true,`
`"internal_files_inspected":17,"gate_passed":true}`. All six anchors re-confirmed.

The gate's headline: the rewire is *mechanically* simple (`make_client(model, None,
settings)` plus a guarded try/except) but carries **nine concrete API differences**
between `ClaudeClient` and `ClaudeCodeClient`. The ones that change behaviour:

1. **The rail's `--json-schema` is POST-HOC VALIDATED WITH INTERNAL RE-PROMPTING**, not
   constrained decoding like the API's `output_config.format`. The guarantees are **not
   equivalent** — though the existing error path fails safe. This is the central finding
   and the gate flagged it as a contradiction between sources it had to resolve.
2. **`temperature=0.0` is unreachable on the rail** (no CLI flag exists) → determinism
   is lost for all six.
3. **`max_output_tokens` is a documented CLI no-op.** Harmless for C1/C3/C4/C5/C6, but
   it may silently *lower* `news_screen`'s 48K cap to a model default — **the one place
   a signal can get worse.**
4. **No caller anywhere passes `config['system']`**, so the house financial-analyst
   prompt is already absent from every rail call (pre-existing, rail-wide → Q3/78.12).
5. **None of the six passes `_role`**, so rewired rows would land in the BARE `cc_rail`
   bucket. Passing `_role` fixes attribution *and* removes the ordering dependency on
   75.5.12.

## Measured flag state (Main, 2026-07-25) — closes a gap the gate disclosed

The researcher's sandbox denies `backend/.env`, so it honestly marked the live flag
state unverified. Read from the **runtime settings object** instead (booleans only):

```
paper_use_claude_code_route  = True     <- the rail IS on
meta_scorer_enabled          = True     C1 RUNS
news_screen_enabled          = True     C2 RUNS
macro_regime_filter_enabled  = True     C3 RUNS
pead_signal_enabled          = True     C4 RUNS
analyst_narrative_enabled    = False    C5 DOES NOT RUN
call_transcript_gpr_enabled  = False    C6 DOES NOT RUN
anthropic_api_key            = SET (non-empty)   -> the empty-key guard does not bail today
github_token                 = absent           -> the GitHub-Models branch is unreachable
```

**This inverts the gate's risk ranking.** The gate called C5/C6 "the only genuine
blocker" on per-ticker latency (32–64 s/call measured externally, 88.9 s observed
in-repo) — but they are **off**, so rewiring them is the *low*-risk half. C1–C4 are the
live surface, and per the 78.0 correction they are the ones plausibly **already failing
every cycle** on dead credits (`0 rows` proves only "no call succeeded").

## Hypothesis

Routing the six through `make_client` puts them under the existing
`PAPER_USE_CLAUDE_CODE_ROUTE` flag, so they survive dead Anthropic credits on the
flat-fee Max rail — removing the phase-72 rail-bypass class — **without changing any
signal's logic, thresholds, or output schema.**

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. All six services obtain their client through make_client (or ClaudeCodeClient) so
   PAPER_USE_CLAUDE_CODE_ROUTE governs them, proven by a test that flips the flag and
   asserts the client TYPE changes for each of the six
2. Model tier per service is unchanged from the census table (all haiku-tier); no signal
   logic, threshold, or output schema is modified — proven by git diff review recorded
   in experiment_results
3. Dummy-key $0-leakage proof per rewired path: with a junk ANTHROPIC_API_KEY the rail
   call still succeeds and no request reaches api.anthropic.com
4. MUTATION: force the flag OFF -> the six fall back to the metered client and a test
   asserting rail-routing goes red (the guard is not a tautology)
5. One-flag revertible: setting PAPER_USE_CLAUDE_CODE_ROUTE=false restores today's
   behavior exactly
6. news_screen's 48K max-tokens cap is explicitly handled: either proven still in effect
   on the rail, or the behavior change is measured and disclosed with the resulting
   output length — NOT left to the CLI's silent no-op
7. The D-KEY and D-SYS decisions are each stated explicitly in experiment_results.md
   with the reason

## Plan

1. **Establish dormant-vs-failing FIRST** (the 78.0 correction requires it, and it
   changes the risk calculus). C1–C4 are enabled; check their callers' logs/telemetry
   for evidence they ran and failed. Note `meta_scorer.py:238` swallows the exception
   and returns `_fallback_all`, so an outage here is silent by construction.
2. **Shared edit per service**: obtain the client via `make_client(model, None,
   settings)`; pass `_role` so rows carry `cc_rail:<service>` attribution rather than
   the bare shape.
3. **D-KEY (decide explicitly)**: make the empty-key guard **rail-aware** rather than
   deleting it — on the rail no API key is needed (`claude_code_invoke` deliberately
   *scrubs* `ANTHROPIC_API_KEY` from the subprocess env), so a missing key would
   otherwise still disable all six in exactly the scenario this step exists to fix. The
   guard must NOT be deleted outright: it is also what keeps the GitHub-Models branch
   unreachable (moot today — `github_token` is absent — but not structurally).
4. **D-SYS (decide explicitly)**: add `config['system']` for the six, or defer to the
   rail-wide parity step 78.12. State which and why.
5. **news_screen's 48K cap (criterion 6)**: measure the actual output length on the
   rail; if the cap is a no-op there, disclose the measured consequence rather than
   letting it pass silently.
6. **Order the work by risk, given the measured flags**: C5/C6 first (disabled → safe
   to prove the pattern), then C1–C4 (live).
7. **Tests**: a flag-flip client-TYPE test per service; fix the three `meta_scorer`
   tests that break because MagicMock settings make every flag truthy; C5/C6 have **no**
   tests today, so they need new ones.
8. **Mutation matrix**: force the flag OFF → rail-routing assertions red (criterion 4);
   neuter one flag-flip fixture → that test red (STUB); revert one service → only its
   test red (no co-firing).
9. Q/A on changed evidence → `harness_log.md` append → masterplan flip (log LAST).

## Boundaries

- The six service modules + their tests. **No signal logic, threshold, or output schema
  changes** — criterion 2 makes this checkable by diff.
- `llm_client.py` / `claude_code_client.py` NOT modified here (their defects are 78.10,
  78.12, 78.13).
- **No flag is flipped**; `PAPER_USE_CLAUDE_CODE_ROUTE` is already `True` and stays so.
- Latency is a live constraint for any per-ticker service: if a measured rewire would
  push a per-ticker path past its cycle budget, say so and stop rather than shipping it.

## Risks (ranked, from the gate + the measured flags)

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Schema guarantees differ (post-hoc validation vs constrained decoding) | Existing error paths fail safe; measure one real output per service and compare shape |
| 2 | `news_screen` 48K cap silently lowered | Criterion 6 forces it to be measured and disclosed |
| 3 | Determinism lost (`temperature=0.0` unreachable) | Disclose; these are scoring overlays, not reproducibility-critical — but say so explicitly rather than omitting it |
| 4 | Per-ticker latency (C5/C6) | They are DISABLED today; gate the decision on measured latency before any future enable |
| 5 | Three meta_scorer tests break on MagicMock truthiness | Known and enumerated by the gate; fix as part of the change |

## References

- `handoff/current/research_brief_78.1.md` (6 sources read in full, 30 URLs, 17 internal files)
- `handoff/current/census_78.json` — rows C1–C6, and the `measurement_note` prohibition
  on reading a zero row count as proof a path was dormant
- `.claude/masterplan.json` steps 78.10 / 78.12 / 78.13 (the out-of-scope defects this
  step must NOT absorb)
