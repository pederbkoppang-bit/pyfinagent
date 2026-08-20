# Sprint Contract -- phase-86.120
Step: CC rail circuit breaker is not weekly-limit-aware

## Research Gate
Workflow `research-gate.js`, run `wf_498c5a7f-2be`, tier=moderate, `gate_passed=true`
(script-recomputed, not self-reported: 6/6 claimed sources found verbatim in the brief,
37 URLs collected >= 10 floor, recency scan present, `brief_status=COMPLETE`).
Brief: `handoff/current/research_brief_86.120.md` (587 lines).

- 6 external sources read in full (floor 5): Claude Code error reference, Anthropic API
  rate-limits doc, AWS exponential-backoff-and-jitter, Azure circuit-breaker pattern, a
  2025-12 preprint systematic review of microservice recovery patterns, and the Claude Code
  headless/programmatic-invocation doc.
- Recency scan: weekly Claude Max limits are a **2025-08-28** addition explicitly targeting
  unattended 24/7 automation -- i.e. this project's exact shape. `run_away_session.sh`'s
  existing limit detector predates that date, which is *why* it has no weekly pattern (a
  dating artefact, not an oversight).
- Key findings that change the design from what the masterplan step's audit_basis assumed:
  1. **The CLI emits THREE distinct, textually-documented limit messages**, not one:
     `"You've hit your session limit - resets 3:45pm"`,
     `"You've hit your weekly limit - resets Mon 12:00am"`,
     `"You've hit your Opus limit - resets 3:45pm"`. Session/weekly are shared across
     models; Opus-only gates just the Opus model. All three need detection, not just the
     "session limit" text the original audit_basis quoted.
  2. **There is no documented JSON field for this in `--output-format json`** (what
     `claude_code_invoke` uses) -- the CLI error-reference doc explicitly states no
     `stop_reason`/`subtype` covers it. Text-pattern matching is not a shortcut around a
     better signal; per the brief, it is "the ONLY documented surface" for this mode.
  3. **But there IS a real, currently-discarded machine-readable signal**:
     `claude_code_client.py:439` raises on `returncode != 0` *before* `:465`'s `json.loads`,
     so on every failure -- including a limit hit -- the JSON envelope (which DOES carry
     `api_error_status: 429` and the full `result` sentence) is never parsed. This is
     measured against the one real captured limit envelope in this repo
     (`handoff/away_ops/session_pm_20260707T200007Z.json`, a 429/session-limit hit from
     2026-07-07), not assumed.
  4. **The existing truncation is a landmine, not a working detector.** `str(exc)` embeds
     only `stdout[:150]`; on the real captured envelope this severs the human sentence at
     `"You've hit y"` -- `api_error_status` survives only by *positional luck* (JSON key
     order), not by contract. A regex over `str(exc)` inherits that fragility.
  5. **No date/year on the reset-time string**: session/Opus give `HH:MMam/pm` with no
     date; weekly gives `Day HH:MMam/pm` with no year. Turning that into an absolute
     cooldown-until instant needs "next occurrence of" arithmetic, which this repo has
     gotten wrong before (`reference_stat_SB_prints_local_time`,
     `reference_fixed_offset_tz_fixture_is_hour_dependent`). The design therefore treats
     exact reset-time parsing as a best-effort refinement with a safe, conservative,
     round-UP default when parsing is ambiguous or fails -- never a silent under-shoot.
  6. **Governing pattern is Azure "accelerated circuit breaking"** (trip at N=1 on a
     *classified* quota signal), not AWS backoff-with-jitter -- jitter cannot manufacture
     quota that will not exist again until the reset. The two patterns are complementary,
     not competing: the NEW quota-specific trip-at-1 cooldown sits alongside the EXISTING
     20-consecutive-failure breaker, which stays exactly as-is for ordinary transient
     failures (network blips, auth hiccups) where retrying-with-backoff is the right
     response.
  7. **`CLAUDE_CODE_RETRY_WATCHDOG=1`** is a real, current env var that enables
     *indefinite* retry for 429/529 in unattended sessions -- i.e. it would make this exact
     problem worse if ever set. Not part of this fix; noted so nobody "fixes" retries with
     it later.
  8. **`--output-format stream-json` exposes a real enumerated classifier**
     (`system/api_retry` events with an `error` category including `rate_limit`, plus
     `retry_delay_ms`) that `--output-format json` does not. The brief flags this as "a
     real option, and a real cost" requiring the rail to reassemble the streamed final
     result -- a materially bigger change than this step's boundary. **Explicitly
     considered and deferred**, not silently skipped (see Out-of-scope below).
  9. **There is zero local evidence of an actual weekly-limit hit** (1 of 85 away-ops
     session files hit any limit, and it was the session/5-hour form). Test fixtures for
     the weekly and Opus message shapes are therefore built from S1's documented strings,
     labelled as synthesized-from-docs, not claimed as reproduced-from-real-data. The
     session-limit and generic-429 cases CAN be tested against the real captured envelope.

## Hypothesis
The CC rail's circuit breaker (`claude_code_client.py`, phase-66.1) is a generic,
per-cycle, in-memory failure counter with no concept of "this specific failure means the
account is out of quota until a known future time." Because of that, once the operator's
weekly (or session, or Opus) Claude Max limit is exhausted, every subsequent cycle for the
rest of the week independently re-discovers the same fact the hard way -- burning up to
~20 doomed `claude` subprocess spawns per cycle, per rail-touching call site -- while
producing zero real analysis. Classifying the CLI's own documented limit-exhaustion
messages (parsed from the JSON envelope's `result` field rather than a truncated `str(exc)`
slice) and persisting a cooldown derived from that classification to disk closes the gap
without weakening the existing breaker's handling of ordinary transient failures, and
without introducing any path to metered `api.anthropic.com` spend.

## Success Criteria (immutable)
```
.venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
```
Plus sub-criteria (copied verbatim from `.claude/masterplan.json` step 86.120):
- claude_code_invoke (or its direct caller in claude_code_client.py) recognizes the CLI's
  specific weekly/session-limit-exhaustion signature in the failure output (stdout, per
  the existing :441-443 comment) as DISTINCT from a generic non-zero-exit failure --
  reusing/aligning with the same message class run_away_session.sh:242 already matches,
  not a new incompatible pattern
- on detecting that signature, a COOLDOWN state is persisted to disk (survives both the
  per-cycle rail_guard_reset() call and a backend process restart) recording when the
  limit was hit and when it is safe to retry -- the in-memory per-cycle _RAIL_GUARD alone
  is not sufficient, because it is wiped every cycle
- while the persisted cooldown is active, the rail is skipped BEFORE any `claude`
  subprocess is spawned on every subsequent cycle -- proven by a test asserting
  subprocess.run/Popen is never invoked (mock-level assertion, not just an empty
  LLMResponse) while the cooldown is active, across at least two simulated cycles
  including one after a simulated process restart
- the cooldown is bounded and self-clearing, not a permanent lockout: it has an
  operator-configurable backoff/retry window (a new Settings field, documented, with a
  sane default e.g. hours not minutes) after which the rail is tried again on its own, and
  a single subsequent success clears the cooldown
- MUTATION: remove the limit-signature detection -> the persisted cooldown never engages
  and behavior reverts to today's generic 20-consecutive-failure per-cycle breaker
  (proving the new detection, not the old breaker, is what is being tested)
- MUTATION: remove the pre-subprocess cooldown-skip guard -> the subprocess-not-invoked
  assertion goes red (proving that guard, independently of the detection above, is
  load-bearing)
- the existing phase-66.1 per-cycle consecutive-failure breaker (default threshold 20) is
  preserved unchanged for genuine transient failures (auth hiccups, network blips) that
  are NOT the limit signature -- this step ADDS limit-awareness, it must not weaken or
  replace the existing breaker's behavior for other failure classes
- at least one of the always-on signal agents that route through
  make_client()/claude_code_invoke (macro_regime, pead_signal, news_screen, meta_scorer,
  analyst_narrative_scorer, or call_transcript_gpr) is driven through the cooldown guard
  directly and shown to respect it, not just the main orchestrator path
- REGRESSION: make_client()'s existing routing-breach ValueError guard
  (backend/agents/llm_client.py, the $0-metered safety net this step's own audit_basis
  measured) is re-verified intact by a test that still fails loud on a routing breach --
  this step touches adjacent failure-handling code and must not accidentally introduce or
  weaken a fallback to metered api.anthropic.com
- the cooldown state (active/not, hit-at, retry-at) is readable by an operator without
  grepping logs -- surfaced via rail_guard_status() or an equivalent existing
  observability seam, not a new one-off script
- mutation-test every new guard per this project's standing discipline: control observed
  GREEN first, each mutant KILLED, byte-identical restore after

**Reading note on criterion 1** (disclosed per this project's "judge, don't just accept"
convention): "reusing/aligning with the same message class run_away_session.sh already
matches, not a new incompatible pattern" is read as *the same class of signal*, not
*identical coverage*. **Correction (Q/A cycle-1 critique):** the original wording here
called `run_away_session.sh:242`'s regex "a proper subset" of the three CLI-documented
messages -- that is imprecise and is corrected to *overlapping, not a subset*:
`usage limit|session limit|credit.*(exhaust|limit)|out of credit` matches `session limit`
(shared with this step's classifier) but ALSO matches `usage limit` and `out of credit`
alternatives this step's classifier does not, while lacking `weekly` and `Opus limit`
entirely (research finding G5) that this step's classifier DOES match. Reusing only the
away-ops regex verbatim would leave the weekly case -- the exact case the operator is
worried about this week -- undetected. The new detector is the same *class* of signal
(the CLI's own human-readable limit sentences) extended to cover all three documented
forms, which is compatible with, not incompatible with, the existing away-ops pattern,
even though the two patterns' matched sets overlap rather than nest.

## Plan (PRE-commit; will NOT diverge in Generate)
1. Add a pure classifier function in `claude_code_client.py` that takes the CLI's raw
   stdout, attempts `json.loads` (best-effort, never raises), and if it parses, inspects
   `api_error_status` + `result` (the full untruncated string) for the three documented
   limit sentences (session / weekly / Opus), returning a structured classification
   (`kind`, `raw_message`, best-effort `retry_at` datetime or `None`). Falls back to a
   direct substring scan of raw stdout if JSON parsing fails (covers the "plain-text, not
   JSON" case the existing :441-443 comment already anticipated).
2. Wire this classifier into `claude_code_invoke`'s failure path, BEFORE it raises
   `ClaudeCodeError`, so the classification happens on every failure without changing the
   function's existing exception-based contract for callers.
3. Add a small persisted-cooldown module (JSON state file under a path consistent with
   this project's existing state-file conventions) with: `hit_at`, `kind`, `retry_at`
   (computed: parsed reset time if confidently parseable, else `hit_at + a new
   Settings-configurable default backoff`, rounded UP never down), `raw_message`. Expose
   `cooldown_active() -> bool` and `cooldown_status() -> dict` and `cooldown_record_hit(...)`
   and `cooldown_clear_on_success()`.
4. In `claude_code_invoke`, check `cooldown_active()` FIRST, before any subprocess spawn --
   mirroring the existing `_rail_guard_blocked()` pre-check shape so all current callers
   (orchestrator, lite fallback, always-on signal agents) get the protection for free
   without their own call sites changing.
5. On a classified limit failure, call `cooldown_record_hit(...)` and ALSO trip
   `_RAIL_GUARD.open = True` immediately (N=1, not N=20) for the rest of the current
   cycle, consistent with Azure accelerated circuit breaking -- the existing generic
   20-failure path is untouched for non-limit failures.
6. On a real success after a cooldown was active, call `cooldown_clear_on_success()`.
7. Surface cooldown state through `rail_guard_status()`'s existing dict (add fields, don't
   replace it) so it stays the one observability seam.
8. Add the new Settings field (backoff default) to `backend/config/settings.py`, documented
   inline, with a conservative default measured in hours.
9. Write `backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py`: classifier unit
   tests (real captured session-limit envelope + synthesized weekly/Opus envelopes,
   explicitly labelled), persistence-across-reset/restart tests, pre-subprocess skip test
   (mock-level, asserts subprocess never invoked), self-clearing test, at least one
   always-on signal agent driven through the guard, the make_client() routing-breach
   regression test, and the two named mutation cells from the immutable criteria (plus
   this project's standing control-first / kill / byte-identical-restore discipline).
10. Run the immutable verification command; write `experiment_results_86.120.md` +
    `live_check_86.120.md` with verbatim command output.

## Scope honesty / out-of-scope
- **`scripts/away_ops/run_away_session.sh`'s own `LIMIT_HIT` regex (research finding G5,
  missing `weekly`/`Opus`) is a real, separate defect on a DIFFERENT rail** (the away-ops
  Claude Code session harness, not the trading analyst). It is adjacent but out of this
  step's stated boundary (`backend/agents/claude_code_client.py` +
  `backend/services/autonomous_loop.py` + the always-on signal agents + tests). Filing it
  as its own follow-up defect step rather than silently folding it in or silently leaving
  it undocumented.
- **Switching the rail to `--output-format stream-json`** to get the documented
  `system/api_retry` enumerated classifier is explicitly NOT done here -- the brief's own
  words are "a real option, and a real cost", requiring reassembly of the streamed final
  `result` message across `claude_code_invoke`'s entire parse path. Deferred as a
  candidate future step, not attempted as a stretch goal inside this one.
- **Exact reset-time arithmetic is best-effort, not guaranteed-precise.** Given this
  repo's history with local-time/date-math bugs (cited above), the fallback path (a fixed,
  configurable backoff from `hit_at`) is the safety net whenever the parsed reset string is
  ambiguous -- precision is a nice-to-have on top of a bound that is always safe by
  construction (rounds up, never down).
- Not touching `make_client()`'s routing logic itself (per the masterplan step's own
  boundary) beyond adding the regression test named in the criteria.

## References
- Research brief: `handoff/current/research_brief_86.120.md`
- Real captured limit envelope: `handoff/away_ops/session_pm_20260707T200007Z.json`
- Precedent 401-latch idiom: `scripts/away_ops/auth_state.py`, `scripts/away_ops/healthcheck.sh`
- Prior phase this step extends: phase-66.1 (`claude_code_client.py`'s rail guard + breaker)
- Related, deliberately out-of-scope: G5 (`run_away_session.sh`'s stale limit regex)
