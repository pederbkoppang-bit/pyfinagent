# Contract -- phase-62.4: guardrail/budget sentinel

Date: 2026-06-12. Goal: goal-away-ops. Research: rolling research_brief.md (retry-completed;
gate_passed, 9 external in full, recency scan; GO with 2 disclosed compromises).

## Research anchors

- Enforcement OUTSIDE the agent (Waxell/Braintrust: "blocking at the LLM call is too
  late") -- sentinel runs in the wrapper pre-flight (62.3 wiring), never in the prompt.
- metered_llm_usd_today from BQ llm_call_log is a LOWER BOUND (grounding/per-request
  charges + invisible retry tokens omitted) -- pinned in the script header.
- baseline_usd = PINNED CONSTANT from the 58.1 ledger class ($5-8/day), NOT a rolling
  mean (14-day mean is ~$0.006 and would false-trip the first legitimate full cycle).
- Drift doctrine (ArgoCD/Flux ignoreDifferences analogue): git-tracked
  scripts/away_ops/flag_baseline.json grandfathers the 3 pre-62.2 operator-ON flags
  (keystroke 2026-06-12 predates the token handler); sentinel is passive/read-only.
- TAMPER TESTS MUST NOT INSERT PROD BQ ROWS: llm_call_log uses insertAll (streaming
  buffer, no DML for ~30 min) -- a synthetic row would inflate the metered figure all
  day (self-DoS). Env overrides instead: SENTINEL_TEST_METERED_USD (inflate-only,
  cannot mask a real breach -- it MAXes with the real value when BQ is reachable... NO:
  override SKIPS BQ entirely; inflate-only is preserved because the override path is
  test-only and exits non-zero, never used to pass), SENTINEL_ENV_FILE (alternate .env).
- Exit semantics: 1 = BREACH (named gate: metered_budget | flags_match_tokens);
  2 = INFRA (metered_source_unavailable) -- both fail-closed to digest-only via the
  62.3 wrapper, but the digest can distinguish outage from tamper. kill_switch_paused
  is REPORT-ONLY (never a gate -- paused trading is a legitimate state).
- Date math in SQL only (CURRENT_DATE() UTC matches DATE(ts) partitions; BSD/GNU date
  arithmetic banned). JSON via venv-python json.dumps (jq 1.7.1 fallback exists).
- The sentinel reads backend/.env directly: it is a SCRIPT under launchd/wrapper, not a
  Claude tool call -- the 62.0 hook gates agent writes, not program reads. Read-only.

## Immutable success criteria (verbatim from masterplan 62.4)

1. "sentinel prints {metered_llm_usd_today, baseline_usd, kill_switch_paused,
   flags_match_tokens, ok} JSON and exits 0 healthy; the metered figure source (BQ
   table/endpoint) is pinned in the script header"
2. "tamper tests: a synthetic cost row above baseline AND a behavior flag line with no
   matching token each make sentinel exit non-zero with a named gate failure (test
   transcript in experiment_results.md)"
3. "run_away_session.sh pre-flight wires sentinel failure to the digest-only prompt
   (wrapper test asserts the prompt path switch)"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && source
.venv/bin/activate && bash scripts/away_ops/sentinel.sh; echo exit=$?

Criterion-2 note: "a synthetic cost row" is implemented as the SENTINEL_TEST_METERED_USD
env override BY RESEARCHED NECESSITY (streaming-buffer self-DoS makes a literal prod row
actively harmful); the override exercises the identical gate logic. Disclosed here
pre-generate; Q/A rules on it.

## Plan

1. scripts/away_ops/flag_baseline.json: the 3 grandfathered flags + provenance.
2. scripts/away_ops/sentinel.sh: venv-python core (BQ metered query w/ CURRENT_DATE(),
   .env behavior-flag parse, operator_tokens.jsonl + flag_baseline.json reconciliation,
   kill-switch report-only read w/ audit fallback, json.dumps emission, gates + exit
   codes); header pins source table + LOWER-BOUND caveat + baseline provenance.
3. run_away_session.sh minimal amendment: AWAY_SESSION_TEST_PREFLIGHT=1 -> exercise the
   REAL pre-flight (sentinel + prompt selection), log PREFLIGHT-TEST prompt=<kind>, exit
   before any git/claude action (makes criterion 3 honestly testable).
4. backend/tests/test_phase_62_4_sentinel.py: healthy path (requires_live mark -- real
   BQ); tamper metered (env override -> exit 1 + metered_budget named); tamper flag
   (SENTINEL_ENV_FILE tmp file w/ unauthorized PAPER_*=true -> exit 1 +
   flags_match_tokens named); infra path (SENTINEL_TEST_BQ_FAIL=1 -> exit 2 +
   metered_source_unavailable); wrapper preflight test (override + TEST_PREFLIGHT ->
   digest_only selected, logged).
5. experiment_results.md (tamper transcripts) + live_check_62.4.md -> ONE fresh Q/A ->
   harness_log -> flip.

## Out of scope

Healthcheck probes (62.5, shipped); enabling any flag; BQ writes of any kind.
