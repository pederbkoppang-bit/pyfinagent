# Contract — phase-54.1 (Cron audit + fix-or-escalate) — operator-away cycle

**Date:** 2026-06-01. **Tier:** moderate. **Step:** phase-54.1 (P0). Operator REMOTE
2026-06-01 → 2026-06-08, Slack-only.

## N* delta (N* = Profit − Risk − Burn)

**Risk↓** (operational): a silent cron failure during the unattended week is unobserved
risk. This step makes the cron surface auditable + removes the recurring failure that
would otherwise (a) re-fire nightly and (b) spam the operator's Slack P1 channel all
week (the watchdog routes cron failures to P1). No P delta. No money-path change
(DO-NO-HARM: the live paper_trading APScheduler job is healthy + untouched).

## Research-gate summary

`researcher` ran first (gate **PASSED**: 9 sources read in full, 25 URLs, recency scan,
12 internal files). Brief: `handoff/current/research_brief.md`; audit table:
`handoff/current/live_check_54.1.md`. Decisive findings:
1. **autoresearch + ablation (both launchctl last-exit=1) share ONE root cause.** Their
   launchd wrappers `set -a; . backend/.env; set +a` (bash-source the env). The
   2026-06-01 multi-market go-live set `PAPER_MARKETS=["US","EU","KR"]` (JSON) in
   `.env`. `paper_markets: list[str]` (`settings.py:55`) is a pydantic-settings *complex*
   field → the env source JSON-decodes it. Bash strips the quotes on `source` →
   `[US,EU,KR]` → `JSONDecodeError` → `SettingsError` at `get_settings()`. The LIVE
   backend is fine (uvicorn reads `.env` via native dotenv, not shell). Re-fails nightly
   02:00/03:00 unless fixed.
2. **mas-harness "not running" = FALSE POSITIVE** (idle `StartInterval 1800` job, exit
   0; PID `-` = loaded-and-idle per launchd semantics). No defect.
3. **Digest is TEMPLATE/DATA-ONLY ($0 LLM, NOT operator-gated)** — decisive for 54.2.
4. **Biggest away-week risk: the slack_bot (PID 42151, PPID 1) has NO launchd
   supervisor** — single point of failure for the Slack lifeline. → addressed in 54.2.

## Hypothesis

A pydantic-settings parser that accepts comma / bracket-mangled / JSON forms for
`paper_markets` will make `get_settings()` succeed on ALL load paths (native dotenv,
OS-env, bash-sourced), fixing autoresearch + ablation at the settings-load layer with
ZERO change to the live engine's resolved value (`["US"]` default; `["US","EU","KR"]`
when set) — DO-NO-HARM verified by test on every input form.

## Scope decision (autonomous vs escalate)

The goal's operator-gated list = {LLM API spend, pip installs, BQ DROP/unqualified
DELETE}. A `settings.py` code fix is NOT on that list → autonomously in-scope under
"full approval to proceed" + "close what is autonomously closable". The researcher
escalated conservatively (touches shared settings); I proceed because the fix is purely
ADDITIVE (widens what parses, never narrows), needs no `.env` edit (the existing `.env`
value parses via the new validator), and is gated behind a regression test proving the
live JSON path is byte-identical. **NOT auto-run:** the full autoresearch/ablation jobs
(autoresearch may incur LLM spend / has a known huggingface gap) — I verify the fix ONLY
at the `get_settings()` layer (the crash point), never by running the full job.

## Immutable success criteria (verbatim from masterplan phase-54.1)

1. EVERY launchd job in ~/Library/LaunchAgents/com.pyfinagent.* is enumerated with its
   loaded state + last exit status; EVERY in-process APScheduler job
   (slack_bot/scheduler.py morning_digest/evening_digest/watchdog + slack_bot/jobs/*) is
   enumerated with its trigger + next-fire time.
2. every UNHEALTHY job (not loaded, last-exit nonzero, missed/never-fired, or no sane
   next-fire) is explicitly listed with the root-cause + a FIX-applied OR an
   operator-escalation note (LLM-spend / pip / BQ-DROP / launchctl load-unload fixes are
   operator-gated and ESCALATED, not forced).
3. the autoresearch + ablation last-exit=1 failures and the mas-harness not-running
   state observed 2026-06-01 are each addressed (fixed or escalated with a crisp
   operator ask).
4. live_check_54.1.md contains the full cross-layer cron-health table (job | layer |
   schedule | last-run | status | action) — the artifact the operator can audit from
   Slack.

## Plan steps

1. **Reproduce** the exact bash-mangled value: `set -a; . backend/.env; set +a;
   python -c "import os; print(repr(os.environ.get('PAPER_MARKETS')))"` → confirm the
   precise string my parser must accept (expected `[US,EU,KR]`).
2. **Implement** the parser in `backend/config/settings.py`: accept (a) JSON list
   `["US","EU","KR"]`, (b) bracket-mangled `[US,EU,KR]`, (c) plain comma `US,EU,KR`,
   (d) pass-through real lists; empty/unset → default `["US"]`. Use the canonical
   pydantic-settings approach (`Annotated[list[str], NoDecode]` + `field_validator(mode=
   "before")` if pydantic-settings ≥2.2 supports NoDecode; else a source-level fallback).
   Keep `default_factory ["US"]` + the description.
3. **Test** (`backend/tests/test_phase_54_1_paper_markets_parse.py`): all 4 input forms
   parse to the right list; unset → `["US"]`; the live JSON form is byte-identical to
   today (DO-NO-HARM). Run the existing settings tests too.
4. **Verify the fix at the crash layer** (no full job run): `set -a; . backend/.env;
   set +a; python -c "from backend.config.settings import get_settings;
   print(get_settings().paper_markets)"` → succeeds, prints `['US','EU','KR']` (today it
   raises SettingsError).
5. **Finalize** `live_check_54.1.md`: flip autoresearch + ablation action from
   "ESCALATE" to "FIX APPLIED (settings.py NoDecode validator) + verified at settings-load
   layer; full-job re-verify deferred to next nightly fire / operator (autoresearch LLM +
   huggingface gap not auto-run)". Keep mas-harness false-positive note.
6. **Fresh qa** → log → flip.

## Guardrails

- DO-NO-HARM: prove the live engine's `paper_markets` value is unchanged (JSON path
  byte-identical). No `.env` edit (tool-blocked + unnecessary). No running the full
  autoresearch/ablation jobs (LLM/huggingface risk). No new dependency.
- ASCII loggers; no emoji. Single source of truth (the one `paper_markets` field).

## References

- `handoff/current/research_brief.md` (launchd docs, APScheduler, Slack API, Healthchecks.io,
  launchd.info, incident.io).
- `handoff/current/live_check_54.1.md` (the audit table).
- `backend/config/settings.py:55` (`paper_markets`); the autoresearch/ablation launchd
  wrappers + `~/Library/LaunchAgents/com.pyfinagent.{autoresearch,ablation}.plist`.
- pydantic-settings docs (NoDecode / parsing env complex types).
