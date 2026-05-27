# Contract -- Cycle 7 (closure correction): 38.12 ships, 27.6 stays pending, 38.13 added

**Cycle:** 7 (closure-correction commit)
**Date:** 2026-05-27
**Class:** verification cycle + small settings change. NO trading-policy change (citation floor does NOT apply).
**Status flips:** `38.12` flipped done; `27.6` STAYS pending; new step `38.13` added (P0).

**File-collision note (NINTH occurrence today):** `handoff/current/contract.md` clobbered by autonomous-loop sprint contract for the 9th time today. Permanent deconfliction is on the follow-up backlog and is itself becoming a P0.

## Research gate

Borrowed from cycles 3 (`aff3444de945e98c2` deep) + 4 (`ab1987d4ec80af4dd` simple), both `gate_passed=true`. Cycle 7 is a 1-field settings bump + verification; no new external research required.

## What cycle 7 shipped

### Code change (2 files modified)

1. `backend/config/settings.py:31` -- `paper_cycle_max_seconds` default raised 1800.0 -> 7200.0.

2. `backend/api/settings_api.py` -- exposed `paper_cycle_max_seconds` in 4 places: FullSettings, SettingsUpdate (ge=300, le=21600), _FIELD_TO_ENV, _settings_to_full.

### Operational steps (live)

3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` -- backend reloaded.
4. `PUT /api/settings/ {paper_cycle_max_seconds: 7200}` -- verified via re-GET = 7200.0.
5. `POST /api/paper-trading/run-now` triggered at 2026-05-27T06:48:53+02:00.
6. Cycle ran to completion at 08:31:33 with `cycle complete: NAV=$23767.00, P&L=18.83%, trades=0, cost=$1.3000`.

## Verification outcome

`38.12` ship: **PASS**. Timeout bump landed; cycle ran 102 min inside the 7200s budget.

`27.6` closure attempt: **FAIL**. The 13 BQ rows are LITE-FALLBACK signatures (standard_model=NULL, deep_think_model=NULL, $0.10 flat cost). Q/A `abbcca28fb3536a63` confirmed 11/13 cycle-7 tickers hit `Full orchestrator failed: credit balance is too low` -- the full orchestrator silently fell back to lite-mode which DOES honor the rail flag, then persisted lite-mode signatures.

## Root cause (Q/A diagnosis)

The cycle-3 rail-routing gate at `backend/agents/llm_client.py:1888-1905` is correct in isolation -- when `model_name.startswith("claude-")` AND `paper_use_claude_code_route=True`, `make_client()` returns `ClaudeCodeClient`. BUT:

- The full orchestrator pipeline (`backend/agents/orchestrator.py:516-518`) has LLM call sites that either bypass `make_client` entirely, pass model_name values that don't match the `claude-` prefix guard (e.g., `deep_think_model=gemini-2.5-pro` -> Gemini Vertex AI), or cache an LLMClient from before the operator flipped the flag.
- The lite-mode `_run_claude_analysis` at `autonomous_loop.py:1444+1481+1537` DOES honor the rail flag (cycle-5 dispatch). That's why lite-mode rows persisted -- orchestrator failed credit-exhausted, fell back to lite, lite used the rail, persisted lite signatures.

Fix scope: `38.13` -- wire the rail into the full-orchestrator's downstream consumers AND audit which call sites bypass make_client.

## Masterplan changes (this commit)

- `38.12` STATUS: pending -> done (with `closure_note` documenting the gap).
- `38.13` ADDED -- P0, harness_required. Verification: `>=5 BQ rows with standard_model LIKE 'claude%'`.
- `27.6` STATUS: unchanged at `pending` (Q/A RECOMMEND-KEEP-PENDING).

## Honesty trail

Operator memory `feedback_full_codebase_audit_before_changes.md` (2026-05-26 23:55) explicitly required full-codebase trace before claiming PASS. Cycle 7 violated this. Cycle 8 will do the trace BEFORE claiming PASS.

## Cycle 7 immutable success criteria (NOT 27.6's)

1. AST parse settings.py + settings_api.py: exit 0. PASS.
2. curl shows paper_cycle_max_seconds=7200.0. PASS.
3. grep paper_cycle_max_seconds in settings_api.py: 4 hits. PASS.
4. Cycle ran within new budget (102 min < 120 min). PASS.
5. Masterplan: 38.12 done, 38.13 added, 27.6 pending. PASS.
6. handoff/current/live_check_27.6.md rewritten with FAIL verdict + Q/A findings. PASS.
7. ZERO frontend changes / ZERO new deps / ZERO emojis. PASS.

## /goal integration gates

1. AST parse green. 2. Log LAST. 3. No self-evaluation (Q/A spawned + returned FAIL). 4. Citation floor N/A. 5. Honest closure: 38.12 done, 27.6 pending, 38.13 is the load-bearing follow-up.
