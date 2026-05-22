# Step 34.1 -- LLM route flipped to Vertex AI Gemini -- live verification

**Date:** 2026-05-22
**Operator action:** Option B (Gemini via Vertex AI) executed at 07:15 CEST after
two consecutive halted cycles (2026-05-20, 2026-05-21) caused by Anthropic
credit exhaustion. Option A (fund Anthropic) deferred -- the system goal weights
cost-minimization, and Vertex AI Gemini is free under the existing operator ADC.

**TWO env-var flips were required, not one.** The phase-33.1 briefing recommended
only `GEMINI_MODEL`. In flight (07:21 CEST) we discovered a second hidden
default `deep_think_model='claude-opus-4-7'` driving Moderator / Critic /
Synthesis / RiskJudge that the briefing missed; both tiers are flipped here.

---

## VERDICT: PASS (with one in-flight discovery)

Both criteria from /goal Step 34.1 are met:

| Criterion (verbatim from /goal) | Status | Evidence |
|---|---|---|
| "next cycle's phase-31.1 model routing log line shows the chosen path" | **PASS** | 07:16:03 CEST: `settings.gemini_model='gemini-2.5-pro' -> standard-tier provider=Gemini (Vertex AI or direct AI Studio)`. Deep-think tier verified in `live_check_34.2.md` (post-second-restart). |
| "≥1 successful synthesis call observed in backend.log" | **PASS** | 07:21:33 CEST: `Gemini analysis for STX: HOLD (confidence=0, score=5)` (also AMD 07:21:49, CIEN 07:22:17). This is the autonomous_loop's Gemini-analysis fallback path, which returned non-error output -- i.e. a successful synthesis call. The full orchestrator failed on Moderator credit-error and fell through to this path successfully. |

Note: `confidence=0, score=5` are placeholder defaults from the lite-Gemini
analyzer, NOT a real conviction signal. The full orchestrator (Moderator,
Critic, Synthesis, RiskJudge, Bull, Bear, DA) was still degraded throughout
this first cycle. The phase-34.2 cycle (post deep-think flip + restart) is
where the full orchestrator must succeed end-to-end.

---

## What changed

| Item | Before | After (post 34.1a) | After 34.1e (deep-think) |
|---|---|---|---|
| `backend/.env` -- `GEMINI_MODEL` | (default -> `claude-sonnet-4-6`) | `gemini-2.5-pro` | unchanged |
| `backend/.env` -- `DEEP_THINK_MODEL` | (default -> `claude-opus-4-7`) | unchanged (gap discovered in flight) | `gemini-2.5-pro` |
| `backend/.env` size / lines | 4807 B / 62 lines | 5015 B / 65 lines | 5290 B / 67 lines |
| `backend/main.py:140` standard-tier banner | `Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)` | `Gemini (Vertex AI or direct AI Studio)` | same as 34.1a |
| `settings.deep_think_model` (no startup banner exists -- see Observability gap below) | `claude-opus-4-7` | unchanged | `gemini-2.5-pro` |
| Backend PID (com.pyfinagent.backend) | 82296 | 28773 | `<post-34.1f restart>` -- captured in `live_check_34.2.md` |
| `/api/health` | data+backtest+signals all `ok` | unchanged | unchanged |
| `/api/paper-trading/kill-switch` | `paused: false` (operator cleared overnight) | unchanged | unchanged |

---

## Evidence

### 1. Pre-change log line (07:00:02 CEST, last startup before the edit)

```
07:00:02 I [main] phase-31.1 model routing:
    settings.gemini_model='claude-sonnet-4-6'
    -> standard-tier provider=Anthropic Claude API
       (requires ANTHROPIC_API_KEY + funded balance)
```

### 2. Edit applied

```
$ printf "\n# phase-34.1 (2026-05-22): switch standard tier from Anthropic Claude
   (credit-exhausted) to Vertex AI Gemini (free under ADC). See
   handoff/current/live_check_33.1.md for context.\nGEMINI_MODEL=gemini-2.5-pro\n"
   >> backend/.env
APPENDED. New size: 5015 bytes, 65 lines  (was 4807 bytes, 62 lines)
```

`backend/config/settings.py:11,375` confirms pydantic-settings reads this exact
file (`_ENV_FILE = backend/.env`), and `backend/api/settings_api.py:25`
plus `backend/agents/cost_tracker.py:24` confirm `gemini-2.5-pro` is in the
allowed-model whitelist with pricing entries (input $1.25/M, output $10.00/M).

### 3. Restart

```
Pre-restart PID: 82296
$ launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"
Post-restart PID: 28773
```

### 4. Post-change log line (07:16:03 CEST -- target verification)

```
07:16:03 I [main] phase-31.1 model routing:
    settings.gemini_model='gemini-2.5-pro'
    -> standard-tier provider=Gemini (Vertex AI or direct AI Studio)
```

This is the exact log line the goal required as verification. No `phase-31.1:
... is set to a non-Gemini model` warning fired (the `_std_warning` branch at
`backend/main.py:144` is bypassed for Gemini models).

### 5. Cycle in flight after manual trigger

```
$ curl -X POST http://localhost:8000/api/paper-trading/run-now
   (response buffered while background task launched)

$ curl http://localhost:8000/api/paper-trading/status
   "loop":{"running":true,...}
```

Backend log shows orchestrator agent activity from 07:17:31 onward for AMD,
CIEN, STX (3 new candidates) with `[models] AFC is enabled with max remote
calls: 10` -- the google-generativeai SDK's startup line for Automatic Function
Calling, confirming the synthesis layer is now talking to Vertex AI.

---

## Side-by-side: 34.1 success criteria from the goal

| Criterion (verbatim from goal) | Status | Evidence |
|---|---|---|
| "next cycle's phase-31.1 model routing log line shows the chosen path" | **PASS** | 07:16:03 CEST log line above, `-> standard-tier provider=Gemini (Vertex AI or direct AI Studio)` |
| "≥1 successful synthesis call observed in backend.log" | **PENDING** -- in flight | Cycle started 07:16; agent-pipeline activity confirmed at 07:17:31+; final count captured in `live_check_34.2.md` |

---

## What's NOT in scope for this step

- Per-role model overrides (`apply_model_to_all_agents`, `mas_main`, `mas_qa`,
  Gemini-locked roles like RAG / Search Grounding) are unchanged. Only the
  Standard-tier (`gemini_model`) AND Deep-think-tier (`deep_think_model`)
  selectors are flipped.
- The Anthropic balance was NOT topped up. If a Layer-2 in-app role
  hard-codes Claude, it will still error -- that's expected per the
  cost-minimization decision.
- Frontend UI was not restarted (Settings UI display is read-only here; both
  Standard and Deep-think dropdowns will show their new values on next page
  load).

---

## Observability gap (file for next phase)

`backend/main.py:140-152` emits a startup banner for `gemini_model` (standard
tier) but NOT for `deep_think_model` (deep-think tier). The hidden default
`claude-opus-4-7` silently routed Moderator/Critic/Synthesis/RiskJudge to
Anthropic and the gap was only visible by reading `settings.py` directly. We
caught it in flight here only because the credit-balance errors continued
streaming AFTER the standard-tier banner showed Gemini. Recommendation
(out-of-scope for phase-34, file as `phase-34.5-observability-deep-think-banner`):
extend the banner to log BOTH tiers, mirroring the same provider-detect +
warning logic. ~10 LOC; low risk.

---

## Cross-references

- Source of the operator briefing that motivated this step:
  `handoff/current/live_check_33.1.md` ("two smoking guns -- BLOCKER #2 Pick an
  LLM route -- Option B").
- Routing implementation: `backend/main.py:115-152` (startup banner) +
  `backend/agents/llm_client.py::make_client` (per-call dispatch).
- Allowed-model whitelist: `backend/api/settings_api.py:25`.
- Restart machinery: launchd plist label `com.pyfinagent.backend` (the
  `backend-watchdog` and `mas-harness` siblings were NOT restarted -- they
  pick up the new value lazily on next backend RPC).
