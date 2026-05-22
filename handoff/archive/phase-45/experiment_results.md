# Experiment Results -- phase-34 LLM-route flip + first clean cycle

**Step ids:** `phase-34.1` + `phase-34.2`
**Date:** 2026-05-22
**Author:** Main (Claude Opus 4.7, this Claude Code session)

---

## What was changed (files edited)

| File | Change | Author | Reversibility |
|---|---|---|---|
| `backend/.env` | Appended `GEMINI_MODEL=gemini-2.5-pro` (+ comment) | Main, 07:15:18 CEST | Trivially reversible (delete the appended lines OR set to a different value) |
| `backend/.env` | Appended `DEEP_THINK_MODEL=gemini-2.5-pro` (+ comment) | Main, 07:24:30 CEST | Trivially reversible |
| `handoff/current/contract.md` | Rewritten for phase-34 (was for phase-33.1) | Main | The 33.1 contract is archived under `handoff/archive/phase-33.1/` |
| `handoff/current/live_check_34.1.md` | Created -- evidence for Step 34.1 verdict | Main | n/a |
| `handoff/current/live_check_34.2.md` | (Created after the second cycle completes) | Main | n/a |
| `handoff/current/evaluator_critique.md` | (Created by Q/A subagent at end) | Q/A | n/a |
| `handoff/harness_log.md` | Appended one cycle block | Main | n/a (append-only by protocol) |
| `.claude/masterplan.json` | Appended `phase-34` block + flipped status to `done` | Main | n/a (auto-commit hook fires the live-check gate) |

**No backend source code was edited.** All changes are config + handoff artifacts.

---

## What was NOT changed

- No edits to `backend/main.py`, `backend/agents/*.py`, `backend/services/*.py`,
  `backend/config/settings.py`. Only env-var values changed.
- No mutating BQ / Alpaca / LLM-billing actions beyond the natural cycle.
- No code change to add the missing deep-think-tier startup banner (filed as
  out-of-scope `phase-34.5-observability-deep-think-banner` recommendation).
- Anthropic balance NOT topped up.

---

## Verbatim verification commands + outputs

### Pre-change routing log line

```
$ grep "phase-31.1 model routing" backend.log | tail -1
07:00:02 I [main] phase-31.1 model routing:
    settings.gemini_model='claude-sonnet-4-6'
    -> standard-tier provider=Anthropic Claude API
       (requires ANTHROPIC_API_KEY + funded balance)
```

### Append #1 (standard tier)

```
$ printf "\n# phase-34.1 (2026-05-22): switch standard tier from Anthropic Claude
   (credit-exhausted) to Vertex AI Gemini (free under ADC). See
   handoff/current/live_check_33.1.md for context.\nGEMINI_MODEL=gemini-2.5-pro\n"
   >> backend/.env
APPENDED. New size: 5015 bytes, 65 lines  (was 4807 bytes, 62 lines)
```

### Restart #1

```
$ echo "Pre-restart PID: $(launchctl list | awk '$3=="com.pyfinagent.backend"{print $1}')"
Pre-restart PID: 82296
$ launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"
$ echo "Post-restart PID: $(launchctl list | awk '$3=="com.pyfinagent.backend"{print $1}')"
Post-restart PID: 28773
```

### Post-change routing log line #1 (standard tier)

```
$ grep "phase-31.1 model routing" backend.log | tail -1
07:16:03 I [main] phase-31.1 model routing:
    settings.gemini_model='gemini-2.5-pro'
    -> standard-tier provider=Gemini (Vertex AI or direct AI Studio)
```

No `_std_warning` branch fired -- the Gemini provider path bypasses the
warning at `backend/main.py:144`.

### Manual cycle trigger

```
$ curl -X POST http://localhost:8000/api/paper-trading/run-now -m 10
(buffered; background task launched)

$ curl -sS http://localhost:8000/api/paper-trading/status | jq '.loop'
{"running": true, "last_run": null, "last_result": null}
```

### Cycle's behavior with ONLY the standard tier flipped

```
$ grep -E "Moderator anthropic|Gemini analysis for" backend.log | head -10
07:21:31 W [debate] Moderator anthropic error -- SDK retries exhausted, propagating: BadRequestError
07:21:33 I [autonomous_loop] Gemini analysis for STX: HOLD (confidence=0, score=5)
07:21:47 W [debate] Moderator anthropic error -- SDK retries exhausted, propagating: BadRequestError
07:21:49 I [autonomous_loop] Gemini analysis for AMD: HOLD (confidence=0, score=5)
07:22:11 W [debate] Moderator anthropic error -- SDK retries exhausted, propagating: BadRequestError
07:22:17 I [autonomous_loop] Gemini analysis for CIEN: HOLD (confidence=0, score=5)
```

**This is the partial-fix scenario.** Standard-tier (Bull / Bear / enrichment)
runs via Gemini. Deep-think tier (Moderator) hits Anthropic credit. The
autonomous_loop's lite-Gemini-analysis fallback then produces a (confidence=0,
score=5) HOLD verdict per ticker, so the cycle DOES progress (no hard halt),
but every ticker degrades to the lite path.

### Append #2 (deep-think tier)

```
$ printf "# phase-34.1e (2026-05-22): standard-tier flip exposed deep-think tier
   as ALSO Anthropic-pinned (Moderator/Critic/Synthesis/RiskJudge).
   settings.deep_think_model default='claude-opus-4-7' -> credit errors.
   Flipping to Gemini-2.5-pro to match.\nDEEP_THINK_MODEL=gemini-2.5-pro\n"
   >> backend/.env
APPENDED. New size: 5290 bytes, 67 lines
```

### Restart #2 + cycle #2 (the "clean" cycle for 34.2)

(Captured in `live_check_34.2.md` once the in-flight partial-fix cycle
completes and the deep-think restart fires.)

---

## Artifact shape

After this step:

- `handoff/current/contract.md`           -- phase-34 contract (this directory)
- `handoff/current/experiment_results.md` -- this file
- `handoff/current/live_check_34.1.md`    -- Step 34.1 verdict + evidence
- `handoff/current/live_check_34.2.md`    -- Step 34.2 verdict + 9-row probe table (pending)
- `handoff/current/evaluator_critique.md` -- Q/A verdict on both live_checks (pending)
- `handoff/harness_log.md`                -- appended block at the tail
- `.claude/masterplan.json`               -- phase-34 entry + status flip
- `backend/.env`                          -- 2 new lines + 2 comment lines (67 total)
- `handoff/cycle_history.jsonl`           -- expected +2 rows (one per cycle)

---

## Honest gaps + risks

1. **The "first clean cycle" data isn't in this file yet.** The deep-think
   tier flip + restart + second cycle is queued behind the in-flight first
   cycle. `live_check_34.2.md` is where the clean-cycle evidence lands.

2. **Anthropic balance not topped up.** Any Layer-2 in-app MAS role that
   hard-codes Claude (e.g. `mas_main` per CLAUDE.md effort policy) will
   still error if it fires. Layer-2 MAS doesn't run on every autonomous-loop
   cycle (per-ticker, separate trigger), so this gap is dormant -- but it
   will bite if the operator manually triggers a per-ticker MAS run.

3. **Observability gap filed.** `backend/main.py:140-152` should also emit
   a deep-think-tier banner; recommend `phase-34.5` for that ~10 LOC fix.

4. **Lite-Gemini fallback output quality.** The (confidence=0, score=5)
   defaults from the partial-fix first cycle are NOT real conviction signals.
   This is fine for verifying the route flipped, but the second cycle MUST
   show non-default confidence/score values to confirm the full orchestrator
   is producing real analysis.
