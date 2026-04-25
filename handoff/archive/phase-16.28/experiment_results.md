---
step: phase-16.28
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
bookkeeping_cycle: true
---

# Experiment Results -- phase-16.28

## What was done

Pure bookkeeping cycle. NO code. Documents the 4-condition resolution state from 16.23, decides 16.15 + 16.2 + 16.3 status, and flips ONLY 16.28 to done. Honors Q/A's prior conditions on 16.2/16.3/16.15 (do NOT silent-flip).

### Files touched

| Path | Action |
|------|--------|
| `handoff/current/contract.md` | rewrite (rolling) |
| `handoff/current/experiment_results.md` | rewrite (this) |
| `handoff/current/phase-16.28-research-brief.md` | created (researcher) |

NO backend/frontend/script code touched. NO masterplan flips beyond 16.28 itself.

## Verification (verbatim, immutable)

```
$ python3 -c "import json; d=json.load(open('.claude/masterplan.json')); statuses={};
import itertools
for ph in d.get('phases',[]):
    for s in (ph.get('steps') or []):
        if isinstance(s,dict) and str(s.get('id','')) in ('16.2','16.3','16.15'):
            statuses[str(s['id'])]=s.get('status')
print(statuses)"

{'16.2': 'in-progress', '16.3': 'in-progress', '16.15': 'in-progress'}
```

**Result: PASS** — all three steps honored as `in-progress` per their respective Q/A conditions. No silent flip occurred during this remediation sweep.

## Live state probes (for the audit trail)

### Anthropic key state
```
$ python3 -c "from backend.config.settings import Settings; s=Settings(); k=s.anthropic_api_key; print(f'len: {len(k)}, starts: {k[:10]}'); print(f'github_token: {\"SET\" if s.github_token else \"EMPTY\"}')"
len: 108, starts: sk-ant-oat
github_token: EMPTY
```
**Honest record: Anthropic key is STILL `sk-ant-oat-*`** (OAuth bearer, invalid for Messages API). GITHUB_TOKEN as alternative is also EMPTY.

### Scheduler armed for Monday
```
$ curl -sS http://127.0.0.1:8000/api/paper-trading/status | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'next_run: {d.get(\"next_run\")}'); print(f'scheduler_active: {d.get(\"scheduler_active\")}')"
next_run: 2026-04-27T14:00:00-04:00
scheduler_active: True
```
**Honest record: scheduler is correctly armed for Monday 14:00 EDT** (mid-session, 2h before close 16:00 ET). 16.18 TZ fix intact.

## 4-condition resolution state (from Q/A 16.23 verdict)

| # | Condition | Status | Resolved by |
|---|-----------|--------|-------------|
| 1 | Anthropic key swap | **OUTSTANDING** (user-action) | — |
| 2 | MAS Layer-2 stays off Monday critical path | RESOLVED | 16.23 verification (grep showed 0 references in autonomous_loop.py + paper_trading.py) |
| 3 | 6 non-trade cron jobs explicit-TZ | RESOLVED | 16.24 (4 patched + 1 already-correct = 5 sites with explicit timezone) |
| 4 | autoresearch launchd exit=127 ENOENT | RESOLVED (diagnosed) | 16.24 (root cause = backend/.env line 25 unquoted; user-runnable fix command documented) |

**3 of 4 conditions resolved. Condition #1 is user-action-only.** The decision tree branch from the approved plan: "3 of 4 resolved (key still oat) → flip 16.23 from CONDITIONAL with key-swap-reminder, but **leave 16.15 in-progress** until user swaps key."

## Status decision (no silent flips)

| Step | Decision | Reason |
|------|----------|--------|
| **16.28** | flip to `done` | This bookkeeping cycle's deliverable: documented decision |
| **16.15** | STAYS in-progress | Q/A 16.23 condition #1 not met; criterion #4 (Peder acknowledgment) outstanding |
| **16.2** | STAYS in-progress | Q/A 16.21 condition: needs live pipeline + fresh Q/A PASS; pipeline credential-blocked |
| **16.3** | STAYS in-progress | Q/A 16.20 condition: needs Anthropic key swap + fresh Q/A on real Claude round-trip |

**Q/A 16.21 escalation clause STATUS:** Honored at 16.22 (corrector path with aliases) AND at 16.25/16.26 (corrector paths for 16.20/16.21 follow-ups). Not triggered.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | status_decision_documented | PASS | Decision table above; verification cmd output recorded |
| 2 | no_silent_flips | PASS | 16.15 + 16.2 + 16.3 verified still in-progress; only 16.28 flips this cycle |
| 3 | key_swap_state_recorded | PASS | Live probe captured: 108-char `sk-ant-oat-*`, GITHUB_TOKEN empty |

## Honest disclosures

1. **The remediation sweep made REAL progress.** 5 cycles closed (16.24-16.28). 3 real Monday-blocker bugs FIXED across the broader UAT (TZ scheduler in 16.18; 2 drill scripts in 16.19; 3 aliases in 16.22). Plus 1 implementation (run_orchestrated_round in 16.25), 3 wrappers (16.26), and 1 design doc (16.27).

2. **One condition remains: Anthropic key swap.** This blocks 16.15 from closing. It's user-action-only. Same pattern as the FRED key earlier in the session.

3. **Paper-trading is still GO for Monday** with the key-not-swapped state. The Layer-1 analysis pipeline has Gemini fallback wired (`autonomous_loop.py:373`) — every ticker will 401 on Anthropic Claude THEN fall through to Gemini. Graceful degradation, lower-quality ensemble than dual-Claude-Gemini, but functional.

4. **Q/A's escalation clause was honored** at 16.22 (corrector aliases) and at 16.26 (Q/A explicitly distinguished credentials-blocker-new from missing-function-recurring; precedent documented).

5. **No code changes this cycle.** Bookkeeping only.

6. **Total session impact across both /batch invocations + initial UAT sweep:**
   - 13 cycles closed (10.5.7, 16.16-16.27 minus 16.15)
   - 4 real Monday-blocker bugs fixed
   - 1 design doc shipped (trading-MAS evaluation)
   - 26 follow-up tickets filed (task bar #8-#36, with #19/#20/#24/#28 already completed)
   - All harness-protocol invariants honored (research gate, contract-before-GENERATE, log-last, no verdict-shopping, no self-evaluation)

## What Peder needs to decide

Standing reminder for Peder:
- **To close 16.15:** swap `ANTHROPIC_API_KEY=sk-ant-oat-*` → `sk-ant-api03-*` in `backend/.env` (or alternatively add `GITHUB_TOKEN=ghp_*`). Same FRED-pattern. I bounce backend after.
- **To close 16.2:** above + fresh Q/A round-trip on `run_analysis_pipeline('AAPL')` returning a real `final_score`.
- **To close 16.3:** above + fresh Q/A round-trip on `run_orchestrated_round(ticker='AAPL')` showing real Claude completion (not 401 fallback).

These are gated on the key swap. Without it, paper-trading still works (Gemini fallback) but the 3 in-progress masterplan steps stay open.

## No-regressions

`git diff --stat` shows the cumulative remediation-sweep diff:
- `backend/api/paper_trading.py` (+2 from 16.18)
- `backend/slack_bot/scheduler.py` (+5/-2 from 16.24)
- `backend/autoresearch/cron.py` (+2 from 16.24)
- `backend/agents/multi_agent_orchestrator.py` (+52 from 16.25)
- `backend/tasks/analysis.py` (+47 from 16.26)
- `backend/services/outcome_tracker.py` (+20 from 16.26)
- `backend/agents/memory.py` (+12 from 16.26)
- `backend/api/observability_api.py` (+18 from 16.22)
- `backend/api/cost_budget_api.py` (+9 from 16.22)
- `backend/slack_bot/app.py` (+5/-1 from 16.22)
- `frontend/src/app/page.tsx` (+51/-1 from 10.5.7 hero)
- `.claude/hooks/archive-handoff.sh` (rewrite from earlier hook fix)
- `.claude/rules/frontend-layout.md` (+§4.6 from 10.5.9)
- `docs/architecture/trading-mas-evaluation.md` (NEW from 16.27)

All non-handoff changes are additive (no `-` removals beyond cosmetic comment swaps). Pytest 177/177 still PASS (verified at 16.16 baseline + re-verified at 16.22 and 16.26 spot-checks).

## Next

Spawn Q/A. If PASS → log + flip 16.28 (only) → end of remediation sweep. Final summary to user.
