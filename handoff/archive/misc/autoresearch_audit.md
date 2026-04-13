# Autoresearch Audit: pyfinAgent Optimizer vs Karpathy's autoresearch

**Date:** 2026-04-06 22:36 GMT+2
**Reference:** https://github.com/karpathy/autoresearch

## Karpathy's Core Design (What Makes It Work)

1. **Single file to modify** — agent only touches train.py. Scope is tight.
2. **Fixed time budget** — 5 min per experiment, always comparable.
3. **Simple metric** — val_bpb (lower = better). One number.
4. **results.tsv** — tab-separated log: commit, val_bpb, memory, status, description
5. **Git-based state** — keep = advance branch, discard = git reset
6. **NEVER STOP** — loop forever until human interrupts
7. **Simplicity criterion** — improvement that adds ugly complexity = not worth it
8. **Crash handling** — easy fix → retry, fundamentally broken → skip, log crash, move on
9. **program.md** — the "skill file" that programs the researcher agent

## Our Optimizer: What Matches

| Karpathy | pyfinAgent | Status |
|----------|-----------|--------|
| Single file (train.py) | Strategy params (JSON) | ✅ Similar — params instead of code |
| Fixed time budget (5 min) | Walk-forward backtest (~2-5 min) | ✅ Similar |
| Single metric (val_bpb) | Sharpe ratio + DSR | ✅ Similar (two metrics) |
| results.tsv | quant_results.tsv | ✅ Same pattern |
| Git-based keep/discard | TSV status: keep/discard/crash | ⚠️ We log but don't use git branches |
| NEVER STOP loop | run_loop(max_iterations=100) | ⚠️ We stop after N, should loop until interrupted |
| Simplicity criterion | Quality criteria (15% weight) | ✅ Have it but weaker |
| Crash handling | try/except → log crash | ✅ Same |
| program.md | agent_definitions.py system prompts | ⚠️ Ours is code, should be markdown |

## What We're Missing (Gaps)

### GAP 1: No "NEVER STOP" mode
Karpathy's loop runs indefinitely. Ours stops after max_iterations.
**Fix:** Add `max_iterations=0` to mean "infinite, stop on signal only"

### GAP 2: No git branch per run
Karpathy keeps good experiments by advancing the git branch.
We only log to TSV — if something crashes, we lose the exact state.
**Fix:** Git commit params + results after each kept experiment

### GAP 3: Results not saved as backtest runs
Optimizer experiments don't appear in the main Run dropdown.
Only the baseline and final result get saved.
**Fix:** Save each kept experiment as a full backtest run (with report)

### GAP 4: No experiment description
Karpathy logs a human-readable description of what each experiment tried.
Our TSV has `param_changed` but it's just "holding_days: 90 -> 60".
**Fix:** Add a `description` column or use param_changed as-is (it's close enough)

### GAP 5: program.md equivalent is code, not configurable
Karpathy's agent behavior is controlled by a markdown file the human edits.
Our agent behavior is hardcoded in Python (agent_definitions.py).
**Fix:** Phase 4.0 moves this to SOUL.md (OpenClaw workspace files)

### GAP 6: Progress visibility
Karpathy's agent runs in a terminal — you see everything.
Our optimizer runs in background with polling.
**Fix:** Better progress bar showing window/experiment detail (partially done)

### GAP 7: Pause/Resume
Karpathy has no pause — it's NEVER STOP.
We need pause because experiments cost money (API calls).
**Fix:** Already have stop_check lambda, need UI buttons

## Action Items (Priority Order)

1. **UI: Run dropdown shows experiments under parent run** ✅ Done
2. **UI: Progress bar shows window detail during optimizer** ✅ Done
3. **UI: Pause/Stop buttons** ✅ Done
4. **Backend: max_iterations=0 for infinite loop** ✅ Done — default is now 0 (forever)
5. **Backend: Git commit per kept experiment** ✅ Done — commits optimizer_best.json + TSV
6. **Backend: Save each kept experiment as full run** ✅ Done — on_result persists via result_store
7. **Architecture: Move to program.md/SOUL.md pattern** → Phase 4.0 (planned, saved for later)
