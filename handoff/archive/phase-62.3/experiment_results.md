# Experiment Results -- Step 62.3 (GENERATE)

**Step:** 62.3 -- Scheduled-session plists + wrapper + kickoff prompts. **Date:**
2026-06-12. **State:** built + behaviorally probed; cycle-2 fixes applied after Q/A
spawn-1 CONDITIONAL.

## What was built (7 files + 1 install + 2 doc fixes)

1. ~/Library/LaunchAgents/com.pyfinagent.away-session-am.plist (07:30 local) and
   ...-pm.plist (22:00 local) -- mas-harness EnvironmentVariables block verbatim,
   RunAtLoad=false, logs to handoff/away_ops/launchd-{am,pm}.log. BOTH BOOTSTRAPPED;
   first fires PM tonight 22:00 + AM tomorrow 07:30 (live rehearsal day, operator home).
2. scripts/away_ops/run_away_session.sh -- noclobber-atomic lockfile + stale-PID reap
   with ps name-check; HALT-DEV honor (AM exits / PM degrades); sentinel pre-flight
   FAIL-CLOSED to digest-only (sentinel ships in 62.4 -- until then real sessions are
   report-only BY DESIGN); dirty-tree -> recovery prompt; pull --rebase with
   rebase-abort + offline fallback; gtimeout -k 60 14400/7200; claude -p
   --dangerously-skip-permissions --model claude-opus-4-8 --max-turns 250/120
   --output-format json via stdin; COST + LIMIT_HIT surfacing; all failures exit 0.
3. scripts/away_ops/prompt_{am,pm,recovery,digest_only}.md -- rules file FIRST +
   overriding (FO-1); AM carries all 10 rails inline (faithful after cycle-2 fix);
   corrected token-application order (validate -> advance_cursor -> .env -> restart ->
   live_check); ONE-step AM scope; PM evidence list; ~80% WIP checkpoint; rail-10
   ambiguity discipline.
4. brew install coreutils (gtimeout 9.11 -- research caught it MISSING).
5. docs/runbooks/away-ops-rules.md: token-order wording fix (original would deadlock
   against the 62.0 hook) + enforcement-layers paragraph aligned with the Q/A-ruled
   prompt architecture.
6. handoff/away_ops/pending_tokens.json (NEW, canonical asks file): SDK-CREDIT decision
   (due 2026-06-15, reply strings included) + MAS-PLIST-ZOMBIE pre-departure action.

## Verification (verbatim)

    plutil: both plists OK | bash -n: clean | grep -c 'END session': >=1 -> PASS
    dry-run: START/prompt/COST/END lines; concurrent kickstart -> SKIP (live_check SA)

Q/A spawn-1 probes (independently run): stale-lock reap, HALT-DEV AM-exit + PM-degrade,
cost-parse malformed-JSON tolerance, sentinel-before-claude code order, token-order
three-way agreement, env-block diff vs mas-harness -- ALL PASS. Incidental: Q/A's own
.env stat attempt was hook-blocked (62.0 guard proven live again).

## Iterations (honest log)

- Spawn-1 Q/A: CONDITIONAL on (B1) missing per-step experiment_results (this file),
  (B2) prompt_am rails 4/5 dropped clauses + 7/8 paraphrase while live_check claimed
  "verbatim" -- rails made faithful, live_check corrected, rules.md architecture
  paragraph aligned, (B3) pending_tokens.json absent -- created with both asks.
- Probe residue (Q/A N5): extra dry-run lines in session.log (gitignored) + json stubs;
  synthetic HALT-DEV tokens file created-and-REMOVED by Q/A (verified absent; the 62.2
  live round-trip owns the first real line).

## Operator asks routed to 62.7

SDK-CREDIT (HARD 06-15 fuse) + MAS-PLIST-ZOMBIE -- both in pending_tokens.json with
exact reply strings.
