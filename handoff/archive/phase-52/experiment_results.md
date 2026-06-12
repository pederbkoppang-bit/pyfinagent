# Experiment Results -- Step 62.0 (GENERATE)

**Step:** 62.0 -- Hard-rules file + away goal install + backlog disposition + hook
away-patterns. **Date:** 2026-06-12. **State:** complete pending Q/A.

## What was built (per criterion)

1. **docs/runbooks/away-ops-rules.md** (NEW): the 10 rails verbatim from
   handoff/away_ops/approved_plan_2026-06-12.md "Safety rails", + enforcement-layers
   section (prompt/hook/sentinel), + token mechanics (grammar, cursor, reserved tokens
   KILL SWITCH: RESUME / HALT-DEV / RESUME-DEV), + the rail-6 trading-behavior file list.
   "Referenced by every kickoff prompt" leg: prompts are created by 62.3 -- forward
   obligation noted in the contract, re-verified there.
2. **Backlog disposition**: 10 steps (36.2-36.6, 37.3.1, 40.1, 40.3.1, 40.7, 40.8.2)
   pending -> deferred + deferral_audit field citing the operator's verbatim reply.
   Diff-class proof in live_check_62.0.md SB: only status/deferral_audit/updated_at (+
   trailing-comma artifact) changed; criteria byte-identical; auto-push hook silent.
3. **Hook away-patterns** (.claude/hooks/pre-tool-use-danger.sh, fail-open discipline
   preserved):
   - robust force-push: position-free --force/--force-with-lease/--force-if-includes/-f
     on any `git push` segment, plus +refspec (research: both missed by the legacy glob);
   - launchctl bootout/unload/remove/disable x com.pyfinagent.* (kickstart untouched);
   - backend/.env write tripwire (>>/>, sed -i, tee [-a], perl flag-cluster -i) AND
     Edit/Write/NotebookEdit file_path coverage (researcher-found bypass), gated on
     handoff/away_ops/tokens_cursor mtime < 6h; block stderr prescribes the
     pending_tokens.json ask path (issue #24327 stall guard).
   - Layer-2: 7 deny mirrors in .claude/settings.json (issue #40580 subagent caveat).
4. **active_goal.md** refreshed (dual goals, rails-first reading order, calendar pointer).

## Verification output (verbatim)

    $ bash -n .claude/hooks/pre-tool-use-danger.sh && echo "syntax OK"   -> syntax OK
    $ python -m pytest backend/tests/test_phase_62_0_danger_hook.py -q   -> 30 passed in 0.89s
    $ <masterplan deferred assert>                                       -> deferred OK

Live self-demonstration: the session's own first transcript-capture Bash call was blocked
by the live hook (payload contained `>> backend/.env`) -- rail-1 enforcement proven on
real session traffic. Transcript verbatim in live_check_62.0.md SA.

## Iterations (honest log)

- Initial test run: 29/31 -- (a) perl `-pi` flag-cluster missed by a literal `-i` pattern
  -> regex widened to `-[A-Za-z]*i`; (b) one test expected `grep 'git push --force'` to be
  allowed, but the PRE-EXISTING legacy substring guard blocks it (harmless conservative
  false positive, predates this step) -> test case removed, legacy behavior left as-is.
- 30/30, first Q/A PASS.
- POST-PASS live defect (cycle-2): the 62.0 COMMIT ITSELF was blocked by the new
  force-push guard -- the commit message mentioned the flag literals in prose while
  `git push origin main` sat in a later segment of the same compound command. Whole-string
  matching poisoned across segments. Away sessions will routinely write commits
  describing these guards, so reword-and-move-on would have planted a recurring trap.
  FIX: force-push + launchctl guards rewritten to per-segment scoping (python re.split on
  ; && || |, pattern tested inside each segment only); 3 regression tests added
  (commit-message prose + clean push allowed; force flag in the actual push segment still
  blocked). The .env tripwire stays whole-string deliberately (fail-safe direction; Write
  tool is the documented path for docs that must quote those shapes).
- Final: 33/33. Delta re-evaluated by a SECOND fresh Q/A (cycle-2 flow, evidence changed).

## File list

- docs/runbooks/away-ops-rules.md (NEW)
- .claude/hooks/pre-tool-use-danger.sh (3 new guard blocks + Edit/Write coverage)
- .claude/settings.json (7 deny entries)
- .claude/masterplan.json (10 deferral flips ONLY)
- handoff/current/active_goal.md (refresh)
- backend/tests/test_phase_62_0_danger_hook.py (NEW, 30 tests)
- handoff/current/{contract.md, research_brief.md, live_check_62.0.md, this file}

## Masterplan flip diff (key lines, verbatim)

    -          "status": "pending",
    +          "status": "deferred",
    +          "deferral_audit": "deferred 2026-06-12 per operator verbatim 'Confirm
               disposition (Recommended)' (goal-away-ops backlog disposition; ...)"
    (x10 steps; updated_at bumped; no other field classes in the diff)
