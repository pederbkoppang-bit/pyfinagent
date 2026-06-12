# Contract -- phase-62.0: Hard-rules file + away goal install + backlog disposition + hook away-patterns

Date: 2026-06-12. Goal: goal-away-ops (install commit 66cb8bc1, operator plan-mode approval).

## Research-gate summary

Brief: handoff/current/research_brief.md (tier moderate, gate_passed: true, 5 sources in
full -- Claude Code hooks reference, git-push docs, launchctl man, Anthropic harness
design, BashFAQ/050 -- plus issues #24327/#40580; recency scan done). Key findings:
- EXTEND .claude/hooks/pre-tool-use-danger.sh (single PreToolUse entry, no matcher, env
  vars CLAUDE_TOOL_NAME/CLAUDE_TOOL_INPUT with stdin-JSON fallback, exit 2 = block,
  fail-open on internal error by design). The MCP_MIGRATE_TOKEN gate at :161-168 is the
  in-file precedent shape for the tokens_cursor gate.
- Force-push surface gaps: flags are position-free (`git push origin main --force`) and
  `+refspec` forces with no flag -- current glob (:141) and settings deny both miss these.
- launchctl removal surface = FOUR verbs: bootout, unload, remove, disable (deny on
  com.pyfinagent.* labels/plists); kickstart stays allowed (rail 9).
- .env detection is a TRIPWIRE on write shapes (>>/>, sed -i, tee, perl -i), not a
  complete parser (BashFAQ/050); daily sentinel reconciliation (62.4) is the backstop.
  GAP found by researcher: Edit/Write tool calls on backend/.env bypass a Bash-only
  check -- must match tool_name in {Edit,Write,NotebookEdit} + file_path too.
- Operator keystrokes (`!` commands) never traverse PreToolUse -- no operator carve-out
  needed.
- Block stderr must prescribe "write a token ask, move on" (issue #24327 stall guard);
  mirror patterns into settings.json deny as layer 2 (issue #40580 subagent caveat).
- Deferred flips are push-silent (archive/auto-commit hooks key ONLY on status=="done");
  hazard: run flips on a clean masterplan (verified clean now). `deferred` is existing
  status vocabulary (8 uses). All 10 target ids exist in bare form, all pending.
- active_goal.md exists (refresh, don't create); handoff/away_ops/tokens_cursor does not
  exist yet (absent = no token = gate closed).

## Hypothesis

Encoding the away rails as (1) a binding rules document, (2) deterministic PreToolUse
blocks with a token-cursor gate, and (3) settings-deny mirrors gives three independent
layers that make the prior away-week failure mode (unauthorized behavior change) require
three simultaneous failures instead of one, without blocking legitimate work.

## Immutable success criteria (verbatim from .claude/masterplan.json, phase-62 step 62.0)

1. "docs/runbooks/away-ops-rules.md contains the 10 numbered rails from the approved plan
   verbatim and is referenced by every kickoff prompt"
2. "the 10 disposition steps are status=deferred with an audit note citing the operator's
   verbatim 'Confirm disposition' reply; a git diff in experiment_results.md proves ONLY
   status/audit fields changed (verification criteria byte-identical)"
3. ".claude/hooks/pre-tool-use-danger.sh (or equivalent PreToolUse hook) blocks: git push
   --force variants, launchctl bootout/unload of pyfinagent labels, and edits
   adding/changing PAPER_* flag lines in backend/.env when handoff/away_ops/tokens_cursor
   has no fresh matching token -- each pattern unit-tested by invoking the hook with a
   synthetic payload"
4. "handoff/current/active_goal.md points at goal_away_ops.md with the away calendar"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && test -f
docs/runbooks/away-ops-rules.md && python3 -c "import json; mp=json.load(open('.claude/
masterplan.json')); ids={'36.2','36.3','36.4','36.5','36.6','37.3.1','40.3.1','40.8.2',
'40.7','40.1'}; steps={s['id']: s for p in mp['phases'] for s in p.get('steps',[])};
bad=[i for i in ids if steps.get(i,{}).get('status')!='deferred']; assert not bad, bad;
print('deferred OK')"

Criterion-1 forward-reference note: kickoff prompts are CREATED by 62.3; at 62.0 close the
"referenced by every kickoff prompt" leg is satisfiable only as a 62.3 obligation. Q/A may
mark that leg deferred-to-62.3; 62.3's contract re-verifies it. The rules file itself +
all other legs complete here.

## Plan

1. Write docs/runbooks/away-ops-rules.md: the 10 rails verbatim from
   handoff/away_ops/approved_plan_2026-06-12.md "Safety rails" section + enforcement-
   layers note + token grammar reference.
2. Extend pre-tool-use-danger.sh (fail-open shell discipline preserved):
   (a) robust force-push: any `git push` segment carrying --force/--force-with-lease/-f
   anywhere, or a `+refspec` argument; (b) launchctl {bootout,unload,remove,disable} on
   com.pyfinagent.* labels or plist paths (kickstart untouched); (c) backend/.env write
   tripwire for Bash shapes (>>, >, sed -i/--in-place, tee [-a], perl -i) AND for
   Edit/Write/NotebookEdit with file_path matching backend/.env -- gated on
   handoff/away_ops/tokens_cursor existing with mtime < 6h; block stderr prescribes
   "record a token ask in handoff/away_ops/pending_tokens.json and move on".
3. Mirror layer-2 deny entries in .claude/settings.json permissions.deny:
   Bash(git push*--force*), Bash(git push*-f *), Bash(git push* +*),
   Bash(launchctl bootout*com.pyfinagent*), Bash(launchctl unload*com.pyfinagent*),
   Bash(launchctl remove*com.pyfinagent*), Bash(launchctl disable*com.pyfinagent*).
4. Masterplan flips: python round-trip setting status=deferred + new deferral_audit field
   on the 10 ids (audit_basis and verification untouched); git diff captured verbatim
   into experiment_results.md.
5. Refresh active_goal.md: dual in-flight goals (phase-61 chain + goal-away-ops), away
   calendar pointer, token grammar, pending token asks.
6. Tests: backend/tests/test_phase_62_0_danger_hook.py -- subprocess the hook with
   synthetic env payloads: BLOCK cases (git push origin main --force; git push -f; git
   push origin +main; launchctl bootout/unload/remove/disable gui/501/com.pyfinagent.
   backend; echo X >> backend/.env without cursor; sed -i on backend/.env; Edit tool
   file_path backend/.env) and ALLOW cases (git push origin main; launchctl kickstart -k
   com.pyfinagent.backend; >> on other files; .env write WITH fresh cursor; rm -rf
   node_modules). Assert exit codes 2/0 and stderr content for blocks.
7. Q/A spawn -> harness_log append -> flip 62.0 (auto-commit hook pushes).

## Out of scope

62.1-62.8 artifacts (plists, wrapper, sentinel, healthcheck, digests); any backend/.env
edit (the hook gate is built and tested with a TEMP cursor fixture, never a real flag
write); phase-61 step work.

## References

- handoff/current/research_brief.md (62.0 gate)
- handoff/away_ops/approved_plan_2026-06-12.md (rails source of truth)
- https://code.claude.com/docs/en/hooks ; git-scm.com/docs/git-push ; ss64.com/mac/
  launchctl.html ; anthropic.com/engineering/harness-design-long-running-apps ;
  mywiki.wooledge.org/BashFAQ/050 ; issues #24327, #40580
