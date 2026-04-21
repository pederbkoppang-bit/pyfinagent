# Evaluator Critique — Cycle 4.15.6

Step: phase-4.15.6 Batches + Files + Citations + Search results

## Q/A verdict: PASS

Acting agent: qa-evaluator (filling in for merged qa.md — session
cache; requires restart per MF-44).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 7 zero-usage grep checks returned 0. ClaudeClient.generate_content builds messages.create kwargs with model/max_tokens/temperature/system/messages/thinking only — no betas= forwarding, confirming structural blocker. autonomous_loop._run_claude_analysis line 382 with sequential asyncio.to_thread per-ticker at 436-441; deprecated claude-sonnet-4-20250514 on line 438. MF-40 to MF-44 fixes all verified.",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "grep_batches=0", "grep_files=0", "grep_citations=0",
    "grep_search_result=0", "grep_pdf=0", "grep_anthropic_beta=0",
    "grep_betas_kwarg=0",
    "llm_client_betas_structural_blocker_confirmed",
    "autonomous_loop_batch_candidate_confirmed_line_436_438",
    "stale_model_id_confirmed_line_438",
    "MF40_permissionMode_2", "MF41_qa_NEVER_constraint_1",
    "MF42_SubagentStop_hook_present",
    "MF43_separation_of_duties_note_1",
    "MF44_session_restart_note_1", "settings_json_valid"
  ]
}
```

## Novel findings verified

- ClaudeClient has no `betas=` kwarg plumbing — confirmed by both
  grep (0 matches) AND code read of llm_client.py:613-630 (kwargs
  dict never sets `betas`).
- `autonomous_loop.py:438` uses deprecated `claude-sonnet-4-
  20250514` on the per-ticker nightly sweep.
- `_run_claude_analysis` at line 382 is the archetypal batch
  candidate — sequential `asyncio.to_thread` per ticker, no
  batching. MF-34 fix has a specific line target.

## Separation-of-duties note (MF-43 applied)

Per MF-43 doctrine, flagging: Main authored MF-40..MF-44 edits
this same session, and this Q/A verification is filling in for
the merged `qa` agent (not yet dispatchable per MF-44). The
deterministic checks are independent of Main's reasoning so the
verification is sound, BUT human second-look on the agent-config
edits is recommended before the next session cycle depends on
them.

**Action for Peder:** review `.claude/agents/qa.md` (permissionMode
+ constraint rewrite) and `.claude/settings.json` (new
`SubagentStop` hook) + CLAUDE.md additions before the next
compliance audit cycle starts depending on the new agent roster.

## Combined verdict: PASS

- Zero-usage claim live-verified across all 4 features
- Structural blocker (missing `betas=` kwarg) confirmed
- ClaudeClient read-through shows the exact 2-3 line addition that
  would unblock 3 separate MUST-FIX items
- MF-40-44 landed cleanly

## Next

Proceed to 4.15.8 Tool-use primitives.
