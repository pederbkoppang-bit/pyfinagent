# RECOVERY away session (goal-away-ops) -- a prior session left a dirty tree

The wrapper found uncommitted changes: a previous session crashed, hit its wall-clock
cap, or hit the Agent SDK credit limit mid-step. Your ONLY job is to restore a clean,
truthful state. Authority: docs/runbooks/away-ops-rules.md -- read FIRST; all 10 rails
bind (see prompt_am.md for the inline text). Rail 3 is the sharp one here:

  NEVER git checkout/restore/stash files. Those silently destroy work (documented
  incident class in scripts/mas_harness/run_cycle.sh:36-40). You may only COMMIT or
  surgically REVERT-BY-EDITING files the crashed session itself edited.

## Procedure

1. Read handoff/away_ops/session.log tail -- which session died, doing what, when.
   Read handoff/away_ops/session_notes.md + any `chore(away-wip)` checkpoint commit
   message for declared resume state.
2. git status --porcelain + git diff -- inventory EVERY dirty file. Classify each:
   (a) belongs to the crashed session's declared step -- complete it if the remaining
   work is small and unambiguous (finish the harness loop: qa, log, flip), otherwise
   commit as `chore(away-wip): <step> checkpoint (recovery)` with a session_notes.md
   entry describing exactly what remains;
   (b) NOT attributable to the crashed session (unexpected) -- do NOT touch it; record
   it verbatim in session_notes.md and pending_tokens.json as an operator ask.
3. Audit-log files (handoff/audit/*.jsonl) and session artifacts: just commit them.
4. After the tree is clean: push (manual fallback if the hook stalls), append a
   `## Recovery -- <date>` section to session_notes.md (what was found, what was done,
   what remains), and EXIT. Do NOT start a new masterplan step in this session -- the
   next AM session resumes the calendar with a clean tree.

Unsure about ANY file => rail 10: leave it, write the ask, exit. A dirty-but-honest
tree beats a clean-but-lossy one.
