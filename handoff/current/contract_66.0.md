# Contract -- 66.0 Recovery re-baseline (goal-phase66-reactivation)

Step: 66.0 | Cycle 67 | 2026-07-06 | Operator present (return day)

## Research-gate summary

research_brief_66.0.md (tier simple, gate_passed: true, 6 read-in-full / 58 URLs /
recency scan / 8 internal files). Load-bearing findings:
- Credential root cause: signature (ECONNRESET 06-15..19 -> persistent 401 from 06-20,
  no self-heal over 34 sessions) matches the claude-code credential-corruption class
  (anthropics/claude-code#61912; concurrent-refresh race #54443). Official auth doc
  publishes NO refresh-token lifetime -> the root-cause note must say "consistent with
  the corruption class", never a fixed-lifetime claim. `claude setup-token` (1-year,
  higher precedence slot) is the structural candidate for 66.4.
- run_away_session.sh:96-102: "clean" excludes ONLY handoff/audit/|handoff/away_ops/|
  handoff/logs/; the 4 handoff-root evidence files + handoff/current/ DO count as dirty.
  Correct exit evidence = AWAY_SESSION_TEST_PREFLIGHT mode echoing PREFLIGHT_PROMPT=am
  (:122-126) on a clean-at-that-moment tree. DRY_RUN=1 skips the dirty check (:80) and
  is NOT valid evidence.
- pending_tokens.json: sole parser scheduler.py:499-503 is .get()-tolerant -> adding
  disposition/resolved_at fields is safe; file lives under an excluded prefix (edits
  don't re-dirty the wrapper's check); keep `asks` an array + raised_at ISO.
- Sentinel/healthcheck are inert to pending_tokens annotations (sentinel.sh:83-123 reads
  .env/flag_baseline/operator_tokens.jsonl; healthcheck.sh:174-181 gates services only).
- SRE canon: skip > double-launch (no make-up sessions); full recovery declared only
  after one complete scheduled cycle; the paging that died lived inside the dying
  sessions -- dead-man's-switch belongs OUTSIDE (66.4's job, noted not built here).

## Hypothesis

The away engine's failure is fully accounted for (credential corruption + fail-silent
loop); after the backlog sweep (899d4a90..68909af1, already pushed this session) the
remaining re-baseline is bookkeeping: disposition the 8 stale operator asks with the
measured root cause, and prove the wrapper selects a non-recovery prompt on a clean
tree, so tomorrow's scheduled sessions resume normal one-step work.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.0)

1. "Away-window backlog committed AND pushed: git log origin/main..HEAD empty at
   verification time, and no modified/untracked files outside hook-appended audit
   streams (handoff/audit/*, handoff/*.jsonl heartbeat/history) and runtime logs"
2. "pending_tokens.json re-dispositioned per goal_phase66 Part-1 table: METERED-BREACH
   entry carries the phantom-cost root-cause note (failed cc_rail calls logging flat
   $0.50 with 0 tokens; real metered spend for the window ~ Gemini-only ~$8), resolved
   asks (MAS-PLIST) marked resolved, operator-gated asks retained open"
3. "Recovery-loop exit evidenced: run_away_session.sh's prompt-selection condition
   quoted verbatim against a current clean 'git status --porcelain' output showing the
   next scheduled session selects a NON-recovery prompt"

Verification command (immutable):
git log origin/main..HEAD --oneline | wc -l && git status --porcelain | grep -vE 'handoff/(audit/|logs/|\.cycle_heartbeat|cycle_history|kill_switch_audit|prompt_leak)' | wc -l && jq -r '.updated' handoff/away_ops/pending_tokens.json

live_check: live_check_66.0.md with push output, pending_tokens before/after summary,
and the prompt-selection evidence.

## Plan

1. Annotate pending_tokens.json: METERED-BREACH += root-cause (phantom $0.50 on failed
   flat-fee cc_rail rows; real window spend ~$8.24 Gemini; fix owned by 66.3);
   MAS-PLIST resolved (mv executed 2026-07-06, operator-approved); FABLE-HEADLESS
   resolved (KEEP OPUS OVERRIDE, operator approved the Part-1 table in-session);
   SDK-CREDIT deferred-to-next-away-window; TEST-TOKEN/WEBHOOK/AUTORESEARCH-SPEND
   remain open (operator-gated) with return-day re-ask note; bump `updated`.
2. Commit residual handoff churn so the tree is clean by the wrapper's definition;
   run AWAY_SESSION_TEST_PREFLIGHT to capture PREFLIGHT_PROMPT=am.
3. Write experiment_results_66.0.md + live_check_66.0.md; run the immutable command.
4. Fresh Q/A; on PASS append harness_log Cycle 67; flip 66.0 -> done (auto-push hook).

## Scope boundaries

No trading code, no .env, no plists beyond the already-approved MAS-PLIST mv, no
sentinel/healthcheck edits (66.3/66.4 own those). Planning notes for 66.4 (setup-token,
dead-man's switch) are recorded in the brief, not built here.

## References

research_brief_66.0.md; goal_phase66_reactivation.md Part 1; run_away_session.sh:80,
96-102, 122-126; scheduler.py:499-503; sentinel.sh:83-123; healthcheck.sh:174-181;
claude-code#61912, #54443; sre.google incident-management + cron chapters.
