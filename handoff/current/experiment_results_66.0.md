# Experiment results -- 66.0 Recovery re-baseline (Cycle 67, 2026-07-06)

Nature of the step: mechanical recovery + bookkeeping (no trading code, no .env, no
plist edits beyond the operator-approved MAS-PLIST mv). All evidence below is verbatim
command output from this session.

## What was done

1. **Backlog sweep + goal install (pre-contract session actions, recovery-class):**
   - Commit `7be476b3`-era backlog (34x 401 session JSONs 06-19..07-06, audit streams,
     analysis draft): commit `<sweep>` in range `899d4a90..68909af1`, pushed.
   - goal-phase66-reactivation installed: masterplan phase-66 (6 steps, ALL pending --
     no status flipped in the install edit), active_goal.md refreshed, spec file
     goal_phase66_reactivation.md. Commit `68909af1`, pushed.
2. **MAS-PLIST-ZOMBIE executed (operator-approved in-session):**
   `mv ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist ~/Library/LaunchAgents/disabled.com.pyfinagent.mas-harness.plist.bak` -> output `MOVED`; `ls` confirms
   `disabled.com.pyfinagent.mas-harness.plist.bak` present, original absent.
3. **Credential restoration verified:** `claude_code_health_probe(timeout_s=30)` ->
   `probe ok: True / msg: ok` (post-/login). The rail is functional ahead of 66.1's
   hardening.
4. **pending_tokens.json: 8/8 asks dispositioned** (commit `576fdb13`):
   - METERED-BREACH-RECURRING -> `root_caused_pending_fix` with the BQ-measured phantom
     accounting note (06-17: 137/137 failed cc_rail calls $16.30; 06-18: 207/207 $42.20;
     0 tokens; flat $0.50 rows from an unpinned writer; real window metered spend ~$8.24
     Gemini + $0.07 claude-code; fix owned by 66.3).
   - MAS-PLIST-ZOMBIE -> `resolved` (mv above). FABLE-HEADLESS -> `resolved`
     (KEEP OPUS OVERRIDE, operator approved the Part-1 table). SDK-CREDIT -> `deferred`.
   - TEST-TOKEN-62.2, WEBHOOK, AUTORESEARCH-SPEND, ENV-LINE-81 -> `open_operator_gated`
     with return-day re-ask notes. `updated` bumped to 2026-07-06T21:30:00+00:00.
   - Schema safety per research_brief_66.0.md: sole parser `scheduler.py:499-503` is
     `.get()`-tolerant; sentinel/healthcheck never read this file.
5. **Recovery-loop exit proven** (criterion 3): see live_check_66.0.md -- wrapper
   condition quoted verbatim (run_away_session.sh:96-102), wrapper-visible dirty set
   empty, and the REAL preflight chain (`AWAY_SESSION_TEST_PREFLIGHT=1`, which per
   :122-126 exercises HALT-DEV -> sentinel -> dirty-tree -> prompt selection with no
   side effects) emitted `PREFLIGHT_PROMPT=am`. NOTE: `DRY_RUN=1` would have skipped
   the dirty check (:80) and was NOT used.

## Verbatim verification command output (immutable command)

```
$ git log origin/main..HEAD --oneline | wc -l && git status --porcelain | grep -vE 'handoff/(audit/|logs/|\.cycle_heartbeat|cycle_history|kill_switch_audit|prompt_leak)' | wc -l && jq -r '.updated' handoff/away_ops/pending_tokens.json
       0
       0
2026-07-06T21:30:00+00:00
```
(First run mid-step returned 3 unpushed commits -- the changelog-hook trailing-commit
race documented in feedback_auto_commit_hook_stalls; resolved by manual `git push
origin main` per that doctrine, then re-run. Output above is the post-push run.)

## Root cause of record (credential)

Per research_brief_66.0.md: the failure signature (ECONNRESET 06-15..19 -> persistent
401 from 06-20 with zero self-heal across 34 sessions) is CONSISTENT WITH the
claude-code credential-corruption class (anthropics/claude-code#61912: refresh during a
transient network window persists a corrupted credential; possible concurrent-refresh
race #54443). The official auth doc publishes no refresh-token lifetime, so no
fixed-lifetime claim is made. Structural mitigation candidate for 66.4:
`claude setup-token` (1-year lifetime, higher credential-precedence slot) + a
dead-man's switch on "last successful away session" OUTSIDE the session process.

## File list

- handoff/current/{research_brief_66.0.md, contract_66.0.md, experiment_results_66.0.md,
  live_check_66.0.md} (NEW)
- handoff/away_ops/pending_tokens.json (dispositions)
- .claude/masterplan.json (phase-66 install -- separate pre-contract commit)
- handoff/current/{active_goal.md, goal_phase66_reactivation.md} (install commit)
- ~/Library/LaunchAgents plist mv (outside repo)
- NO trading code, NO backend/.env, NO sentinel/healthcheck edits.

## Artifact shape

Five-file protocol: research_brief_66.0.md (gate_passed true) -> contract_66.0.md ->
this file -> evaluator_critique_66.0.md (Q/A) -> harness_log.md Cycle 67 append ->
status flip. live_check: live_check_66.0.md.
