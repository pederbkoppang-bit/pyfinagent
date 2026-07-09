# Contract -- phase-62.6: ops hygiene batch (rotation + autoresearch + ablation + 39.1)

Date: 2026-06-12. Goal: goal-away-ops. Research: rolling research_brief.md (gate_passed,
5 in full, recency scan; per-sub-item GO).

## Research anchors

- ROTATION: uvicorn FDs 1/2 carry O_APPEND (lsof-verified) -> cp + truncate is safe, NO
  restart (POSIX write re-derives EOF offset); newsyslog REJECTED (rename-based, the
  launchd FD would follow the renamed inode; root-territory). Mechanism: size-gated
  block in healthcheck.sh (30-min cadence; >50MB -> cp to handoff/logs/backend.log.<ts>
  -> truncate -> gzip; ~1-3s loss window, schedule away from 18:00/02:00/03:00 UTC).
  Archives hold the FRED key -> gitignored handoff/logs/, compressed, never deleted
  (forensics for the deferred rotation). 385MB backlog rotated manually in-step.
- AUTORESEARCH: nightly ALREADY exits 0 via the 51.4 preflight skip (deps missing).
  Constrained install REQUIRED (unconstrained pip silently upgrades langchain-core
  1.2.30 -> 1.4.6): pip install -c <(echo langchain-core==1.2.30) langchain-huggingface
  sentence-transformers -> 9 packages, versions recorded. SPEND TRAP (Ack 2): once
  importable, every 02:00 run executes GPTResearcher on Anthropic models (~$0.10-0.50/
  night) -- conflicts with the operator's verbatim "$0 - Max plan only". RESOLUTION:
  add --preflight-only to run_memo.py (exit 0 after _embedding_preflight, $0) and wire
  run_nightly.sh to pass it for the away window; spend resumption = operator token
  "AUTORESEARCH SPEND: RESUME" (ask added to pending_tokens.json). This PRESERVES
  today's effective $0 behavior while fixing the import -- not a trading-behavior
  change (research job; rail-6 surface untouched).
- ABLATION: NOT reproducible -- launchctl last exit 0, 16 runs; 37/37 features tested
  since 05-24 -> all-tested branch returns 0; original failing night's log truncated by
  housekeeping (unrecoverable). Disposition: fix-not-needed, documented-with-evidence;
  stays loaded (self-resumes via --next-untested). No disable needed.
- 39.1: the literal command can NEVER match (the window produced only -ERROR- files;
  success memos never carry a -PASS token; pipeline exit is head's). Evidence-by-output
  per its success_criteria: (a) launchd exit-0 streak -- 11 ERROR-free nights since
  06-01 + launchctl exit 0 (lenient) and/or 3 dep-live nights post-install (strict,
  closes ~06-15 via PM sessions); (b) root_cause.md EXISTS; (c) operator action = the
  approved constrained install recorded in the audit trail. Q/A rules lenient-now vs
  strict-Monday; criteria immutable, closure cross-referenced from 62.6.
- Residual check: test_phase_23_2_6_sector_cap_emit.py:234 greps backend.log -- verify
  its skip-guard before claiming a green suite post-truncate.

## Immutable success criteria (verbatim from masterplan 62.6)

1. "backend.log live file is under 50MB with rotation in place (newsyslog entry or
   copytruncate script wired to a schedule); the historical log is archived compressed,
   not deleted, pending the deferred FRED rotation"
2. "the autoresearch nightly cron exits 0 on a dry invocation with langchain_huggingface
   importable from ITS venv; ablation exit=1 is root-caused with the fix applied or the
   job documented-disabled with an audit note"
3. "masterplan step 39.1 is closed via its own immutable verification (cross-referenced,
   not duplicated)"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && test
$(stat -f%z backend.log) -lt 52428800 && source .venv/bin/activate && python -c "import
langchain_huggingface; print('lh OK')"

## Plan

1. Manual first rotation (cp -> truncate -> gzip into handoff/logs/) NOW (13:30 UTC,
   clear of job windows); verify backend still logging (fresh lines appear).
2. healthcheck.sh: size-gated rotation block (>50MB), same recipe, logged to the JSON
   line (rotated_log field).
3. Constrained install + record versions; add --preflight-only to run_memo.py; wire
   run_nightly.sh; dry invocation: run_nightly.sh path exits 0 with deps importable, $0.
4. Ablation evidence section in experiment_results (doc-only disposition).
5. 39.1: write live_check_39.1.md (evidence-by-output: launchd streak + root_cause.md +
   the recorded install), append its harness_log entry, flip 39.1 done IF Q/A accepts
   the lenient path; otherwise leave for the Monday PM session (strict path documented).
6. pending_tokens: AUTORESEARCH SPEND ask. 7. sector-cap test guard check.
8. experiment_results + live_check_62.6 -> ONE fresh Q/A -> harness_log -> flip 62.6.

## Out of scope

Resuming nightly LLM spend (operator token only); FRED rotation (deferred); any
masterplan criteria edits.
