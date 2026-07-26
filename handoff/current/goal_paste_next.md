Drain the pyfinagent masterplan. Full harness cycle per step: researcher -> contract -> GENERATE ->
ONE fresh Q/A -> harness_log append -> status flip. No self-eval. Read CLAUDE.md +
.claude/rules/research-gate.md first.

STATE (measured, HEAD 9ec84b7f, pushed):
- CLOSED today: 36.7, 80.40, 36.12 (all Q/A PASS, logged, flipped). 231 pending, 23 P0.
- Backend NOT restarted: launchd pid 76381, armed:true, peak_nav 24666.57, trailing_dd 3.36%.
  36.7 + 80.40 code is LIVE. 36.12's is COMMITTED BUT NOT LOADED (absent baseline_provenance on
  GET /kill-switch proves it) -- so the defect 36.12 fixes is STILL LIVE on the order-placing path.
  A restart now delivers it. That is an operator decision; do not restart without one.
- kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e (8 lines). VERIFY FIRST.

START HERE: 36.8, cycle 5. It has FAILED four times and every FAIL was right -- each found a LIVE
safety regression I shipped. Do NOT write a sixth patch.

  THE FOUR-TIMES-CONFIRMED PATTERN: `_read_audit_rows`'s `complete` flag is NEGATIVELY derived --
  it starts True and drops to False on each failure mode someone remembered. Five routes have now
  beaten it: (1) no gate at all, (2) gate keyed on the wrong signal, (3) chmod-000 dir (glob returns
  empty WITHOUT raising), (4) unparseable lines, (5) present-but-silent sources -- 0-byte file,
  ABSENT LIVE FILE (_audit_source_paths appends it only `if .exists()`), unglobbed name, nested
  subdir. The cycle-4 Q/A: "closing hole #5 by hand is the same move that produced holes #2-#5."

  THE DIRECTIVE, already written into experiment_results_36.8.md: an anchor may claim authority
  ONLY if it NAMES what it superseded -- `prior_peak` must be a value the writer actually observed,
  never None. All five routes produce an anchor from None, so none could ever be authoritative.
  That kills them structurally instead of by enumeration. Cross-check it against the research
  (research_brief_36.8.md): the authorized re-anchor is `peak_reset` (token-gated) and NO production
  path writes an intentional lower anchor, so the marker may be production-dead by construction --
  decide that explicitly, do not leave it ambiguous. Re-run the research gate before the redesign.

THEN: 36.9, 36.13 (P0: execute_buy has NO kill-switch gate; the MCP signals path bypasses it),
36.15, 36.16, 36.17, 36.18, 36.14, then the phase-80 tail and phase-79.

WHAT WORKED, keep doing it:
- Tell every Q/A that YOUR closure claim is the least reliable statement in the artifact, and name
  the positions you did NOT probe. That found 5 wiring holes on 36.12 and 5 routes on 36.8.
- Measure last, write once: run the whole mutation matrix in ONE batch AFTER the final test lands.
  Three artifacts went stale because I measured, then added a test.
- Regenerate artifacts, never edit numbers in place.
- "N killed" licenses only "these N were killed at this baseline" -- never "no vacuous guard remains".

TRAPS THAT COST REAL CYCLES:
- `dict.get(k) is None` is satisfied by a MISSING key -- two of my assertions could not fail.
- A mutant that cannot change behaviour is not evidence (I shipped one; disclose and rewrite).
- Test filenames must contain `kill_switch` or the immutable `-k` gate selects ZERO of them.
- Mutations that a test reads FROM DISK must be disk mutations; in-memory reads as a false survivor.
- Redirect ks._AUDIT_PATH to tmp BEFORE any exec that builds the module singleton (an evaluator
  wrote 54 rows into live safety state that way).
- alerting.py:167 posts to REAL Slack with no test guard; 17 false P1s reached #ford-approvals.
  Patch raise_cron_alert_sync in any harness reaching pause() or the disarmed path.
- Use PLAYWRIGHT_DIST_DIR (next.config.js:9) for any :3100 rig; a wrong var shares the operator's
  .next and 404s :3000/login. Probe /login, never just /.

DO NO HARM: paper only; no .env; no flag flips; historical_macro FROZEN; limits/stops/caps/DSR/PBO
byte-untouched; NO peak reset (owed token 79.6); never drive :3000; `git add -An` before EVERY flip
and commit each step separately first.

END OF SESSION: write the next goal prompt in chat, under 4000 chars.
