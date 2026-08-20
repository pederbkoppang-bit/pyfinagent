STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.120
WRITTEN: 2026-08-18T08:41:20Z

# Q/A write-first record -- step 86.120, Cycle 3 (re-grade on changed evidence)

Role: Layer-3 Q/A, Workflow rail. Read .claude/agents/qa.md in full at 08:41:20Z.

## Prior-attempt evidence (gathered, not a trigger)
- `qa_wip.py 86.120 --spawned-at 2026-08-18T08:41:20Z`: source_present=true,
  attempt_number=3 (status "ok", is_lower_bound=true), prior_attempts=2,
  records_retained=3 (gauge, incl. mine).
- `verdict_history_86_21.py --step 86.120 --evidence-only`:
  status = **no_rows_for_step**, verdicts = (none).
- CROSS-CHECK: prior_attempts (2) > ledger rows (0) -> **LEDGER IS STALE for
  this step**; sequence: UNKNOWN from the authoritative source. Main's
  advisory disclosure (C1=CONDITIONAL, C2=CONDITIONAL) recorded as ADVISORY
  ONLY. I did not scan prior_records bodies for verdict words.

## A. Harness compliance -- CLEAN (5/5)
1. research-gate-before-contract: research_brief_86.120.md exists (36,779 B,
   09:31:11) < contract (10:15:25). Envelope: brief_status COMPLETE,
   external_sources_read_in_full=6 (>=5), urls_collected=37 (>=10),
   recency_scan_performed=true (section present, 4 findings), gate_passed=true.
2. contract-before-generate: OK. Production mtime 10:37:47 > contract, which is
   the mutation apply/restore cycle -- sha256 proves zero net change.
3. experiment_results_86.120.md present (16,575 B).
4. log-last: `git diff .claude/masterplan.json` = +73 lines only, adding
   86.120/86.121/86.122 all `"status": "pending"`. NOT flipped.
5. no-verdict-shopping: evidence CHANGED (30 -> 31 tests; new test
   `test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling`,
   mtime 10:38:14 > critique 10:36:45; prose corrections). Legitimate cycle-N.

## B. Deterministic -- ALL GREEN
- IMMUTABLE CMD: 31 passed, 1 warning, 2.24s; **bare exit 0** (re-run twice,
  incl. AFTER the mid-eval peer commit). 31 progress dots == "31 passed" (no splice).
- sha256 claude_code_client.py = 76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1
  -- MATCHES the claimed byte-identical hash, re-verified post-peer-commit.
- ruff F821/F401/F811 on the step's 3 files: **All checks passed! exit 0**.
  (Derived-scope run also flagged 1 F401 in scripts/qa/rank_stability_86_59.py
  -- PEER 86.59 file, symbol present at HEAD:100 and :845, NOT this step's.)
- Runtime smoke: `import backend.agents.claude_code_client` OK;
  rail_guard_status() keys include cooldown_active; cooldown_status() =
  {'cooldown_active': False} (no leaked state); Settings field default 6.0
  with Ge(0.5)/Le(192.0).
- Wider sweep `-k "claude_code or rail or llm_client or routing or settings"`:
  293 passed, 1 failed = test_phase_40_2_...json_still_valid_json_after_edit
  (asserts effortLevel=='xhigh'; CLAUDE.md records the operator raised it to
  'max' 2026-08-04). PRE-EXISTING, unrelated, .claude/settings.json not in diff.
- Scope: only claude_code_client.py + settings.py + the new test file. Untracked
  backend/agents/_cache/ holds ONLY a self-scoping .gitignore ("*\n!.gitignore");
  git check-ignore confirms cc_rail_cooldown.json is ignored.
- Consumer grep on rail_guard_status(): llm_client.py:2157 and
  autonomous_loop.py:1951 both use `.get()`; test_phase_72_0_2:131's exact-dict
  compare is symmetric. Additive keys => no consumer-contract break.
- MID-EVAL HEAD MOVE: 8af51173 -> ecb709b3 (peer 86.59 cycle-5). `git diff
  --name-only 8af51173..HEAD` touches NONE of 86.120's scope; 86.120's files
  remain uncommitted (M/M/??). Re-verified immediately before returning.

## C. INDEPENDENT mutation matrix -- 13 source-level cells, run BY ME
Method: read the real source, mutate IN MEMORY, inject into sys.modules before
pytest collection. The disk tree is NEVER touched (no restore step needed, no
window in which a peer `git add -A` could commit a mutant). Each anchor asserted
to occur EXACTLY once. CONTROL (M0, zero edits) through the same harness:
**31 passed, RC=0** -- so the harness neither over- nor under-kills.

| # | Mutation (claude_code_client.py unless noted) | Result |
|---|---|---|
| 1 | M0 control -- no edit | 31 passed (GREEN) |
| 2 | M17 `active = True` -> `False` (corrupt-record fail-safe) | **KILLED** 1 failed/30 passed -- exactly `test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling`. **Cycle-2 blocker independently CONFIRMED CLOSED.** |
| 3 | delete the pre-subprocess cooldown guard (`if False:`) | **KILLED** 5 failed (incl. the 2-cycle+restart test and the signal-agent test) -- criterion 6 proven by a REAL deletion, stronger than the in-suite proxy |
| 4 | `_limit = classify_limit_failure(...)` -> `None` | **KILLED** 2 failed -- criterion 5 |
| 5 | `cooldown_record_hit` never persists | **KILLED** 14 failed |
| 6 | drop `cooldown_clear_on_success()` from the success path | **KILLED** 1 failed |
| 7 | `_rail_guard_open_for_quota` -> no-op | **KILLED** 1 failed |
| 8 | always feed `_rail_guard_record_failure` (drop cooldown_blocked bypass) | **KILLED** 1 failed |
| 9 | M6 tz fallback -> `now.tzinfo` | **KILLED** 1 failed |
| 10 | remove the past/implausible retry_at clamp | **KILLED** 1 failed |
| 11 | `status.update(cooldown_status())` removed from rail_guard_status | **KILLED** 1 failed |
| 12 | `cooldown_clear_on_success` never clears | **KILLED** 2 failed |
| 13 | **llm_client.py**: routing-breach `raise ValueError` unreachable | **KILLED** 1 failed -- criterion 9's regression guard is load-bearing |
| 14 | **delete the JSON-envelope `result` extraction in `classify_limit_failure`** | **SURVIVED -- 31 passed, RC=0** |

### The survivor is NOT an equivalent mutant (measured, twice)
Differential A -- on the ONE REAL captured envelope in the repo
(`handoff/away_ops/session_pm_20260707T200007Z.json`, 769 B, phrase at offset 145):
- REAL   raw_message len=56  -> "You've hit your session limit - resets 1am (Europe/Oslo)"
- MUTANT raw_message len=500 -> '{"type":"result","subtype":"success","is_error":true,...'
That string is persisted into the cooldown record and surfaced as
`cooldown_message` through `rail_guard_status()` -- the criterion-10 seam.

Differential B -- scan-scope narrowing. Envelope where the limit sentence sits
OUTSIDE `result` (a SUCCESSFUL call: result="AAPL: BUY, confidence 0.81"):
- REAL   classified = None   (correct)
- MUTANT classified = weekly (would engage a >=6h persisted cooldown and take
  the analyst rail down on a successful call)

### Why the suite cannot see it
`test_classify_reads_full_message_not_the_150_char_truncation` ADVERTISES this
property -- docstring: "Prove the classifier reads the untruncated stdout, not a
pre-truncated string". Measured under both arms: envelope len 420, "session
limit" at offset 389; `assert "session limit" in c.raw_message` is True for BOTH
(raw_message len 45 real vs 420 mutant). The assertion discriminates a 150-char
truncation only, never the envelope extraction. Vacuity shape #3 -- the scanned
literal survives while the behaviour the test NAMES is removed.

### Why the author's criterion-11 claim does not hold
experiment_results §"Success criteria coverage" row 11 argues MET as:
"Q/A cycle-1 found 3 guards (M10/M3/M6) ... cycle-2 found a 4th (M17), now also
closed". That is a CLOSED-LIST argument over the guards a previous evaluator
happened to find, not the criterion's universal text ("every new guard"). The
population of new guards was never enumerated; a 5th member falls out of a
13-cell independent sweep. Three consecutive cycles, three findings, none from
the author's own matrix.

## Criterion-by-criterion
1 MET  -- classify_limit_failure covers all 3 documented forms + plaintext
          fallback; `test_classify_returns_none_for_generic_failure` is the
          control; cell 4 kills. Contract's "overlapping, not a subset" reading
          note re run_away_session.sh:242 INDEPENDENTLY VERIFIED CORRECT: shared
          `session limit`; shell-only `usage limit`/`out of credit`/`credit.*exhaust`;
          new-only `weekly`/`Opus`. Honest disclosure, not an overclaim.
2 MET  -- disk JSON; cell 5 kills 14.
3 MET  -- guard at the top of claude_code_invoke; cell 3 kills 5 incl.
          `test_generate_content_skips_subprocess_across_two_cycles_and_a_restart`.
4 MET  -- claude_rail_cooldown_default_hours 6.0 / 0.5..192.0, documented;
          expiry + success-clear proven by cells 6 and 12.
5 MET  -- cell 4.
6 MET  -- cell 3 (real deletion).
7 MET  -- generic path byte-unchanged for non-limit failures (getattr default
          False => `not False` => records, as before); cells 8 + the two
          generic-breaker tests.
8 MET  -- `test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point`
          killed by cell 3. Verified pead_signal.py:300 is verbatim
          `make_client(getattr(settings, "pead_signal_model", "claude-haiku-4-5"),
          None, settings, enable_prompt_caching=False)` -- the citation is accurate.
9 MET  -- cell 13.
10 MET -- rail_guard_status() carries cooldown_active/kind/hit_at/retry_at
          (verified live); cell 11 kills.
11 **NOT MET** -- cell 14: a new guard whose mutant survives 31/31.

## NOTE-level (do NOT degrade the verdict)
- N1 contract Plan step 1 says the classifier "inspects `api_error_status` +
  `result`"; the shipped code inspects `result` only. Harmless (arguably safer --
  429 is not exclusively a quota signal) but the plan says "will NOT diverge".
- N2 the accelerated N=1 breaker trip interacts with phase-72.0.2 fail-forward:
  once `paper_rail_failforward_enabled` is promoted (DARK/false today), a
  classified quota hit routes to METERED Vertex at N=1 instead of N=20. Not a
  criterion violation and plausibly desired; worth queueing as an observation.
- N3 pre-existing unrelated red: test_phase_40_2 effortLevel xhigh vs max.
- N4 peer 86.59 F401 in scripts/qa/rank_stability_86_59.py, present at HEAD.

## Named fix for the single blocker
Add to the suite (either or both):
  (a) `c = classify_limit_failure(REAL_ENVELOPE_TEXT); assert not
      c.raw_message.lstrip().startswith("{")` -- or assert equality with the
      exact sentence -- pinning that the persisted/operator-facing message is
      the sentence, not the JSON slab; and
  (b) a false-positive control: an envelope whose limit phrase sits OUTSIDE
      `result` must return None.
Then re-run cell 14 and confirm it goes RED.

COMPLETED: 2026-08-18T08:52:49Z
(self-correction: an earlier draft of this line carried a time I had not read
from the clock -- replaced with the actual `date -u` reading.)
