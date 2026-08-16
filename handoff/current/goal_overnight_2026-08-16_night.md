Binding: CLAUDE.md, .claude/rules/research-gate.md, MEMORY.md, this file.
OVERNIGHT AUTONOMOUS DRAIN -- 2026-08-16 ~20:45 through ~07:00 Oslo. You work
ALONE. Drain the masterplan; do not wait on Peder.

════════════════════════════════════════════════════════════════════════
§0. PREFLIGHT -- RUN THIS FIRST. IF IT FAILS, STOP AND DO NOT DRAIN.
════════════════════════════════════════════════════════════════════════
The operator's explicit concern: do not spend a night of tokens on a broken
development machine. So the harness proves itself BEFORE any step work.

Run all six. Every command below was executed at 20:39 and its expected output
is recorded, so a deviation is real and not a typo.

  source .venv/bin/activate
  node scripts/qa/verify_prompt_render_86_90.mjs      # ALL GREEN: 95 passed
  node scripts/qa/verify_research_gate_workflow.mjs   # ALL GREEN: 124 passed
  node scripts/qa/verify_escalation_86_78.mjs         # ALL CHECKS PASS
  node scripts/qa/verify_rail_retry.mjs               # ALL GREEN: 38 passed
  curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 \
       http://127.0.0.1:8000/api/health                # 200
  git status --short | grep -E "^(UU|AA|DD)" || echo "no merge conflicts"

KNOWN RED, NOT NEW BREAKAGE -- do NOT treat as a preflight failure:
  node scripts/qa/verify_workflow_args_boundary.mjs   # FAILED: 84 passed, 3 failed
That is step 86.92 itself (red since phase-86.37, cause proven unrelated).

ABORT CONDITIONS -- if ANY of these, write handoff/current/overnight_halt.md
naming which check failed with its verbatim output, commit, push, and STOP.
Do not "work around" a red preflight; a broken harness is exactly the state
this section exists to refuse.
  - any of the four green gates is no longer green
  - /api/health is not 200
  - a merge conflict exists in the working tree
  - the FIRST research-gate spawn of the night returns gate_passed:false twice

════════════════════════════════════════════════════════════════════════
§1. HARD RAILS -- these bound the night. They are not advisory.
════════════════════════════════════════════════════════════════════════
MEASURED 2026-08-16: 22 workflow runs, 4,472,219 tokens in ~11h (~406K/h), and
**0 steps closed**. An unbounded night at that rate is ~4.0M more tokens for a
measured close rate of zero. These rails exist because of that number.

R1. PER-STEP ATTEMPT CAP = 3, not 5. F1 allows 5; overnight you take 3. On the
    3rd non-PASS: PARK the step (status stays pending, annotate notes, append
    harness_log) and MOVE ON. One step must not eat the night.

R2. CIRCUIT BREAKER. If TWO CONSECUTIVE steps park at the cap without a PASS,
    STOP ALL STEP WORK. That is evidence the HARNESS is the blocker, not the
    steps -- which is precisely today's finding (8 of 15 verdicts said every
    criterion was MET and returned CONDITIONAL anyway). Write
    handoff/current/overnight_halt.md with both steps' verdicts, commit, push,
    and spend the remaining time on §4 instead.

R3. TOKEN CEILING = 3.0M workflow tokens for the night. Check it at every step
    boundary with the snippet in §5. On breach: finish the in-flight cycle, then
    stop starting new steps.

R4. METERED SPEND = IMMEDIATE STOP. Standing $0-metered constraint. If anything
    indicates usage credits rather than the Max pool, halt and report.

R5. NO GATE-SEMANTICS EDITS. .claude/agents/qa.md, .claude/workflows/qa-verdict.js
    and research-gate.js are OFF LIMITS tonight. Changing the thing that grades
    you, unsupervised, is the one action with no safe failure mode.

R6. Explicit pathspecs on every commit. NEVER git add -A, never git stash. A peer
    edits backend/api/sovereign_api.py and frontend/src/*. Flip the masterplan via
    python to bypass the auto-commit hook, then commit explicitly and run
    .claude/hooks/post-commit-changelog.sh by hand.

R7. Launch Layer-3 rails by scriptPath, never by name. Pass history as DATA in
    args.verdict_sequence. Keep spawn args LEAN -- two spawns died at the args
    boundary today with 0 agents and 0 tokens (86.96).

R8. PAPER ONLY. No flag promotion, no .env write, no manual cycle, and never
    loosen a gate to get green. When an immutable command goes RED mid-work,
    answer it by CLASSIFYING the new member, not by relaxing the check.

════════════════════════════════════════════════════════════════════════
§2. DO NOT START -- operator-blocked. Starting these wastes the night.
════════════════════════════════════════════════════════════════════════
  86.90, 86.91, 86.88  ESCALATED [CONDITIONAL x4]. Await Peder's decision:
                       handoff/current/escalation_86.90_86.91.md
  86.85                PARKED, and it is the BLOCKING DEPENDENCY for the
                       harness-termination work. Needs the operator's call.
  86.71                Depends on 86.85. Do not build on an unsourced counter.
  86.98                Criterion 7 REQUIRES an operator sign-off recorded in the
                       artifact. A Q/A PASS alone is deliberately insufficient.
                       Do not route around it.
  86.99                Edits qa.md -> gate semantics -> R5.
  86.84                Owed: separation-of-duties review.
  86.89                Cycle 2 returned FAIL. Its cycle-3 prerequisites are in
                       goal_next_2026-08-17.md item 0; not overnight work.

════════════════════════════════════════════════════════════════════════
§3. THE DRAIN -- in this order. Full harness loop on every one.
════════════════════════════════════════════════════════════════════════
research gate -> contract -> generate -> Q/A -> harness_log -> flip. No skipping.

1. 86.92 (P1) FIRST, because it restores a GATE and every later step is more
   trustworthy once it signals. verify_workflow_args_boundary.mjs has been RED
   since phase-86.37 for a reason unrelated to what it guards: section [3]
   asserts a "healthy run" against handoff/current/research_brief_86.17.md, a
   2026-08-09 brief with `grep -c brief_status` = 0, and 86.37 made that marker
   mandatory on 2026-08-10 (d3bb1dfb). It is the ONLY checker driving the args
   boundary of BOTH Layer-3 scripts. DO NOT loosen enforceGate to make an old
   fixture pass -- replace or pin the fixture.

2. 86.97 (P2) three bash exit-0 paths run BEFORE the changelog detector and emit
   no decision line (10 commits vs 5 lines), and the production CALL
   _log_decision(bump_type) is unguarded -- deleting it leaves the guard ALL
   GREEN because detector_source() collects only FunctionDef/Assign nodes.

3. 86.94 (P2) the now-relative-window class. `--since=<bare date>` is applied at
   the CURRENT time of day. Its criterion 1 was rewritten to forbid pinned
   figures -- do not reintroduce them.

4. 86.95 (P3) harness-self-audit.js:68 has the same concat shape 86.90 fixed.
   Small; good if a slot is short.

5. 86.96 (P2) the args channel. BISECT the trigger from the two failing payloads.
   Do NOT write down size or escaped quotes as the cause -- both are already
   contradicted by cycle-2 payloads that were large, contained escaped quotes and
   parsed fine.

6. 86.87 (P2) the lite risk_assessment fabricates its own audit trail.

7. 86.93 (P2) only if time remains. Measure SUBJECT IDENTITY per step first --
   a re-grade of a step whose subject has moved is not a re-grade.

════════════════════════════════════════════════════════════════════════
§4. IF THE CIRCUIT BREAKER TRIPS (R2) -- do this instead of more steps
════════════════════════════════════════════════════════════════════════
Do NOT keep spawning. Spend the remaining hours on evidence the operator can
act on in the morning:
  a. For each parked step, a one-page summary: which criterion was actually
     missed vs which finding was a quality gap. That distinction is the input to
     86.98 and nobody has assembled it.
  b. Extend today's 15-verdict measurement across ALL sessions in
     ~/.claude/projects/<slug>/*/workflows/*.json -- verdict, whether the reason
     states criteria MET, and finding count. A population figure would move 86.98
     from one session's evidence to the repo's.
  c. Do NOT implement 86.98. Measure for it.

════════════════════════════════════════════════════════════════════════
§5. BUDGET CHECK -- run at every step boundary
════════════════════════════════════════════════════════════════════════
FIRST, at preflight, stamp the baseline ONCE:

  date +%s > /tmp/pyfin_night_start

Then at every step boundary:

  python - <<'PY'
  import json, glob, os
  base = os.path.expanduser('~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent')
  start = float(open('/tmp/pyfin_night_start').read().strip())
  tot = n = 0
  for p in glob.glob(base + '/*/workflows/wf_*.json'):
      try: r = json.load(open(p))
      except Exception: continue
      if (r.get('startTime') or 0) / 1000 >= start:
          tot += r.get('totalTokens') or 0; n += 1
  print(f'night so far: {n} runs, {tot:,} tokens  (ceiling 3,000,000)')
  if tot > 3_000_000:
      print('  CEILING BREACHED -> finish the in-flight cycle, then stop starting steps')
  PY

A ROLLING WINDOW WOULD BE WRONG HERE and the first draft of this goal had that
bug: a 14-hour lookback sweeps in the 4,472,219 tokens the DAY session already
spent, so the very first check reports a breach before the night has begun.
Baseline from a stamp, never from `now - N hours`.

════════════════════════════════════════════════════════════════════════
§6. DISCIPLINE -- today's findings, not generic advice
════════════════════════════════════════════════════════════════════════
- A claim about an ARTIFACT is true only once you have opened that artifact.
  Serialise it and diff the two states.
- A no-match str.replace looks identical to success. Assert the bytes changed.
- A guard covering N duplicated sites with ONE mutation cannot see a regression
  in one of them. Mutate each site separately.
- A cardinality floor catches a DELETED assertion, never a NEUTERED one.
- An N-id fixture is defeated by an N-id whitelist. Use a runtime-derived value
  and STATE the residual.
- A mutant that does not BUILD is UNSCORABLE, not KILLED. Score three outcomes.
- Prove the probe is LIVE before believing a clean result.
- A fingerprint must EXCLUDE the region it was derived from.
- Never wire a disk-mutating checker into the file it mutates -- `git add -A` is
  one interrupt away from committing a truncated source file.
- Regenerate captures from a live run. Derive counts; never type them.
- "Queued" in prose is not queued. File the step in the SAME turn.
- When a subagent reports a defect in its own INPUT, that is a harness finding.

════════════════════════════════════════════════════════════════════════
§7. MORNING HANDOFF -- do this before you stop, whatever happened
════════════════════════════════════════════════════════════════════════
  1. Day report: handoff/current/day_report_2026-08-17.md -- steps closed, steps
     parked, tokens spent (MEASURED with §5, never estimated), and every claim
     you could not verify.
  2. Regenerate handoff/current/goal_next_2026-08-17.md from what actually
     happened.
  3. git push origin main and confirm origin == HEAD.
  4. State plainly at the top whether the preflight passed, whether the circuit
     breaker tripped, and how many steps actually reached PASS. If the answer is
     zero, say zero -- that is the number the operator needs to decide on 86.98.
