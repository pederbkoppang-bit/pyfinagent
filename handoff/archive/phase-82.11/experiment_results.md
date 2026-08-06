# Experiment Results -- phase-82.11 (cycle 1)

**Step:** 82.11 (P1). **Date:** 2026-08-06.
**Contract:** `handoff/current/contract_82.11.md`.
**Research brief:** `handoff/current/research_brief_82.11.md` (`gate_passed: true`).

---

## 1. What was built

### D1 -- the rail exit (implements the operator's recorded decision)

`scripts/autoresearch/run_nightly.sh:78` (pre-edit line number):
`${AUTORESEARCH_USE_MAX_RAIL:-0}` -> `${AUTORESEARCH_USE_MAX_RAIL:-1}`, plus a
comment block recording the operator's verbatim instruction, the measured
justification, and the one-line revert. **Nothing else in that file changed** --
the `/health` preflight, the loud-fail `exit 78`, the dummy-key override and
`_record_fail_and_page` are byte-identical.

Why the default and not `backend/.env`: `.env` is gitignored (`.gitignore:5`),
so an `.env` flip would be invisible to review and to any future audit. The
default lives in tracked code; the env var still overrides it, so the operator's
revert is unchanged in cost.

### D1a -- the superseded phase-76.9.2 pin, disclosed not hidden

`backend/tests/test_phase_76_9_2_max_bridge.py` deliberately pinned the default
OFF. Two of its tests observed that default and both were updated:

- `test_nightly_flag_off_is_inert` -- its fixture omitted the flag and leaned on
  the default, so it was silently doubling as a default-pin. Now sets
  `AUTORESEARCH_USE_MAX_RAIL=0` explicitly, which restores the guard to what its
  name claims ("flag OFF is inert") and keeps its protective value intact.
- `test_nightly_default_documented_off` -> `test_nightly_default_documented_on`.
  Repinned to `:-1`, **plus** a new assertion that `:-0` does not linger
  anywhere in the file. Its two other assertions (loud-fail echo lives in
  executed code, `exit 78` present) are untouched.

This is a deliberate supersession of a prior step's decision on the operator's
standing instruction. It is stated in the contract, in the test docstring, and
here.

### D2 -- the audible, escalating, Python-drivable notification path

New: `backend/services/autoresearch_health.py` (**387 lines** --
`wc -l backend/services/autoresearch_health.py`), mirroring the phase-82.10
`freshness_cron.py` module shape.

- `classify_failure(exc)` -- narrow, in the style of `run_memo._is_network_weather`;
  walks the `__cause__`/`__context__` chain. Returns `credit_exhausted` / `auth`
  / `generic`.
- `count_consecutive_failures(memo_dir, today)` -- walks backwards over failing
  DATES; a date failed iff it has a `*-ERROR-*` memo AND no success memo. Stops
  at the first non-failing date.
- `report_run_outcome(...)` -- the ladder + edge trigger, fail-open.

Emitter: `raise_cron_alert_sync` from
`backend/services/observability/alerting.py`, imported **function-locally**, at
`P1`/`P0` only (a `P2` is logged and dropped by `alerting.py:219-224` while
`slack_webhook_url` is empty). **No second notifier was built.**

Ladder (contract §5): config-class failures page at n=1 because they are
immediately actionable and never self-heal; generic failures wait for
`PAGE_AFTER_N=3`; both escalate at `ESCALATE_AFTER_N=7` with a **distinct
`error_type`** so `AlertDeduper` keys the escalation separately; steady state is
silent; a slow `REMIND_EVERY_DAYS=7` safety net keeps a persisting fault from
going quiet forever. A successful run emits nothing at all.

### D3 -- wiring into the production entry point

`scripts/autoresearch/run_memo.py`: new `_report_outcome()` helper (function-local
import, fail-open, never changes the exit code) called from **both** branches of
`_main_async` -- in the failure branch **after** the ERROR file is written so the
count includes tonight, and in the success branch.

### D4 -- the guard file

`backend/tests/test_phase_82_11_autoresearch_failure_paging.py`, 20 tests.

---

## 2. Verbatim verification command output

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_11_autoresearch_failure_paging.py -q
....................                                                     [100%]
20 passed in 0.15s
```

Regression sweep over every suite that touches the changed files:

```
$ python -m pytest backend/tests/test_phase_76_9_2_max_bridge.py \
    backend/tests/test_phase_76_9_launchd_fixes.py \
    backend/tests/test_phase_39_1_autoresearch_env.py \
    backend/tests/test_phase_75_deps.py backend/tests/test_phase_51_4_crons.py \
    backend/tests/test_phase_82_10_freshness_paging.py \
    backend/tests/test_phase_82_11_autoresearch_failure_paging.py -q
80 passed, 1 warning in 11.31s
```

Syntax:

```
OK backend/services/autoresearch_health.py
OK scripts/autoresearch/run_memo.py
OK backend/tests/test_phase_82_11_autoresearch_failure_paging.py
OK backend/tests/test_phase_76_9_2_max_bridge.py
OK run_nightly.sh syntax          # bash -n
```

---

## 3. LIVE evidence

### 3a. The bridge answers a real Anthropic request (measured before any edit)

```
$ curl -sf -m 5 http://127.0.0.1:18797/health
{"ok":true,"proxy":"claude-code-cli"}

$ curl -s -m 90 -X POST http://127.0.0.1:18797/v1/messages \
    -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
    -d '{"model":"claude-haiku-4-5","max_tokens":32,
         "messages":[{"role":"user","content":"Reply with exactly: BRIDGE_OK"}]}'
{"id": "msg_1785995776470", "type": "message", "role": "assistant",
 "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "BRIDGE_OK"}],
 "stop_reason": "end_turn", ...}
```

### 3b. The REAL `run_nightly.sh`, no env flag, against the LIVE bridge

Fixture repo via the `AUTORESEARCH_REPO` test seam; its `.env` contains a real
metered-looking key and **no `AUTORESEARCH_*` key at all** (`grep -c AUTORESEARCH`
-> `0`), i.e. the measured shape of the production `backend/.env`.

```
=== running the REAL run_nightly.sh against the LIVE bridge, default flag ===
rc=0
=== observed env inside run_memo ===
{
  "ANTHROPIC_API_URL": "http://127.0.0.1:18797",
  "ANTHROPIC_BASE_URL": "http://127.0.0.1:18797",
  "ANTHROPIC_API_KEY": "max-rail-dummy-key"
}
=== log ===
[2026-08-06T08:16:05+02:00] START nightly autoresearch
[2026-08-06T08:16:05+02:00] max-rail ON -- routing via http://127.0.0.1:18797 (dummy key, $0 metered)
[2026-08-06T08:16:05+02:00] END nightly autoresearch OK
```

The real metered key was overridden by `max-rail-dummy-key`, so any leak to
`api.anthropic.com` 401s -- provable `$0` metered.

### 3c. Baseline measurements this step relies on (all re-derived by Main)

| Fact | Value | How |
|---|---|---|
| ERROR dates in `handoff/autoresearch/` | 62 | structural scan of `*.md` filenames |
| Success memo dates | 2 | same scan |
| Dates carrying BOTH | `['2026-07-24','2026-07-25']` | same scan |
| **Consecutive** failing dates to 2026-08-06 | **12** | walk back, stop at first non-failing date |
| `autoresearch_fail_state.json` | `{"consecutive_fails": 13}` | file read |
| `launchctl list` autoresearch last exit | `1` | `launchctl list` |
| `backend/.env` `AUTORESEARCH_*` keys | **none** | key-name scan (values never printed) |
| `SLACK_BOT_TOKEN` / `SLACK_CHANNEL_ID` | present, non-empty | key-name scan |

---

## 4. Mutation matrix

Script: `scratchpad/mutate_82_11.py`. Every mutant asserts its target text
exists **before** the replace and hashes the file **after**, so a no-op replace
cannot masquerade as an applied mutant. Every mutant restores the file and the
run ends by re-verifying the restored tree is green.

| # | Mutant (production call site) | Result | Killed by |
|---|---|---|---|
| M1 | delete `_report_outcome(True, e, topic)` from `_main_async` | KILLED | `test_credit_exhaustion_emits_operator_alert`, `test_production_entry_point_actually_reports_both_outcomes` |
| M2 | severity -> `"P2"` (which `alerting.py` drops) | KILLED | criterion-1 + criterion-2 guards |
| M3 | same `error_type` at every tier | KILLED | `test_prior_failures_escalate_...` |
| M3b | same `severity` at every tier | KILLED | `test_prior_failures_escalate_...` |
| M4 | make the emitter unconditional (fire on success too) | KILLED | `test_success_short_circuits_before_the_ladder` |
| M5 | suppress the tier-crossing edge trigger | KILLED | 9 tests |
| M6 | count TOTAL error files instead of the consecutive run | KILLED | `test_consecutive_is_not_total_...`, `test_live_memo_directory_...` |
| M7 | revert the rail default to `:-0` | KILLED | `test_rail_decision_is_implemented_in_tracked_code` |
| M8 | paraphrase the operator instruction in the artifact | KILLED | `test_operator_decision_recorded_verbatim_...` |
| M9 | drop the recorded DECISION line | KILLED | same |
| M10 | hoist the emitter import to module scope | KILLED | `test_wrong_patch_target_does_not_exist` |
| M11 | make `classify_failure` a catch-all | KILLED | 3 tests |
| M12 | drop the WARN exclusion | KILLED | `test_warn_files_are_not_failures` |

**13 of 13 mutants died.** That licenses exactly the claim "these 13 mutants
were killed" -- it is NOT a claim that no survivor exists.

A separate, additive recall test on the behavioural rail guard: reverting
`run_nightly.sh` to `:-0` and running only
`test_default_takes_the_max_rail_branch_with_no_env_flag` produced
`AssertionError: ... got rc=0` (the metered path ran `run_memo`), confirming
that guard observes real behaviour rather than passing vacuously.

### The survivor I found and fixed mid-cycle

**M4 survived the first matrix run.** Making the emitter unconditional left the
suite green, because `test_successful_run_emits_no_alert` runs after the
production path has written today's success memo -- so the consecutive count is
0 and the ladder would have been silent anyway. My criterion-3 guard was
passing for a benign reason rather than because the success branch
short-circuits. Fixed by adding
`test_success_short_circuits_before_the_ladder`, whose fixture puts an ERROR
file dated TODAY on disk as well (the real 2026-07-24/25 shape) so the ladder is
sitting on a tier crossing, with a positive control proving that same directory
DOES fire when the run failed. M4 then died.

---

## 4b. Cycle-2 corrections (Q/A CONDITIONAL -> fixed)

The cycle-1 Q/A returned CONDITIONAL with all four criteria MET and three
blockers, all of them prose. Verbatim verdict:
`handoff/current/evaluator_critique_82.11.md` (+ raw return at
`handoff/current/qa_returns/82.11_cycle1.output.json`).

- **B1** "349 lines" -> measured **387**. Fixed, with the command shown.
- **B2** `+30/-11` reported for two different files -> measured **22/8** and
  **27/3**. The Q/A's diagnosis is correct: `30` is `git diff --stat`'s
  total-changed column and `11` is the repo-wide deletion total, mis-applied
  per file. The whole table is now transcribed from derived output.
- **B3** "gets its own research-gated masterplan step" -> **no such step
  existed**. Fixed by making the claim true: 82.49 is queued (§8).
- **B4 (found by my own class sweep, not by the Q/A)** -- this step's edit added
  35 lines above `_main_async`, so several `run_memo.py:NNN` anchors in the
  contract went stale as soon as the code landed. Re-derived; the contract now
  says *re-derive* rather than carrying numbers that this step itself moves.

No production code, no test and no criterion changed in cycle 2.

## 5. Corrections made during this cycle (stated, not buried)

1. **The step description's premise is half wrong.** "the loop writes a failure
   file and exits silently, so 59 failures produced zero operator signal" -- a
   bash paging seam exists (`run_nightly.sh:49-69`) and it *ran* last night
   (fail-state mtime 02:00:13, `13 >= PAGE_AFTER_N=3`). What is true is that its
   result is discarded (`>/dev/null 2>&1 || true`), Slack returns HTTP 200 on
   `{"ok":false}`, it never escalates, and no pytest can capture it.
2. **The research brief's "30 consecutive ERROR dates" is wrong by the brief's
   own rule.** Re-derived structurally: **12** (2026-07-26..2026-08-06), stopping
   at 2026-07-25 which carries both file kinds. The brief's other two numbers
   (13, 62) reproduced exactly.
3. **I retracted my own "gap-safe" claim.** An earlier draft of the contract and
   the module docstring said the edge trigger was gap-safe ("a skipped night
   that jumps 2 -> 4 still fires"). That is FALSE: the counter walks failing
   DATES and stops at the first non-failing one, so a missed night RESETS rather
   than jumps. The claim is retracted in both files and the real behaviour is
   now pinned by `test_a_missed_night_resets_the_counter_and_the_ladder`, which
   also records the consequence (a missed night silently rewinds the ladder).
4. **The research brief said the rail flip "cannot be 82.11's code
   deliverable" because it is an operator-owned `.env` line. I rejected that**
   and put the default in tracked code instead -- see D1.

---

## 6. Files changed

Every number below is DERIVED, not typed. New-file sizes come from `wc -l`;
modified-file counts come from `git diff --numstat` (added/deleted per file);
the test count comes from an AST walk, not a grep, because grep cannot tell a
test function from a helper.

```
$ wc -l backend/services/autoresearch_health.py backend/tests/test_phase_82_11_autoresearch_failure_paging.py
     387 backend/services/autoresearch_health.py
     638 backend/tests/test_phase_82_11_autoresearch_failure_paging.py

$ git diff --numstat -- scripts/autoresearch/run_memo.py scripts/autoresearch/run_nightly.sh backend/tests/test_phase_76_9_2_max_bridge.py
27      3       backend/tests/test_phase_76_9_2_max_bridge.py
35      0       scripts/autoresearch/run_memo.py
22      8       scripts/autoresearch/run_nightly.sh

$ python3 -c "import ast; t=ast.parse(open('backend/tests/test_phase_82_11_autoresearch_failure_paging.py').read()); \
print(len([n.name for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]))"
20
```

| File | Change |
|---|---|
| `backend/services/autoresearch_health.py` | NEW, 387 lines |
| `backend/tests/test_phase_82_11_autoresearch_failure_paging.py` | NEW, 638 lines, 20 test functions |
| `scripts/autoresearch/run_memo.py` | `+35 / -0` -- `_report_outcome()` + 2 call sites |
| `scripts/autoresearch/run_nightly.sh` | `+22 / -8` -- default `:-0` -> `:-1` + rationale comment |
| `backend/tests/test_phase_76_9_2_max_bridge.py` | `+27 / -3` -- two default-observing tests repinned (D1a) |
| `.claude/masterplan.json` | `+1 step` -- queues 82.49 (see §8) |
| `handoff/current/contract_82.11.md`, `contract.md` | contract (+ the corrections above) |
| `handoff/current/research_brief_82.11.md` | research gate |
| `handoff/current/evaluator_critique_82.11.md`, `qa_returns/82.11_cycle1.output.json` | Q/A verdict, verbatim |

No production file outside this list was touched.

The rest of the dirty tree does NOT belong to this step and is not staged --
commits use deliberate `git add <paths>`, never `git add -A`. Corrected after the
cycle-2 PASS (Q/A NOTE 1): an earlier version of this paragraph called that set
"untracked noise", which was wrong -- several members are TRACKED-MODIFIED, and
the paragraph named a glob instead of deriving the set. That is the same
claim-about-an-underived-set class as B1-B4. Derived with `git status --short`
at commit time:

```
TRACKED-MODIFIED, not this step's:
  .claude/.archive-baseline.json
  .claude/agent-memory/qa/MEMORY.md
  .claude/agent-memory/researcher/MEMORY.md
  handoff/.cycle_heartbeat.json
  handoff/audit/instructions_loaded_audit.jsonl
  handoff/audit/pre_tool_use_audit.jsonl
  handoff/away_ops/auth_probe_last.json
  handoff/away_ops/autoresearch_fail_state.json
  handoff/away_ops/health.jsonl
  handoff/cycle_history.jsonl
  handoff/kill_switch_audit.jsonl
  handoff/prompt_leak_redteam_audit.jsonl
UNTRACKED, not this step's:
  .claude/agent-memory/qa/feedback_two_mutant_forms_separate_artifact_from_kill.md
  .claude/agent-memory/researcher/project_autoresearch_paging_82_11.md
  .claude/agent-memory/researcher/project_dsr_trial_count_reset_82_25.md
  .claude/agent-memory/researcher/project_exit_quality_ratio_blowup_82_5.md
  .claude/agent-memory/researcher/project_freshness_alarm_browser_driven_82_10.md
  .claude/agent-memory/researcher/project_macro_preload_refusal_82_13.md
  .claude/agent-memory/researcher/project_non_forward_labels_82_16.md
  .claude/agent-memory/researcher/project_pbo_level_and_dead_gate_82_27.md
  .claude/agent-memory/researcher/project_phantom_columns_82_39.md
  .claude/agent-memory/researcher/project_vacuous_bq_guards_82_12.md
  handoff/autoresearch/2026-08-05-ERROR-topic07.md
  handoff/autoresearch/2026-08-06-ERROR-topic08.md
  threshold
```

None is a production file and none is in this step's scope.

---

## 7. Behaviour change to the running system

- **From tonight's 02:00 launchd run**, autoresearch routes through the Max-rail
  bridge instead of the metered direct API. If the bridge is down at 02:00 the
  run exits 78 and pages -- it can never silently reach the metered API.
- **A nightly failure now emits a `P1`** through the same Slack channel the rest
  of the system pages on, escalating to `P0` at 7 consecutive nights with a
  distinct `error_type`, silent in steady state, with a 7-day reminder.
- **Disclosed:** at today's live state (12 consecutive failures, config-class),
  the ladder is already at tier 2 and steady, so shipping this emits nothing
  immediately. The next signal would be the 7-day reminder -- or silence,
  because the rail fix should make tonight succeed. That is the intended
  behaviour, recorded here so nobody reads the silence as a broken alarm.
- **Disclosed cost:** the Max rail is `$0` *metered* but draws the shared weekly
  Max plan pool. The operator's constraint is specifically `$0 metered`, which
  this satisfies.

## 8. Non-scope, and the defect queued out of it

The **dead-man's-switch hole** is real, was surfaced by the recency scan, and is
NOT fixed here. `run_memo.py` has three paths that return 0 having written no
memo and emitted no alert -- positions re-derived AFTER this step's edit shifted
them (the helper added 35 lines above `_main_async`):

```
$ grep -n 'warn_path = MEMO_DIR\|_skip_msg = _embedding_preflight\|if args.preflight_only' scripts/autoresearch/run_memo.py
204:            warn_path = MEMO_DIR / f"{dt.date.today().isoformat()}-WARN-topic{idx:02d}.md"
341:    _skip_msg = _embedding_preflight()
348:    if args.preflight_only:
```

and a missed night rewinds the ladder (§5 point 3). None of it is reachable by
82.11's four criteria.

**It is now queued as masterplan step 82.49** (P2, `harness_required: true`, 5
immutable criteria), written for an executor with no memory of this discovery:
it restates the measurements, re-derives the line numbers as a task rather than
citing these, names the reuse targets, and pre-registers the obvious trap (a
naive "no memo today" check fires every morning before 02:00 and forever if the
job is intentionally disabled). Verify with:

```
$ python3 -c "import json; d=json.load(open('.claude/masterplan.json')); \
s=[x for x in d['phases'][105]['steps'] if str(x.get('id'))=='82.49']; \
print(len(s), s[0]['status'], len(s[0]['verification']['criteria']))"
1 pending 5
```

The bash paging seam is deliberately left in place so there is never a window
with no paging at all; retiring it is a follow-up.
