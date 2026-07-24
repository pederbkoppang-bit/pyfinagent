# Experiment results — Step 75.4.2 (skill_optimizer post-write delivery invariant)

Date: 2026-07-24. Execution model: **sonnet-tagged → delegated Sonnet executor
GENERATE** (operator tiering directive; first delegated executor this session),
Main review + mutation matrix; Researcher gate opus/max (wf_91b00dc2-3ea, PASSED,
6 read-in-full — DbC/DSPy/APE/OPRO/VeriGuard canon; brief carried VERBATIM
implementation code the executor followed).

## What was built

1. **`backend/agents/skill_optimizer.py`** — unconditional, fail-CLOSED delivery
   postcondition on `apply_modification`:
   - module-level `DELIVERY_MIN_RETAIN_RATIO = 0.80`, `_PLACEHOLDER_RE`, and
     `_delivery_invariant_ok(before, after) -> (ok, reason)` with two INDEPENDENT
     guards: placeholder-set subset (no delivered `{{placeholder}}` may vanish)
     and 80% length retention (catches heading-promotion truncation of
     placeholder-free body);
   - pre-write baseline `delivered_before = load_skill(agent)` (fail-closed skip
     if the baseline itself cannot be established);
   - post-write: the existing load-exception revert is KEPT, then the invariant
     runs on (before, after); on violation → byte-exact revert via the in-function
     `skill_path.write_text(content)` pattern (never `revert_modification`'s
     `git checkout`), reload + cache invalidate, `return False` — all BEFORE the
     `_git` commit. No flag gate: the guard is always on, deterministic, $0.
2. **`backend/tests/test_phase_75_4_2_optimizer_invariant.py`** (new, 4 tests, all
   through the REAL `apply_modification` + REAL `load_skill` on a temp copy of the
   real `quant_model_agent.md`): T1 `###`→`##` promotion refused + sha256
   unchanged; T2 placeholder drop via the UNIQUE 2-line old_text refused +
   byte-identical (the bare placeholder occurs TWICE in the file — every fixture
   asserts `count(old_text)==1` so the pre-existing occurs-once guard can never
   make a test vacuously green); T3 negative control: prose-only edit ACCEPTED,
   file changed, delivery still carries `{{quant_model_data}}`; T4
   length-guard-only trip refused.

## Closed hole (why this matters)

Before: the optimizer's only post-write check was `load_skill()` not raising — a
heading promotion loads FINE while truncating the delivered prompt from 7532→190
chars (the exact 75.4 regression), then `_git` commits it, `reload_skills` makes
it live, and the PENDING metric path never auto-reverts. After: the write is
refused and reverted byte-exactly at the moment it happens.

## Files changed

`backend/agents/skill_optimizer.py`, `backend/tests/test_phase_75_4_2_optimizer_invariant.py`
(new), `.claude/masterplan.json` (75.4.2 → in_progress), handoff artifacts.
prompts.py / skills/*.md / the 71.4 review block: untouched.

## Verbatim verification output

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py -q
4 passed
$ .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py backend/tests/test_phase_75_skill_delivery.py -q
31 passed, 1 warning in 2.69s
```

Lint: new test file clean; skill_optimizer.py class-census delta vs HEAD = +1
BLE001 (the file's documented fail-open idiom, 7th instance). Main fixed two
executor nits during review (an ISC004 implicit-concat + 2 PLR0402 import
aliases in the test).

## Mutation matrix (criterion 4 + qa.md §4c) — 5 mutations, 5 killed

Runner + verbatim log in scratchpad; table + per-guard independence analysis in
live_check_75.4.2.md §3. The two criterion-4-required mutations are M1 (invariant
call removed → 3 tests fail) and M2 (helper weakened to always-ok → 3 tests
fail); M3/M4 prove each guard independently load-bearing (each kills exactly the
test designed for it); M5 mutates the FIXTURE (T2's unique old_text degraded to
the bare twice-occurring placeholder → the fixture's own count==1 trap-assert
kills it).

## Delegation record

The Sonnet executor implemented the brief's verbatim code with zero design
deviations; Main independently re-ran both suites, reviewed the full diff line by
line, fixed the two lint nits, re-derived the lint scope from git after the last
edit, and executed the mutation matrix. (The executor's own report channel had
not flushed by close; all its claims were verified first-hand by Main rather
than trusted.)
