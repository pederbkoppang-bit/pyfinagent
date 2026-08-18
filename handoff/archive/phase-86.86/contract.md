# Contract -- phase-86.86

**Step:** 86.86 (P1, LIVE MONEY) -- D6: the lite risk-judge paths destroy an
explicit 0% position verdict at the CONSTRUCTION seam.
**Date:** 2026-08-15. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_86.86.md`,
enforced `gate_passed: true` (script-recomputed, self-report agreed, zero
violations), **audit_class** with `dry: true` after 12 rounds / 2 dry,
**16 sources read in full**, 39 URLs collected, 8 internal files inspected,
16/16 claimed URLs verified literally present in the brief.

---

## 1. Research-gate summary (what the gate changed about the plan)

Five findings from the brief are load-bearing here. Two of them changed the
design; one of them I re-verified myself because the plan now leans on it.

1. **86.74 hardened the CONSUMER; the collapse is at the PRODUCER.** The
   three-state resolver at `portfolio_manager.py:1093` is *blind*, not wrong --
   by the time it runs, `PositionVerdict(SIZE, 3.0)` is a correct reading of an
   already-falsified value. Any fix that touches the resolver cannot reach this.

2. **`paper_risk_judge_shape_fix_enabled` has ZERO production readers.**
   *Re-verified by Main, not taken on trust* -- a repo-wide grep returns the
   `settings.py:350` definition, the `settings_api.py:283` env mapping, a
   docstring mention at `portfolio_manager.py:1116`, and test files. There is no
   production `if settings.paper_risk_judge_shape_fix_enabled` anywhere.
   **Consequence for criterion 5, stated so a green test is not over-read:**
   parametrising over that flag proves the code is *insensitive* to it; it does
   NOT prove a gated branch was exercised, because there is no gated branch.
   The brief's own words: *"do not hang the fix on it."* `reject_binding` DOES
   have production readers (`portfolio_manager.py:385`, `autonomous_loop.py:1139,
   2384, 2398`) and is the flag whose two states carry real information.

3. **A mutation cell is UNSCORABLE until a test drives the REAL producer.** The
   existing `scripts/qa/mutation_matrix_86_74.py` is complete only over the
   subject it names and has no producer cell. This is the
   `feedback_a_shipped_fix_that_never_ran` class: the fault must be injected
   into production code that the test actually executes, not into a copy.
   **This is why the fix extracts a real, callable production function** rather
   than editing two dict literals in place -- a test cannot drive a dict literal
   that lives inside a 300-line async LLM function.

4. **The zero SURVIVES wherever the dict is a plain literal** --
   `autonomous_loop.py:2358` (`_data_integrity_blocked_analysis`),
   `risk_debate.py:152`, `orchestrator.py:2415` all write `0` directly and are
   unaffected. The defect is confined to the two LLM lite paths. This bounds the
   change: no other producer needs touching.

5. **Prior art converges on "carry a presence bit; do not tune the default."**
   protobuf restored explicit `optional` after implicit presence made "set to 0"
   indistinguishable from "unset" (source 2); PEP 661 sentinels are deliberately
   **truthy** so an `or`-default cannot destroy one (source 1);
   `dataclasses.MISSING` exists *"because `None` is a valid value ... with a
   distinct meaning"* (source 4); RFC 7396 treats absent-vs-null as the two most
   different states (source 3); CWE-1188 names a permissive default as the
   defect and the remedy as explicit initialise-to-denied (source 5).
   **Adversarial finding, recorded because it cuts against the fix:** Ruff
   FURB110 actively pushes code *toward* `x or y`, and no Python linter detects
   this class -- CodeGuru and quantifiedcode's `dict.get` rules are about
   `KeyError`, not falsy collapse. So no tool will keep this fixed; only a test
   and an AST assertion will.

---

## 2. Hypothesis

`float(risk_dict.get("recommended_position_pct") or _LITE_RISK_DEFAULT[...])`
destroys the single most restrictive verdict the risk judge can issue. Routing
both lite paths through **one** production function that reuses the existing
`_resolve_position_pct` / `PositionVerdict` three-state rule will make an
explicit `0.0` survive to `decide_trades` (where it already blocks correctly),
while leaving a genuinely absent verdict defaulting to 3.0 exactly as today --
and will do so without introducing a second parallel idiom.

---

## 3. Evidence gathered BEFORE any code change (pre-fix baseline)

Run 2026-08-15 against the shipped tree, driving the **real** `decide_trades`.

**(a) The seam alone.** Shipped expression, real imported `_LITE_RISK_DEFAULT`:

```
judge emits 0.0  (explicit no-buy) -> persisted pct = 3.0
judge emits 3.0                    -> persisted pct = 3.0
judge silent (key absent)          -> persisted pct = 3.0
```

Rows 1 and 3 are identical: **judge-said-zero is indistinguishable from
judge-said-nothing.**

**(b) Downstream, real `decide_trades`, NAV 23,997.71, all four flag combos:**

```
shape_fix=False binding=False | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=False binding=True  | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=True  binding=False | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
shape_fix=True  binding=True  | judge 0.0 = BUY $719.93 | 3.0 = BUY $719.93 | absent = BUY $719.93
```

**(c) Control -- the same inputs when the seam is honest:**

```
shape_fix=False binding=False | true 0.0 -> no order
shape_fix=False binding=True  | true 0.0 -> no order
shape_fix=True  binding=False | true 0.0 -> no order
shape_fix=True  binding=True  | true 0.0 -> no order
```

$719.93 reproduces the 86.74 Q/A's figure exactly. Note the decision here is
`APPROVE_REDUCED`, so **`reject_binding=True` does not protect** -- this is
exposure case (a) from the D6 brief, and it means the live-harm bound is weaker
than "an .env line": with a non-REJECT decision the .env line is irrelevant.

**(d) Two further measured absurdities in the same expression:**

```
'0'  (string zero)   -> 0.0     <-- the STRING zero SURVIVES
0.0  (float zero)    -> 3.0     <-- the FLOAT zero DIES
'high' (garbage str) -> RAISES ValueError
''   (empty string)  -> 3.0
False                -> 3.0
```

The falsy test precedes `float()`, so the serialisation of the same number
decides whether it survives.

---

## 4. The CLASS, enumerated from source and classified by MEASUREMENT

AST walk (`ast.BoolOp` / `ast.Or` with a `_LITE_RISK_DEFAULT` subscript operand)
over `backend/services/autonomous_loop.py`: **10 sites, 5 keys x 2 lite paths.**
Positive control: the scan finds all five of its own known members (0 missing).

**Enumeration rule, written down as criterion 3 requires:** a member is
**decision-inverting** iff its falsy trigger value is (i) in the judge's
legitimate emitted domain, and (ii) semantically *distinct from absence*, and
(iii) substituting the default changes what `decide_trades` returns. Clause
(iii) is settled by **driving `decide_trades`**, not by reading it.

| key | falsy trigger | measured effect on `decide_trades` | class |
|---|---|---|---|
| `recommended_position_pct` | `0.0` | `no order` -> `BUY $719.93` | **DECISION-INVERTING** |
| `decision` | `""` | `""` -> BUY $719.93; `APPROVE_REDUCED` -> BUY $719.93 (**identical**; only exact `REJECT` blocks) | audit-fabricating; **latent** fail-open |
| `risk_level` | `""` | `""` / `MODERATE` / `EXTREME` all -> BUY $719.93 (**not decision-bearing**; 0 reads in `portfolio_manager.py`) | audit-fabricating |
| `reasoning` | `""` | not read by any decision | audit-fabricating |
| `risk_limits` | `{}` | shipped stop 90.0 vs honest 92.0 -- a stop is *installed* where none existed | protective |

**This corrects the research brief on two counts, by measurement.** The brief
classified `decision` as "HARMFUL (fail-open)" and `risk_level` as
"HARMFUL-MILD". Driven through the real `decide_trades`, neither changes any
order today. The brief's underlying observation is still right and is kept: the
`decision` collapse *would* invert if the gate were ever changed from
"block exact REJECT" to an allow-list ("only APPROVE* may buy"), so it is
recorded as a **latent** fail-open rather than dismissed.

**Scope call, stated explicitly so the Q/A can judge it rather than discover
it.** This step fixes the **decision-inverting** member only. The three
audit-fabricating members are real -- the substituted `reasoning` literally
reads *"risk-judge parse failed; falling back to conservative default sizing"*
when the parse SUCCEEDED and only that field was blank, which writes a false
statement into a persisted audit column -- but repairing them is a *design*
question (what should be persisted instead of a fabricated value?) that does not
belong in a P1 money fix. They are **queued as their own masterplan step
(86.87)** with the measurements above attached, per the standing
queue-discovered-defects instruction. `risk_limits` is left alone deliberately:
its substitution is protective.

---

## 5. Plan

1. **Extract one production function**, `_build_lite_risk_assessment(risk_dict)`,
   at module level in `autonomous_loop.py` next to `_LITE_RISK_DEFAULT`. Both
   lite paths (`:3085-3097` Claude, `:3333-3343` Gemini) call it. The two blocks
   are byte-identical today, so this removes a duplicate rather than adding an
   idiom.
2. **Inside it, resolve the pct through `_resolve_position_pct`** (imported from
   `portfolio_manager`, which `autonomous_loop.py:24` already imports from -- no
   import cycle):
   - `SIZE` -> the judge's number, **`0.0` included**;
   - `UNPARSEABLE` -> `0.0` and a **loud WARNING** (fail closed AND fail loud);
   - `ABSENT` -> `_LITE_RISK_DEFAULT["recommended_position_pct"]` (3.0) -- the
     only state that reaches the default.
3. **Leave the other four keys byte-identical.** Their `or` idioms move into the
   shared function unchanged, so the diff for them is pure relocation.
4. **Add a re-runnable AST checker**, `scripts/qa/verify_lite_risk_seam_86_86.py`:
   enumerates the class from source, asserts `recommended_position_pct` no longer
   appears in any `or _LITE_RISK_DEFAULT[...]` node, asserts exactly one call
   site can reach the pct default, and **fails if its own positive control
   cannot find the members it knows about** (a scan that cannot find its own
   known members is a FAILED gate).
5. **Add tests to `backend/tests/test_phase_66_2_risk_judge_shape.py`** (the file
   the immutable command already runs) that drive the **real**
   `_build_lite_risk_assessment` and then the **real** `decide_trades`,
   parametrised over both `reject_binding` states and both `shape_fix` states.
6. **Add producer mutation cells** to `scripts/qa/mutation_matrix_86_74.py` (or a
   sibling), restoring `or _LITE_RISK_DEFAULT["recommended_position_pct"]`
   byte-identically at the fixed site, with the control observed GREEN first.
7. **Queue 86.87** for the audit-fabricating trio.
8. **Write `handoff/current/live_check_86.86.md`** with every command and its
   verbatim output.

## 6. What this step will NOT do

- Not promote, flip, or add any flag; not write `backend/.env`.
- Not touch `risk_debate.py`, `orchestrator.py`, or
  `_data_integrity_blocked_analysis` -- their zeros already survive.
- Not change `decide_trades`, `_resolve_position_pct`, `_sizing_pct`, or any
  86.74 code. The consumer is correct; only the producer lies to it.
- Not restart the backend (batched to session end; the fix is **NOT IN FORCE**
  in the running process until then, and will be recorded as such).

---

## 7. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. the defect is REPRODUCED before anything is changed, with the command and its verbatim output, on BOTH lite paths -- and the reproduction must show that judge-said-0.0 and judge-silent are INDISTINGUISHABLE after the line, not merely that 0.0 is wrong
2. the fix is at the SEAM: both lite paths resolve the position size through the SAME three-state resolver the full path already uses (_resolve_position_pct / PositionVerdict), NOT through a second parallel idiom; and the number of places in the lite path that can reach _LITE_RISK_DEFAULT['recommended_position_pct'] is stated and is exactly one
3. the CLASS is enumerated FROM SOURCE rather than hand-listed: every `or _LITE_RISK_DEFAULT[...]` site in backend/services/autonomous_loop.py is enumerated by an AST walk, the enumeration command is quoted with its verbatim output, and each member is classified HARMFUL or BENIGN with the reason stated per member. The enumeration rule itself must be written down. A scan that cannot find its own known members is a FAILED gate
4. the positive control is shown after the fix: 0.0 -> 0.0 while None/absent -> 3.0. Mapping BOTH to 0.0, or both to 3.0, swaps one collapse for another and fails this criterion
5. behaviour is proven under BOTH states of paper_risk_judge_shape_fix_enabled AND BOTH states of paper_risk_judge_reject_binding, because the shipped production state is shape_fix OFF and OFF is the broken one
6. every behaviour change OTHER than the falsy-zero repair is disclosed and justified -- including what happens to a present-but-unparseable value, which the shipped expression today does not default but RAISES ValueError on. An undisclosed change of a raise into a value is a scope breach
7. mutation-tested: restore `or _LITE_RISK_DEFAULT['recommended_position_pct']` byte-identically at EACH fixed site and show the new assertion goes RED, with the control observed GREEN first and the restore verified byte-identical
8. the downstream consequence is MEASURED, not asserted: a 0.0 lite verdict must produce no order, shown by driving the real decide_trades rather than by reading the source
9. no gate is loosened, no flag is promoted, and no .env is written in order to obtain a green result

**Immutable command:**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q'`
Run BEFORE the criteria were frozen and green (41 passed, 2026-08-15).

---

## 8. Anticipated behaviour changes to disclose (criterion 6)

Enumerated ahead of the change so the disclosure is a prediction, not a
post-hoc rationalisation. Each will be re-measured after the fix.

| input | today | after | why |
|---|---|---|---|
| `0.0` | 3.0 | **0.0** | the fix |
| `0` (int) | 3.0 | **0.0** | same collapse |
| absent | 3.0 | 3.0 | unchanged -- ABSENT is the only default path |
| `null` (explicit) | 3.0 | 3.0 | `_coerce_pct(None)` returns None -> ABSENT |
| `3.0` | 3.0 | 3.0 | unchanged |
| `'0'` (string) | 0.0 | 0.0 | unchanged (already survived) |
| `'high'` | **raises ValueError** | **0.0 + WARNING** | UNPARSEABLE fails closed and loud; a raise becoming a value is the one non-falsy-zero change and is disclosed here |
| `''` | 3.0 | **0.0 + WARNING** | `float('')` raises -> UNPARSEABLE; a value we cannot read is not evidence of safety |
| `False` | 3.0 | **0.0** | `float(False)` == 0.0 -> SIZE 0.0 |

---

## 9. References

- `handoff/current/research_brief_86.86.md` (16 sources read in full; envelope
  `gate_passed: true`, script-enforced)
- `handoff/current/queued_defects_from_86.74.md` section D6
- PEP 661 (truthy sentinels); protobuf field-presence guide; RFC 7396;
  `dataclasses.MISSING`; CWE-1188; SEC Rule 15c3-5 (controls written in PREVENT
  form); "Parse, don't validate"
- `.claude/rules/research-gate.md`, `CLAUDE.md` harness protocol
- Prior step 86.74 (`portfolio_manager.py:993-1124`) -- the consumer-side fix
  this one sits upstream of
