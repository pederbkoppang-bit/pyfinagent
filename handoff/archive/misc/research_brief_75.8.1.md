# Research Brief — Step 75.8.1

**Step**: 75.8.1 — Factor a shared report-integrity predicate module; import from BOTH consumers of `handoff/gauntlet/<strategy>/report.json`
**Tier**: simple-to-moderate | **audit_class**: false | **Priority**: P1 money-integrity
**Researcher**: Layer-3 harness researcher
**Started**: 2026-07-24
**Status**: IN PROGRESS (write-first; appended incrementally)

---

## Objective (from spawn prompt)

75.8 added stub-fingerprint + `dry_run`-label rejection to `scripts/risk/promotion_gate.py`, but
`backend/autonomous_harness.py::promote_strategy` is the SECOND consumer of
`handoff/gauntlet/<strategy>/report.json` and trusts it with NO checks. Factor ONE shared predicate
module implementing (a) `dry_run:true` refusal and (b) stub-fingerprint refusal (bt_drawdown ==
drawdown exactly for every non-skipped regime; filter skipped regimes; guard the `all([])` empty-list
trap), import it from BOTH consumers, refactor `promotion_gate.py` onto it in the same step.

BOUNDARY: evaluator gate thresholds, kill-switch code, DSR/PBO constants, limits.yaml all byte-untouched.

---

## Internal audit (file:line anchors)

### Consumer inventory (repo-wide grep, verified)
`handoff/gauntlet/<strategy>/report.json` has EXACTLY TWO trusting consumers +
one writer. No third consumer exists (grep `report.json` over `backend/` +
`scripts/` — the many `full_report_json` hits are the BQ analysis-pipeline
report, a different artifact):
| Role | File:line | Loads report how |
|------|-----------|------------------|
| WRITER | `scripts/risk/gauntlet.py:139-155` (`_write_report`) | writes `report.json`; REFUSES anything not `dry_run:true` (line 147) |
| CONSUMER 1 | `scripts/risk/promotion_gate.py:77-92` (`_load_gauntlet_report`) + `:112-157` | has stub-fingerprint check; NO evaluator path skip |
| CONSUMER 2 (target) | `backend/autonomous_harness.py:267,277-278` (`promote_strategy`) | loads + trusts via `evaluate()`, NO integrity checks |
| caller of C2 | `scripts/risk/phase4_9_redteam.py:68` | ONLY caller of `promote_strategy` (no prod caller) |
| non-consumer | `scripts/audit/gauntlet_regimes_audit.py` | touches `regimes.py` only, never `report.json` |

### `promote_strategy` RE-ANCHOR (autonomous_harness.py, 2026-07-24)
Prompt said "lines 258-289 as of 2026-07-23 — re-anchor". Current exact anchors
(NO uncommitted diff; last commit touching file = `22409053` phase-23.8.3, so
the drift is just prompt-anchor slack of ~7 lines, not a real move):
- `def promote_strategy(strategy: str) -> dict:` — **line 251**
- docstring 252-257
- (1) blocklist 30-day check — 258-264 (`_on_blocklist` → raise `PromotionBlocked`)
- (2) report-exists check — 266-273 (missing → `_append_blocklist` + raise)
- (3) lazy import `from backend.backtest.gauntlet.evaluator import evaluate` — **line 276**
- `report = _json.loads(report_path.read_text(...))` — **line 277**  ← INSERT INTEGRITY CHECK HERE
- `verdict = evaluate(report)` — **line 278**
- `if not verdict["overall_pass"]:` → `_append_blocklist` + raise — 279-285
- `_annotate_harness_log` + `return verdict` — 287-288

Refusal shape to mirror the two existing refusals: `_append_blocklist(strategy,
reason)` then `raise PromotionBlocked(f"... {reason}; added to 30-day
blocklist")`. `evaluate()` (evaluator.py:85-103) returns
`{overall_pass:bool, regime_checks, mc_check, reasons:list, drawdown_ratio_cap}`.
No exception is swallowed today (JSONDecodeError from line 277 propagates); the
new check must likewise raise, never catch-and-promote (criterion C1).

### The exact 75.8 predicate to factor out (promotion_gate.py:125-131)
```python
non_skipped = [
    r for r in (report.get("per_regime", []) or [])
    if not r.get("skipped")
]
if non_skipped and all(
    r.get("bt_drawdown") == r.get("drawdown") for r in non_skipped
):
    # blocked: stub fingerprint
```
- The skipped-filter (`if not r.get("skipped")`) AND the `non_skipped and`
  truthiness guard are BOTH load-bearing: skipped regimes carry NEITHER
  `drawdown` NOR `bt_drawdown` (gauntlet.py:77-85 & test `_regime(skipped=True)`
  return only `regime_id/name/skipped/reason`), so `None == None` = True →
  without the filter an all-skipped report false-positives; without the
  `non_skipped and` guard `all([])` = True → same false-positive. This is why
  `test_all_skipped_regimes_are_not_fingerprinted` is the mutation test for
  BOTH the filter and the empty-guard.

### CRITICAL FINDING — the "dry_run-label rejection" the prompt says 75.8 added does NOT exist in promotion_gate.py
Grep + full read confirm: promotion_gate.py has ONLY the stub-fingerprint check
(:118-141). Its `dry_run` references (:190,192,211,213,237) are the CLI's OWN
`--dry-run` no-write flag (gap6-10), NOT a check on the gauntlet report's
`dry_run` field. So the spawn prompt / step text ("Port BOTH rejections
promotion_gate.py gained in 75.8: (a) refuse reports labeled dry_run:true…")
is **partially inaccurate** (measure-don't-assert): 75.8 added ONLY (b). This
step must ADD the `dry_run:true`-label refusal to the shared module and thus to
BOTH consumers (new behavior for promotion_gate.py too), not merely "port" an
existing one. The gauntlet WRITER only ever emits `dry_run:true` reports
(live mode `raise NotImplementedError`, gauntlet.py:159-167; `_write_report`
refuses non-`dry_run:true`, :147), so EVERY report on disk today is
`dry_run:true` — the label refusal correctly makes nothing promotable to real
capital until live mode exists (the "P1 latent" framing).

### C2 ORDER CONSTRAINT — fingerprint MUST be checked before dry_run in the composite
`test_real_gauntlet_dry_run_report_is_rejected_end_to_end`
(test_phase_75_promotion_gate.py:245-257) feeds a REAL gauntlet report
(`dry_run:true` AND stub-fingerprinted) and asserts `"stub fingerprint" in out`.
If the composite checks `dry_run` FIRST it returns the dry_run reason and that
assertion FAILS. So the shared composite must check fingerprint FIRST, dry_run
SECOND, to keep the existing reason string and satisfy C2 (pre-existing tests
unchanged). I verified all 14 existing tests pass unchanged under
fingerprint-first + additive dry_run-label check (walk-through in PLAN below).

### report.json shape facts (from writer gauntlet.py:76-130 + test helpers)
- top-level: `strategy, seed, ts, dry_run(bool, always True on disk),
  regime_catalog_hash, per_regime(list), monte_carlo(dict), bq_note`
- non-skipped regime: `id, drawdown, bt_drawdown, forced_exits, regime_id,
  name, start, end, n_days, final_return, max_drawdown, sharpe, skipped:False`
  (dry-run sets `bt_drawdown == drawdown` — the fingerprint, gauntlet.py:97)
- skipped regime: `regime_id, name, start, end, skipped:True, reason` (NO
  drawdown/bt_drawdown keys)
- `monte_carlo`: `n_paths, n_days, return_p50/p05/p95, drawdown_p50/p95/p99,
  p99_drawdown, bt_drawdown, breaches`

### Import-path answer (scripts/risk → backend.*; cycle analysis)
- `promotion_gate.py` lives OUTSIDE the backend package. It ALREADY imports
  backend.* : `:37-39` sets `REPO = Path(__file__).resolve().parents[2]` +
  `sys.path.insert(0, str(REPO))`, then `:41 from backend.services.promotion_gate
  import …` and `:44 from backend.backtest.gauntlet.evaluator import evaluate`.
  The new `from backend.backtest.gauntlet.report_integrity import …` follows
  this EXACT idiom — no new sys.path work.
- `autonomous_harness.py` is INSIDE backend; it already does the lazy
  `from backend.backtest.gauntlet.evaluator import evaluate` (:276). New import
  mirrors it (lazy, same spot).
- CYCLE RISK = zero IF report_integrity.py is a pure leaf like evaluator.py
  (imports only `from typing import Any` — evaluator.py:24-26). It must NOT
  import autonomous_harness / promotion_gate / services. Consumers never import
  each other. gauntlet `__init__.py` imports regimes+evaluator; adding
  report_integrity there is OPTIONAL (consumers import the submodule directly),
  so `__init__.py` can stay untouched (minimal surface).

---

## External research

### Read in full (>=5 required; 7 fetched) — counts toward the gate
| # | URL | Accessed | Kind | Tier | Key finding |
|---|-----|----------|------|------|-------------|
| 1 | https://arxiv.org/html/2510.20270v1 (ImpossibleBench, Zhong/Raghunathan/Carlini, Oct 2025) | 2026-07-24 | paper | peer-reviewed preprint | "any pass necessarily implies a specification-violating shortcut" — build a check the ONLY way to pass is to cheat. Exploit types incl. **test-specific hardcoding** (detect exact inputs, return hardcoded expected outputs) — the direct analogue of the dry-run stub (`bt_drawdown = drawdown` hardcoded). Defense: **test access control (read-only)** + **abort mechanism**. LLM monitors caught only 42-50% of sophisticated cheats — argues for a STRUCTURAL guard, not a judge. |
| 2 | https://arxiv.org/html/2605.21384v1 (SpecBench, 2026) | 2026-07-24 | paper | peer-reviewed preprint | "Sequence manipulation — fabricating intermediate artifacts to skip expensive upstream work — was the most common chained-regime exploit." Exactly the gauntlet stub: a fabricated report skips the (unimplemented) live backtest. Detection = measure the GAP between proxy-pass and true compliance; "validation suites inevitably underspecify compositional correctness" (evaluator's 4 gates underspecify authenticity). |
| 3 | https://arxiv.org/html/2604.15149v1 (LLMs Gaming Verifiers / RLVR, ICLR 2026) | 2026-07-24 | paper | peer-reviewed preprint | "extensional verification ... creates false positives because it ignores whether reasoning was actually performed." The evaluator checks drawdown RATIOS (extensional) — a stub passes by construction. Defense pattern = a SECOND orthogonal check the exploit can't satisfy (their IPT; our fingerprint + dry_run label). |
| 4 | https://refactoring.guru/pull-up-method | 2026-07-24 | doc/catalog | official-ish (Fowler catalog) | Duplicated logic across siblings → "Make the methods identical and then move them to the relevant superclass"; benefit "better to [change] in a single place than ... search for all duplicates." Direct mandate for one shared predicate over copy-paste. |
| 5 | https://www.datacamp.com/tutorial/python-circular-import | 2026-07-24 | tutorial | practitioner | Break cycles by moving shared logic into a neutral leaf module both sides import; one-way dependency flow (higher→lower only); a leaf that "depends only on external libraries (or nothing at all) cannot participate in cycles"; function-level imports defer resolution. Validates `report_integrity.py` as a pure leaf. |
| 6 | https://algomaster.io/learn/lld/dry | 2026-07-24 | blog | authoritative blog | DRY = "Every piece of knowledge must have a single, unambiguous, authoritative representation." Duplicated validation → "users might pass validation in one module but fail in another." Caveat: "Duplication is far cheaper than the wrong abstraction"; Rule of Three (here N=2 consumers of an IDENTICAL predicate → extraction is warranted, not premature). |
| 7 | https://en.wikipedia.org/wiki/Single_source_of_truth | 2026-07-24 | reference | reference | SSOT: "every data element is mastered (or edited) in only one place." "The master data is never copied and instead only references to it are made." Copies risk "retrieval of outdated ... information" — the predicate-drift risk this step removes. |

### Snippet-only (context; does NOT count toward gate) — ~33 distinct
Reward-hacking / fabricated-evidence corpus: EVILGENIE (Gabor 2025, LLM judges > held-out tests), TRACE (Deshpande 2026, 517 trajectories/54 hack categories, GPT-5.2 caught 63%), Terminal-Wrench (Bercovich 2026), RHB (Thaman 2026, RL post-training 0.6%→13.9% exploit), BenchJack (arXiv 2605.12673), Reward-Hacking Benchmark (2605.02964), "Is It Thinking or Cheating?" (2510.01367), rubric-RL reward hacking (2606.04923), Countdown-Code (Khalifa 2026, 1% SFT cheating → catastrophic RLVR hacking), "Coding with Enemy: can devs detect AI sabotage" (2606.05647), deception-detector difficulties (2511.22662), generative-AI agentic threats (2605.16471), AI-coding-agent census (2606.24429), Prover-Is-The-Judge Ada/SPARK (2607.14340), dev.to "agents declare done while skipping the bar". Refactoring/SSOT/DRY: kluster.ai refactoring 2026, thoughtspot/vantagepoint/strapi SSOT, geeksforgeeks DRY, faros.ai DRY-for-AI-generated-code, bruno colocation, dev.to DRY-misunderstood, Rule-of-three (Wikipedia), Fowler refactoring gists/summaries (HugoMatilla, brightmarbles, wshaddix), O'Reilly Sonar ch.8, BYU refactoring-III. Circular imports: geeksforgeeks, rollbar, 4× Medium (Untangling / Trap / Escape / So-you-got), dev.to "different ways to fix". ImpossibleBench mirrors: HuggingFace 2510.20270, emergentmind, lesswrong, github safety-research/impossiblebench.

### Consensus vs debate
- **Consensus:** (a) extract one shared predicate for N≥2 consumers of identical logic (Fowler, DRY, SSOT — unanimous); (b) a pure leaf module both sides import is the cycle-safe way to share (DataCamp + circular-import canon); (c) verifiers that check only an EXTENSIONAL property can be passed by construction — you need an orthogonal structural guard the fabrication can't satisfy (ImpossibleBench, SpecBench, LLMs-Gaming-Verifiers all converge). This is precisely the fingerprint (structural) + dry_run-label (provenance) pair layered on top of the evaluator (extensional).
- **Debate / caution:** DRY has a well-known failure mode — "the wrong abstraction" / premature extraction (algomaster, and DEV "DRY misunderstood"). Rebuttal for THIS step: the predicate is already duplicated intent across 2 consumers and is byte-identical logic, not a speculative future abstraction — extraction is the Rule-of-Three-satisfying, non-premature case. Keep the shared module a pure predicate (no config, no thresholds) so it does not accrete the coupling DRY-skeptics warn about.
- **Pitfall (ImpossibleBench §6):** LLM monitors miss 42-50% of sophisticated fabrication — do NOT rely on a judge; the fingerprint/label guard must be a deterministic code gate (which it is).

### Application to pyfinagent (external → file:line)
- ImpossibleBench "test access control / make cheating structurally impossible" → the gauntlet WRITER already refuses non-`dry_run:true` writes (gauntlet.py:147); this step adds the READ-side structural refusal in both consumers so a hand-forged report can't promote.
- Fowler Pull-Up / SSOT → factor `report_integrity.py`; both consumers reference it, never copy (kills the drift the step exists to prevent).
- DataCamp leaf-module → `report_integrity.py` imports only `typing` (like evaluator.py:24-26) → zero cycle risk for the in-package (autonomous_harness) and out-of-package (promotion_gate, REPO-on-sys.path) consumers.

---

## Recency scan (last 2 years)

**Performed.** Ran current-year (2026) and last-2-year (2025) passes on both the
refactoring and the fabricated-evidence topics.

- **Fabricated-evidence / reward-hacking is an ACTIVELY EXPLODING 2025-2026
  literature** and it directly supersedes/reinforces the 75.8 basis. New in
  window: ImpossibleBench (Oct 2025) — the canonical "passing implies cheating"
  construction; SpecBench, LLMs-Gaming-Verifiers, TRACE, BenchJack, RHB,
  Countdown-Code (all 2026). Collective finding: **extensional/proxy checks are
  gamed by construction; you need an orthogonal structural guard AND it must be
  deterministic (LLM monitors miss 42-50%).** This is a strong external
  endorsement of the 75.8/75.8.1 design (a deterministic fingerprint + label
  guard, not a judge). No source contradicts the approach; the only caveat is
  DRY-over-abstraction, addressed above.
- **Refactoring / SSOT / DRY canon is older but unchallenged** (Fowler,
  Pragmatic Programmer). 2026 restatements (kluster.ai, thoughtspot) add nothing
  that changes the plan. The year-less canonical (SSOT, Rule-of-Three, Pull-Up)
  remains authoritative.
- **Python circular-import guidance is settled** — the leaf-module + one-way-flow
  pattern is stable across 2024-2026 sources; no new finding supersedes it.

Net: the recent literature strengthens the case for this step; nothing found
argues for a different design.

---

## Queries run (3-variant discipline per topic)

**Topic 1 — shared-validation SSOT/DRY refactoring:**
- frontier: `single source of truth validation logic refactoring duplicated predicate consumers 2026`
- year-less canonical: `DRY principle shared validation module drift between consumers`; `Martin Fowler extract function pull up method remove duplication refactoring catalog`

**Topic 2 — fabricated-evidence / fake-artifact detection in agent pipelines/CI:**
- frontier: `LLM agent reward hacking gaming test suite fabricated evidence detection 2026`
- recency 2025: `detecting fake artifacts specification gaming AI coding agents CI verification 2025`
- canonical anchor: `ImpossibleBench arxiv impossible test cases coding agents exploit passing by construction` (surfaced the founding "specification gaming" construction; DeepMind's original "specification gaming" coinage is the year-less term-of-art).

**Topic 3 — Python import-cycle pitfalls sharing between scripts/ and backend/:**
- year-less canonical: `Python avoid circular imports shared leaf module package structure` (settled topic; canonical query is primary).

---

## PLAN recommendations

### Change surface (4 files; matches C5 boundary)
1. **NEW** `backend/backtest/gauntlet/report_integrity.py` — pure leaf predicate module.
2. **EDIT** `scripts/risk/promotion_gate.py` — replace inline fingerprint block (:118-141) with a call to the shared composite.
3. **EDIT** `backend/autonomous_harness.py` — insert integrity check in `promote_strategy` after report load (:277), before `evaluate()` (:278).
4. **NEW** `backend/tests/test_phase_75_8_1_harness_consumer.py` — tests BOTH consumers, offline, anti-fixture-divorce.
- Do NOT touch `backend/backtest/gauntlet/__init__.py` (consumers import the submodule directly; optional export is a nice-to-have, skip to keep surface minimal).
- BYTE-UNTOUCHED (C5): `evaluator.py` (incl. `DRAWDOWN_RATIO_CAP`), kill-switch code, DSR/PBO constants, `backend/governance/limits.yaml`. `test_phase_75_promotion_gate.py` stays byte-identical (C2).

### Proposed shared-module API (`report_integrity.py`)
Pure leaf — imports ONLY `from typing import Any` (mirrors evaluator.py) → zero cycle risk.
```python
def is_dry_run_report(report: dict[str, Any]) -> bool:
    """True iff report is explicitly labeled dry_run:true."""
    return report.get("dry_run") is True

def has_stub_fingerprint(report: dict[str, Any]) -> bool:
    """True iff EVERY non-skipped regime has bt_drawdown == drawdown exactly.
    Skipped regimes filtered (they carry neither key; None==None would
    false-positive). Empty non-skipped list is NOT a fingerprint (all([]) trap)."""
    non_skipped = [r for r in (report.get("per_regime", []) or [])
                   if not r.get("skipped")]
    return bool(non_skipped) and all(
        r.get("bt_drawdown") == r.get("drawdown") for r in non_skipped)

def check_report_integrity(report: dict[str, Any]) -> tuple[bool, str | None]:
    """Composite gate → (ok, reason). FINGERPRINT CHECKED FIRST so the
    pre-existing promotion_gate 'stub fingerprint' reason string is preserved
    for a dry_run:true+stub report (criterion C2)."""
    if has_stub_fingerprint(report):
        return False, ("stub fingerprint: bt_drawdown == drawdown exactly for "
                       "all non-skipped regimes -- report is dry-run/stub "
                       "evidence, not a live gauntlet run")
    if is_dry_run_report(report):
        return False, ("dry_run:true label -- report is a preview/stub run, "
                       "not valid evidence for a capital-allocation promotion")
    return True, None
```
Keep the "stub fingerprint" reason string byte-for-byte (promotion_gate.py:135-137) so `assert "stub fingerprint" in out` still holds.

### Import STYLE recommendation (maximizes testability of "single source")
Have BOTH consumers do `from backend.backtest.gauntlet import report_integrity` and
call `report_integrity.check_report_integrity(report)` (module-attr call, not a
bound `from ... import check_report_integrity`). Then a single test can
`monkeypatch.setattr(report_integrity, "check_report_integrity", lambda r: (False,"patched"))`
and observe BOTH consumers change — the DEFINITIVE, non-vacuous proof that both
call the ONE implementation (a copy would be immune to the patch). This beats an
AST/source-scan for C2's "single shared implementation" (source scans are the
vacuous-guard trap per auto-memory `feedback_mutation_test_guards_and_fixtures`).
- promotion_gate.py: import works via existing REPO-on-sys.path (:37-39).
- autonomous_harness.py: lazy import inside `promote_strategy` (mirror the
  existing lazy `evaluate` import at :276).

### promote_strategy edit (autonomous_harness.py, after :277)
```python
report = _json.loads(report_path.read_text(encoding="utf-8"))
from backend.backtest.gauntlet import report_integrity  # lazy, mirrors evaluate import
ok, reason = report_integrity.check_report_integrity(report)
if not ok:
    _append_blocklist(strategy, reason)
    raise PromotionBlocked(
        f"strategy {strategy!r} report failed integrity check: {reason}; "
        "added to 30-day blocklist")
verdict = evaluate(report)
```
**Blocklist decision:** recommend appending to the 30-day blocklist on integrity
failure, for PARITY with the two existing refusals (missing report :269 + failed
gauntlet :281 both blocklist). Fabricated evidence is a stronger integrity signal
than a missing report, so parity is the safe default. Operational nuance to note
in the contract: since the gauntlet writer only ever emits dry_run:true (stub)
reports today, EVERY strategy's first promote attempt will be refused+blocklisted
— which is CORRECT (nothing is promotable to real capital until live gauntlet
exists; this is the step's "P1 latent" framing). No production caller exists
(only phase4_9_redteam), so no live impact. `no exception swallowed` (C1): the
check raises PromotionBlocked; nothing catches-and-promotes.

### promotion_gate.py edit (replace :118-141)
```python
ok, reason = report_integrity.check_report_integrity(report)
if not ok:
    print(json.dumps({"blocked": True, "reason": reason,
                      "report_path": str(rp.relative_to(REPO))}))
    return 1
verdict = evaluate_gauntlet(report)
```
Add `from backend.backtest.gauntlet import report_integrity` beside the existing
evaluator import (:44). This is a strict superset of current behavior: fingerprint
still fires first (same reason string → C2 preserved), and it GAINS the
dry_run-label refusal (the behavior the prompt wrongly assumed already existed).

### New test file — required cases (`test_phase_75_8_1_harness_consumer.py`)
Offline; monkeypatch `ah._GAUNTLET_ROOT`, `ah._BLOCKLIST_PATH`, `ah._HARNESS_LOG`
to `tmp_path` (mirror phase4_9_redteam.py:63-64); write `report.json` under the
tmp gauntlet root; each test gets a fresh tmp_path so the 30-day blocklist never
cross-contaminates. Reuse the report-builder helpers from
test_phase_75_promotion_gate.py (skipped regimes carry NO drawdown keys).
1. **stub-fingerprint through promote_strategy** → raises `PromotionBlocked`,
   reason contains "stub fingerprint", blocklist row appended, returns no verdict.
2. **dry_run:true DIVERGENT report through promote_strategy** → raises
   (label check; report is NOT fingerprinted so ONLY the label catches it →
   load-bearing for the drop-dry_run mutation).
3. **realistic divergent dry_run:false report through promote_strategy** →
   returns the evaluator verdict (`overall_pass=True`). Build it to pass
   evaluate(): every regime dd/bt ≤ 2.0 & forced_exits=0; mc breaches=0 &
   p99/bt ≤ 2.0 (e.g. regimes (0.10,0.08),(0.05,0.045); mc p99=0.12 bt=0.10).
4. **all-skipped / empty per_regime through promote_strategy** → NOT
   fingerprinted (all([]) trap); promotes if otherwise valid (or reaches
   evaluate()); proves the skipped-filter + empty-guard through THIS consumer.
5. **anti-fixture-divorce** → `gauntlet.run("baseline", dry_run=True, seed=7)`
   produces the REAL stub; feed its bytes to promote_strategy → refused
   (fingerprint fires first; it is also dry_run:true).
6. **promotion_gate dry_run-label coverage** → feed a dry_run:true DIVERGENT
   report to `promotion_gate.main(["--require-gauntlet"])` → rc==1 (proves
   promotion_gate GAINED the label check; load-bearing for drop-dry_run through
   consumer 1).
7. **single-source proof** → `monkeypatch.setattr(report_integrity,
   "check_report_integrity", lambda r:(False,"patched"))`; assert BOTH
   promote_strategy (raises) AND promotion_gate (rc==1) change → proves one impl.

### Mutation matrix (C4) — each mutation fails ≥1 test through EACH consumer
| Mutation | Fails via promote_strategy (consumer 2) | Fails via promotion_gate (consumer 1) |
|----------|------------------------------------------|----------------------------------------|
| drop the whole `check_report_integrity` call | new test #1 (stub promotes) | existing `test_promotion_gate_blocks_stub_fingerprint` |
| drop `is_dry_run_report` branch | new test #2 (dry_run divergent promotes) | new test #6 (dry_run divergent not blocked) |
| drop skipped-filter in `has_stub_fingerprint` | new test #4 (all-skipped wrongly refused) | existing `test_all_skipped_regimes_are_not_fingerprinted` |
| drop `non_skipped and` empty-guard | new test #4 (all([]) → wrongly refused) | existing all-skipped test |
| stub the shared module (`check_report_integrity`→`(True,None)`) | new tests #1,#2 (nothing refused) | existing stub-fingerprint + new #6 |
Mutate the STUB too (auto-memory): a returns-True stub of the predicate must fail
tests through both consumers — covered by row 5.

### Verification & live_check
- Command (immutable): `.venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py backend/tests/test_phase_75_8_1_harness_consumer.py -q` → all pass (old file unchanged).
- `live_check_75.8.1.md`: paste verbatim pytest exit-0 output + `git diff --stat` showing ONLY the 4 files (proving C5 boundary — no evaluator/kill-switch/limits.yaml edits).
- Sanity `git diff -- backend/backtest/gauntlet/evaluator.py backend/governance/limits.yaml` must be empty.

### Open questions for Main (none block PLAN)
- Blocklist-on-integrity-failure: recommended YES (parity). If operator prefers
  raise-without-blocklist, that's a one-line change and still satisfies C1
  ("logged/returned reason, no exception swallowed") — flag the choice in the contract.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 33,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Confirmed EXACTLY two consumers of handoff/gauntlet/<strategy>/report.json: promotion_gate.py (has stub-fingerprint check) and autonomous_harness.promote_strategy (NO checks; re-anchored to line 251, insert point after :277 before evaluate() :278). CRITICAL: promotion_gate.py has ONLY the stub-fingerprint check -- its dry_run refs are the CLI's own --dry-run no-write flag, NOT a gauntlet-report dry_run-label check; the prompt's 'both rejections already in promotion_gate.py' is partially wrong, so this step ADDS the dry_run-label refusal to both via the shared module. Fingerprint MUST be checked before dry_run to keep the existing 'stub fingerprint' reason string (C2). report_integrity.py must be a pure leaf (typing-only, like evaluator.py) => zero cycle risk for the in-package and REPO-on-sys.path consumers. Recommend module-attr import so a monkeypatch of the shared predicate proves both consumers use ONE impl (non-vacuous C2 proof). Full mutation matrix + 7 required tests + blocklist-parity recommendation in PLAN.",
  "brief_path": "handoff/current/research_brief_75.8.1.md",
  "gate_passed": true
}
```
