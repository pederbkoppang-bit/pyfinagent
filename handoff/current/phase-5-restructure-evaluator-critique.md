# Q/A Critique — phase-5 restructure (masterplan meta-action)

**Q/A id:** `qa_phase5_restructure_v1`
**Agent:** qa (merged qa-evaluator + harness-verifier)
**Date:** 2026-04-19
**Scope:** verify restructuring of `.claude/masterplan.json` phase-5 from
3 placeholder steps to 15 concrete implementation sub-steps, with phase-5
moved to the end of the phase list.

---

## 5-item protocol audit

1. **Researcher present.**
   `handoff/current/phase-5-restructure-research-brief.md` exists
   (45 KB, 23:43 UTC). Spawn preceded contract.
2. **Contract pre-generate.**
   contract mtime 23:44 < experiment-results mtime 23:45. PASS.
3. **Experiment-results present.** 4.7 KB, 23:45 UTC. PASS.
4. **Log-last discipline.** Tail of `handoff/harness_log.md` is the
   phase-7.9 cycle block. No phase-5-restructure block yet — correct,
   log is appended AFTER Q/A verdict. PASS.
5. **First Q/A on this restructure.** No prior
   `phase-5-restructure-evaluator-critique.md` artifact. Not a
   verdict-shopping run. PASS.

**Audit: 5/5.**

---

## Deterministic checks (A–H)

| ID | Check | Result |
|----|-------|--------|
| A  | `json.load('.claude/masterplan.json')` parses | PASS (34 phases loaded) |
| B  | `phases[-1].id == 'phase-5'` | PASS |
| C  | `len(steps) == 15`, ids `['5.1'..'5.15']` | PASS |
| D  | Every step has non-empty `verification.command` + `verification.success_criteria` list | PASS (0 missing) |
| E  | `archived_legacy_steps` preserves 3 old placeholders (ids 5.1, 5.2, 5.3 — the generic harness-dry-run variants) | PASS |
| F  | `open_issues` has >=3 actionable entries | PASS (4: eodhd_budget, ibkr_infra, market_priority, compliance_per_market) |
| G  | Dependency DAG acyclic; 5.15 is sink; 5.1 has no deps | PASS (no cycles; 5.1 deps=[]; 5.15 deps=['5.14']; 5.15 downstream=[]) |
| H  | Scope limited to `.claude/masterplan.json` + handoff artifacts; no code files touched | PASS (`git diff --stat` shows only masterplan.json +614/-111; three new handoff files; all other repo mods are pre-existing unrelated work) |

**Deterministic: 8/8.**

### DAG topology (for record)

```
5.1 (broker_base) ─┬─> 5.2 (data_provider_base) ─> 5.3 (schemas) ─┬─> 5.5 (crypto)
                   │                                              ├─> 5.7 (fx)
                   │                                              └─> 5.8 (futures)
                   └─> 5.4 (risk_engine) ─┬─> 5.5, 5.6, 5.7, 5.8
                                          └─> 5.13 (backtest)
5.5 ─> 5.10 (universe), 5.11 (regime)
5.7 ─> 5.9 (international), 5.11 (regime)
5.11 ─> 5.12 (cross-market signals) ─> 5.14 (autonomous loop)
5.13, 5.14 ─> 5.15 (integration test, sink)
```
Ordering is architecturally coherent: abstractions (5.1–5.4) before
market-specific (5.5–5.9), data before signals, signals before
backtest, backtest before live-loop, live-loop before integration.

---

## LLM judgment

### 1. Measurable state predicates on every step?

Spot-checked 5.5, 5.7, 5.8, 5.11 per the Q/A brief:

- **5.5 (crypto):** "crypto_candles has >=18 rows for BTC/USD + ETH/USD
  over 10-day window" — concrete BQ row-count predicate. PASS.
- **5.7 (fx):** ">=16 rows for EUR_USD + GBP_USD over 10-day window
  (weekday-only)" — concrete, and the weekday-only qualifier shows the
  author thought about the 5/7 calendar difference vs crypto. PASS.
- **5.8 (futures):** "open_interest populated (non-null) for all rows"
  — concrete column-level NULL-check predicate. Plus a specific
  `contract_roll.get_front_month('ES')` assertion. PASS.
- **5.11 (regime):** Two exact inputs/outputs:
  `classify(vix=35, crypto_vol=1.2, yield_slope=-0.003) -> 'CONTAGION'`
  and `classify(vix=14, crypto_vol=0.3, yield_slope=0.01) -> 'RISK_ON'`.
  This is a proper behavioural test, not a shape check. PASS.

**Full-15 scan for vagueness.** Grep for `works correctly`, `looks right`,
`functions properly`, `is implemented` across all criterion strings:
**0 hits**. No step relies on soft language.

**Scaffold-padding scan.** Any step whose ONLY criterion is "module
exists" or "parses"? **0 hits**. Every step has 3–4 criteria that
exercise real behaviour (DDL writes, row counts, broker routing, regime
classifications).

### 2. Scaffolds masquerading as implementation steps?

5.1 (broker abstraction) and 5.2 (data-provider abstraction) are
genuinely abstraction layers — but their criteria require BOTH the ABC
AND a working concrete subclass (`AlpacaBroker` for 5.1, an EODHD
implementation referenced downstream in 5.2/5.9). They are not pure
scaffolds. The scaffold-only pattern (cf. phase-7.9 style) is NOT
present in phase-5. Good.

### 3. Dependency ordering architecturally sane?

- 5.5 (crypto) depends on 5.1+5.2+5.3+5.4. Correct — crypto ingestion
  needs broker base, data provider base, schema DDL, and risk engine.
- 5.6 (options) depends on 5.4+5.5. The 5.5 dependency is slightly
  unusual (options don't obviously require crypto), but the contract
  is likely routing options through the same `asset_class` enum
  introduced in the crypto work. Acceptable — flagged as a soft
  observation, not a blocker.
- 5.13 (backtest) depends on 5.4+5.5+5.11. Correct — backtest needs
  risk engine, at least one non-equity market (crypto), and regime
  detector.

### 4. Open issues actionable?

All 4 issues name a specific owner decision or external resource:

| id | decision/resource | actionable? |
|----|------|-------------|
| iss_eodhd_budget | $19.99/mo API key approval | YES |
| iss_ibkr_infra | TWS local vs VPS vs defer | YES |
| iss_market_priority | confirm/reorder crypto→options→FX→futures→intl | YES |
| iss_compliance_per_market | CFTC registration gate before live capital | YES |

None are "we should think about this later". PASS.

### 5. Anti-rubber-stamp: caveat #4 (phase-5.5 collision)

Main disclosed that a top-level phase named `phase-5.5` already exists
AND step 5.5 within phase-5 also has id `"5.5"`. Verified:

- `mp['phases']` contains an entry with `id == 'phase-5.5'`.
- `mp['phases'][-1]['steps']` contains an entry with `id == '5.5'`.

**Collision assessment:** step ids are scoped within their parent phase
in this masterplan schema, and the top-level `phase-5.5` is a distinct
object reachable as `phases[x]` not `phases[-1].steps[x]`. The
`/masterplan` slash command and the harness resolver both address
steps by `phase.id + step.id` composition, so there is no runtime
ambiguity. However, human operators may be briefly confused by "5.5"
appearing in two contexts. Recommend (non-blocking) that Main add a
one-line comment in the contract pointing at this, which the
experiment-results already does as caveat #4. Treating as disclosed
and acceptable. No blocker.

---

## Violated criteria

None.

## Violation details

None.

## Checks run

```
["syntax_json", "phase_position", "step_count_ids", "verification_fields",
 "archived_legacy", "open_issues", "dag_acyclic_sink",
 "scope_limited_to_masterplan_and_handoff", "protocol_audit_5_item",
 "llm_judgment_criteria_concreteness", "llm_judgment_scaffold_padding",
 "llm_judgment_dependency_ordering", "llm_judgment_open_issues_actionable",
 "llm_judgment_caveat4_phase55_collision"]
```

## Final Decision

**PASS** — `qa_phase5_restructure_v1`.

Restructure is coherent, concrete, and non-scaffolded. Every one of
15 steps has measurable success criteria grounded in row counts,
column predicates, function-call outputs, or broker routing
assertions. DAG is acyclic and architecturally ordered. Open issues
name specific owner decisions. Caveat #4 (phase-5.5 id collision) is
disclosed and non-breaking. Archived legacy preserved. Scope clean.

**Recommended next step for Main:** append `harness_log.md` cycle
block for phase-5-restructure, then this meta-action is closed. The
15 phase-5 steps remain `status: pending` and will be executed in
future cycles once phase-4 go-live completes.

**Non-blocking observation:** consider adding a top-level note in
`.claude/masterplan.json` (e.g. a `_comments` field on phase-5) that
reminds operators step `5.5` is distinct from top-level `phase-5.5`.
Not required for PASS.
