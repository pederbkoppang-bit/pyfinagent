# Experiment Results — phase-80.27

**Step:** `80.27` (P0) — a data outage was laundered into a confident NEUTRAL trading
verdict that reached the live paper-trading loop, while the data-quality gate reported
100%. Date 2026-07-25. Contract: `handoff/current/contract_80.27.md`.
Gate: `research_brief_80.27.md` (`gate_passed: true`, audit-class, 8 sources in full,
38 URLs, 24 internal files, `coverage.dry: true` after 6 rounds / 2 dry rounds).

**This is the only step in the drain that changes live decision behaviour.**

---

## 1. What was built — a deliberately ASYMMETRIC shipping shape

The two halves have different risk profiles, so they ship differently. That asymmetry is
the central design decision.

### Half A — the DETECTOR. Ships **ON** (un-flagged).

`backend/agents/info_gap.py`:
- `_assess_source_status` gains a **non-finite numeric scan** (`_has_non_finite`,
  recursive over dict/list/tuple/set, depth-capped) and a **`nan`/`inf` token scan of the
  summary prose**.
- `NO_DATA` is now handled. `backend/tools/alt_data.py` already returned it and the
  detector had **no case for it**, so an explicit failure was being classified
  **SUFFICIENT** — the same defect shape as the NaN one, found by the research gate.
- `quant_model` added to `_SOURCE_CRITICALITY`. It was **absent** (11 keys, not 12), and
  `detect_info_gaps` iterates that dict — so quant_model was never assessed, never
  counted, and could never enter `critical_gaps`. **Flipping it to ERROR would have done
  nothing without this.**

Why un-flagged: it is what the immutable verification command exercises (a flag defaulting
OFF would fail it), and it is a pure *tightening of a detector* — it can only move a
source toward MISSING, i.e. **more gating, never less**. It cannot manufacture a trade.

### Half B — the TOOL VERDICTS. Ship **DARK**.

New `tools_nonfinite_fail_safe_enabled: bool = Field(False, ...)` in
`backend/config/settings.py`, read through the house idiom
`getattr(settings, flag, False)` in a **function-local** import (nothing in
`backend/tools/` imports settings today; a module-level import risks a circular-import
regression), failing **open to legacy** so the gate can never itself crash an analysis.

- `sector_analysis.get_sector_analysis` — guard placed **BEFORE** the comparison ladder.
- `quant_model.get_quant_model_signal` — guard placed **BEFORE** `_classify_signal`, and
  `mda_source` becomes `'non_finite_inputs'` (criterion 6).

**OFF = byte-identical legacy**, matching `sign_safe_overlays` (`settings.py:37`),
`paper_data_integrity_enabled` (`:46`), `paper_synthesis_integrity_enabled` (`:198`),
`paper_swap_churn_fix_enabled` (`:350`), `momentum_52wh_tilt_enabled` (`:449`),
`paper_atomic_swap_enabled` (`:459`).

Why dark: this half changes a **trading verdict**. Every documented gradient points at
fewer/more-gated trades (see §5), but an LLM is in the loop, so it is not a *proof*. A P0
that changes trading behaviour gets operator sign-off.

### Files

| File | Δ |
|---|---|
| `backend/agents/info_gap.py` | `_NAN_TOKEN_RE`, `_has_non_finite()`, rewritten `_assess_source_status`, `quant_model` criticality |
| `backend/tools/sector_analysis.py` | `math` import, `_nonfinite_fail_safe_enabled()`, guard before the ladder |
| `backend/tools/quant_model.py` | `math` import, `_nonfinite_fail_safe_enabled()`, guard before `_classify_signal` + `mda_source` |
| `backend/config/settings.py` | the flag, default `False` |
| `backend/tests/test_phase_80_27_nonfinite_fail_safe.py` | **new**, **29 tests** (24 in cycle 1 + 5 added in cycle 2) |

---

## 2. Verification (verbatim in `live_check_80.27.md`)

```
immutable command                              -> status: MISSING   (pre-fix: SUFFICIENT)
pytest 80.27 + 80.1 + 80.2                     -> 61 passed (cycle 2; was 56)
ruff --select F401,F811,F821 (derived scope)   -> 1 finding, reproduces at HEAD
flag default                                    -> False
```

Live, both flag states, real yfinance data:

| | flag OFF (default) | flag ON |
|---|---|---|
| `sector.signal` | `NEUTRAL` | **`ERROR`** |
| sector non-finite floats | **31** | **0** |
| `quant.signal` | `NEUTRAL` | **`ERROR`** |
| `quant.score` | `nan` | `None` |
| `quant.mda_source` | `'backtest'` | **`'non_finite_inputs'`** |
| `info_gap` verdict | **MISSING** | **MISSING** |

The last row is the safety property: **the detector fires in both states**, so gap
detection and the retry loop are live immediately; only the verdict change waits.

---

## 3. Mutation matrix — **11/11 killed, 0 survived**

Full table in `live_check_80.27.md` §D. The three that matter most:

- **M3 — widen the nan regex to a substring match.** `"financial"`, `"finance"`,
  `"governance"`, `"covenant"`, `"tenant"` and `"nanotech"` all contain `nan`. A naive
  `"nan" in summary.lower()` would mark nearly every real summary MISSING — a
  **self-inflicted analysis outage, strictly worse than the bug being fixed.** The token
  regex uses non-letter boundaries. Independently probed by the researcher: 7/7 positives
  match, 13/14 negatives correctly do not (only a contrived `nan-o` false-positives, and
  a hyphen is a boundary by design).
- **M9 — flag defaults ON.** Guards the dark-launch contract itself.
- **M11 — guard becomes fail-ALWAYS.** A guard returning ERROR on *finite* data is a
  denial-of-service, not a safety feature. Two tests explicitly assert the tools still
  produce real verdicts on finite input.

Plus **M10, a FIXTURE mutation** — the class 80.1 cycle 1 shipped broken and the Q/A
caught. The fixture pins in this suite bind to the subject (`_POISONED_RETURNS`, and the
settings default) rather than asserting a library fact.

---

## 4. Criteria → evidence

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | NaN-poisoned payload classified NOT-SUFFICIENT → critical_gaps → retry fires | immutable command → `MISSING` (was `SUFFICIENT`); `MISSING` is the only status reaching `critical_gaps`; `quant_model` added to the criticality table or it was never assessed at all | **MET** |
| 2 | sector_analysis + quant_model return documented `ERROR`, never NEUTRAL; default to the failure state | live table §B; guards placed **before** the ladders; M6/M7 | **MET (dark)** |
| 3 | No prose or JSON containing `nan` reaches an LLM prompt — assert on the rendered summary AND the serialised payload | **Cycle 1 did NOT meet this** — see §3.1. Cycle 2 added a payload-completeness guard: partial-non-finite input now yields `ERROR` with `json has NaN: False`, while an all-finite control still yields a real `NEUTRAL`. Mutations QM13 + M-payload | **MET (dark) after cycle 2** |
| 4 | Every threshold ladder guarded with `math.isfinite` BEFORE the comparisons; enumerate what you checked, do not assert a count you did not measure | audit-class research ran **6 rounds to 2 dry rounds**, membership rule written down first, then applied: **18 in-set ladders + 9 auditable exclusions**. In this step's scope: L1 (`sector_analysis` `:132-153` + `:55/:70/:78/:87`) and L2 (`quant_model` `:167-181` + the falsy-guard cluster). The rest are enumerated and dispositioned in §6 | **MET** |
| 5 | MUTATION-TEST each guard; "no crash" does not count | **Cycle 1 did NOT meet this** — 7 of 10 Q/A-authored mutations survived, incl. the flag-activation path executed by zero tests (§3.1 D2). Cycle 2: my 11/11 hold AND Q/A's 5 substantive survivors are now killed 5/5; suite 24 → 29 | **MET after cycle 2** |
| 6 | `mda_source` must not report `'backtest'` on non-finite factors | `'non_finite_inputs'`; M8 | **MET (dark)** |

**"MET (dark)"** is stated honestly: criteria 2, 3 and 6 are met *by the code*, verified in
both flag states, and are **inert in production until the operator sets the flag.**

---

## 5. Why this is fail-safe — traced, not assumed

- **ERROR cannot open or close a position that NEUTRAL would not.** The enrichment payload
  never reaches a trade decision as a number: it reaches the per-source LLM agent as JSON,
  the debate prompt as `{signal, summary, analysis}`, and the synthesis prompt as
  `{signal, summary}`. The trade action is `synthesis["recommendation"]["action"]`.
  Removing a *fake* NEUTRAL cannot manufacture bullish evidence; the debate loses a
  fictional data point and gains an explicit "this source failed" marker plus, at ≥2
  errors, a `source_diversity` bias flag telling the synthesiser to lower confidence.
- **ERROR is never *more* actionable than NEUTRAL.** Every consumer treats it as less
  usable: excluded from session memory, not merged on retry, stripped from compacted
  debate prompts, counted into a confidence-lowering bias flag.
- **It cannot crash or stall the loop.** Zero field-level dereferences of these payloads
  exist outside the tools, so an ERROR payload missing `stock_returns`/`score`/
  `top_factors` cannot `KeyError`. `_assess_source_status` already had an ERROR branch;
  the retry helper wraps every call in `try/except`; `_run_single_analysis` sits inside a
  `try/except Exception` whose `_degraded` marker is converted to `None` and **never
  enters `decide_trades`**.
- **It is degradation, not outage.** Every *ticker* is affected but only **2 of 12
  sources**. The other HIGH-criticality yfinance-backed tools were measured live and are
  clean (`monte_carlo` 0 non-finite, `anomaly` 0, `options` 0) — because
  `monte_carlo.py:40` does `hist["Close"].dropna()`. With `quant_model` now tracked,
  `critical_gaps = 2`, so `recommendation_at_risk` (`>=3`) stays **False** and
  `dq = 10/12 = 0.83` vs `data_quality_min` 0.5 — **debate and risk assessment still
  run**, on 10 real sources instead of 10 real + 2 fabricated.

### The real cost, disclosed rather than buried

The failure is **deterministic**, so both `max_retries=2` attempts are guaranteed to fail.
Instrumented: `get_sector_analysis` makes **23** `history()` calls (not the ~14 first
estimated) plus 1 `.info`; `get_quant_model_signal` makes 1 + 1. That is ≈**52 extra
round-trips per ticker per cycle** → ≈**1,040–1,560 per cycle** at 20–30 tickers. Retries
are **sequential with NO backoff** (`info_gap.py:205`; `grep sleep|backoff|jitter` returns
nothing) and there is **no repo-level yfinance rate limiter**. Real HTTP-429 risk.

This cost lands with **Half A**, which ships ON — it is *required* by criterion 1 ("the
existing retry loop fires"), so it is intended behaviour, not an accident. **It is the
strongest argument for sequencing the bad-bar repair BEFORE the operator flips Half B.**

### One behaviour change beyond NaN, stated so it is not discovered later

`_has_non_finite` scans the whole payload, so a **legitimately infinite** ratio — e.g. P/E
for a zero-earnings company — would also mark an otherwise-healthy source MISSING. That is
directionally fail-safe (more gating) and defensible, but it is a change beyond the NaN
case and is recorded rather than left for a reviewer to find.

---

## 6. Out of scope — each with its reason

- **The bad-bar repair (dropping the NaN-OHLC tail row). HARD STOP for this step.**
  Dropping the row *restores real values*, which can turn today's fabricated `NEUTRAL`
  into a genuine `DOUBLE_TAILWIND`/`BULLISH` — a change in the **less conservative**
  direction that could open a position that would not otherwise open. Queue with
  before/after trade-diff evidence. The repo's own working precedent already exists in
  three places (`monte_carlo.py:40`, `screener.py:169-170`, `anomaly_detector.py:66`),
  which is exactly why those tools are clean today.
- **`monte_carlo` (L3).** It currently fabricates `EXTREME_RISK` on NaN; guarding it would
  *remove an alarming input* — the one directionally-less-conservative change in the set.
  Excluded deliberately, not overlooked.
- **L11 / L15 / L16 / L17 — risk-control bypasses.** Non-finite values silently disable the
  **sector-concentration warning** (`orchestrator.py:347-374`) and the **per-ticker limit,
  total-exposure limit and KILL SWITCH** (`mcp_servers/signals_server.py:926`, `:935`,
  `:1285-1287`). **This is the most alarming thing the audit surfaced that is not in this
  step**, and it needs its own research gate.
- **L4/L12/L13** (`anomaly_detector`) — same file as open step **80.31**; coordinate.
- **The 16 prompt-serialisation sites** using stdlib `json.dumps` (all `allow_nan=True`,
  zero with `allow_nan=False`).
- **`price_quality.py` cannot catch this** — US is a fast-path no-op, every rule is
  `.fillna(False)`-masked, and `is_bad_bar(nan,nan,nan,nan,vol)` returns `False`.

---

## 7. DO-NO-HARM

| Item | Status |
|---|---|
| Live book | **Cannot move.** Half A only gates more; Half B is OFF → byte-identical legacy |
| `.env` / flags | Flag **declared** OFF, not enabled. No `.env` edit |
| Optimizer / `historical_macro` | No run; FROZEN |
| Kill-switch / stops / sector caps / DSR / PBO | Not in the diff |
| Crash/stall risk | Traced to zero (§5); the gate itself fails open to legacy |
| Operator `:8000` | Not restarted; `79.55` still open |
| Retry-storm cost | Measured and disclosed (§5), goes to the ask list |

## 8. Tier ledger

| Phase | Role | Model / effort | Why |
|---|---|---|---|
| RESEARCH | Agent-tool `researcher` | **T3** Opus 5 / max | Audit-class, 6 rounds; the shared-code and ERROR-consumer traces were the whole risk |
| GENERATE | Main | **T3** Opus 5 / xhigh | Design fully determined by the gate; the work was careful implementation, not judgment under uncertainty |
| EVALUATE | fresh Q/A | **T3** Opus 5 / max | Independent verdict on the one step that touches trading |

**On Fable/T4:** the goal reserves it for this step. I did not spend it, and the reason is
substantive rather than budgetary: every hard question here — does ERROR crash anything,
is ERROR ever more actionable, how many ladders exist, what does the flag cost — was
answered by **measurement and code-tracing**, not by model judgment. The residual risk is
an LLM-in-the-loop behaviour change, which no amount of model capability at authoring time
resolves; it is resolved by shipping dark and letting the operator decide. Spending scarce
Fable quota where a measurement already gives a decisive answer would be ceremony.

---

## 3.1 CYCLE 2 — two real defects Q/A found, both mine

Cycle-1 verdict was **CONDITIONAL**, violating criteria **3** and **5**. Neither was a
judgment call; both were demonstrated by execution.

### D1 — criterion 3 was NOT met (Overgeneralization). BLOCKING.

My sector guard tested only `sec_3m`/`spy_3m`/`stock_3m` — all three the **3mo** horizon.
But the payload carries `1mo`/`6mo`/`1y` plus five other float maps, and it is that whole
dict that `json.dumps(sector_data, indent=2)` at `backend/config/prompts.py:537` hands to
the sector agent. Q/A demonstrated the leak:

```
[PARTIAL: only 1mo NaN, 3mo finite] flag=True
   signal='NEUTRAL'   json.dumps has NaN: True
   LEAKED LINES: ['"1mo": NaN,', '"1mo": NaN,', '"1mo": NaN,', '"1mo": NaN,']
```

The input class is realistic, not contrived: `_compute_return` is
`Close.iloc[-1]/Close.iloc[0]`, so a bad bar at the **start** of one window poisons that
horizon alone. And `MISSING` from Half A does **not** remove the payload from the prompt —
`orchestrator.py:2056` merges any non-ERROR retry result straight back.

**Fix (option (a) of the two Q/A offered):** a payload-completeness guard —
`_count_non_finite(payload)`; any non-finite anywhere ⇒ `ERROR`. I chose widening over
narrowing the claim because the fail-safe reading is that a partially-fictional analysis is
not fit to be reasoned over. Measured in `live_check_80.27.md` §G, including an all-finite
control proving it is not fail-always.

`quant_model` never had this hole (`nonfinite_feats` already scanned every feature).

### D2 — criterion 5: the flag-activation path was executed by ZERO tests.

Every Half-B test monkeypatched `_nonfinite_fail_safe_enabled` itself, so the **production
flag-read never ran**. Q/A mutated the helper to `return False`, and to read a *misspelled*
settings attribute — **the whole 24-test suite still passed**. The guard could have shipped
permanently dead and green.

This is the third instance of the same family in three consecutive steps (80.1: a
library-fact fixture pin; 80.2: a guard mutating the array instead of the wiring; 80.27:
a guard whose *activation* path is stubbed in every test). The common shape:
**the test replaces the very thing whose correctness it is supposed to establish.**

**Fix:** `test_the_real_flag_read_path_executes` drives the real helpers against a stubbed
`backend.config.settings.get_settings`, plus
`test_flag_read_fails_open_to_legacy_when_settings_explode` pinning the fail-open path.

### Q/A's surviving mutations, now killed — 5/5

```
[QM12-sector]  KILLED -- flag helper hard-wired to return False
[QM12b-sector] KILLED -- flag helper reads a MISSPELLED settings attribute
[QM12-quant]   KILLED -- quant flag helper hard-wired to return False
[QM13]         KILLED -- sector payload-completeness guard removed (D1 leak returns)
[QM14]         KILLED -- quant scans only the score, not every feature
```

Suite **29 passed** (was 24). Five new tests: the D1 partial-NaN pin, the quant
partial-feature pin, the real-flag-read pin, the fail-open pin, and a prose-`inf` case
(D4).

### Also corrected in cycle 2

- **The retry-cost figure.** The contract and the `settings.py` flag description carried
  the researcher's first estimate, **~500-900/cycle**. Its later instrumentation measured
  **23** `history()` calls per `get_sector_analysis` (not ~14), giving ≈**1,040-1,560 per
  cycle**. `settings.py` now carries the measured figure, since a code comment outlives a
  handoff file. The contract's F6 is superseded by this section.
- **The overgeneralised criterion-3 sentence** in `live_check_80.27.md` §B, replaced with
  the measured scope plus an explicit correction note.

### NOTE-level, accepted from Q/A and not fixed

- **D5** — one new regex false positive: **`"Nan Ya Plastics"`**, a real listed
  petrochemical company (`Nan` + space is a non-letter boundary by design). It would mark
  a source MISSING, i.e. fail-safe; recorded rather than special-cased.
- **D7** — the `NO_DATA` addition reduces `dq` whenever `alt_data` rate-limits, an
  always-on effect I did not quantify.

## 3.2 CYCLE 2 (post-verdict) — closing the last WARN, and a vacuity I caught on myself

Q/A cycle 2 returned **PASS** (14/17 mutations killed; 2 proven *equivalent* mutants — the
ladder guard is fully subsumed by the payload guard — and 1 WARN). The WARN:

> **N-A** — narrowing `_count_non_finite(payload)` to `payload["stock_returns"]` leaves the
> suite green at 29/29 while behaviourally reopening the leak for `peers`-only and
> `sector_performance`-only inputs. Production code is correct; only the pin is narrow.

Closed with `test_non_finite_in_sector_performance_only_still_trips_the_guard`. Suite
**30 passed**, and the narrowing mutation now dies (`1 failed, 29 passed`).

**My first version of that test was itself vacuous, and my own mutation caught it.**
I stubbed `_compute_return` to return finite values for *every* ticker, so no NaN ever
reached `sector_performance` — the test passed while the narrowing mutation still
survived. The fix was to make the poison **ticker-specific** (`XLE` returns NaN; AAPL maps
to Technology/XLK, so `stock_returns`/`sector_returns`/`spy_returns` all stay finite and
the ladder guard provably cannot be what fires).

That is the fourth instance this session of the same shape — *the test replaces the very
thing whose correctness it is meant to establish* — and the first one I caught myself,
because I ran the mutation before believing the green. Recorded in
`feedback_mutation_test_guards_and_fixtures`.

### Accepted from cycle 2, queued not fixed

- **N-B — queue a research-gated step.** `_score_ticker` launders a NaN **MDA weight**
  into a clean `0.0` via the `total_weight > 0` guard, producing a confident `NEUTRAL`
  with `mda_source='backtest'`. **Not live today** (Q/A measured 37/37 cache weights
  finite), no NaN reaches a prompt, and it is outside all six criteria — but it is exactly
  this step's bug class on a different input, and it deserves its own gate.
