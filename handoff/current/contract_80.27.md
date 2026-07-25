# Contract — phase-80.27

**Step id:** `80.27` (phase-80, **P0**, `harness_required: true`)
**Title:** *[P0 — A DATA OUTAGE IS LAUNDERED INTO A TRADING VERDICT THAT REACHES THE LIVE
LOOP]* — the most consequential finding of the 2026-07-25 audit, and **not a UI bug**.

Date 2026-07-25 | Wave 1 (NaN family), step 2 of 3.
**Tier T4** — the one step in this drain that changes live decision behaviour.

> **HARD CONSTRAINT, from the goal:** FAIL-SAFE ONLY. A non-finite input must yield
> `ERROR`/`NO_DATA` (→ fewer, more-gated trades), **never** a new trade. Anything that
> could make the system *less* conservative is out of scope and a HARD STOP.

---

## 1. Research gate — PASSED (audit-class, loop-until-dry)

`handoff/current/research_brief_80.27.md`. Envelope: `gate_passed: true`,
`external_sources_read_in_full: 8`, `urls_collected: 38`, `recency_scan_performed: true`,
`internal_files_inspected: 24`, and — because this is audit-class —
`coverage: {audit_class: true, rounds: 6, dry_rounds: 2, K_required: 2, dry: true}`.

### Findings that determine the build

**F1 — the step's stated mechanism is WRONG, and I measured it independently first.**
The bad bar is **not** a "still-forming current session" placeholder. Measured
2026-07-25 17:45 UTC, a **Saturday**, markets closed ~22h:

```
ticker  rows  lastIdx      lastClose      lastVol   NaN-close?
AAPL     125  2026-07-24         nan     47402209   YES
SPY      125  2026-07-24         nan     41480799   YES
XLK      125  2026-07-24         nan      7725125   YES
NVDA     125  2026-07-24         nan    110844026   YES
MSFT     125  2026-07-24         nan     27544509   YES
JPM      125  2026-07-24         nan      7513084   YES
```

AAPL `history(period="1mo")`: NaN-Close rows **1 of 21**; rows with NaN Close AND
Volume > 0: **1**. Exactly one bad row — the most recent **completed** session — with
NaN OHLC and a real Volume. It is a persistent yfinance 1.2.0 defect, and the real
Volume is precisely why yfinance's own `keepna` mask (`.all(axis=1)`) does not drop it.
**Consequence: every ticker, every cycle, all the time — not market-hours-gated.**

**F2 — flipping to ERROR is NOT a total analysis outage.** `data_quality_score` falls
`1.0 → 0.91` (0.83 if `quant_model` is tracked), both far above the `0.5`
`data_quality_min`. Debate, risk assessment and synthesis still run. The pipeline loses
two of twelve enrichment inputs — **the two currently contributing pure fiction.**

**F3 — ERROR is fully handled downstream; no crash or stall is possible.** Zero
field-level dereferences of these payloads exist outside the tools; `_assess_source_status`
already has an `ERROR` branch (`info_gap.py:47`); the retry helper wraps every call in
`try/except` (`:180-183`); and `_run_single_analysis` is inside a `try/except Exception`
(`autonomous_loop.py:1944`) whose `_degraded` marker is converted to `None` and **never
enters `decide_trades`**.

**F4 — ERROR is never *more* actionable than NEUTRAL.** Every consumer treats it as less
usable: excluded from session memory (`:2003`), not merged on retry (`:2046`), stripped
from compacted debate prompts (`:2181`), and it raises a confidence-lowering
`source_diversity` bias flag (`bias_detector.py:231-246`). **No code path lets ERROR
unlock an action NEUTRAL does not.**

**F5 — `quant_model` is NOT in `_SOURCE_CRITICALITY`** (11 keys, not 12). Criterion 1
needs that key added or the quant path is never assessed at all.

**F6 — the real cost, and it must not be buried.** The failure is *deterministic*, so both
retries are guaranteed to fail. One `get_sector_analysis` makes ~14 `yf.Ticker().history()`
round-trips (stock ×4 periods + sector ETF ×4 + SPY ×4 + 11 sector ETFs for the rotation
chart) plus 1-6 `.info` calls. At ~20-30 tickers/cycle that is **~500-900 extra yfinance
requests per cycle** — a real HTTP-429 risk on a free endpoint.

**F7 — line-number correction:** the `sector_analysis` initialiser is **`:132`**, not
`:136` as the step body states.

---

## 2. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

> 1. A NaN-poisoned enrichment payload is classified NOT-SUFFICIENT by _assess_source_status, so a HIGH-criticality source routes into critical_gaps and the existing retry loop fires
> 2. sector_analysis and quant_model return the DOCUMENTED signal: 'ERROR' (or an explicit NO_DATA) on non-finite inputs -- never NEUTRAL. Initialising signal to NEUTRAL before a comparison ladder is itself the bug: default to the failure state
> 3. No prose or JSON containing 'nan' can reach an LLM prompt -- assert on the rendered summary string and on the serialised payload the sector agent receives
> 4. Every threshold ladder that can receive a computed float is guarded with math.isfinite BEFORE the comparisons (sector_analysis.py:140-153, quant_model.py:171-181). Enumerate the ladders you checked; do not assert a count you did not measure
> 5. MUTATION-TEST each guard: feed float('nan') and assert the verdict is ERROR/NO_DATA, then remove the guard and confirm the test FAILS. A test that only asserts 'no crash' does not count
> 6. quant_model must not report mda_source='backtest' on a payload whose factors are non-finite

**Immutable verification command** (verbatim):

```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import math; from backend.agents.info_gap import _assess_source_status; p={'signal':'NEUTRAL','summary':'3M return: +nan% vs sector +nan%','stock_returns':{'1mo':float('nan')}}; print('status:', _assess_source_status('sector', p))"
```

**live_check:** `handoff/current/live_check_80.27.md` — (a) `_assess_source_status`
returning a non-SUFFICIENT status for a poisoned payload, (b) sector_analysis and
quant_model returning ERROR/NO_DATA rather than NEUTRAL on non-finite input, (c) the
mutation runs proving each guard fails without the fix.

---

## 3. Plan — a deliberately ASYMMETRIC shipping shape

The two halves have different risk profiles, so they ship differently. This is the
central design decision of the step.

### Half A — the DETECTOR. Ships **ON** (un-flagged).

`backend/agents/info_gap.py::_assess_source_status` gains a **non-finite numeric scan**,
and `quant_model` is added to `_SOURCE_CRITICALITY` (F5).

Why un-flagged: (i) it is what the **immutable verification command actually exercises**,
and that command must pass — a flag defaulting OFF would fail it; (ii) it is a pure
*tightening of a detector*: it can only move a source toward MISSING/NOT-SUFFICIENT, i.e.
**strictly more gating, never less**. It cannot manufacture a trade.

### Half B — the TOOL VERDICTS. Ship **DARK**, behind a default-OFF flag.

`sector_analysis.get_sector_analysis` and `quant_model.get_quant_model_signal` return the
documented `signal: 'ERROR'` on non-finite input, gated by a new
`tools_nonfinite_fail_safe_enabled: bool = Field(False, ...)` in
`backend/config/settings.py`, read via the house idiom `getattr(settings, "<flag>", False)`.

**OFF = byte-identical legacy output**, matching the established precedent in this repo:
`sign_safe_overlays` (`settings.py:37`), `paper_data_integrity_enabled` (`:46`),
`paper_synthesis_integrity_enabled` (`:198`), `paper_swap_churn_fix_enabled` (`:350`),
`momentum_52wh_tilt_enabled` (`:449`), `paper_atomic_swap_enabled` (`:459`).

Why dark: this is the half that changes a **trading verdict**. F4 says every documented
gradient points at fewer/more-gated trades, but an LLM is in the loop, so it is *not a
proof*. A P0 that changes trading behaviour gets operator sign-off. **The flip token goes
on the ask list.**

**Import caution (F-brief):** `backend/tools/*` currently import **no** settings module.
Use a function-local `from backend.config.settings import get_settings` — a module-level
import risks a circular-import regression.

### Criterion 4 — the ladder enumeration (measured, not asserted)

The audit ran 6 rounds to 2 dry rounds and enumerated **18 in-set ladders + 9 auditable
exclusions**, with the membership rule written down. In 80.27's scope: **L1** (
`sector_analysis` `:132-153` + `:55/:70/:78/:87`) and **L2** (`quant_model` `:167-181` +
the falsy-guard cluster `:47-48`, `:84-85`, `:167`, `:234`). The contract carries the
count I measured; I do not claim the set is exhaustive beyond the stated rule.

### Tests + mutation matrix (criterion 5)

Every guard mutation-tested: feed `float('nan')`, assert `ERROR`/NO_DATA, then remove the
guard and confirm the test **FAILS**. Plus a **flag-OFF byte-identity test** (the 52.2/70.3
precedent) and a **fixture mutation** (feed a finite float and assert the test fails).

---

## 4. Explicitly OUT of scope — each with its reason

- **The bad-bar repair (dropping the NaN-OHLC tail row).** This is the tempting root fix
  and it is a **HARD STOP for this step**: dropping the row *restores real values*, which
  can turn today's fabricated `NEUTRAL` into a genuine `DOUBLE_TAILWIND`/`BULLISH`. That is
  a change in the **less conservative** direction and could open a position that would not
  otherwise open. Queue as its own research-gated step with before/after trade-diff
  evidence. Precedent for the repair already exists at `screener.py:169-170` — which is
  exactly why the screener funnel is *not* NaN-poisoned today.
- **`monte_carlo` (L3).** It currently fabricates `EXTREME_RISK` on NaN. Guarding it would
  *remove an alarming input* — the one directionally-less-conservative change in the set.
  Documented and queued, not ridden in silently.
- **L11, L15, L16, L17** — non-finite values silently suppressing **sector-concentration,
  position-limit and kill-switch controls** on the portfolio/MCP path. Higher stakes,
  different blast radius; deserves its own research gate. **This is the most alarming thing
  the audit surfaced that is not in this step.**
- **L4/L12/L13** (`anomaly_detector`) — same file as open step **80.31**; coordinate rather
  than collide.
- **The 16 prompt-serialisation sites using stdlib `json.dumps`** (all `allow_nan=True` by
  default, zero with `allow_nan=False`). Criterion 3 is satisfied for the two tools in
  scope by their ERROR summaries; a repo-wide prompt-serialisation fix is its own step.

---

## 5. DO-NO-HARM

- **The live book does not move in the risky direction.** Half A can only gate more.
  Half B ships OFF → byte-identical legacy until the operator flips it.
- No `.env` edit, no flag *flip* (the flag is *declared* OFF, not enabled), no optimizer
  run, `historical_macro` FROZEN. Kill-switch limits, stops, sector caps, DSR and PBO
  byte-untouched.
- **The retry-storm cost is disclosed, not buried** (F6): Half A shipping ON means the
  retry loop starts firing on these sources — ~500-900 extra yfinance requests/cycle,
  HTTP-429 risk. It is *required* by criterion 1 ("the existing retry loop fires"), so it
  is intended behaviour, but the operator must know. Goes in the log and the ask list.
- **Inert until restart regardless** — `phase-79.55` is an open RESTART BLOCKER.
- `git add -An` before the flip.

## 6. Evidence to produce

`experiment_results_80.27.md` · `live_check_80.27.md` · `evaluator_critique_80.27.md`
(Q/A verdict, transcribed verbatim) · `harness_log.md` append **before** the flip.
