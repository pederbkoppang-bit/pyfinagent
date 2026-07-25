# Cycle block summary + batched operator ask list — 2026-07-25

Session: masterplan drain, Wave 0 + Wave 1. **Three P0s closed and pushed.**

| step | verdict | commit | what |
|---|---|---|---|
| **80.2** | PASS (cycle 2) | `9457a88d` | 500s now carry CORS + OWASP + a PerfTracker row |
| **80.1** | PASS (cycle 2) | `68427db6` | `/api/signals/{ticker}` 500 → 200, NaN → `null` |
| **80.27** | PASS (cycle 2) | `8b1c7158` | a data outage can no longer become a trading verdict |

Plus `0c569eb6` — tracked the phase-80 UI-audit evidence base (31 files, 5.0M) that three
earlier sessions had left untracked and that would otherwise have been swept into whichever
step flipped next.

**Measured backlog (re-derived, not inherited):** 222 open steps; P0 **26** / P1 **44** /
P2 **92** / P3 40 / P4 8 / unset 12; **53** are `harness_required: false` operator actions
(all phase-79). The draft's P1 45 / P2 91 / phase-80 31 were off by one each.

---

# OPERATOR ASK LIST

## A. Blocking this session's own work — please action first

**A1. `phase-79.55` — RAIL-MODEL TIER CONFIRMATION (P0, `pending`).**
An explicit `[RESTART BLOCKER -- answer BEFORE the next backend restart]`. It is why all
three fixes above are **built but inert on the running `:8000`**. Restarting would activate
the phase-78.2 rail re-tiering (the six signal overlays down to `claude-haiku-4-5`, the lite
trader and lite risk judge to `settings.gemini_model`) before you have answered.
**Answer it, then restart.** Until then every phase-80 backend fix ships dark.

> `phase-79.2` (BACKEND RESTART) is already `done` — a restart happened at 2026-07-25
> 11:39:05, pid 70791. `79.55` gates the *next* one.

**A2. `tools_nonfinite_fail_safe_enabled=true` — the 80.27 flip token.**
Declared OFF. **Until you flip it, the 80.27 defect is still live**: a NaN-poisoned sector
payload still reaches the LLM prompt. Two things to weigh first:
- **Cost, instrumented:** the failure is deterministic so both retries are guaranteed to
  fail — ~**1,040–1,560 extra yfinance requests per cycle** at 20–30 tickers, **sequential,
  no backoff, no rate limiter**. Real HTTP-429 risk.
- **Sequencing:** fixing the malformed tail bar first removes the cause and makes the retry
  storm moot. That repair is a separate queued step because it is *less* conservative
  (it restores real values, which can turn a fabricated NEUTRAL into a real BULLISH).

**A3. `rm -rf frontend/.next-audit-3100`** — 208M, gitignored, harmless but reclaimable.
Policy-denied to me four times this session.

## B. The other 50 phase-79 operator actions

Unchanged and still owed. P0s: **79.1** (PROMOTE-66.2 flags, approved 2026-07-09 and never
applied), **79.3** (Anthropic direct-API credit decision), **79.4**
(`AUTORESEARCH_USE_MAX_RAIL=1`), **79.5** (ops-bridge bootstrap).

Worth surfacing early from the P1/P2 tail: **79.21** `backend/.env` line 81 has an
unbalanced quote · **79.25** the leaked FRED API key is still the live key · **79.24**
11 of 11 secrets overdue for rotation · **79.20** plaintext `AUTH_SECRET` /
`AUTH_GOOGLE_SECRET` in the frontend launchd plist · **79.15** Slack bot 42 days of
unloaded code including a P0 pager fix.

---

# DEFECTS DISCOVERED AND QUEUED (not silently fixed)

Per `feedback_queue_discovered_defects_in_masterplan`. Each needs its own research-gated
step; none were bundled into the steps that found them.

**Highest severity — risk-control bypasses, found by the 80.27 audit and NOT in that step:**
non-finite values silently disable the **sector-concentration warning**
(`orchestrator.py:347-374`) and the **per-ticker limit, total-exposure limit and KILL
SWITCH** (`mcp_servers/signals_server.py:926`, `:935`, `:1285-1287`). Same bug class as
80.27, on the portfolio/MCP path, higher stakes.

1. **The bad-bar repair.** yfinance leaves the most recent *completed* session with NaN
   OHLC and a real Volume, on every ticker, permanently. Repairing it is **less
   conservative** (restores real values → can open a position that would not otherwise
   open), so it needs its own gate with before/after trade-diff evidence. Precedent exists
   in-repo: `monte_carlo.py:40`, `screener.py:169-170`, `anomaly_detector.py:66`.
2. **`monte_carlo` (L3)** fabricates `EXTREME_RISK` on NaN — guarding it *removes* an
   alarming input, the one directionally-less-conservative change in the set.
3. **`_score_ticker` launders a NaN MDA weight** into `0.0` via `total_weight > 0` →
   confident NEUTRAL with `mda_source='backtest'`. Not live today (37/37 cache weights
   finite).
4. **Three components bypass `apiFetch`** and get none of 80.2's error handling:
   `ResearchInvestigator.tsx:33`, `Sidebar.tsx:155`, `StockChart.tsx:94`.
5. **`frontend/eslint.config.mjs:11`** ignores `.next/**` but not `.next-*/**`, so any
   audit-rig dist dir breaks `npx eslint .` repo-wide — silently degrades the frontend lint
   gate for every future step.
6. **`nlp_sentiment.py:161`** — `np.mean([])` clamped into a silently wrong `1.0`.
7. **The analysis-report poll route** embeds the same `quant_model` dict and may 500
   identically. **NOT VERIFIED** — flagged, not asserted.
8. **Doc drift:** `.claude/rules/security.md` requires `X-XSS-Protection: 1; mode=block`;
   `main.py` sets `0` (the modern-correct value). Reconcile the doc.
9. **The 16 prompt-serialisation sites** all use stdlib `json.dumps` (`allow_nan=True`),
   zero with `allow_nan=False`.
10. **Frontend `null` rendering** — a null signal value should render as an explicit "data
    unavailable" state, not blank or zero, or "green" still hides missing data.

---

# THE PATTERN WORTH CARRYING FORWARD

Four vacuous guards in four steps, one shape:
**the test replaces the very thing whose correctness it is supposed to establish.**

- **80.2** — mutated the exported array the test imports, not the *wiring*. Q/A reverted the
  call site and every test stayed green.
- **80.1** — `assert not math.isfinite(float("nan"))`: a library fact posing as a fixture
  pin. Passed under the fixture mutation it claimed to guard.
- **80.27 (D2)** — every test stubbed the flag-read helper, so the *production* flag read
  ran in **zero** tests; wiring it to a misspelled settings key left 24 tests green.
- **80.27 (N-A)** — my closure test stubbed `_compute_return` for *every* ticker, so the
  poison never reached the code path. **Caught by my own mutation** — the only one of the
  four I found myself, because I ran the mutation before believing the green.

Derived pre-flight check, now written into
`feedback_mutation_test_guards_and_fixtures`: *name what this test stubs, mocks, imports
directly or monkeypatches — then ask whether the criterion is a claim ABOUT that thing.*
Also: **never claim "0 vacuous"** — a matrix licenses only "these N mutations were killed".

---

# NEXT

Wave 2: `80.3` (agent-map renders zero edges) and `80.4` (false "Disconnected"), then the
remaining open P0s oldest-first across phases 27, 61, 62, 63, 65, 68, 72.
