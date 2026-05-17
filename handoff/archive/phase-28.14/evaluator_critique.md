# Evaluator Critique -- phase-28.14 -- Defense/war-stocks reference case

**Step ID:** phase-28.14
**Date:** 2026-05-17
**Cycle:** 1
**Verdict:** **PASS**
**Author:** Q/A subagent (Claude Code, Opus 4.7 xhigh)

---

## Harness-compliance audit (5-item)

| # | Item | Status |
|---|------|--------|
| 1 | Researcher spawned before contract | PASS — `handoff/current/phase-28.14-research-brief.md` (155 lines; 6 sources READ IN FULL; 15 URLs collected; recency scan performed; hard blockers all checked) |
| 2 | Contract written pre-GENERATE | PASS — `handoff/current/contract.md` exists with all 5 immutable criteria copied verbatim from masterplan, hypothesis, plan steps |
| 3 | Experiment_results exists with verbatim verification command output | PASS — `handoff/current/experiment_results.md` |
| 4 | Log-last discipline | DEFERRED to Main (post-PASS append, pre-status-flip) — 0 prior phase-28.14 entries in harness_log.md (clean slate) |
| 5 | No-verdict-shopping | PASS — 0 prior phase-28.14 CONDITIONALs; this is cycle 1 (fresh evaluation, not a re-spawn on unchanged evidence) |

---

## Deterministic checks (Q/A reproduction)

### 1. Immutable verification command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/defense_signal.py').read()); print('syntax OK')" && grep -q 'defense_signal_enabled' backend/config/settings.py && grep -qE 'ITA|XAR' backend/services/defense_signal.py && echo "IMMUTABLE: PASS"
syntax OK
IMMUTABLE: PASS
```
Exit 0.

### 2. 4-file syntax
```
OK backend/services/defense_signal.py
OK backend/config/settings.py
OK backend/tools/screener.py
OK backend/services/autonomous_loop.py
```

### 3. Settings fields (all 7 expected present)
```
286: defense_signal_enabled: bool = Field(False, ...)
287: defense_xar_window_days: int = Field(5, ...)
288: defense_xar_min_momentum: float = Field(0.0, ...)
289: defense_tickers: str = Field("LMT,NOC,RTX,GD,LHX,BA,LDOS,HII,KTOS,BAE.L,RHM.DE,SAAB-B.ST", ...)
290: defense_boost: float = Field(0.05, ...)
291: defense_budget_pledge_keywords: str = Field("NATO budget,defense spending,Zeitenwende,...", ...)
```
Default-OFF verified at line 286.

### 4. Public-API imports
```
$ python -c "from backend.services.defense_signal import fetch_defense_trigger, apply_defense_boost_to_score, DefenseSignal; print('IMPORT OK')"
IMPORT OK
```

### 5. GPR reuse (no duplicate fetch)
```
backend/services/defense_signal.py:104:    from backend.services.macro_regime import _fetch_gpr_acts
backend/services/defense_signal.py:105:    gpr_info = await _fetch_gpr_acts(cache_hours=gpr_cache_hours, quantile=gpr_quantile)
```
Upstream: `backend/services/macro_regime.py:111` confirmed exists. No duplicate FRED / Caldara-Iacoviello fetch logic in defense_signal.py.

### 6. rank_candidates kwarg
```
backend/tools/screener.py:231:    defense_signal=None,
backend/tools/screener.py:352-354:  apply block (after social_velocity)
        if defense_signal:
            from backend.services.defense_signal import apply_defense_boost_to_score
            score = apply_defense_boost_to_score(score, stock.get("ticker"), defense_signal)
```

### 7. autonomous_loop cycle-level pre-fetch
```
backend/services/autonomous_loop.py:303-319: cycle-level fetch when flag on; non-fatal on failure
backend/services/autonomous_loop.py:502: defense_signal=defense_signal_obj passed into rank_candidates
```

### 8. Unit tests on apply_defense_boost_to_score (Q/A wrote + ran)
| Test | Input | Expected | Actual | Result |
|------|-------|----------|--------|--------|
| T1 triggered + defense ticker | (10.0, 'LMT', sig_t) | 10.5 | 10.5 | PASS |
| T2 not triggered | (10.0, 'LMT', sig_f) | 10.0 | 10.0 | PASS |
| T3 non-defense + triggered | (10.0, 'AAPL', sig_t) | 10.0 | 10.0 | PASS |
| T4 lowercase ticker | (10.0, 'lmt', sig_t) | 10.5 | 10.5 | PASS (case-insensitive) |
| T5 EU ticker | (10.0, 'BAE.L', sig_t) | 10.5 | 10.5 | PASS |
| T6 None signal | (10.0, 'LMT', None) | 10.0 | 10.0 | PASS |
| T7 None ticker | (10.0, None, sig_t) | 10.0 | 10.0 | PASS |

7/7 PASS.

### 9. Live fetch_defense_trigger (real GPR + XAR)
```
triggered=False
  gpr_above=True current=285.3481 thr=184.9315
  xar_5d_momentum=-0.0176 above=False
  boost_multiplier=1.0
  defense_tickers (count)=12
LIVE FETCH OK
```
Real-time AND-gate evaluation verified: GPR above threshold, XAR below threshold → conservative non-firing, matches design intent.

---

## LLM judgment (5 dimensions)

### Contract alignment
All 5 immutable criteria evidenced in experiment_results.md success-criteria mapping table. Hypothesis (AND-gate on GPR + XAR with US+EU defense ticker boost, default OFF, Gap 1 evidence) is implemented exactly as specified.

### Default-OFF & blast radius
- `defense_signal_enabled = False` (settings.py:286) — production behavior unchanged.
- Cycle-level fetch (one per cycle, not per-ticker) bounds external network cost to 1 yfinance call + reuse of cached GPR.
- Graceful degradation: any fetch failure → triggered=False → identity boost (multiplier=1.0).

### AND-gate vs OR-gate rationale
Module docstring lines 14-17 explicitly explains:
> "GPR alone fires on any geopolitical event — including ones that DON'T move defense stocks. XAR positive momentum confirms institutional flow is actually pricing the GPR signal into defense. Both together = high-confidence convergence."

Today's live data validates the design — GPR=285.35 (above) but XAR=-1.76% (below) → triggered=False. The conservative AND-gate prevents firing on stale GPR signal that defense markets aren't pricing. This is exactly the noise-suppression behavior described.

### GPR reuse (phase-28.3 fetcher)
Imports `_fetch_gpr_acts` directly from `backend.services.macro_regime` (line 104). No duplicate FRED API code, no duplicate quantile computation, no duplicate Caldara-Iacoviello dataset access. Free (cached) GPR check — the criterion explicitly required reuse of the phase-28.3 fetcher.

### XAR-preferred-over-ITA documentation
Documented in (a) module docstring line 12 ("XAR preferred over ITA (ITA 19% GE/commercial-aviation noise)"), (b) live_check_28.14.md ("Why XAR was preferred over ITA"), and (c) research brief. The immutable verification grep accepts `ITA|XAR` — XAR is present, satisfying the OR-grep.

### US+EU ticker coverage
12 tickers: 9 US (LMT, NOC, RTX, GD, LHX, BA, LDOS, HII, KTOS) + 3 EU (BAE.L, RHM.DE, SAAB-B.ST). Researcher noted BAE/RHM most GPR-sensitive — both present. SAAB-B.ST extends to Swedish prime. Live check confirmed `defense_tickers (count)=12`.

---

## Code-review heuristics scan

| Dimension | Findings |
|---|---|
| Security audit | No secrets, no command injection, no eval/exec, no insecure deserialization. yfinance is the only external API. |
| Trading-domain correctness | Default OFF preserves all kill-switch / stop-loss / max-position / perf-metrics invariants. Boost is multiplicative (1.05x); does not bypass `risk_engine.py` position sizing or `paper_trader.py` guards downstream. No `crypto` re-enable. |
| Code quality | The two `except Exception` blocks (defense_signal.py:64, 80) are around yfinance fetch / compute with explicit fallback (return None → triggered=False) — graceful degradation pattern, NOT the broad-except-silences-risk-guard anti-pattern. The block at line 106 (GPR fetch) and line 131 (pledge_hit_provider) follow the same pattern. ASCII-only logger messages confirmed. Type hints present on public API. Pydantic `extra="forbid"`. |
| Anti-rubber-stamp | Behavioral tests cover 7 paths (triggered + ticker, not triggered, non-defense ticker even when triggered, case-insensitivity, EU ticker recognition, None signal, None ticker). No tautological assertions. Real live-data fetch exercised. |
| LLM-evaluator | Cycle 1 — no prior verdict to flip, no sycophancy risk. Code-review findings cited with file:line evidence per Cloudflare pattern. |

No BLOCK or WARN findings.

---

## Verdict justification

All 5 immutable success criteria evidenced. Deterministic checks: 9 PASS, 0 FAIL. 7/7 unit tests PASS. Live AND-gate evaluation against real GPR + real XAR data confirms the design correctly suppresses on partial signal (conservative non-firing). GPR fetcher reuse verified (no duplicate code). XAR-over-ITA rationale documented. US+EU ticker coverage complete (12 tickers, BAE+RHM included). Default-OFF blast-radius minimal. Research gate: 6 sources read in full, 15 URLs collected, hard blockers checked. No code-review heuristic findings.

**Verdict: PASS.**

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable criteria met: defense_signal module created using phase-28.3 GPR fetcher (no duplicate code); XAR (not ITA) momentum implemented with rationale documented; pledge keyword set in settings.py:291; default-OFF at settings.py:286; live_check_28.14.md documents real GPR=285.35 above + XAR -1.76% below -> AND-gate not triggered + synthetic triggered case showing +5% on LMT/NOC/BAE.L. Deterministic: immutable cmd exit=0, 4-file syntax OK, 7/7 unit tests PASS, live fetch_defense_trigger OK.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "settings_defaults", "public_api_import", "gpr_reuse_grep", "rank_candidates_signature", "unit_tests_apply_boost", "live_fetch_end_to_end", "code_review_heuristics", "harness_compliance_audit"]
}
```

---

## Next steps for Main

1. Append cycle entry to `handoff/harness_log.md` (`## Cycle 29 -- 2026-05-17 -- phase=28.14 result=PASS`).
2. Flip `.claude/masterplan.json` phase-28.steps[14].status -> done.
3. Auto-commit-and-push hook should fire (live_check_28.14.md present, gate satisfied).
4. Continue with Supplement tier 2/4 work.
