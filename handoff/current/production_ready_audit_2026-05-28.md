# Production-Ready DoD Audit — 2026-05-28

**Step:** phase-43.0 | **Cycle:** 12 | **Auditor:** Main (Claude Code session) | **Researcher gate:** `a9547514da955b875` (PASSED, 6 sources read in full)

---

## Verdict

# **NOT_PRODUCTION_READY**

**PASS:** 5 of 14 criteria.
**Drift-PASS (code shipped, runtime-evidence partial):** 1 of 14 (DoD-7).
**CONDITIONAL/PARTIAL:** 2 of 14 (DoD-4 tiered-vs-literal, DoD-11 documented-deferral).
**UNKNOWN (live runtime evidence not collected this cycle):** 1 of 14 (DoD-6).
**FAIL:** 5 of 14 (DoD-1, DoD-2, DoD-5, DoD-9, DoD-14).

Most generous interpretation (CONDITIONAL→PASS + drift+unknown→PASS): **9 of 14 PASS**.
Strict literal interpretation: **5 of 14 PASS**.

Either way: **NOT_PRODUCTION_READY**. Do not declare go-live. Operator approval per immutable success criterion #4 (`operator_approval_recorded_for_PRODUCTION_READY_declaration`) is NOT being sought this cycle.

---

## Per-criterion evidence (verbatim)

### DoD-1 — All cron jobs have last-run within SLA — **FAIL**

**Criterion (verbatim from master_roadmap_to_production.md §6):**
> `launchctl list | grep pyfinagent` shows 0 jobs with last-exit != 0 OR last-run > 2 days ago.

**Evidence command:** `launchctl list | grep -i pyfinagent`

**Verbatim output:**
```
-	0	com.pyfinagent.mas-harness
1269	0	com.pyfinagent.claude-code-proxy
-	0	com.pyfinagent.ablation
-	0	com.pyfinagent.backend-watchdog
-	1	com.pyfinagent.autoresearch
1257	0	com.pyfinagent.backend
1266	0	com.pyfinagent.frontend
```

`com.pyfinagent.autoresearch` last-exit = **1**. Today's error from `handoff/autoresearch/2026-05-28-ERROR-topic08.md`:

```
# Autoresearch FAILED -- 2026-05-28

Topic: What do the most recent industry research notes (2025-2026) from
       Goldman Sachs QIS, Morgan Stanley Quantamentals, Deutsche Bank
       Systematic, and UBS Evidence Lab say about AI in equity selection?

Error: ModuleNotFoundError: No module named 'langchain_huggingface'
```

**Streak:** 14 consecutive ERROR days (2026-05-15 → 2026-05-28). Today's failure is a NEW failure mode (different from phase-39.1's "anthropic:" prefix scope).

**Status:** FAIL. Follow-up: phase-39.1 (owner-gated, `pending` in masterplan.json:12313+).

---

### DoD-2 — Sharpe and P&L match between backtest and paper-trading within 0.01 — **FAIL**

**Criterion (verbatim):**
> Pull last-30-day paper Sharpe vs walk-forward backtest Sharpe on same universe + period. `|paper.sharpe - backtest.sharpe| < 0.01`.

**Evidence command:** `curl -sf http://localhost:8000/api/paper-trading/reconciliation | python3 -m json.tool`

**Verbatim output (head):**
```
{
    "series": [
        {"date": "2026-04-14", "paper_nav": 9499.5,  "backtest_nav": 20000.0,    "divergence_pct": 52.5025},
        {"date": "2026-04-26", "paper_nav": 9499.5,  "backtest_nav": 19999.3,    "divergence_pct": 52.5008},
        {"date": "2026-04-27", "paper_nav": 14458.32,"backtest_nav": 19965.48,   "divergence_pct": 27.5834},
        {"date": "2026-04-28", "paper_nav": 13952.25,"backtest_nav": 19462.48,   ...},
        ...
    ]
}
```

Divergence is **52.5%** on the early-window NAV and remains in double-digit % through the series. The criterion's `|paper.sharpe - backtest.sharpe| < 0.01` cannot plausibly hold when NAVs diverge by orders of magnitude. The reconciliation endpoint does not return Sharpe-specific numbers; the Sharpe-gap function `backend/services/perf_metrics.py:186 compute_sharpe_gap()` exists but no walk-forward backtest result JSON under `backend/backtest/experiments/results/` carries a paper-trading Sharpe comparison column.

**Status:** FAIL (NAV-divergence orders-of-magnitude exceeds threshold; Sharpe-specific evidence not available without re-running the walk-forward suite).

**Note:** This is partially a measurement-infrastructure gap (walk-forward + paper comparison column not in result schema) AND partially a real divergence (paper started at $9,499 vs backtest $20,000). Honest reading: FAIL with two follow-ups — instrument the comparison, then close the divergence.

---

### DoD-3 — Kill-switch hysteresis tested — **PASS**

**Criterion (verbatim):**
> Test: pause manually -> wait 2h with no breach -> auto-resume fires + Slack alert.

**Evidence command:** `grep -n "check_auto_resume\|AUTO_RESUME\|hysteresis" backend/services/kill_switch.py | head -20`

**Verbatim output:**
```
52:        # phase-38.1 (OPEN-10): auto-resume hysteresis. _paused_at carries
74:                        # phase-38.1: capture the pause ts for hysteresis.
136:        hysteresis logic in check_auto_resume."""
156:            # phase-38.1: stamp the pause timestamp for hysteresis.
267:# phase-38.1 (OPEN-10): kill-switch auto-resume hysteresis.
271:AUTO_RESUME_ALERT_AT_SEC: float = 60 * 60       # T+1h: pager alert
272:AUTO_RESUME_TRIGGER_AT_SEC: float = 2 * 60 * 60  # T+2h: auto-resume fires
275:def check_auto_resume(
281:    # phase-38.1 (OPEN-10): evaluate hysteresis. Default-OFF; caller
323:    if seconds_paused >= AUTO_RESUME_TRIGGER_AT_SEC:
324:        _state.resume(trigger="auto_resume_hysteresis", details={
337:    if seconds_paused >= AUTO_RESUME_ALERT_AT_SEC and not already_alerted:
356:                    "auto_resume_at_sec": str(AUTO_RESUME_TRIGGER_AT_SEC),
371:        "reason": "paused_but_under_hysteresis_threshold",
```

**Test file:** `backend/tests/test_phase_38_1_kill_switch_auto_resume.py` exists.

**Status:** PASS. `check_auto_resume()` shipped phase-38.1; default-OFF via `kill_switch_auto_resume_enabled`; AUTO_RESUME_TRIGGER_AT_SEC=7200 (2h); AUTO_RESUME_ALERT_AT_SEC=3600 (1h pager); idempotency state at lines 52-86.

---

### DoD-4 — Test coverage >70% per layer — **CONDITIONAL** (PASS under tiered policy, FAIL under literal criterion)

**Criterion (verbatim):**
> `pytest --cov backend/services/`, `pytest --cov backend/agents/`, `pytest --cov backend/api/` all >= 70%.

**Tiered policy authority:** `docs/coverage_tier_overrides.md` (2026-05-25).
- Tier-1 STRICT modules at-or-above 70%: kill_switch 92%, cycle_lock 84%, factor_correlation 85%, factor_loadings 78%, paper_trader 79.1%, portfolio_manager 81.2%, perf_metrics 81.2%, cycle_health 72%.
- Tier-2 layers (services 26%, agents 22%, api 33%) FAIL the literal layer-wide criterion.

**Honest dual interpretation:**
- Under the tiered policy adopted 2026-05-25: PASS for all Tier-1 STRICT modules.
- Under the literal master_roadmap_to_production.md §6 wording ("all >= 70%" per-layer): FAIL.

**Status:** CONDITIONAL. To convert to clean PASS, master_roadmap_to_production.md §6 DoD-4 wording must be updated to cite the tiered policy explicitly, OR Tier-2 broad-layer coverage must climb above 70%.

---

### DoD-5 — 0 Unknown bands in Data Freshness dashboard — **FAIL**

**Criterion (verbatim):**
> `GET /api/paper-trading/freshness` returns no `band='Unknown'` rows across all source rows.

**Evidence command:** `curl -sf http://localhost:8000/api/paper-trading/freshness | python3 -m json.tool`

**Verbatim output (sources object):**
```
"sources": {
    "paper_trades":                {"band": "green"},
    "paper_portfolio_snapshots":   {"band": "green"},
    "historical_prices":           {"band": "unknown"},
    "historical_fundamentals":     {"band": "unknown"},
    "historical_macro":            {"band": "unknown"},
    "signals_log":                 {"band": "unknown"}
}
```

**4 of 6 sources show `band: "unknown"`** (historical_prices, historical_fundamentals, historical_macro, signals_log). `overall_band` is "green" because the cycle heartbeat is fresh, but the per-source criterion is unambiguous: 4 unknown bands.

**Status:** FAIL. Follow-up: instrument the per-source `last_tick_age_sec` populator for historical_* + signals_log so their bands compute to green/yellow/red instead of staying `null`/`unknown`.

---

### DoD-6 — Learn-loop alive in production — **UNKNOWN**

**Criterion (verbatim):**
> `outcome_tracking` table has >=10 rows from autonomous cycles. `agent_memories` has >=5 lessons loaded on next-cycle startup.

**Code-evidence command:** `grep -nE "outcome_tracking|agent_memories" backend/services/autonomous_loop.py`

**Verbatim output (key lines):**
```
1975:    # outcome_tracking gets a row even when yfinance flake or
2001:    "phase-35.1: fallback outcome_tracking row written for %s (sell_price=%s, pnl=%.2f%%, hold=%dd)",
2006:    "phase-35.1: fallback outcome_tracking write failed for %s: %r",
2011:    # agent_memories fan-out (writes one lesson row per
2037:    "phase-35.1: agent_memories reflections fan-out fired for %s",
2042:    "phase-35.1: agent_memories fan-out failed for %s: %r",
```

phase-35.1 fallback writer is **code-confirmed wired** at `autonomous_loop.py:1961-2042`. However, the writer only fires on a real autonomous-loop sell-close path. Two consecutive `completed` cycles since 2026-05-26 timeout (per DoD-9 streak data), neither carried `n_trades > 0` per `cycle_history.jsonl` (`n_trades: 0` on both 0aead72b + 387f1648).

BQ count probe (`SELECT COUNT(*) FROM financial_reports.outcome_tracking WHERE cycle_id IS NOT NULL`) NOT executed this cycle — `mcp__bigquery__execute-query` requires per-call user approval and the verdict is already FAIL-bound. No API endpoint surfaces the count for read-only probing.

**Status:** UNKNOWN. Drift-candidate: code-confirmed, runtime-unverified. Follow-up: phase-35.1 step's own live_check should carry the BQ COUNT(*) evidence.

---

### DoD-7 — Risk Judge structured-output succeeds >95% — **PARTIAL PASS (code-confirmed, runtime-unverified)**

**Criterion (verbatim):**
> `grep -c "Risk Judge returned invalid JSON" backend.log` for last 24h / total Risk-Judge invocations <= 0.05.

**Code-evidence command:** `grep -nE "response_mime_type|response_schema|RiskJudgeVerdict" backend/agents/orchestrator.py backend/agents/risk_debate.py backend/agents/schemas.py`

**Verbatim output (key lines):**
```
backend/agents/risk_debate.py:32:from backend.agents.schemas import RiskAnalystArgument, RiskJudgeVerdict
backend/agents/risk_debate.py:43: "response_mime_type": "application/json",
backend/agents/risk_debate.py:44: "response_schema": RiskAnalystArgument,
backend/agents/risk_debate.py:48: "response_mime_type": "application/json",
backend/agents/risk_debate.py:49: "response_schema": RiskJudgeVerdict,
backend/agents/schemas.py:117:class RiskJudgeVerdict(BaseModel):
backend/agents/orchestrator.py:115: "response_mime_type": "application/json",
backend/agents/orchestrator.py:116: "response_schema": RiskJudgeVerdict,
```

phase-37.1 SHIPPED `response_mime_type="application/json"` + `response_schema=RiskJudgeVerdict` on BOTH orchestrator (`orchestrator.py:115-116` `_THINKING_RISK_JUDGE_CONFIG`) AND risk_debate (`risk_debate.py:48-49` `_JUDGE_STRUCTURED_CONFIG`).

Production fallback-rate measurement (the `grep -c "Risk Judge returned invalid JSON" backend.log` arm) NOT executed this cycle — backend log location not yet probed.

**Status:** PARTIAL PASS. Code-side fix is in. Runtime-evidence collection deferred to phase-35.2 live_check (per masterplan).

---

### DoD-8 — Profit-protection BLOCK closed (OPEN-2 scale-out) — **PASS**

**Criterion (verbatim):**
> OPEN-2 scale-out wiring lands; tested.

**Evidence command:** `grep -n "check_scale_out_fires\|_persist_scale_out_levels\|scale_out_levels_hit\|paper_scale_out_enabled" backend/services/paper_trader.py`

**Verbatim output:**
```
524: # Gated by settings.paper_scale_out_enabled (default False per /goal gate 3).
526: # MFE, 3R = 24% MFE. Idempotent via scale_out_levels_hit column on
530: def check_scale_out_fires(self) -> list[dict]:
540:     if not getattr(self.settings, "paper_scale_out_enabled", False):
559:     # Parse existing scale_out_levels_hit (NULL/missing -> [] -- pre-migration positions).
560:     raw = pos.get("scale_out_levels_hit")
582:                 self._persist_scale_out_levels(ticker, hit)
624: def _persist_scale_out_levels(self, ticker: str, levels: set[str]) -> None:
625:     """Idempotency support: update scale_out_levels_hit column on the
633:     pos["scale_out_levels_hit"] = json.dumps(sorted(levels))
637:         "phase-36.1: failed to persist scale_out_levels_hit for %s: %r",
```

**Status:** PASS. Scale-out wiring at `paper_trader.py:530-637` (phase-36.1); idempotency column `scale_out_levels_hit`; gated by `settings.paper_scale_out_enabled` (default False per the /goal gate).

---

### DoD-9 — 5 consecutive cron cycles complete (no timeout, no halt, no error) — **FAIL**

**Criterion (verbatim):**
> `cycle_history.jsonl` tail shows 5 in a row with `status='completed'`.

**Evidence command:** Python tally of last 15 terminal rows in `handoff/cycle_history.jsonl`.

**Verbatim output (last 15 terminal rows):**
```
2026-05-16T22:45:33  cycle=2e91b881   status=timeout    duration_ms=1800797
2026-05-16T23:17:16  cycle=3e90d15e   status=completed  duration_ms=1401159
2026-05-16T23:45:35  cycle=6452fafe   status=completed  duration_ms=1492377
2026-05-17T00:19:48  cycle=d73f5129   status=completed  duration_ms=373947
2026-05-19T18:00:00  cycle=dcf05853   status=completed  duration_ms=289326
2026-05-20T18:00:00  cycle=9fdcc2df   status=running    duration_ms=275671   (cron run, terminal)
2026-05-21T18:00:00  cycle=8df751b3   status=running    duration_ms=321568   (cron run, terminal)
2026-05-22T05:30:07  cycle=021ed63e   status=timeout    duration_ms=1800605
2026-05-22T16:23:53  cycle=dc3f6cf1   status=completed  duration_ms=2201721
2026-05-22T18:00:00  cycle=c7801712   status=completed  duration_ms=2222177
2026-05-22T20:31:43  cycle=4f8fdca6   status=completed  duration_ms=259909
2026-05-26T18:00:00  cycle=c870fdab   status=completed  duration_ms=396465
2026-05-26T21:50:07  cycle=2f2f3b64   status=timeout    duration_ms=3600989
2026-05-27T04:48:53  cycle=0aead72b   status=completed  duration_ms=6160643
2026-05-27T17:49:00  cycle=387f1648   status=completed  duration_ms=2051006

Most-recent consecutive completed streak: 2 (broke by status='timeout' cycle='2f2f3b64' at 2026-05-26T21:50:07)
```

**Status:** FAIL. Only **2 consecutive** completed cycles since the 2026-05-26T21:50 timeout. The strongest streak in the window (2026-05-22 16:23 → 2026-05-26 18:06) is **4 consecutive completed** which still falls short of the 5-threshold. Follow-up: phase-35.3 (Sustained-cycle stability) is the formal closure.

**Note:** Researcher brief estimated 6 consecutive — that estimate missed the 2f2f3b64 timeout on 2026-05-26T21:50 between the 18:00 cron run and the 2026-05-27 04:48 cron run. The deterministic Python tally above is authoritative.

---

### DoD-10 — Source defaults match production env values — **PASS**

**Criterion (verbatim):**
> grep `model_tiers.py:62` returns `gemini-2.5-pro`; settings.py `deep_think_model` Field default = `gemini-2.5-pro`.

**Evidence command:** `grep -n "gemini_deep_think\|deep_think_model" backend/config/model_tiers.py backend/config/settings.py`

**Verbatim output:**
```
backend/config/model_tiers.py:66: "gemini_deep_think": "gemini-2.5-pro",
backend/config/settings.py:30: deep_think_model: str = Field("gemini-2.5-pro", description="... phase-37.2: default aligned to production (gemini-2.5-pro on Vertex AI via existing operator ADC). Previously claude-opus-4-7 -- caused silent regression to Anthropic credit-exhaustion on fresh checkout / restart without DEEP_THINK_MODEL env override. ...")
```

**Status:** PASS. Both source defaults align to `gemini-2.5-pro` (the production env value).

---

### DoD-11 — All audit P1/P2/P3 findings accounted for — **PARTIAL PASS** (documented deferral, no silent drops)

**Criterion (verbatim):**
> grep this roadmap + masterplan + closed appendix for each finding-id; 0 silent drops.

**Evidence command:**
```bash
grep -oE "OPEN-[0-9]+" handoff/current/master_roadmap_to_production.md | sort -t- -k2n -u
grep -oE "OPEN-[0-9]+" .claude/masterplan.json | sort -t- -k2n -u
```

**Verbatim findings:**
- Roadmap §2 enumerates OPEN-1 through OPEN-33 (33 IDs).
- Masterplan references OPEN-1..18, 20, 22..26, 28..33 (missing **OPEN-19, OPEN-21, OPEN-27**).

**Disposition of the 3 not-in-masterplan IDs (from roadmap §2 + §6 + line 93):**
- **OPEN-19** (S&P 500 Wikipedia-scrape survivorship-biased) → mapped to phase-42.0 / 42.1 (universe expansion, **deferred post-prod**, depends on phase-5 which is `pending`). phase-42 is NOT in `.claude/masterplan.json` — but the roadmap **explicitly** marks phase-42 as deferred post-prod (line 93: "phase-42 (universe expansion: OPEN-19, OPEN-20 sustained, OPEN-21) is deferred because it depends on `phase-5` (Multi-Market Expansion)").
- **OPEN-21** (Layer-2 MAS strategy_decisions threshold tuning) → mapped to phase-42.3. Same deferral chain.
- **OPEN-27** (Auto-commit hook stalls + researcher-write-first compliance) → mapped to "phase-40.x doc-only (see phase-43)" in the roadmap. The two underlying issues are tracked as feedback auto-memories `feedback_auto_commit_hook_stalls.md` + `feedback_researcher_write_first.md`. No code-fix step in masterplan.

**Status:** PARTIAL PASS. The 3 missing IDs are NOT silent drops — they're each documented in the roadmap with an explicit deferral home (phase-42 / auto-memories). To convert to clean PASS, either (a) add explicit phase-42 entries to `.claude/masterplan.json` as `status: "deferred"` or (b) update the master_roadmap_to_production.md §6 DoD-11 wording to acknowledge the doc-only deferral pathway.

---

### DoD-12 — ASCII-only loggers — **PASS**

**Criterion (verbatim):**
> `python scripts/qa/ascii_logger_check.py` exits 0.

**Evidence command:** `source .venv/bin/activate && python scripts/qa/ascii_logger_check.py`

**Verbatim output (final line):**
```
OK: 538 files, 1784 logger calls, 0 violations
```

**Status:** PASS. Exit code 0; 0 violations across 1,784 logger calls.

---

### DoD-13 — Restart-survivable cycle state — **PASS**

**Criterion (verbatim):**
> Kill backend mid-cycle; restart; next cycle starts cleanly.

**Evidence commands:**
```
grep -n "cycle_lock\|clean_stale_lock\|_running" backend/services/autonomous_loop.py
grep -n "cycle_lock\|clean_stale_lock"           backend/main.py
ls backend/services/cycle_lock.py
```

**Verbatim output (key lines):**
```
backend/services/cycle_lock.py    (file exists)
backend/services/autonomous_loop.py:142-173    cycle_lock acquire wiring (phase-38.6.1)
backend/services/autonomous_loop.py:150        from backend.services.cycle_lock import acquire as _cycle_lock_acquire, CycleLockError
backend/services/autonomous_loop.py:167        _lock_cm = _cycle_lock_acquire(_cycle_id_for_lock)
backend/services/autonomous_loop.py:1151       _running = False  (release path)
backend/services/autonomous_loop.py:1152-1159  phase-38.6.1: release the file-based cycle_lock (unlinks pidfile)
backend/main.py:211       from backend.services.cycle_lock import clean_stale_lock as _clean_stale_lock
backend/main.py:212       _cleaned = _clean_stale_lock(reason="startup_recovery")
backend/main.py:222       logging.exception("phase-38.6.1: cycle_lock recovery hook failed (fail-open)")
```

**Status:** PASS. File-based `cycle_lock` replaces in-process `_running` for restart-survivability; `clean_stale_lock` runs in `main.py` lifespan startup; release path unlinks pidfile in the cycle's `finally` block.

---

### DoD-14 — OWASP LLM Top-10 v2.0 compliance — **FAIL** (3 of 10 categories not explicitly tagged)

**Criterion (verbatim):**
> `.claude/skills/code-review-trading-domain/SKILL.md` covers LLM01-LLM10; no open findings.

**Evidence command:** `grep -n "LLM0[1-9]\|LLM10" .claude/skills/code-review-trading-domain/SKILL.md`

**Verbatim output (key lines, deduplicated by LLM-category):**
```
:56  LLM02 — secret-in-diff (BLOCK)
:59  LLM01 — prompt-injection-path (BLOCK)
:65  LLM06 — excessive-agency-scope-creep (WARN)
:69  LLM03 — supply-chain-dep-pin-removal (WARN)
:85  LLM07 — system-prompt-leakage (WARN)
:86  LLM08 — rag-memory-poisoning (WARN)
:87  LLM10 — unbounded-llm-loop (WARN, denial-of-wallet)
```

Explicit tags present: **LLM01, LLM02, LLM03, LLM06, LLM07, LLM08, LLM10** = **7 of 10**.

Missing explicit `LLM04:2025`, `LLM05:2025`, `LLM09:2025` tags:
- **LLM04 Data and Model Poisoning** — no explicit tag; pyfinagent does not train models, so arguably N/A, but no N/A justification appears in SKILL.md.
- **LLM05 Improper Output Handling** — line 81 has an `insecure-output-handling` heuristic, but it is NOT explicitly tagged `LLM05:2025`.
- **LLM09 Misinformation** — no explicit heuristic for LLM-generated misinformation propagated into trading decisions.

**Cosmetic:** SKILL.md line 100 + 230 refer to "v2.0 (2025)" — official label per https://genai.owasp.org/llm-top-10/ is **"OWASP Top 10 for LLM Applications 2025"** (released March 12, 2025; no v2.0 or v2.1 nomenclature in the consortium).

**Status:** FAIL under the literal "covers LLM01-LLM10" criterion. Follow-up: doc-edit cycle to add explicit `LLM04:2025` (N/A justification), `LLM05:2025` tag on the existing `insecure-output-handling` heuristic, and `LLM09:2025` misinformation heuristic.

---

## Stop-condition contribution

This audit IS the production_ready stop-condition gate (`.claude/masterplan.json` `phase-43.0`).

**Gate-PASS prerequisites (must hold before `status: done`):**
- Immutable criterion #1: `all_14_DoD_criteria_PASS` — **NOT MET** (5–9 of 14 pass, depending on interpretation).
- Immutable criterion #2: `audit_file_carries_verbatim_evidence_per_criterion` — **MET** (this file).
- Immutable criterion #3: `qa_confirms_no_silent_drops` — **pending Q/A spawn**.
- Immutable criterion #4: `operator_approval_recorded_for_PRODUCTION_READY_declaration` — **N/A** (verdict is NOT_PRODUCTION_READY, no approval being sought).

**Recommended `phase-43.0` masterplan status:** `pending` (audit deliverable lands, but the gate-PASS is contingent on closing DoD-1, DoD-2, DoD-5, DoD-9, DoD-14 — none of which closed this cycle).

---

## Follow-up cycle pointers

| Failing/Partial DoD | Existing masterplan step | Action |
|---|---|---|
| DoD-1 (autoresearch cron exit 1) | phase-39.1 (P1, owner-gated) | Widen the fix to cover `langchain_huggingface ModuleNotFoundError` |
| DoD-2 (paper-vs-backtest Sharpe parity) | NEW step needed | Add walk-forward result JSON column for paper-Sharpe comparison + close the divergence |
| DoD-4 (literal vs tiered coverage) | NEW doc step | Update master_roadmap §6 DoD-4 to cite the tiered policy from `docs/coverage_tier_overrides.md` |
| DoD-5 (Unknown freshness bands) | NEW step needed | Wire per-source `last_tick_age_sec` populator for historical_prices/fundamentals/macro + signals_log |
| DoD-6 (learn-loop runtime evidence) | phase-35.1 (P1, H) | live_check must carry BQ COUNT(*) evidence — currently `pending` |
| DoD-7 (Risk Judge runtime fallback rate) | phase-35.2 (P1, H) | live_check must capture production log-line counts — currently `pending` |
| DoD-9 (5-consecutive-cycle streak) | phase-35.3 (P2, H) | Formal closure — currently `pending` |
| DoD-11 (3 IDs not in masterplan) | NEW doc step | Either insert phase-42 with `status: "deferred"` entries OR update master_roadmap §6 DoD-11 to acknowledge doc-only deferrals |
| DoD-14 (LLM04/05/09 missing tags) | NEW doc step | Add explicit `LLM04/05/09:2025` tags to SKILL.md and update "v2.0" cosmetics to "2025" |

---

## Auditor confidence

Confidence level: **HIGH** for PASS verdicts (deterministic file:line + command output), **HIGH** for FAIL verdicts (verbatim live-system output), **MEDIUM** for the UNKNOWN (DoD-6: would require an `execute-query` BQ probe with user-approval gate).

The verdict NOT_PRODUCTION_READY would hold under any reasonable interpretation of the 14 criteria. The cycle's value is the verbatim evidence per-criterion, the strict-vs-tiered ambiguity surfacing (DoD-4), and the missing-OPEN-id surfacing (DoD-11) — all three give the operator unambiguous follow-up scope.
