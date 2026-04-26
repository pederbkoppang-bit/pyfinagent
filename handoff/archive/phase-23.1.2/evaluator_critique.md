---
step: phase-23.1.2
verdict: PASS
cycle_date: 2026-04-27
qa_agent: qa (merged qa-evaluator + harness-verifier)
qa_pass: 1
---

# Q/A Critique — phase-23.1.2

## 5-item harness-compliance audit

1. Researcher brief at `handoff/current/phase-23.1.2-research-brief.md` — EXISTS (37286 bytes), envelope at EOF: `tier: moderate`, `external_sources_read_in_full: 8`, `snippet_only_sources: 10`, `urls_collected: 18`, `recency_scan_performed: true`, `internal_files_inspected: 8`, `gate_passed: true`. PASS.
2. Contract front-matter `step: phase-23.1.2` matches title; `verification:` field is the immutable `python -c "..."` one-liner; matches verbatim what's in `experiment_results.md`. PASS.
3. `experiment_results.md` includes verbatim verification output (`ok ticker=AAPL tag=insufficient_history sent=0.82 surprise=0.0`) + `exit=0`. Three E2E-discovered bugs (httpx headers, index.json URL, type-vs-filename pattern) are explicitly documented in the "Bugs surfaced and fixed" section — anti-rubber-stamp evidence. PASS.
4. `harness_log.md` NOT yet appended for `phase=23.1.2` (grep -c returned 0). Log-LAST rule honored. PASS.
5. First Q/A spawn for phase-23.1.2 — pre-existing `evaluator_critique.md` was for phase-23.1.1 (now overwritten by this critique, per single-rolling-file convention). No prior CONDITIONAL/FAIL on disk for this step-id. PASS.

## Deterministic checks (checks_run)

| Check | Result |
|---|---|
| C. Syntax check (5 files) | `all syntax ok` |
| B. pytest tests/services/test_pead_signal.py + test_macro_regime.py | `30 passed in 0.03s` (18 PEAD + 12 macro_regime — no regression) |
| A. Immutable verification command (real EDGAR + real Claude) | `ok ticker=AAPL tag=insufficient_history sent=0.82 surprise=0.0` exit=0 — reproduced live in this Q/A run |
| D. Default-OFF safety | `autonomous_loop.py:127-128` initializes `pead_signals = {}`, only fetches when `pead_signal_enabled=True`; `rank_candidates(..., pead_signals=pead_signals or None)` at line 142. `apply_pead_to_score` short-circuits on falsy input per test_apply_pead_no_signals_passes_through. PASS. |
| E. Strong-negative filter | `pead_signal.py:374-375`: `if tag == "negative_surprise" and surprise < -0.3: return None`. `screener.py:206-211`: `new_score = apply_pead_to_score(...); if new_score is None: continue` — candidate dropped, not ranked with score=0. PASS. |
| F. EDGAR client correctness | (1) `_FILING_INDEX_URL` (line 41) is `.../{acc_nodash}/index.json` — correct, brief's `{accession}-index.json` was wrong; (2) Exhibit 99 identified at line 188 by FILENAME pattern (`ex99` / `ex-99` / `exhibit99` substring) — NOT by `type` field (line 180-181 comment explicitly notes type is the icon name); (3) `httpx.AsyncClient(headers=SEC_HEADERS)` at line 242 ensures User-Agent on every request. All three E2E-discovered bugs fixed in code AND documented in experiment_results. PASS. |
| G. Git diff scope | `git diff --name-only HEAD` returns only contract-scoped files: `backend/services/pead_signal.py` (NEW), `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`, `tests/services/test_pead_signal.py` (NEW), the three handoff/current files, plus auto-modified audit jsonl/heartbeat/lock and a few unrelated pre-existing untracked deltas (`perf_results.tsv`, `frontend/next-env.d.ts`, `frontend/tsconfig*`, `.archive-baseline.json`, `cycle_history.jsonl`, `frontend/handoff/harness_log.md`) which are not in `backend/agents/` or substantive frontend code. NO unexpected edits to backend/agents/ or out-of-scope areas. PASS. |

## LLM judgment

| Question | Verdict | Notes |
|---|---|---|
| Plan accomplished | YES | All 8 plan steps in contract have evidence in experiment_results: `pead_signal.py` NEW (~290 LOC) with PeadSignalOutput + compute_pead_signal_for_ticker + fetch_pead_signals_for_recent_reporters + apply_pead_to_score + file cache; EDGAR client reuses SEC_HEADERS/_resolve_cik from sec_insider.py; screener extended with pead_signals kwarg; autonomous_loop wired (default-off); 3 settings fields added; 18 unit tests pass. |
| Mutation-resistant | YES | Verification command exercises real SEC EDGAR (CIK lookup → submissions → filing index → Exhibit 99) + real Claude Haiku 4.5 with structured Pydantic output. Any break in `_FILING_INDEX_URL` constant, the FILENAME-pattern Exhibit-99 filter, the SEC_HEADERS injection, or the schema-strip helper would fail it. Reproduced live in this Q/A run. |
| Anti-rubber-stamp | YES | Three E2E-discovered bugs (httpx default headers, index.json URL pattern, type-vs-filename) are explicitly enumerated in `## Bugs surfaced and fixed during E2E` — not omitted. Honest disclosure that `tag=insufficient_history` is correct because no prior quarters in cache (surprise_score=0.0 by design, not a failure). |
| Scope honesty | YES | Out-of-scope section excludes BQ table migration `pead_signal_history` (Phase 2), backtest validation (phase-23.2.5), UI (phase-23.1.6), and `earnings_tone.py` replacement (different signal source — both kept). Diff confirms no work in those areas. |
| Research-gate compliance | YES | Contract front-matter `research_brief:` field points to brief on disk; gate_passed:true; 8 sources read in full (above 5-floor); recency_scan_performed; mix of QuantPedia practitioner + Anthropic structured-output docs + arXiv + SEC EDGAR official docs. |
| Default-off discipline | YES | `pead_signal_enabled: bool = Field(False, ...)` in settings.py:155. Pre-existing autonomous_loop callers see no behavior change — `rank_candidates` receives `pead_signals=None` and short-circuits in screener.py loop. Confirmed by reading the code path. |
| Cost discipline | YES | <$0.05/cycle target documented with derivation: ~125 earnings/quarter ÷ 60 trading days ≈ 2 calls/day at Haiku 4.5 ~$0.005/call. File cache per `(ticker, quarter)` at `backend/services/_cache/pead/` prevents re-billing same press release. |

## Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 harness-compliance items, all 7 deterministic checks (A-G), and all 7 LLM-judgment dimensions pass. Real SEC EDGAR + real Claude verification command reproduced live (ok ticker=AAPL tag=insufficient_history sent=0.82 surprise=0.0, exit=0). 30/30 tests pass (18 PEAD + 12 macro_regime no regression). Default-OFF discipline verified. Strong-negative filter (drop on tag==negative_surprise AND surprise<-0.3) implemented in both pead_signal.py and screener.py loop. Three E2E-discovered EDGAR bugs honestly documented in experiment_results.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "pytest", "verification_command_real_edgar_real_claude", "default_off_safety", "strong_negative_filter", "edgar_client_correctness", "git_diff_scope", "llm_judgment"]
}
```

Green light to: append `handoff/harness_log.md`, add masterplan.json step entry with status=done, archive handoff, commit on main, move to phase-23.1.3 (worldwide news idea generator, no API keys).
