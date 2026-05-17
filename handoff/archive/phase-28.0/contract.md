# Contract — phase-28.0 — Drift fix: remove unused `min_market_cap` parameter

**Step ID:** phase-28.0
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1 (single-cycle, XS drift fix)
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.0-research-brief.md` (Researcher subagent; `gate_passed: true`)
- 5 external sources read in full: PEP 702, PyAnsys deprecation guide, Seth Larson's "Deprecations via warnings don't work for Python libraries", CFI S&P 500 article, official Python deprecations doc.
- 15 URLs collected, 10 snippet-only (yfinance GitHub issues, S&P DJI press release, etc.).
- Recency scan (2024-2026) confirmed: Yahoo Finance tightened rate limits 2024; S&P 500 min market cap raised to $22.7B effective Jul 2025; no PEP-level change to deprecation cadence.
- Internal audit: all 6 callers/test sites of `screen_universe` and `min_market_cap` inspected — zero callers pass the parameter.

### Note on Researcher
The Researcher subagent (`phase-28-0-researcher`) wrote the stub frontmatter then was slow to append. Main began parallel fallback fetches (per user goal: "If the Researcher agent crashes again, log the failure signature and fall back to direct web fetch; do not block the run"). Researcher then completed successfully within the same window with five sources read in full and a matching REMOVE recommendation. Both research paths converged on the same conclusion; the Researcher's brief is the canonical record. No crash occurred; the brief was just slow to populate.

---

## Hypothesis

The `min_market_cap` parameter at `backend/tools/screener.py:65` is dead code: accepted in the signature but never referenced in the function body. Removing it is a one-line non-breaking change because:

1. Zero callers in the repository pass it (verified by grep).
2. The current universe (S&P 500 via Wikipedia scrape) already has a $22.7B minimum-market-cap floor by index inclusion rules, so a $1B filter would never fire even if wired.
3. Wiring the filter via `Ticker.info["marketCap"]` would add O(500) per-ticker HTTP requests per screen cycle with high 429 rate-limit risk.
4. Removal aligns the in-code reality with the documentation in the primary brief (which had incorrectly described the picker as enforcing market-cap ≥ $1B).

## Immutable success criteria (copied verbatim from `.claude/masterplan.json::phase-28.steps[0].verification.success_criteria`)

1. `min_market_cap_parameter_either_applied_or_removed`
2. `syntax_OK`
3. `no_regression_in_existing_screener_callsites`

Plus the verbatim verification command (immutable):

```bash
source .venv/bin/activate && python -c "import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')"
```

Plus the `live_check` shape (immutable):

> live_check_28.0.md: one cycle log line showing the screener filter chain with market-cap status (applied / removed)

---

## Plan steps

1. **Research gate** — DONE. Brief at `handoff/current/phase-28.0-research-brief.md`. Recommendation: REMOVE.
2. **Edit `backend/tools/screener.py`** — remove `min_market_cap: float = 1e9,` from `screen_universe` signature (line 65 in pre-edit numbering). Update docstring with phase-28.0 note explaining the removal and the $22.7B S&P 500 floor.
3. **Run the immutable verification command** — must exit 0.
4. **Live smoke** — invoke `screen_universe(tickers=['AAPL','MSFT','NVDA'], period='1mo')`; confirm 3 results with expected fields.
5. **Write `experiment_results.md`** — files touched, verification output verbatim, smoke output verbatim.
6. **Write `live_check_28.0.md`** — capture the live filter-chain log line.
7. **Spawn Q/A** — fresh `qa` subagent reads all artifacts and returns verdict + violations.
8. **On Q/A PASS** — append cycle entry to `handoff/harness_log.md`, then flip status to `done` in `.claude/masterplan.json`.
9. **On Q/A CONDITIONAL/FAIL** — fix per critique, update `experiment_results.md` + append follow-up in `evaluator_critique.md`, spawn fresh Q/A. Never overturn unchanged evidence.

## References

- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief — phase-28 source of truth)
- `docs/research/candidate_picker_improvements_2026-05-16-supplement.md` (supplement brief)
- `docs/research/candidate_picker_masterplan_proposal_2026-05-17.md` (phase-5 integration proposal, applied 2026-05-17)
- `handoff/current/phase-28.0-research-brief.md` (this step's research gate)
- `.claude/masterplan.json::phase-28.steps[0]` (immutable step spec)

## Risk / blast radius

- **None to autonomous-loop money path.** The parameter is unused; removal cannot change behavior of any caller.
- **No backward-compat concern.** Internal API only; no published consumers.
- **No regression risk to tests.** Both `tests/services/test_screener_sector_propagation.py` and `tests/verify_phase_23_1_13.py` assert other parameters, not `min_market_cap`.
- **No cost change.** Zero LLM cost; no new HTTP requests.
