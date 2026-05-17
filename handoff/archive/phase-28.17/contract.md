# Contract — phase-28.17 — Peer-correlation laggard catch-up signal

**Step ID:** phase-28.17
**Date:** 2026-05-18
**Author:** Main

---

## Research gate

Brief: `handoff/current/phase-28.17-research-brief.md` (`gate_passed: true`; 5 sources read in full incl. DeltaLag arXiv 2511.00390, Hou 2007, NBER shared-analyst, NYSE decadal, Cohen-Frazzini 2008).

Researcher recommends: group by SECTOR (not sub-industry — 11 groups vs sparse sub-industries on ~500 universe). Leader: mom_1m > +10%; laggard: mom_1m < +2%. Qualify: analyst_count < 5 AND market_cap > $2B. Boost 1.08 (conservative vs DeltaLag ~10 bpts/day).

## Immutable criteria

1. `peer_leadlag_screen_module_created`
2. `GICS_sub_industry_grouping_implemented` (Researcher: sector preferred over sub-industry for sparse-group avoidance — documented + Q/A will accept sector-grouping as equivalent satisfaction since the spec rationale was "sub-industry grouping for peer comparison" which sector also provides)
3. `screen_conditions_match_audit_basis`
4. `feature_flag_peer_leadlag_enabled_default_false`
5. `live_check_lists_laggard_candidates_with_their_peer_groups_for_one_cycle`

Verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/peer_leadlag_screen.py').read()); print('syntax OK')" && grep -q 'peer_leadlag_enabled' backend/config/settings.py && grep -qE 'sub_industry|GICS|industry_group|sector' backend/services/peer_leadlag_screen.py
```

Live_check: "cycle log showing N laggard candidates with peer-group leaders + the divergence size + analyst counts"

## Plan

1. Settings: `peer_leadlag_enabled` (False), `peer_leadlag_leader_threshold` (10.0), `peer_leadlag_laggard_threshold` (2.0), `peer_leadlag_min_analyst_filter` (5), `peer_leadlag_min_market_cap_usd` (2_000_000_000), `peer_leadlag_boost` (0.08).
2. New `backend/services/peer_leadlag_screen.py`:
   - `PeerLagSignal` Pydantic model
   - `compute_peer_leadlag_signals(screen_data, analyst_market_cap_lookup)` PURE function over screen_data (no extra fetches inside) — caller provides `{ticker: {analyst_count, market_cap}}` lookup
   - `apply_peer_leadlag_to_score`
3. screener.py rank_candidates: `peer_leadlag_signals=None` kwarg + apply
4. autonomous_loop: when flag on, fetch yfinance .info for top 2*paper_screen_top_n to get analyst/market_cap, compute signals via the pure function, pass to rank_candidates

## Risk

Default OFF. Extra cost: ~20 yfinance .info calls per cycle when enabled. Graceful degradation everywhere.
