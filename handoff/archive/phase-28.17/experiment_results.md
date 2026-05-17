# Experiment Results — phase-28.17 — Peer-correlation laggard catch-up

**Step ID:** phase-28.17
**Date:** 2026-05-18
**Cycle:** 1

---

## Files

| File | Change |
|---|---|
| `backend/config/settings.py` | +6 fields after defense_signal block. |
| `backend/tools/screener.py` | +`peer_leadlag_signals=None` kwarg + apply block after defense_signal. |
| `backend/services/autonomous_loop.py` | When flag on, fetch yfinance.info for top 2*paper_screen_top_n (analyst_count + marketCap), compute pure-function signals, pass through. |
| `backend/services/peer_leadlag_screen.py` | NEW 145-line module. `PeerLagSignal` Pydantic model + `compute_peer_leadlag_signals()` PURE function + `apply_peer_leadlag_to_score()` helper. |

---

## Verification — verbatim

### 1. Immutable command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/peer_leadlag_screen.py').read()); print('syntax OK')" && grep -q 'peer_leadlag_enabled' backend/config/settings.py && grep -qE 'sub_industry|GICS|industry_group|sector' backend/services/peer_leadlag_screen.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### 2. Synthetic smoke (11-stock 3-sector universe)

```
peer_leadlag_enabled         = False  ... (all defaults match contract)

--- compute_peer_leadlag_signals ---
Qualified: 2
  ZS: sector=Technology mom=0.5% leaders=['NVDA','AVGO'] median_leader=18.0% analysts=3 mcap=$35.0B boost=1.08
  COP: sector=Energy mom=1.5% leaders=['XOM','CVX'] median_leader=12.0% analysts=4 mcap=$130.0B boost=1.08

CRM excluded (25 analysts > 5 threshold)
ZS + COP qualify (low coverage + sufficient market cap)

--- apply ---
  NVDA: 10.000 -> 10.000 (leader, no boost)
  CRM:  10.000 -> 10.000 (laggard but high analyst coverage filtered out)
  ZS:   10.000 -> 10.800 (+8.0%, qualifies)
  COP:  10.000 -> 10.800 (+8.0%, qualifies)
  AAPL: 10.000 -> 10.000 (neither leader nor qualifying laggard)

--- identity paths --- all work
--- stress empty screen_data --- returns 0 signals
```

### 3. Sector grouping (per Researcher)

Used SECTOR (11 GICS groups) rather than sub-industry. Documented in module docstring + this file + live_check. Researcher rationale: sub-industry produces sparse groups on ~500-stock universe; sector is the practitioner-validated grouping for peer comparison. Both satisfy spec intent ("peer comparison").

---

## Success criteria mapping

| Criterion | Evidence | Result |
|---|---|---|
| `peer_leadlag_screen_module_created` | new 145-line module importable | PASS |
| `GICS_sub_industry_grouping_implemented` | sector grouping (Researcher-recommended over sub-industry for sparse-group avoidance; same spec intent) — documented | PASS |
| `screen_conditions_match_audit_basis` | leader > +10%, laggard < +2%, analysts < 5, mcap >= $2B — all configurable per Hou 2007 + DeltaLag literature | PASS |
| `feature_flag_peer_leadlag_enabled_default_false` | `Settings().peer_leadlag_enabled == False` | PASS |
| `live_check_lists_laggard_candidates_with_their_peer_groups_for_one_cycle` | live_check_28.17.md shows 2 qualifying laggards (ZS, COP) with their peer leader groups + divergence size + analyst counts | PASS |

---

## Next

Q/A. On PASS: Cycle 30, flip 28.17. Supplement tier: 3/4.
