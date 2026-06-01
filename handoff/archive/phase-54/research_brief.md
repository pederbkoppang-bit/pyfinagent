# Research: phase-50.6 — Multi-market UI (backtest page + multi-currency NAV widget + paper_markets settings toggle)

Tier: **moderate**. Caller-set. (Overwrites the 54.2 brief — that one is archived.)

Step (`.claude/masterplan.json:13699`): Multi-market UI — market filter +
currency badges + multi-currency NAV breakdown + market-hours indicator on
the paper-trading + backtest pages; a `paper_markets` toggle in settings.
Cockpit (paper-trading) UX already shipped (`goal-multimarket-ux` / commit
`ac93f67f`). REMAINING per `active_goal.md` scope item 1:
(a) backtest-page market/currency/market-hours treatment,
(b) multi-currency NAV-breakdown widget,
(c) `paper_markets` settings toggle,
(d) `handoff/current/live_check_50.6.md` (build/types/API proofs + operator visual).

Success criteria (immutable, verbatim from masterplan):
1. paper-trading + backtest pages show per-position market/exchange + local
   currency + a multi-currency NAV breakdown (USD total + per-currency
   sub-totals) + a market-open/closed indicator
2. a paper_markets toggle exists in settings UI wired to the backend setting;
   icons via @/lib/icons, no emoji
3. `cd frontend && npm run build` SUCCEEDS with the changes
4. live_check_50.6.md records build pass + API wiring + OPERATOR-TO-CONFIRM
   visual section

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- |
| _pending_ | | | | |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| _pending_ | | |

## Recency scan (2024-2026)
_pending_

## Search-query variants (3-pass discipline)
_pending_

## Key findings
_pending_

## Internal code inventory
| File | Lines | Role | Status |
| --- | --- | --- | --- |
| _pending_ | | | |

## Consensus vs debate (external)
_pending_

## Pitfalls (from literature)
_pending_

## Application to pyfinagent
_pending_

## Recommended scoped plan
_pending_

## Risks (DO-NO-HARM)
_pending_

## Research Gate Checklist
- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Full papers / pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "gate_passed": false
}
```
