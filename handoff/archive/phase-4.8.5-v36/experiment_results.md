# Experiment Results -- Cycle 87 / phase-4.8 step 4.8.10

Step: 4.8.10 2026 regulatory memo + wash-sale filter

## Research-gate upheld

Spawned researcher (22 URLs: SEC 34-96930 T+1, IRS Pub 550,
FINRA SR-2025-017 intraday margin, DTCC Net Debit Cap, FINRA 2026
Oversight Report) + Explore in parallel BEFORE writing code.
Researcher flagged the CALENDAR-day (not business-day) pitfall for
the wash-sale window.

## What was generated

1. **NEW** `docs/compliance/2026-regulatory-memo.md`
   7-section memo with SEC 34-96930, IRC Sec 1091, FINRA 4210 /
   SR-2025-017 citations.
2. **NEW** `backend/services/wash_sale_filter.py`
   `WashSaleLedger` (61-day CALENDAR window via timedelta(days=30)),
   `record_loss` rejects gains, `is_wash_sale` auto-prunes stale
   entries, `filter_candidates` partitions (allowed, blocked).
3. **NEW** `backend/services/funding_guard.py`
   `t1_funding_guard` blocks unsettled-same-day; `realtime_margin_
   guard` blocks over-cap; both return (allowed, enum reason).
4. **NEW** `scripts/compliance/wash_sale_filter.py --test`
   11 discriminating fixtures: wash boundaries (+15, +30, +31, +61,
   diff ticker), filter partitioning, T+1 (unsettled blocked,
   settled allowed), margin (deficit, under, threshold).
5. **NEW** `scripts/audit/regulatory_memo_audit.py`
   5 teeth: memo sections, citations, library API, test rc=0,
   calendar-day proof (Sat buy 3 days after Wed sell flagged).

## Verification (verbatim, immutable)

    $ test -f docs/compliance/2026-regulatory-memo.md && \
      python scripts/compliance/wash_sale_filter.py --test
    {"verdict": "PASS", "failed_count": 0}
    exit=0

    $ python scripts/audit/regulatory_memo_audit.py --check
    {"verdict": "PASS", "memo": true, "lib": true,
     "test": true, "calendar": true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| memo_landed | PASS (7 sections + 3 citations) |
| wash_sale_filter_active | PASS (5 boundary fixtures + partition) |
| t1_funding_guard_active | PASS (unsettled blocked, settled allowed) |
| realtime_margin_handler_active | PASS (3 discriminating margin states) |

## Known limitations (tracked follow-up)

- Guards are library-only this cycle. Wiring into paper_trader.py
  BUY path is a same-phase follow-up.
- Wash-sale ledger is process-local; multi-process would need BQ
  centralization. Documented in memo "Open items".
- FINRA 4210 intraday margin enforcement begins ~2027-04-17;
  monitoring-only today.
