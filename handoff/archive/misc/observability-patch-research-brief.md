---
step: observability-patch
date: 2026-04-24
tier: simple
researcher: researcher-agent
---

## Research: Observability Patch -- Risk Judge logging + buy_amount skip logging

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.index.dev/blog/avoid-silent-failures-python | 2026-04-24 | blog/practitioner | WebFetch | "Robust logging is the cornerstone of error detection." Every operation -- successful, failed, or skipped -- must generate an observable, traceable record. |
| https://www.hrekov.com/blog/python-structured-logging | 2026-04-24 | blog/practitioner | WebFetch | Use `extra` dict for key-value context on every decision log; log the decision VALUE not just the outcome. |
| https://chronosphere.io/learn/logging-best-practices/ | 2026-04-24 | blog/practitioner | WebFetch | "Log events should legitimately be generated to indicate everything is running as expected." Warn when code takes alternative (non-happy-path) routes. |
| https://signoz.io/blog/structured-logs/ | 2026-04-24 | blog/practitioner | WebFetch | Include unique identifiers + context-specific data; consistency across all modules is mandatory. |
| https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/ | 2026-04-24 | blog/practitioner | WebFetch | 2026 guide: `extra` parameter or structured logging for audit context; `logger.exception()` for critical paths. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|----------------|
| https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure | industry | 403 |
| https://medium.com/@vaibhavtiwari.945/backend-observability-made-simple-logging-tracing-and-monitoring-726baeb6d801 | blog | snippet only |
| https://jpfrancoia.github.io/2025/05/29/logging-in-python.html | blog | snippet only |
| https://medium.com/@muruganantham52524/observability-reliability-in-python-data-pipelines-new-peps-tools-best-practices-for-2025-704442bcfeeb | blog | snippet only |
| https://www.quantconnect.com/docs/v2/writing-algorithms/logging | official doc | snippet only -- quant-native logging guidance |
| https://medium.com/@online-inference/mlops-best-practices-for-quantitative-trading-teams-59f063d3aaf8 | industry | snippet only |
| https://edgedelta.com/company/knowledge-center/distributed-systems-observability | guide | snippet only |

### Recency scan (2024-2026)
Searched: "anti-silent-failure logging design python backend observability 2025", "observability-first backend python logging no-op decision audit trail 2026", "python logging best practices complete guide 2026". Result: strong 2025-2026 consensus that structured logging with `extra` context and explicit logging of every decision branch (including skipped/no-op paths) is the current industry floor. No new finding supersedes the basic "log your no-op" principle; the 2026 guidance reinforces it with JSON-structured formats.

### Queries run
1. Year-less canonical: "structured logging best practice quant trading system never silently skip"
2. 2025 window: "anti-silent-failure logging design python backend observability 2025"
3. 2026 frontier: "observability-first backend python logging no-op decision audit trail 2026"

---

### Key findings

1. **Log every branch, including no-ops.** Chronosphere (2026): "Log events should legitimately be generated to indicate everything is running as expected" -- the skip path is a legitimate event. (Source: chronosphere.io/learn/logging-best-practices/)

2. **Log the decision value, not just the effect.** hrekov.com: Use `extra` to surface the specific key-value causing the skip -- caller can then search logs for `decision=REJECT` rather than inferring from absent BUY orders. (Source: hrekov.com/blog/python-structured-logging)

3. **Use WARNING for silent guard activations.** autonomous_loop.py:248 sets the pattern in this codebase: `logger.warning(f"Dropping BUY for {order.ticker}: price={price}")`. The buy_amount<50 skip is the same class of silent guard -- it warrants the same level.

4. **Risk Judge has 4 values:** `APPROVE_FULL`, `APPROVE_REDUCED`, `APPROVE_HEDGED`, `REJECT` (confirmed at `backend/agents/schemas.py:118`). Only non-APPROVE_FULL decisions are worth flagging as informational; REJECT is the high-signal case.

---

### Internal code inventory
| File | Lines (relevant) | Role | Gap |
|------|-----------------|------|-----|
| `backend/services/portfolio_manager.py` | 153-158 | Extracts `risk_judge_decision` into `buy_candidates` dict | Decision is stored but NEVER logged -- no warning if REJECT/APPROVE_REDUCED/APPROVE_HEDGED |
| `backend/services/portfolio_manager.py` | 176-177 | `if buy_amount < 50: continue` | Silent skip -- no log line; gap confirmed |
| `backend/services/portfolio_manager.py` | 194-195 | Summary `logger.info` (sells/buys count) | Only summary; no per-candidate decision trace |
| `backend/agents/schemas.py` | 118 | `decision: Literal["APPROVE_FULL","APPROVE_REDUCED","APPROVE_HEDGED","REJECT"]` | 4 values confirmed |
| `backend/agents/skills/risk_judge.md` | 33 | Position pct by decision | REJECT = no position; APPROVE_REDUCED = 2-5%; important to surface |
| `backend/services/autonomous_loop.py` | 248 | `logger.warning("Dropping BUY for {ticker}: price=...")` | Style reference -- use same pattern |

**Gap (a) confirmed:** `portfolio_manager.py:153` sets `risk_judge_decision` in the candidate dict. Lines 163-192 loop candidates and build BUY orders, but there is no log call when `risk_judge_decision` is `REJECT`, `APPROVE_REDUCED`, or `APPROVE_HEDGED`. If position_pct resolves to 0 or near-0 for a REJECT, the candidate falls through to the `buy_amount < 50` silent skip with zero visibility.

**Gap (b) confirmed:** `portfolio_manager.py:176-177`: `if buy_amount < 50: continue` -- no `logger.warning` before the continue.

**Virtual-fund-readiness brief envelope:** `handoff/current/virtual-fund-readiness-research-brief.md` ends at line 97 with the Research Gate Checklist. Grep for `gate_passed` returns no matches. The JSON envelope is **confirmed absent**.

---

### Recommended log-line wording

All messages use ASCII only (per `security.md`: no Unicode/arrows/em-dashes in logger calls).

**Gap (a) -- Risk Judge non-APPROVE_FULL decision:**
Insert immediately after the `buy_candidates.append(...)` block (after line 158), inside the `for analysis in candidate_analyses:` loop:

```python
if cand_data["risk_judge_decision"] and cand_data["risk_judge_decision"] != "APPROVE_FULL":
    logger.info(
        "buy_candidate risk_judge decision=%s ticker=%s position_pct=%s final_score=%s",
        cand_data["risk_judge_decision"], ticker,
        cand_data["position_pct"], round(final_score, 3),
    )
```

(Use `info` not `warning` -- a non-APPROVE_FULL decision is expected behavior; log it for observability not alarm.)

**Gap (b) -- buy_amount < 50 silent skip:**
Replace `continue` at line 177 with:

```python
logger.warning(
    "Skipping BUY %s: buy_amount=%.2f below $50 minimum (nav=%.2f position_pct=%s available_cash=%.2f)",
    cand["ticker"], buy_amount, nav, position_pct, available_cash,
)
continue
```

(Use `warning` -- same level as the price<=0 guard at `autonomous_loop.py:248`; both are silent drop scenarios.)

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (12 collected: 5 read in full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions/consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
