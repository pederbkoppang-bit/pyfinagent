# Research Brief — step 85.6 (P0 DEADLOCK: the book cannot be un-paused)

Tier: **complex**. Audit-class: **false**. Started 2026-08-08. Write-first: created
before any reading, appended incrementally.

Status: COMPLETE. `gate_passed: true` (6 sources read in full, 22 URLs, recency
scan performed, three-variant query discipline visible below).

## Scope

Q1 — a correct way to roll the start-of-day anchor that does NOT depend on the
analysis phase completing, without weakening phase-36.9's `armed:true` guarantee.
Q2 — 409 wording that is TRUE and phase-36.12-compliant; was the `:611 Verified`
claim ever true?
Q3 — is the daily-anchor staleness guard correctly scoped (sod_date None/stale in
PRODUCTION)? Overlaps 85.5.1 — findings labelled per step.
Q4 — weekend reality: resume WITHOUT a cycle, without hand-writing a
`sod_snapshot` row, without weakening the guard.

---

## Internal code inventory (running)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/kill_switch.py` | 1053 | `KillSwitchState` singleton, `evaluate_breach`, `_sod_date_is_stale`, `check_auto_resume` | live |
| `backend/services/paper_trader.py` | 1588 | `check_and_enforce_kill_switch` (:1214), `sod_anchor_needs_reroll` (:68) | live |
| `backend/services/autonomous_loop.py` | ~1900 | the REAL trading cycle; kill-switch at Step 5.5 (:1370-1375) | live |
| `backend/autonomous_loop.py` | 620 | **NOT the trading loop** — this is the harness Plan/Generate/Evaluate loop | live |
| `backend/api/paper_trading.py` | 1459 | `/resume` 409 chain (:586-652) | live |

### PREMISE CORRECTION #1 — the step cites the wrong file

The step says the MARK/TRADE region is `autonomous_loop.py ~:1271+`. There are TWO
files with that basename. `backend/autonomous_loop.py` is **620 lines** and is the
*harness* loop (`_plan_phase` :349, `_generate_phase` :395, `_evaluate_phase` :421)
— it contains **zero** kill-switch references. The trading cycle is
`backend/services/autonomous_loop.py`. Anchors below use the `services/` path.

### CONFIRMED — the SOD roll has exactly ONE production trigger

`grep -rn "update_sod_nav" --include="*.py"` over the repo (excluding `.venv`)
returns exactly **one** production call site:

- `backend/services/paper_trader.py:1298` — `state.update_sod_nav(nav, date=today)`,
  guarded by `if sod_anchor_needs_reroll(snap, today):` at `:1297`.

Every other hit is a test or a comment. And `check_and_enforce_kill_switch` (the
method containing :1298) has exactly **one** production caller:

- `backend/services/autonomous_loop.py:1375` —
  `ks_check = await asyncio.to_thread(trader.check_and_enforce_kill_switch)`

(`backend/slack_bot/scheduler.py:906` is a *docstring* mention, not a call.)

### CONFIRMED — the roll sits AFTER the analysis phase

Phase order in `backend/services/autonomous_loop.py`:

| Line | Step |
|---|---|
| :514 | Step 1 Screen universe |
| :1086 | Step 2 Filter candidates |
| :1148 | Step 3 **Analyze candidates** |
| :1223 | Step 3+4 dispatch (phase-85.4) |
| :1338 | Step 5 Mark to market |
| :1345 | Step 5.4 Scale-out ladder |
| **:1370-1375** | **Step 5.5 Kill-switch evaluation** ← the only SOD-roll trigger |
| :1414 | Step 5.6 Stop-loss |
| :1488 | Step 6 Decide trades |
| :1555 | Step 7 Execute trades |

So step 85.6's chain link 3 is **CONFIRMED**: a cycle that dies in Step 3 never
reaches :1375, so `update_sod_nav` never runs, so `sod_date` never advances.

### CONFIRMED — the paused book skips decide/execute anyway

`backend/services/autonomous_loop.py:1380` logs
`"Paper trading: kill-switch active (%s) -- skipping decide/execute"`, appends
`kill_switch_halted` to `summary["steps"]` (:1383) and sets
`summary["status"] = "halted_kill_switch"` (:1400) — i.e. it returns BEFORE Step 6
(:1488). Chain link 4 confirmed: even a *completing* cycle trades nothing while
paused.

### The three-state machine as built

`evaluate_breach` (`kill_switch.py:720-853`) computes:

```
daily_baseline_missing   = not (sod is not None and sod > 0)          # :768
trailing_baseline_missing= not (peak is not None and peak > 0)        # :769
daily_baseline_stale     = _sod_date_is_stale(s["sod_date"], sod)     # :780
daily_leg_unevaluable    = daily_baseline_missing or daily_baseline_stale  # :781
armed = not (daily_leg_unevaluable or trailing_baseline_missing)      # :782
baselines_present = baselines_present_in(s)                           # :798
```

`_sod_date_is_stale` (`:876-901`) returns **False** when `sod_nav` is None/<=0
(`:893-894`) — absence is reported by `daily_baseline_missing`, not by staleness.
An unparseable or missing date on a PRESENT baseline is stale (`:895-896`).

`armed` is the strict "can this leg fire NOW" flag; `baselines_present` is the
pre-36.9 "do we still have baselines" flag. The comment at `:787-797` records WHY
the split exists: `paper_trader`'s 36.12 gate measures state BEFORE the roll, so on
the first cycle of every UTC day the pre-roll anchor is yesterday's **by
construction** — folding staleness into that flag made every ordinary morning look
like LOST HISTORY (P1 page + fabricated `lost_history_anchor` row).

### MEASURED — live kill-switch state (read-only, `handoff/kill_switch_audit.jsonl`)

52 rows, `2026-07-25T11:35:09Z` .. `2026-08-08T19:59:35Z`. Histogram:
`pause=40, sod_snapshot=7, resume=5`.

- Last `resume`: **2026-07-27T06:20:38Z**. Every row after it is a `pause`, so the
  replay (`_load_from_audit` :266-273) leaves `_paused=True`. **PAUSED confirmed.**
- Last `sod_snapshot`: **`{"ts":"2026-08-05T19:34:47Z","nav":23830.46,"date":"2026-08-05"}`**.
  So `sod_date='2026-08-05'` — matches the step's premise. **Anchor is 3 days stale.**
- All 7 `sod_snapshot` rows carry an explicit `date` field.

#### SIDE FINDING (not 85.6's fix, but it changes the design) — 40 `pause` rows, `trigger:"manual"`, still arriving

Rows are still being appended **today**: `2026-08-08T08:15:52Z, 08:23:52Z,
08:30:58Z, 08:31:38Z, 08:35:16Z, 19:59:35Z`, all `event:"pause"`,
`trigger:"manual"`, `details:{}` — against a book whose last `resume` was
2026-07-27. Consequences:

1. `_load_from_audit:270` sets `self._paused_at = row.get("ts")` on **every**
   pause row, so the phase-38.1 hysteresis clock in `check_auto_resume`
   (`kill_switch.py:969-977`) is **reset to zero every time one of these lands**.
   The `AUTO_RESUME_TRIGGER_AT_SEC = 2h` threshold (`:939`) can therefore never
   mature while they keep arriving. Auto-resume is default-OFF
   (`settings.py:390`), so this is currently latent — but **any 85.6 design that
   leans on `check_auto_resume` inherits this latch.**
2. The file is git-tracked (last committed in `8aa3f52e`), so this is committed
   operator state (same coupling class as the phase-85.2 finding).

I did not determine the writer (out of scope, read-only). Flagging it because a
design that "just enables auto-resume" would look correct and still never fire.

### Q3 (85.5.1 overlap) — MEASURED, and the step's framing is only half right

**Belongs to 85.5.1 (fixture):** `backend/tests/test_book_safety_69.py:79`
monkeypatches `st.snapshot` to return `{"sod_nav": 100.0, "peak_nav": 100.0}` —
**the `sod_date` key is absent entirely**. `evaluate_breach:780` calls
`_sod_date_is_stale(s.get("sod_date")=None, sod=100.0)`; `:895` (`not
isinstance(None, str)`) returns **True** → `daily_leg_unevaluable` → `armed=False`
→ `:830` skips the daily leg → `daily_loss_breached=False`. The test's first
assertion fails. Production `_snapshot_locked` (`kill_switch.py:440-451`) **always
emits the `sod_date` key**, so the fixture's dict shape is not what production
produces. That much is a fixture defect.

**Belongs to 85.6 (production):** but `sod_date is None` with `sod_nav > 0` IS
production-reachable, by exactly one path — `_load_from_audit:285-295`:

```python
sod_date = row.get("date")
if not sod_date:
    ts = row.get("ts")
    if ts:
        try:    sod_date = datetime.fromisoformat(...).date().isoformat()
        except Exception: sod_date = None
self._sod_date = sod_date
```

A legacy `sod_snapshot` row (pre-23.2.19, no `date`) whose `ts` is also
unparseable restores `sod_nav` positive + `sod_date=None`. **None of the 7 live
rows is in that shape**, so this is not the current live state.

**The measured blast radius is NARROWER than "a real drawdown does not fire":**

- The **trailing** leg is date-independent (`:836-838`) and still fires. A 20%
  drawdown vs peak breaches `paper_trailing_dd_limit_pct` regardless of
  `sod_date`. In the RED test itself `any_breached` is in fact **True** — only
  `daily_loss_breached` is False.
- The exposed window is drawdowns in `[daily_loss_limit_pct, trailing_dd_limit_pct)`
  measured against their respective baselines — i.e. the daily leg alone.
- And in the ORDINARY cycle it self-heals one line later:
  `sod_anchor_needs_reroll` (`paper_trader.py:68-88`) returns True when
  `snap.get("sod_date") != today`, so `:1298` re-anchors inside the same function
  call.

**The case where it does NOT self-heal is 85.6's deadlock itself** — the roll is
unreachable, so a stale/None `sod_date` persists for days. Measured: the live
anchor has been stale since 2026-08-05, i.e. the daily leg has been unevaluable
for 3 days. It is masked only because the book is also paused.

### Q2 — was the `:611 Verified` claim ever true?

`git log -S "Verified: a PAUSED book still reaches the roll"` →
**`1657e25a`, 2026-07-26, "phase-36.9: armed:true must mean the leg can actually
fire NOW"**. The claim asserts one thing and the reader infers another:

- **What it says, and it is TRUE:** *pausedness* does not block the roll.
  `check_and_enforce_kill_switch` has no early-return for a paused book before
  `:1298`; and in the loop the paused short-circuit at
  `services/autonomous_loop.py:1377-1412` runs **after** the `:1375` call.
  Verified by reading both files: a paused book that reaches Step 5.5 does roll.
- **What it implies, and it is FALSE:** "cannot wedge". The verification covered
  the `paused → roll` edge and silently assumed `cycle → Step 5.5`. That second
  edge is the one that broke. The roll's only production trigger is
  `paper_trader.py:1298`, reached only from
  `services/autonomous_loop.py:1375`, which is the **7th** of 10 steps and sits
  behind the analysis phase (Step 3, `:1148`). The refusal self-clears "within one
  cycle" only if a cycle *completes past Step 5.5* — and per phase-85.4 no cycle
  has done so since 2026-07-31.

So: the verification was **scope-correct and conclusion-wrong**. The wording
constraint from 36.12 was satisfied by avoiding the banned phrases; the factual
claim about liveness was never in the tested scope.

---

### PREMISE CORRECTION #2 — the pause did not start on 2026-08-04

The step says "latched `paused` since 2026-08-04T11:43:31Z". Measured from the
audit replay's own state-machine semantics (`_load_from_audit:266-273` — only a
`resume` clears `_paused`), the **state-change** rows since the last resume are:

```
2026-07-27T06:20:38Z  resume  manual     <-- last resume, book RUNNING
2026-08-03T09:03:17Z  pause   manual     <-- LATCH STARTS HERE
```

36 further `pause` rows follow, all redundant. `2026-08-04T11:43:31Z` **is** a row
in the file — but it is one of 12 rows on 2026-08-04 and is not a state change. The
book has been continuously paused since **2026-08-03T09:03:17Z**, ~1 day earlier
than the step states. This matters because the outage is one day longer than
assumed and it predates 85.4's first missed cycle window.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-08-08 | Official docs (tier 2) | WebFetch, full page (3586 w) | "**Failed operations testing:** In the Open state, rather than using a timer to determine when to switch to the Half-Open state, a circuit breaker can **periodically ping** the remote service ... This ping can either attempt to invoke a previously failed operation **or use a special health-check operation**." AND "**Manual override:** If the recovery time for a failing operation is extremely variable, you should provide a **manual reset option** that enables an administrator to close a circuit breaker and reset the failure counter." AND "**Recoverability:** ... if the circuit breaker remains in the Open state for a long period, it can raise exceptions **even if the reason for the failure is resolved**." AND "System recovery is based on **external operations**." |
| 2 | https://resilience4j.readme.io/docs/circuitbreaker | 2026-08-08 | Official docs (tier 2) | WebFetch, full | The exact defect, named and configurable. With `automaticTransitionFromOpenToHalfOpenEnabled=false` (the **default**): "the transition to HALF_OPEN **only happens if a call is made**, even after `waitDurationInOpenState` is passed." With it true: "the CircuitBreaker will automatically transition ... **no call is needed** to trigger the transition. **A thread is created to monitor all the instances** of CircuitBreakers." Also documents `DISABLED` / `FORCED_OPEN` / `METRICS_ONLY` as first-class states. |
| 3 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-08 | Authoritative book (tier 3) | WebFetch, full chapter | Stable-degraded equilibrium: reducing load "will almost certainly not stop the crashes ... only a small fraction of servers will usually be healthy enough". "**Stop Health Check Failures**" — disable automated health checks when "health-checking itself makes the service unhealthy". Restart servers when "**The servers are deadlocked**", but "Make sure that you **identify the source** of the cascading failure **before** you restart". And on rarely-exercised recovery code: "**the code path you never use is the code path that (often) doesn't work**." |
| 4 | https://www.sec.gov/files/rules/final/2010/34-63241-secg.htm | 2026-08-08 | Regulatory (tier 2) | `curl -A ... \| tag-strip` (WebFetch 403s on sec.gov) — full text | Controls must be "reasonably designed to: ... **prevent the entry of orders unless there has been compliance with all regulatory requirements that must be satisfied on a pre-order entry basis**" and the firm must have "**direct and exclusive control** of its financial and regulatory risk management controls". Also: "establish, document, and maintain a system for **regularly reviewing the effectiveness** of the risk management controls ... and for **promptly addressing any issues**." |
| 5 | https://www.legislation.gov.uk/eur/2017/589/chapter/II/section/3/2016-07-19/data.xht?view=snippet&wrap=true | 2026-08-08 | Regulatory / legislative (tier 2) — RTS 6 | WebFetch, full section | Art. 12 kill functionality: capability to "cancel **immediately, as an emergency measure**, any or all of its unexecuted orders". Art. 15 pre-trade controls incl. "**repeated automated execution throttles**" that automatically disable trading systems. Art. 14 business continuity: "**arrangements for shutting down** the relevant trading algorithm" + relocation procedures + **annual review and testing**. Notably: the regulation mandates the *stop* and the *restart procedure*, and treats them as separate, tested arrangements. |
| 6 | https://blog.bolshakov.dev/2025/12/06/why-circuit-breaker-recovery-needs-coordination.html | 2026-08-08 | Practitioner blog (tier 3) — **[QUALIFYING / counter-view]** | WebFetch, full | Argues the canonical half-open diagram is under-specified: "The implicit assumption in every state diagram: **one probe at a time. One evaluation. One decision.** With concurrent execution, this assumption breaks immediately." Proposes **serializing the entire probe sequence under a lock**, not just the final state write, else concurrent probes produce a state "**disconnected from any coherent view of service health**". |
| 7 | https://www.mql5.com/en/blogs/post/773545 | 2026-08-08 | Community (tier 5) — **recency, 2026-08-02** | WebFetch, full | The anti-pattern for two of Q1's candidates, stated exactly: an EA re-anchors the day on `OnInit` (a **restart-class event**), and "**From every anchor's point of view the limit was never breached**" — "down 3%, re-anchor, down 4% more ... re-anchor". Prescribed fix: anchor keyed by account + **date**, so "the **date** stamp makes the anchor **self-expiring**", persisted to disk immediately so it "**survives restart events that might otherwise clear the limit**". |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0 | Regulatory FAQ | WebFetch 403 (sec.gov blocks the agent UA); the SECG (#4) covers the same requirements and was fetched via curl |
| https://www.sec.gov/files/rules/final/2010/34-63241.pdf | Regulatory (adopting release) | Binary PDF; SECG summary sufficed at this tier |
| https://www.finra.org/rules-guidance/guidance/reports/2021-finras-examination-and-risk-monitoring-program/market-access | Regulatory exam guidance | Snippet established that FINRA examines "whether they use ... kill switches, to monitor and respond to aberrant behavior by trading algorithms"; no new mechanism |
| https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-3/ | Regulatory | Fetch returned the FCA Handbook landing page, not the article text; legislation.gov.uk (#5) served the same instrument |
| https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0589 | Regulatory | PDF; superseded by #5 |
| https://github.com/App-vNext/Polly/wiki/Circuit-Breaker | Industry impl | Same state machine as #2; #2 documents the automatic-transition knob explicitly, which is the load-bearing detail |
| https://www.pollydocs.org/strategies/circuit-breaker.html | Industry impl | as above |
| https://linkerd.io/2.15/reference/circuit-breaking/ | Official docs | Snippet gave the key distinction (probe requests "should not be confused with HTTP readiness and liveness probes"); recorded, not load-bearing |
| https://www.javacodegeeks.com/2026/07/the-circuit-breaker-patterns-hidden-assumptions-when-fail-fast-fails-slow.html | Blog | Recency hit; overlaps #6 |
| https://oneuptime.com/blog/post/2026-01-24-circuit-breaker-open-errors/view | Blog | Ops-troubleshooting, no new mechanism |
| https://oneuptime.com/blog/post/2026-02-06-monitor-circuit-breaker-state-changes-opentelemetry-metrics/view | Blog | Observability angle only |
| https://github.com/resilience4j/resilience4j/issues/1362 | Issue tracker | Confirms the "stays open too long" complaint in the wild |
| https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit | Industry (prop firm) | Recency; corroborates date-keyed reset |
| https://apextraderfunding.com/help-center/additional-helpful-items/daily-loss-limit-explained/ | Industry | "pauses trading for the remainder of the day ... continue on the next trading day" — the *session*-scoped semantics referenced in Q4 |
| https://aifo.com/blog/guide/daily-loss-reset-time-in-prop-firms/ | Industry | Reset-time variance across firms (17:00-18:00 ET) |
| https://eleonex.com/blog/when-does-daily-loss-limit-reset-prop-firm-accounts/ | Industry | as above |
| https://www.tradovate.com/daily-loss-limit/ | Vendor | as above |
| https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | Regulatory (2026-02) | PDF; recency-scan hit, logged below |
| https://databento.com/microstructure/market-access-rule | Industry explainer | Secondary to #4 |
| https://learn.microsoft.com/en-us/previous-versions/msp-n-p/dn589784(v=pandp.10) | Official docs (archived) | Superseded by #1 |
| https://distributedsystemauthority.com/circuit-breaker-pattern | Blog | Community tier |
| https://dev.to/axiom_agent/nodejs-circuit-breaker-pattern-in-production-opossum-fallbacks-and-resilience-engineering-1mj4 | Community | Lowest tier |

**Total unique URLs collected: 22** (7 read in full + 15 snippet-only).

### Search-query composition (three-variant discipline)

| Variant | Query run |
|---|---|
| **Year-less canonical** | `circuit breaker pattern half-open state reset semantics` |
| **Year-less canonical** | `SEC Rule 15c3-5 market access risk management controls kill switch pre-trade` |
| **Year-less canonical** | `algorithmic trading kill functionality RTS 6 Article 12 cancel outstanding orders reactivate` |
| **Current-year (2026)** | `2026 automated trading system deadlock recovery daily loss limit reset start-of-day mark` |
| **Last-2-year (2025)** | `2025 circuit breaker stuck open state recovery probe independent of request path liveness` |

## Recency scan (2024-2026) — PERFORMED

Two dedicated passes (2026 and 2025 variants above). Result: **3 new findings that
COMPLEMENT, and 1 that SHARPENS, the canonical sources.**

1. **2026-08-02, MQL5 (source #7)** — six days old. Directly names the anti-pattern
   for Q1 candidate C ("roll on backend startup"): re-anchoring on an
   initialisation event "hands you a fresh full allowance", and *"From every
   anchor's point of view the limit was never breached."* Prescribes the
   **date-keyed, self-expiring, disk-persisted** anchor — which is precisely what
   `sod_anchor_needs_reroll` + `update_sod_nav` already implement. **This
   supersedes nothing in the repo; it independently validates the existing
   predicate and rules out the naive startup-roll design.**
2. **2025-12-06, Bolshakov (source #6)** — qualifies the canonical half-open state
   machine: the diagram assumes a single probe, which breaks under concurrency.
   Relevant here because the SOD roll is *already* concurrency-exposed (a
   `threading.Lock` in `KillSwitchState`, plus a cross-process `handoff/.autonomous_loop.lock`).
   Any new scheduled roller becomes a **second writer** and must be serialized
   against the cycle's roller.
3. **2026-02, ESMA Supervisory Briefing on Algorithmic Trading in the EU**
   (snippet-only) — confirms kill-functionality supervision is still live
   regulatory focus in 2026; no mechanism change vs RTS 6 (2017).
4. **2026-07, Java Code Geeks "hidden assumptions"** (snippet-only) — same thesis
   as #6.

**Nothing found in the 2024-2026 window supersedes the Azure/resilience4j state
machine or the SEC/RTS 6 obligations.** The canonical prior art is still canonical.

---

## Key findings

1. **The literature names pyfinagent's exact bug, and it is a known default, not an
   exotic one.** resilience4j ships with
   `automaticTransitionFromOpenToHalfOpenEnabled = false`, documented as: *"the
   transition to HALF_OPEN only happens if a call is made, even after
   `waitDurationInOpenState` is passed"* (source #2). pyfinagent's recovery
   precondition (the SOD roll) likewise only advances *if a cycle is made*. The
   library's own remedy is the fix: *"A thread is created to monitor all the
   instances of CircuitBreakers to transition them"* — i.e. **drive the recovery
   transition from a scheduler, not from the protected path.**

2. **The recovery probe must not be the failed operation.** Azure (source #1)
   offers two options for testing recovery — *"attempt to invoke a previously
   failed operation **or** use a special health-check operation"*. pyfinagent
   currently uses the first, and the "previously failed operation" is a
   ~7200s-budget, LLM-heavy, 10-step cycle. Google SRE (source #3) adds the
   sharper version: when *"health-checking itself makes the service unhealthy"*,
   the health check is the bug. Here the recovery check is gated behind the
   analysis phase that is itself the failure.

3. **Long-Open is a recognised failure mode, not an edge case.** Azure
   *"Recoverability"*: *"if the circuit breaker remains in the Open state for a
   long period, it can raise exceptions **even if the reason for the failure is
   resolved**."* Measured live: paused since 2026-08-03T09:03:17Z, 5 days.

4. **A manual, audited operator reset is canonical prior art — not a hack.** Azure:
   *"you should provide a **manual reset option** that enables an administrator to
   close a circuit breaker and reset the failure counter."* RTS 6 Art. 14 treats
   shutdown AND the arrangements to come back as **separate, documented, annually
   tested** procedures (source #5). pyfinagent already has the shape for this
   (`KS-PEAK-RESET` operator token, `reset_peak` at `kill_switch.py:641`, DARK by
   default). **The gap is that there is no equivalent operator-gated path for the
   SOD anchor.**

5. **Regulatory posture forbids the tempting shortcut.** SEC 15c3-5 requires
   controls "reasonably designed to ... prevent the entry of orders unless there
   has been compliance with all ... pre-order entry" requirements, under the
   firm's "direct and exclusive control" (source #4). A design where pressing
   *resume* also *creates* the baseline that makes resume legal is a control that
   authorises itself. That is the phase-36.12 defect relocated, not fixed.

6. **Rarely-run recovery code rots.** Google SRE: *"the code path you never use is
   the code path that (often) doesn't work"*, with the remedy *"regularly running
   a small subset of servers near overload in order to exercise this code path."*
   The `:611 "Verified"` comment is an instance: a recovery claim asserted once,
   never exercised, wrong the first time it mattered.

---

## Consensus vs debate (external)

**Consensus** — (a) three states minimum, with an explicit trial state between
"stopped" and "running"; (b) the Open→Half-Open transition is time-driven; (c) a
manual administrator override is expected in production designs; (d) daily-loss
anchors must be **date-keyed and self-expiring**, never event-keyed.

**Debate** — *who drives the transition.* resilience4j's default and Linkerd's
"probation" both drive it from **real traffic** (Linkerd explicitly: probe
requests *"should not be confused with HTTP readiness and liveness probes"* — a
health check passing is not sufficient evidence of recovery). Azure's "Failed
operations testing" and resilience4j's optional monitor thread drive it from an
**independent prober**. Source #6 adds that the independent prober must be
**serialized** or concurrent probes yield an incoherent state.

**Resolution for pyfinagent:** the two camps disagree about proving a *remote
dependency* is healthy. pyfinagent's blocked transition is not a health probe at
all — it is a **bookkeeping precondition** (stamp today's opening NAV). Nothing
about it requires the trading cycle to have succeeded. The traffic-driven camp's
rationale therefore does not apply, and the independent-prober design is correct
here.

---

## Application to pyfinagent — Q1 candidate evaluation

Evaluated against the phase-36.9 invariant: **`armed:true` must mean the daily leg
can fire NOW**, and nothing may let a real drawdown go unfired.

### Candidate A — roll at cycle START (move the roll before Step 3)

- **Fixes the deadlock?** Partially. It moves the roll ahead of the analysis
  phase, so a cycle that dies in Step 3 (`services/autonomous_loop.py:1148`) still
  rolls.
- **Weakens 36.9?** No, *if and only if* the `pre` measurement at
  `paper_trader.py:1241` stays strictly before the roll — 36.12's explicit
  ordering doctrine ("MEASURE THE ARMED STATE BEFORE MUTATING THE BASELINES",
  `:1230`). Naively hoisting the whole `check_and_enforce_kill_switch` call would
  ALSO hoist the breach evaluation ahead of Step 5 mark-to-market (`:1338`), which
  evaluates the breach on an unmarked NAV. **The roll must be split from the
  enforcement; the call must not simply be moved.**
- **Q4 (weekend)?** **FAILS.** Still requires a cycle.
- Verdict: necessary hardening, insufficient alone.

### Candidate B — a separate scheduled daily roll (RECOMMENDED PRIMARY)

The resilience4j `automaticTransitionFromOpenToHalfOpenEnabled=true` design
(source #2), and Azure's "periodically ping ... or use a special health-check
operation" (source #1).

- **Weakens 36.9?** No. Rolling the anchor is what *restores* `armed`; it does not
  bypass any evaluation. The forgiveness risk (MQL5, source #7) is neutralised by
  the guard that already exists: `sod_anchor_needs_reroll`
  (`paper_trader.py:68-88`) fires only when `sod_nav<=0/None` **or**
  `snap["sod_date"] != today`. A job that runs once per UTC day is idempotent by
  construction; a same-day re-run is a no-op. This is precisely MQL5's
  "date stamp makes the anchor self-expiring".
- **Strictly safer than today** on the anchor's *value*: a fixed pre-open schedule
  anchors nearer the true session open than "whenever the first cycle of the day
  happens to reach Step 5.5" does.
- **Q4 (weekend)?** **PASSES** — a scheduler can run Saturday.
- **Must-address (source #6):** it is a **second writer** to `_sod_nav`/`_sod_date`
  and to the audit file. `KillSwitchState._lock` covers in-process races
  (`kill_switch.py:547`), but the roll is a **read-then-write** across
  `state.snapshot()` (`:1295`) and `update_sod_nav` (`:1298`) — not atomic. If the
  scheduled roller runs in the same process as the cycle, wrap the
  reroll-decision + write in one critical section. If it can run in a different
  process, `handoff/.autonomous_loop.lock` (measured `state:"released"` at
  2026-08-08T19:59:54Z) is the existing cross-process seam.
- **Must-address:** the job's NAV source. `check_and_enforce_kill_switch` reads
  `self.get_or_create_portfolio()["total_nav"]` (`:1226-1227`) — the last
  *persisted* NAV, not a fresh mark. For a start-of-day anchor that is arguably
  the more correct input (it is the prior session's close). Do NOT add a
  mark-to-market to the roller; that would make the anchor an intraday mark.
- **Must-address:** `update_sod_nav` already refuses a non-positive anchor
  (`:538-546`), so a degraded BQ read leaves `sod_nav` None → daily leg reported
  *missing*, not silently anchored at 0. Keep that; do not add an `or 0.0`.

### Candidate C — roll on backend startup: **REJECT**

Source #7 is decisive and six days old: re-anchoring on an initialisation event
means *"From every anchor's point of view the limit was never breached."* Worse,
Google SRE (source #3) lists restart as a *legitimate remediation for a deadlocked
server* — so in exactly the situation 85.6 describes, an operator restarting the
backend is likely, and a startup roll would silently re-anchor mid-drawdown.
Partial mitigation exists (the date gate makes a same-day restart a no-op), but
the residual is the dangerous case: a restart on a day whose anchor was never set
anchors at the already-drawn-down NAV. **Do not key the anchor on a
process-lifecycle event.**

### Candidate D — roll inside `/resume`: **REJECT HARD**

`/resume` refuses *because* the daily leg is unevaluable. Rolling inside `/resume`
means the refusal manufactures its own precondition: operator clicks resume →
anchor set to the current (drawn-down) NAV → daily leg reads healthy → resume
succeeds. That is verbatim the phase-36.12 defect ("that cycle's silent anchor WAS
the defect; it forgives the real drawdown", `paper_trading.py:638-641`) moved from
the cycle into the button, and a control that authorises itself under SEC 15c3-5's
"direct and exclusive control" framing (source #4). It is also an IEC 61511
Cl. 16.2.4 bypass-with-no-exit, the same citation `update_sod_nav:535` already
uses against its own prior defect.

**A safe *variant* of D exists and is worth designing:** `/resume` may
**re-derive** today's opening NAV from durable history rather than from the
current NAV. `PaperTrader.save_daily_snapshot` (`paper_trader.py:1080`) persists a
daily NAV snapshot, so a prior-session close is recoverable. Anchoring to a
**measured historical value** is restoration; anchoring to `nav`-at-call-time is
forgiveness. The distinction is the whole safety argument — and it is the same
distinction phase-36.8 already drew for the peak (`update_peak:567-572`: authority
is granted only to a row that **names what it superseded**). **Apply the identical
rule to the SOD anchor: an operator-path re-anchor must name the source snapshot
it restored from.**

### Candidate E — an audited, operator-token-gated re-anchor endpoint

Azure's *"manual reset option"* (source #1) + RTS 6 Art. 14's documented restart
arrangements (source #5). The repo already has the exact pattern:
`kill_switch.reset_peak` (`:641`), DARK behind `kill_switch_peak_reset_enabled`,
with an `operator` attribution and an audit row.

**This is NOT the forbidden hand-written `sod_snapshot` row.** The differences are
substantive, not cosmetic: it goes through `update_sod_nav`'s non-positive refusal
(`:538-546`), it is attributed, it is replay-consistent, and it is
mutation-testable. Hand-editing `handoff/kill_switch_audit.jsonl` bypasses all
four. Combine with the Candidate-D-variant rule (must name the restored source)
and it is auditable.

### Q4 — can Monday trade without a cycle running over the weekend?

**Yes, via Candidate B and/or E** — both roll the anchor without a trading cycle,
without touching the audit file by hand, and without loosening the guard. B is
preferable as the standing fix; E is the operator escape hatch that Azure and RTS
6 both say a production kill switch should have.

**But note the calendar subtlety, and be careful with it.** On Sat/Sun there is no
session, so "today's open" does not exist and a Friday anchor is arguably the
current session's open. Making `_sod_date_is_stale` **session**-aware rather than
**calendar-date**-aware would be a correctness refinement, not a loosening — but it
would make a safety predicate depend on `backend/services/markets.py`, which has a
recorded latent defect (`is_trading_day:149` uses `cal.days`, removed in
`exchange_calendars` 4.13.2, so it can return True unconditionally). **If session
awareness is adopted, an unknown/failed calendar lookup must resolve to STALE**,
matching `_sod_date_is_stale`'s existing conservative doctrine (`:883-887`:
"freshness is a claim that must be provable").

### SCOPE WARNING — 85.6 is necessary but NOT sufficient for Monday trading

The step's framing ("this is why nothing trades") is **incomplete**. Two
independent gates sit between the current state and a trade:

1. The kill-switch halt at `services/autonomous_loop.py:1377-1412`, which returns
   before Step 6 Decide (`:1488`). **85.6 fixes this.**
2. The cycle timing out in the analysis phase (Step 3, `:1148`) so it never
   reaches Step 5.5 at all. **85.4 measured this; its remedies (asks #23/#24/#25)
   are operator-gated and NOT applied.**

Evidence both are real and separable: the last `sod_snapshot` is
`2026-08-05T19:34:47Z` — i.e. on 2026-08-05 a cycle DID reach Step 5.5 and DID
roll the anchor, on an already-paused book, then halted at `:1377`. So (a) the
roll is live-proven reachable on a paused book, and (b) a completing cycle exists
but traded nothing purely because of the pause. **Fixing 85.6 alone makes trading
possible only on days when the cycle also survives Step 3.**

---

## Pitfalls (from literature, mapped to this change)

| Pitfall | Source | Where it bites here |
|---|---|---|
| Recovery transition coupled to the protected path | #2 | The defect itself: `paper_trader.py:1298` reachable only via `services/autonomous_loop.py:1375` |
| Re-anchor on a lifecycle event returns full allowance | #7 | Kills Candidate C; also why Candidate D must not use call-time NAV |
| Concurrent probes → incoherent state | #6 | A scheduled roller is a 2nd writer; `snapshot()`→`update_sod_nav` at `:1295-1298` is read-then-write, not atomic |
| Health check that itself makes things unhealthy | #3 | Do not add a mark-to-market or an analysis call to the roller |
| Unexercised recovery code doesn't work | #3 | The `:611 "Verified"` comment; whatever replaces it needs a test that actually drives the deadlock |
| Long-Open raises errors after the fault is fixed | #1 | 5 days paused; and 40 redundant `pause` rows keep resetting `_paused_at` |
| Self-authorising control | #4 | Candidate D naive form |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7: 5 via WebFetch, 1 via
      curl+tag-strip after a 403, plus 1 community recency source)
- [x] 10+ unique URLs total (22)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read, not abstracts
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered kill_switch.py, paper_trader.py,
      services/autonomous_loop.py, api/paper_trading.py, the live audit file, and
      the 85.5.1 test
- [x] Contradictions / consensus noted (traffic-driven vs prober-driven recovery)
- [x] Claims cited per-claim
- [ ] **GAP:** I did not identify what is writing the 36 redundant `trigger:"manual"`
      pause rows (read-only constraint + out of scope). Flagged as a side finding.
- [ ] **GAP:** I did not verify whether `save_daily_snapshot`'s persisted rows are
      queryable at the granularity Candidate-D-variant needs (a prior-session
      closing NAV). Design-time verification required before committing to that path.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 15,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The deadlock is a named, documented default: resilience4j ships automaticTransitionFromOpenToHalfOpenEnabled=false, where 'the transition to HALF_OPEN only happens if a call is made'. pyfinagent's SOD roll (paper_trader.py:1298) has exactly one production trigger, services/autonomous_loop.py:1375 at Step 5.5, behind the analysis phase at :1148. Two premises corrected: the cited file is backend/services/autonomous_loop.py (backend/autonomous_loop.py is the harness loop), and the pause latched 2026-08-03T09:03:17Z, not 08-04. The :611 'Verified' claim was scope-correct (a paused book does reach the roll -- live-proven by the 2026-08-05 sod_snapshot) and conclusion-wrong (it assumed the cycle reaches Step 5.5). Recommended: a separate scheduled daily roller (resilience4j's monitor-thread design) plus an audited operator-token re-anchor (Azure's mandated manual reset); REJECT rolling on backend startup (MQL5 2026-08-02: re-anchoring on init 'hands you a fresh full allowance') and REJECT rolling inside /resume (self-authorising control). 85.5.1's RED test is a fixture defect (sod_date key absent), but sod_date=None is production-reachable via _load_from_audit:285-295 and the trailing leg still fires. 85.6 is necessary but not sufficient: 85.4's timeout also blocks Step 6.",
  "brief_path": "handoff/current/research_brief_85.6.md",
  "gate_passed": true
}
```
