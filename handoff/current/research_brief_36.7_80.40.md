# Research Brief — masterplan steps 36.7 + 80.40

**Tier:** T3 / `complex` (Opus 5, effort max) — safety-critical.
**Spawned:** 2026-07-25/26 (third attempt; two prior spawns died on transient `API Error: 529
Overloaded` before writing anything).
**Nature of this gate:** DISCLOSED PROTOCOL-ORDER CORRECTION. Both steps were already
implemented + Q/A'd (CONDITIONAL) with internal code research but no external-literature
artifact. Purpose is to VALIDATE OR CHALLENGE shipped design decisions, not rubber-stamp.

**Bottom line up front:** the shipped designs hold up. Three of the four decisions are
independently corroborated by reference implementations or canonical pattern docs; one (36.7's
merge-with-`max()` ratchet) is corroborated only *because* the implementation also sorts by
timestamp and treats `peak_reset` as a compensating event — an order-free `max()` merge, which
is what the spawn prompt described, would be unsound, and the literature says exactly why.
**One NEW residual defect found that is not covered by the six queued follow-ups**: the
trading-path gate never reads the `armed` flag, and it re-anchors baselines *before* evaluating
— so 36.7's fail-loud state is unreachable on the path that actually places orders. Details in
§A.3 / "Ranked assessment".

## Questions

- A. Kill-switch / circuit-breaker re-arm + state persistence across restarts; fail-loud vs fail-open.
- B. Audit-log / event-sourcing rotation-safe state reconstruction (merge+extremum vs snapshot+epoch).
- C. NaN/Infinity hardening in financial risk calculations; documented incidents.
- D. Max-drawdown sign convention across quant libraries; verify the R PerformanceAnalytics claim.
- Recency scan: 2024–2026 guidance on adversarial review of AI-authored safety-critical financial code.

---

## Search log (3-query-variant discipline)

| # | Variant | Query |
|---|---|---|
| 1 | current-year `2026` | circuit breaker state persistence across restarts fail-safe default 2026 |
| 2 | **year-less canonical** | trading kill switch design state persistence restart fail-safe |
| 3 | last-2-year `2025` | event sourcing rebuild state after log rotation archived events snapshot 2025 |
| 4 | **year-less canonical** | maximum drawdown sign convention negative empyrical quantstats pyfolio |
| 5 | **year-less canonical** | NaN comparison silently disabled risk check postmortem financial software IEEE 754 |
| 6 | current-year `2026` | LLM generated code review safety-critical financial adversarial 2026 arXiv |
| 7 | **year-less canonical** | SEC Rule 15c3-5 market access risk controls kill switch pre-trade capital thresholds |
| 8 | **year-less canonical** | FIA recommended practices exchanges pre-trade risk controls kill switch drop copy |
| 9 | current-year `2026` | AI agent authored code review checklist safety-critical trading risk control verification 2026 |

All three variants exercised. The year-less queries were decisive: they produced the FIA
whitepaper, the CFR text, the R reference manual and the reference-implementation source files
— i.e. every source that actually settled a question. The `2026`-locked queries produced
mostly SEO content-farm pages (see snippet-only table) plus two useful arXiv hits.

---

## Read in full (12; gate floor is 5)

| # | URL | Accessed | Kind / tier | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-07-26 | Official docs (Microsoft; `ms.date` 2025-02-05, page updated 2026-07-02) | WebFetch, full (~3586 w) | 14-bullet "Problems and considerations" list includes **Manual override**, Monitoring, Failed request replay — and **NO bullet on durable state across restarts**. Breaker state is *derived* from a rolling counter that "automatically resets at periodic intervals." |
| 2 | https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing | 2026-07-26 | Official docs (Microsoft; `ms.date` 2026-03-27) | WebFetch, full (~4332 w) | **The decisive source for 36.7.** "You can determine the current state of an entity only by replaying all of the events that relate to it against the original state of that entity." "Snapshots are an optimization, not a replacement for the eventstream. The eventstream remains the source of truth." Event-ordering bullet: "The consistency of events in the event store and the order of events that affect a specific entity's current state are crucial. Adding a timestamp to every event can help you avoid problems. Another common practice is to annotate each event that results from a request with an incremental identifier." Compensating events: "The only way to update an entity or undo a change is to add a compensating event." |
| 3 | https://www.kurrent.io/blog/snapshots-in-event-sourcing | 2026-07-26 | Vendor eng. blog (Kurrent / EventStoreDB) | WebFetch, full | "If we received a snapshot, then besides reading the snapshot, you need to also read the events that happened after the snapshot was created." "Closing the books" pattern before archiving: "When the lifecycle is finished, store the *summary event* … we can safely schedule a task to move them to *cold storage* and delete them from the event store." Caution: "The need to use snapshots may hint to the model's design flaw." |
| 4 | https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | 2026-07-26 | Industry standard-setter (FIA, July 2024, 22 pp.) | curl + pdfplumber, full text extracted + keyword-indexed | §1.5 Kill Switches: "A kill switch is a control that, when activated, immediately disables all trading activity … typically preventing the ability to enter new orders and cancelling all working orders." "kill switches offer just one of many different types of risk controls … only invoked based on a qualitative decision taken as a last resort when other actions have failed." "this functionality should serve in addition to and as a final backstop for the pre-trade risk functionality." Non-overridable: "the automated trader should not be able to override a kill switch invoked by the broker." **Measured: the string `restart` occurs 0 times in the whole document; `recovery` 0 times.** |
| 5 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-07-26 | Primary regulation (17 CFR 240.15c3-5) | WebFetch, full | (c)(1): controls must "Prevent the entry of orders that exceed appropriate pre-set credit or capital thresholds" and "Prevent the entry of erroneous orders, by rejecting orders that exceed appropriate price or size parameters." Standard is to "**systematically** limit the financial exposure." (e): annual CEO certification + "a system for regularly reviewing the effectiveness of the risk management controls." **No text prohibiting a control from silently self-disabling** — the obligation is framed as an effectiveness-review duty, not a runtime invariant. |
| 6 | https://arxiv.org/html/2601.14059v1 — *Verifying Floating-Point Programs in Stainless* | 2026-07-26 | Peer-reviewed/preprint (arXiv, Jan 2026) | WebFetch, full HTML | **Names the exact bug class 36.7's `_coerce_nav` guard prevents.** §2: "NaN's unintuitive behaviour: any comparison involving NaN evaluates to false, including NaN == NaN." "If the input threshold is NaN, the comparison evaluates to false, causing the function to return the original vector, effectively ignoring the invalid input." §5.4, on real library code: "In two functions, this omission causes the implementation to return a valid output even when the input is NaN, effectively silently ignoring NaN inputs." Fig. 6 caption: code incorrectly assumes "a false comparison implies a valid (non-NaN) input." Prior incident cited: "NumPy previously contained a bug in its max function, which returned an arbitrary number instead of the maximum when NaN values were present." Remedy adopted: "Stainless automatically inserts NaN checks for all comparisons and equality operations." |
| 7 | https://www.agner.org/optimize/nan_propagation.pdf — Fog, *Parallel floating point exception tracking and NaN propagation* | 2026-07-26 | Named expert monograph (Agner Fog; **"Last updated 2026-06-16"**) | curl + pdfplumber, 17 pp. full | §7.3 "Loss of NaN": "We propose a similar feature for preventing situations where NaNs would not propagate. There are only a few cases where the IEEE 754 specifies that NaNs do not propagate … pow(NaN,0) … hypot(INF, NaN)". Confirms `min`/`max` are "implemented as a branch, for example min(a, b) = a < b ? a : b" — i.e. an extremum over a NaN-containing series silently drops the NaN. §7.4: R uses a **NaN payload of 1954 to mean "data not available"** as distinct from payload 0 for "invalid value" — i.e. a mature stats system separates *unmeasurable* from *invalid*, the same distinction 80.40 draws with `None`. |
| 8 | https://raw.githubusercontent.com/quantopian/empyrical/master/empyrical/stats.py | 2026-07-26 | Reference implementation (empyrical; engine behind zipline + pyfolio) | curl, full file | `max_drawdown` (`:352-402`) returns `nanmin((cumulative - max_return) / max_return)` → **negative fraction**. Degraded path `:379-383`: `if len(returns) < 1: out[()] = np.nan` → **NaN, explicitly not 0.0**. |
| 9 | https://raw.githubusercontent.com/ranaroussi/quantstats/main/quantstats/stats.py | 2026-07-26 | Reference implementation (quantstats) | curl, full file (114 KB) | `max_drawdown` (`:2451`) docstring: "Returns: float: Maximum drawdown (**negative value**)". `to_drawdown_series` (`:2499`): "pd.Series: Drawdown series (**negative values** showing decline from peak)". |
| 10 | https://search.r-project.org/CRAN/refmans/PerformanceAnalytics/html/maxDrawdown.html | 2026-07-26 | Official package reference (CRAN, PerformanceAnalytics 2.0.4) | curl + tag-strip, full | **Verifies the project's code comment verbatim.** `maxDrawdown(R, weights = NULL, geometric = TRUE, invert = TRUE, ...)`. "The default option `invert=TRUE` will provide the drawdown as a positive number. This should be useful for optimization (which usually seeks to minimize a value), and for tables … Practitioners will argue that drawdowns denote losses, and should be internally consistent with the quantile (a negative number), for which `invert=FALSE` will provide the value they expect. … we provide the option, but make no value judgment on which approach is preferable." |
| 11 | https://arxiv.org/html/2603.01494v1 — *Inference-Time Safety for Code LLMs (SOSecure)* | 2026-07-26 | Preprint (arXiv, Mar 2026) | WebFetch, full HTML | "developers often place substantial trust in LLM-generated code and may integrate it into production systems with limited security review"; "in adversarial settings, such trust can lead to exploitable vulnerabilities … or degraded system reliability." LLMs "may inherit vulnerable or outdated coding patterns from their training data." Measured fix rates 71.7 / 91.3 / 96.7% vs 49.1 / 56.5 / 37.5% baselines; 0.0% new-vulnerability introduction rate. Positions itself as "a complementary safety layer rather than a complete solution"; "Static analysis, secure coding guidelines, and training-time interventions all play important roles." |
| 12 | https://resilience4j.readme.io/docs/circuitbreaker | 2026-07-26 | Official library docs (resilience4j) | WebFetch, full | "The state of a CircuitBreaker is stored in a AtomicReference"; "Resilience4j comes with an **in-memory** CircuitBreakerRegistry." **No persistence mechanism and no restart-recovery guidance documented at all.** |

Tier mix: 2 primary regulation/standard-setter (5, 4), 3 official pattern/library docs (1, 2, 12),
2 peer-reviewed/preprint (6, 11), 1 named-expert monograph (7), 1 official package reference (10),
2 reference implementations (8, 9), 1 vendor eng. blog (3). No community-tier source is
load-bearing for any claim.

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://blog.alignment-systems.com/2016/01/kill-switch.html | Practitioner blog | FPGA/packet-capture kill-switch state persistence; superseded by FIA (#4) |
| https://www.institutionalinvestor.com/article/2bsvo5d0v4bynppdnbshs/portfolio/kill-switches-come-to-life | Trade press | Paywalled |
| https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0 | Primary (SEC staff FAQ) | Would have been valuable; CFR text (#5) covered the requirement. Queued if 15c3-5 detail is ever needed |
| https://www.sec.gov/litigation/admin/2013/34-70694.pdf (Knight Capital order) | Primary (SEC enforcement) | **Fetch FAILED — see "Gaps" below.** SEC returns a 1925-byte HTML interstitial to both `curl` and a browser UA at both known paths |
| https://www.mql5.com/en/articles/22532, /22613 | Community (retail EA) | Tier-5, retail-platform-specific |
| https://dev.to/alex_aslam/when-event-sourcing-fails-war-stories-from-production-1nk2 | Community | Tier-5; the "3TB replay" anecdote is colour only |
| https://github.com/quantopian/pyfolio/blob/master/pyfolio/timeseries.py | Source | pyfolio delegates to empyrical (#8); no independent evidence |
| https://medium.com/@sohail_saifii/... "Floating Point Standard That's Silently Breaking Financial Software" | Community | Tier-5; unverifiable "major European bank" anecdote — deliberately NOT used as evidence |
| https://devsecopsschool.com/blog/fail-safe-defaults/, https://aiopsschool.com/blog/circuit-breaker/, https://apiscout.dev/blog/... | Content-farm | Low-quality `2026`-locked SEO output |
| https://www.truefoundry.com/blog/ai-audit-checklist, https://iternal.ai/ai-agent-security-checklist, https://blog.lastpass.com/..., https://sphr.world/blog/owasp-agentic-top-10-checklist/, https://www.metacto.com/blogs/... | Vendor marketing | 2026 "AI audit checklist" pages; all agentic-*security* (prompt injection, tool perms), none about numerical/financial correctness review — a finding in itself, recorded in the recency scan |
| https://arxiv.org/abs/2605.00706 (FinSafetyBench), https://arxiv.org/abs/2604.26506 (SafeReview) | Preprint | Scope is LLM refusal/red-teaming of financial *advice*, not review of financial *code*. Off-target |
| https://arxiv.org/pdf/2601.14059 | Preprint (PDF form) | Read via the `/html/` route instead, per `.claude/rules/research-gate.md` |

**Unique URLs collected: 34** (12 read in full + 22 distinct snippet-only/candidate URLs).

---

## Recency scan (last 2 years, 2024–2026)

Performed, with results — this is not an empty section.

1. **2024 (FIA, July 2024).** The current industry best-practice paper for automated-trading
   risk controls contains **zero** occurrences of "restart", "recovery", or any discussion of
   how a kill switch recovers its own baseline state (measured by keyword count over the full
   extracted text of source #4). Kill switches are framed as *human-invoked last-resort
   backstops*, not as continuously-evaluated automatic limits with persistent baselines.
   pyfinagent's kill switch is the latter, so **the industry canon does not cover this
   design point at all** — a gap, not an endorsement either way.
2. **2026-03 (Microsoft Event Sourcing, `ms.date` 2026-03-27).** Freshly revised. Its
   *Event ordering* and *Versioning events / compensating events* bullets are the most
   directly applicable current guidance to 36.7 and are what the implementation actually
   follows. Supersedes nothing older; it is simply the right canon for this problem.
3. **2026-06 (Fog, updated 2026-06-16).** Live-maintained. Adds the `min`/`max`-as-branch
   detail that explains *why* an extremum silently swallows NaN, and the R
   payload-1954 "not available" vs payload-0 "invalid" distinction.
4. **2026-01 / 2026-03 (arXiv 2601.14059, 2603.01494).** Both new. 2601.14059 is the first
   source found that states the "guard is vacuously false under NaN, so the function returns
   a valid-looking answer" bug class as a *measured finding in real library code*, not as
   folklore.
5. **On the specific recency question — adversarial review of AI-authored safety-critical
   financial code: NO substantive guidance exists.** Every 2026 hit is agentic-AI *security*
   (prompt injection, tool permissions, OWASP Agentic Top 10, FINRA 2026 GenAI/cyber
   attention) or generic AI-PR-triage checklists. Nothing addresses numerical correctness,
   sign conventions, sentinel-value semantics, or guard vacuity — the four things that
   actually went wrong here. The nearest real guidance is arXiv:2603.01494's finding that
   developers "integrate [LLM code] into production systems with limited security review,"
   plus its explicit positioning of any single automated layer as "a complementary safety
   layer rather than a complete solution." **Practical read-across: the review checks that
   would have caught 36.7/80.40's original defects are not in any published AI-code-review
   checklist; this project's own Q/A discipline (independent evaluator, mutation-tested
   guards, presence-not-value discrimination) is ahead of the external literature here, and
   the pyfinagent-local rules are the better authority.**

---

## Findings

### A. Kill-switch re-arm, state persistence, fail-loud vs fail-open

**A.1 — "Fail loud on missing state" is correct, and the SRE circuit-breaker canon cannot be
cited against it, because that canon is about a different kind of state.**
Both the pattern doc and the leading implementation treat breaker state as ephemeral:
resilience4j keeps it in "a AtomicReference" in "an **in-memory** CircuitBreakerRegistry"
with no documented persistence
(https://resilience4j.readme.io/docs/circuitbreaker, accessed 2026-07-26), and Microsoft's
14-bullet considerations list never raises restart durability at all
(https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker, accessed
2026-07-26). That silence is *sound for their problem and unsound for pyfinagent's*: a
web-service breaker's state is a rolling failure count that re-derives itself from live
traffic within seconds, whereas a **trailing high-water NAV mark is not re-derivable from
current state at any cost** — the information exists only in history. So the "just reset to
CLOSED on boot" default that the SRE canon implies is not transferable, and 36.7 is right not
to transfer it. **Conclusion: the shipped fail-loud choice is not contradicted by prior art;
it is outside the scope of the prior art usually cited.**

**A.2 — The regulatory framing supports fail-loud but is weaker than one might hope.**
17 CFR 240.15c3-5(c)(1) requires controls that "**systematically** limit … financial
exposure," and (e) adds an annual CEO certification plus "a system for regularly reviewing
the effectiveness of the risk management controls"
(https://www.law.cornell.edu/cfr/text/17/240.15c3-5, accessed 2026-07-26). A control that
reports "no breach" because it has no baseline is not systematically limiting anything, so a
loud `armed: false` is the compliant behaviour in spirit. But note honestly: the rule
regulates *broker-dealers* (pyfinagent is neither), and it contains **no runtime invariant
forbidding a control from silently self-disabling** — the duty is a periodic-review duty.
FIA is closer to a practice standard and says a kill switch "should serve in addition to and
as a **final backstop** for the pre-trade risk functionality" and that participants "should
not be able to override" it
(FIA July 2024, §1.5, accessed 2026-07-26). "Cannot be overridden" is the closest published
analogue to "cannot silently disarm", and it supports the shipped direction.

**A.3 — CHALLENGE / NEW DEFECT: the fail-loud state is unreachable on the order-placing
path.** This is the one substantive finding that should change the disposition of 36.7.

`evaluate_breach` now returns `armed`, and three consumers honour it:
- `backend/api/paper_trading.py:593` — `if not breach.get("armed", True): raise HTTPException(409, ...)` (resume refused)
- `backend/services/kill_switch.py:601` — auto-resume refused, `reason="kill_switch_disarmed_baseline_missing"`
- `frontend/src/components/OpsStatusBar.tsx:318` and `frontend/src/components/KillSwitchPanel.tsx:137` — `breach.armed === false` → "DISARMED" badge

But the **trading-path** gate does not:
`backend/services/paper_trader.py:1069` `check_and_enforce_kill_switch` — documented "Call
this at the top of every autonomous cycle BEFORE deciding trades" — branches only on
`breach["any_breached"]` (`:1097`) and **never inspects `armed`**. Worse, it *mutates the
baselines before measuring*: `state.update_peak(nav)` at `:1080` and
`state.update_sod_nav(nav, date=today)` at `:1089-1090` run before `evaluate_breach` at
`:1092`. Since `update_peak` writes unconditionally when `_peak_nav is None`
(`kill_switch.py:379-380`), a post-rotation cycle with unrecoverable history does not surface
`armed: false` — it **silently re-anchors the peak to today's NAV and reports ARMED and
healthy.** The trailing-DD limit then measures from a depressed peak, forgiving the entire
real drawdown. That is the same class of harm 36.7 was written to prevent (the original bug
destroyed 541.80 of peak = 2.17pp of headroom, per the comment at `kill_switch.py:239-243`),
now reachable through the self-heal path rather than the assignment path — and it is
*quieter* than the bug that was fixed, because `armed` reads `true`.

This is precisely what FIA warns against by insisting the backstop not be overridable, and
what 15c3-5(c)(1)'s "systematically limit" language is for. It is also an instance of this
repo's own catalogued anti-pattern: absence becoming an affirmative SAFE
(`.claude/agent-memory/researcher/project_fabricated_safe_80_36.md`).

Note the tension with phase-69.1: `reset_peak` (`kill_switch.py:383-414`) is deliberately DARK
behind `settings.kill_switch_peak_reset_enabled` because re-anchoring a peak is a
guard-behaviour change requiring an operator token (`KS-PEAK-RESET: APPROVED`). Meanwhile
`update_peak`'s `None` branch performs an *unaudited, un-tokened, silent* re-anchor whenever
the baseline is missing. The two paths hold opposite policies on the same act. Recommend a
new masterplan step: `check_and_enforce_kill_switch` must evaluate `armed` **before** the
re-anchor and, when the switch would come up disarmed, either refuse to trade for that cycle
or emit a P1 — never both re-anchor and report healthy. (Per
`feedback_queue_discovered_defects_in_masterplan`, this belongs in its own research-gated
step, written for an executor with no memory of this discovery. I have NOT verified whether
one of the six already-queued follow-ups covers it; Main should check before filing.)

**A.3b — Precision, in fairness to the design: per-leg independence is a genuinely good
decision, and it bounds the §A.3 severity.** `evaluate_breach` (`kill_switch.py:496-506`)
evaluates each leg independently — `daily_baseline_missing` and `trailing_baseline_missing` are
computed separately (`:475-476`) and each leg is skipped only if *its own* baseline is missing.
The docstring at `:455` records the reasoning explicitly: a wholesale
`if not sod or not peak: return disarmed` was rejected because it would disable a *working*
leg (the `-v4`-archive-only case still fires correctly via the daily leg at a 50% drawdown).
That is the right call and matches FIA's "suite of controls" framing — degrade one control,
keep the others. It bounds §A.3: losing one baseline leaves the other enforcing; only losing
**both** leaves the book unprotected.

Confirmed mechanism for §A.3 (read at `kill_switch.py:496-515`): a missing baseline leaves its
`*_breached` flag `False`, so `any_breached` at `:515` is `False` — which is precisely why
`paper_trader.py:1097` neither flattens nor pauses. The `armed` flag is the *only* signal that
distinguishes "healthy" from "unmeasurable", and the trading path is the one consumer that
ignores it.

**A.4 — Fail-open on an older payload is defensible and correctly commented.**
`.get("armed", True)` at `api/paper_trading.py:593` fails open against a dict predating the
key, and `KillSwitchPanel.tsx:134-137` deliberately uses `=== false` rather than
`!breach.armed`. That matches the repo's presence-not-value rule and Azure's *Manual override*
bullet (an operator must retain a path to act). Fine as shipped.

### B. Rotation-safe reconstruction — the design is right, but for a reason the spawn prompt's framing omits

**B.1 — "Replay all available history" IS the canonical strategy.** "You can determine the
current state of an entity only by replaying all of the events that relate to it against the
original state of that entity," and "Snapshots are an optimization, not a replacement for the
eventstream. The eventstream remains the source of truth"
(https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing, accessed
2026-07-26). Kurrent concurs: a snapshot read must be followed by "the events that happened
after the snapshot was created"
(https://www.kurrent.io/blog/snapshots-in-event-sourcing, accessed 2026-07-26). So reading
the rotated archives rather than only the live file is not a workaround — **it is the
pattern**, and the pre-36.7 behaviour (live file only) was the deviation.

**B.2 — CHALLENGE the framing: "merge all history and take the extremum" is NOT by itself a
recognized safe pattern, and the known failure mode the project found is the textbook one.**
The spawn prompt describes 36.7 as "merge … and take the extremum (max/min) that matches the
safety direction," and asks whether the "stale archive overrides a fresh re-anchor" bug is a
known class. It is — but it is not usually named as a merge/extremum bug. It is an **event
ordering + compensating event** failure. Microsoft's canon is explicit on both halves:
"The consistency of events in the event store and the order of events that affect a specific
entity's current state are crucial. Adding a timestamp to every event can help you avoid
problems. Another common practice is to annotate each event that results from a request with
an incremental identifier." And: "The only way to update an entity or undo a change is to add
a compensating event to the event store" (same source). A pure order-free extremum fold
**cannot honour a compensating event** — that is a mathematical property, not a bug to be
patched: `max` is commutative, so a legitimate later downward re-anchor can never win against
any earlier higher value. If 36.7 were what the prompt describes, it would be unsound.

**It isn't.** The implementation does the canonical thing:
- `kill_switch.py:168-196` `_read_audit_rows` sorts every row from every source by
  `(ts, source_index, line_index)` — a timestamp-primary total order with deterministic
  tie-breaks, and rows lacking `ts` collate first so "a timestamp-less row can never override
  a genuinely later one" (`:174-176`).
- `kill_switch.py:245-246` `peak_update` ratchets with `max`, matching the writer's own
  invariant in `update_peak` (`:376-381`) — i.e. the extremum is used only where the event
  *type* is itself monotonic.
- `kill_switch.py:255` `peak_reset` **assigns**, so it is a genuine compensating event that
  can move the peak down, and later `peak_update` rows ratchet up from it *because the fold
  is ordered*.

So the ordered fold + type-specific reducers is the sound design, and the `max()` ratchet is
safe *only as a consequence of the ordering*, not in place of it. **This distinction matters
for the handoff**: any future refactor that "simplifies" `_read_audit_rows` by dropping the
sort, or that adds a new downward-moving event type without giving it assignment semantics,
silently reintroduces the permanent-lockout bug. Worth a comment or a test that mutates the
sort away.

**B.3 — Residual: sorting on a wall-clock ISO string is weaker than the canonical
recommendation.** Microsoft recommends a timestamp **and** "an incremental identifier."
36.7 has the timestamp plus *positional* tie-breaks (`source_index` from glob order,
`line_index` within a file) but no monotonic sequence number in the row itself. Two
consequences, both low severity today:
(a) lexicographic ordering of `ts` is only equivalent to chronological ordering while every
writer emits the same UTC offset. `_append_audit` uses
`datetime.now(timezone.utc).isoformat()` (`:262`) → always `+00:00`, so this holds now, but a
row written by any external tool with a `+02:00` offset would sort wrongly and could
resurrect a stale peak.
(b) `source_index` derives from `sorted(archive.glob(...))` (`:100-103`) plus the live file, so
cross-file ordering for same-`ts` rows depends on filename collation rather than on causality.
A per-row monotonic `seq` would remove both. Not a blocker; a cheap hardening.

**B.4 — The archive-classification root cause is a known anti-pattern.** The comment at
`kill_switch.py:51-56` records that the housekeeping backfill classified live kill-switch
state as an archivable artifact. Kurrent's "closing the books" guidance is the counter-rule:
archive only *after* the lifecycle closes and a summary event is written
(https://www.kurrent.io/blog/snapshots-in-event-sourcing). A monotonic NAV peak has **no
lifecycle end**, so this stream should never have been eligible for rotation. The
belt-and-braces fix (allowlist the file *and* read the archives) is stronger than either
alone, and matches "the eventstream remains the source of truth."

### C. NaN/Infinity hardening — strongly validated, with the exact bug class named in the literature

**C.1 — `math.isfinite()` guarding at ingest is the standard defensive practice, and the
"guard is vacuously false so the function returns a valid-looking answer" failure is a
documented, measured bug class.** Stainless (arXiv:2601.14059, §2 and §5.4, accessed
2026-07-26) states it three ways: "any comparison involving NaN evaluates to false, including
NaN == NaN"; "If the input threshold is NaN, the comparison evaluates to false, causing the
function to return the original vector, effectively ignoring the invalid input"; and, as a
finding in real code, "In two functions, this omission causes the implementation to return a
valid output even when the input is NaN, effectively silently ignoring NaN inputs." Their
remedy is to insert a check at *every* comparison ("Stainless automatically inserts NaN checks
for all comparisons and equality operations"). 36.7's `_coerce_nav`
(`kill_switch.py:111-136`) does the stronger and cheaper thing: reject at the **boundary**, so
no non-finite value ever reaches a comparison. That is the correct placement.

**C.2 — The project's specific `inf`-peak reasoning is corroborated at the hardware/standard
level.** The code comment at `kill_switch.py:130-136` argues an `inf` peak makes
`trailing_dd_pct` nan, every threshold comparison false, and the switch "silently, permanently
dead while `armed` still reports True." Fog confirms the mechanism: `min`/`max` are
"implemented as a branch, for example min(a, b) = a < b ? a : b", so an extremum over a
NaN-bearing series silently drops the NaN rather than propagating it, and IEEE 754 propagates
NaN through arithmetic only — never through a comparison
(https://www.agner.org/optimize/nan_propagation.pdf, §7.2–7.3, updated 2026-06-16). Combined
with `max()`'s inability to heal downward, the comment's conclusion is exactly right.

**C.3 — Documented incidents.** The strongest *verifiable* one is NumPy's: "NumPy previously
contained a bug in its max function, which returned an arbitrary number instead of the maximum
when NaN values were present in the input array" (arXiv:2601.14059 §2) — same primitive
(`max` over a possibly-NaN series), same silent-wrong-answer outcome as the kill-switch peak.
Honest negative result: **I found no public postmortem of a NaN silently disabling a
production trading risk control.** The widely-circulated "major European bank miscalculated
interest for three years" story appears only in an unsourced Medium post and I decline to cite
it as evidence. The Knight Capital SEC order — the canonical stale-state trading postmortem,
and the case this repo already invokes at `api/paper_trading.py:613` — **could not be
retrieved**; see Gaps.

**C.4 — 80.40's `None`-not-`0.0` rule is the reference-implementation semantics, improved.**
empyrical returns `np.nan` (not `0.0`) when a drawdown is unmeasurable
(`empyrical/stats.py:379-383`). Fog documents that R goes further and uses a *distinct NaN
payload* (1954) for "data not available" versus payload 0 for "invalid value"
(nan_propagation.pdf §7.4) — a mature statistical system that separates *unmeasurable* from
*invalid*, exactly the distinction 80.40 draws. 80.40 picks `None` instead of `nan` because
`nan` is not JSON-serializable (the phase-80.1/80.27 class in this repo). **That is the
reference semantics with the serialization bug removed — a strictly better choice for an API
boundary, and it should not be softened to `0.0` in any follow-up.**

### D. Max-drawdown sign convention — claim verified; characterization fair, with one refinement

**D.1 — The R counter-example is real, verbatim, and stronger than the project claimed.**
`maxDrawdown(R, weights = NULL, geometric = TRUE, invert = TRUE, ...)`; "The default option
`invert=TRUE` will provide the drawdown as a positive number"
(PerformanceAnalytics 2.0.4 reference, accessed 2026-07-26). The maintainers explicitly
decline to arbitrate: "we provide the option, but make no value judgment on which approach is
preferable." Their stated reasons for positive-by-default are *presentational and
optimization-ergonomic* ("having negative signs in front of every number may be considered
clutter"; "usually seeks to minimize a value"), and they attribute the negative convention to
practitioners wanting consistency "with the quantile (a negative number)."

**D.2 — REFINEMENT: within the Python quant stack the convention is unanimous, not split.**
- empyrical (engine behind zipline and pyfolio): negative fraction (`stats.py:352-402`)
- pyfolio: delegates to empyrical — same
- quantstats: "float: Maximum drawdown (**negative value**)" (`stats.py:2451-2463`), and
  `to_drawdown_series` → "negative values showing decline from peak" (`:2499-2511`)

So the accurate statement is: **the split is across the R↔Python boundary (and between
presentation and computation), not within the Python ecosystem, where negative is the norm.**
The project's word "genuinely split" is defensible — the convention really does differ between
two widely used libraries, and this repo *itself* holds both signs simultaneously
(`paper_go_live_gate.py::_snapshot_max_dd_pct` is positive magnitude) — but it slightly
overstates the ambiguity as a justification. **This does not weaken the decision; it
strengthens it.** Negative is both the Python-ecosystem norm *and* the sign the cockpit's
`maxDd > -10 ? SAFE : ...` ladder requires. The docstring's reason #2 (a positive magnitude
would satisfy `> -10` at every depth and render emerald SAFE forever, inverting a safety
verdict) is the load-bearing argument and it is sound on its own. Recommend a one-line
docstring softening to "the convention differs between R's presentational default and the
Python quant stack, where negative is standard (empyrical, pyfolio, quantstats)" — accuracy
only, no behaviour change.

**D.3 — Pinning the sign in a test rather than a docstring is right.** Since the same repo
publishes both signs from two helpers, the only durable guard is an assertion. The existing
pin (`backend/tests/test_phase_80_40_perf_metrics_drawdown.py`) is the correct mechanism, and
per `feedback_mutation_test_guards_and_fixtures` it should be mutation-checked: flip the sign
in the helper and confirm the test fails.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/kill_switch.py` | 660 | Baseline persistence + breach evaluation. `_audit_source_paths` `:86-108`, `_coerce_nav` `:111-136`, `_read_audit_rows` `:168-196`, `_load_from_audit` `:198-257`, `_append_audit` `:260`, `update_sod_nav` `:360-374`, `update_peak` `:376-381`, `reset_peak` `:383-414` (DARK), `evaluate_breach` `:427-520`, `_log_disarmed_once` `:522-547`, `check_auto_resume` `:549-604` | 36.7 changes present and sound (ordered fold + `max` ratchet + assignment on `peak_reset` + `isfinite` boundary reject + `armed`) |
| `backend/services/perf_metrics.py` | 757 | `compute_max_drawdown_from_snapshots` `:127+`, ~75-line docstring pinning sign / `None`-not-`0.0` / ASC sort / window / flow-blindness | 80.40 present; delegates arithmetic to `analytics.compute_max_drawdown` per the single-source rule |
| `backend/api/paper_trading.py` | — | `:575-603` resume gate: 409 on `any_breached`, **and** 409 on `armed is False` (`:593`). `:613` comment "kill-switch loss limits are deliberately NOT here (Knight Capital safety)" | Honours `armed` — correct |
| `backend/services/paper_trader.py` | — | `check_and_enforce_kill_switch` `:1069-1116`. `update_peak` `:1080`, SOD roll `:1089-1090`, `evaluate_breach` `:1092`, breach branch `:1097` | **DEFECT (§A.3): never reads `armed`, and re-anchors baselines before evaluating** |
| `frontend/src/components/KillSwitchPanel.tsx` | — | `:134-137` presence-not-value `armed === false`; `:214` mirrors server 409 | Correct |
| `frontend/src/components/OpsStatusBar.tsx` | — | `:318-319`, `:355` DISARMED badge; `:367-368` disables resume | Correct |
| `backend/config/settings.py` | — | `:39` `kill_switch_peak_reset_enabled` (KS-PEAK-RESET token, DARK); `:360` `kill_switch_auto_resume_enabled` | Policy asymmetry vs `update_peak`'s silent re-anchor — see §A.3 |
| `backend/governance/divergence.py` | — | `:30-32` `daily_loss_kill_switch`, `trailing_dd_kill_switch` divergence rows | Unchanged; consumes limits not baselines |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | 841+ | 36.7 regression suite; `:841` references the paper_trader call site | Present; §A.3 suggests it does not assert the trading-path `armed` behaviour |
| `backend/tests/test_phase_80_40_perf_metrics_drawdown.py` | — | Sign pin for 80.40 | Present (per docstring `:157-159`) |

---

## Gaps / honest failures

1. **SEC Knight Capital order (34-70694) could not be fetched.** Both
   `https://www.sec.gov/litigation/admin/2013/34-70694.pdf` and
   `.../files/litigation/admin/2013/34-70694.pdf` return a 1925-byte HTML interstitial to
   `curl`, with and without a browser User-Agent. This is the single most relevant primary
   postmortem for §A/§C (stale state in a trading risk control) and this repo already cites it
   by name at `api/paper_trading.py:613`. Its absence does not change any conclusion — the
   conclusions rest on FIA, the CFR, and Microsoft's canon — but it is a real hole. Suggested
   route for a follow-up: fetch via the SEC ALJ/administrative-proceedings index page rather
   than the direct PDF, or via a law-school mirror.
2. **No public NaN-disabled-a-risk-control postmortem found** (§C.3). Recorded as a negative
   finding rather than padded with the unsourced Medium anecdote.
3. **I did not verify whether the §A.3 defect is already covered by one of the six queued
   follow-up steps.** Main should check before filing a duplicate.

---

## Ranked assessment — does the shipped design hold up?

| Rank | Decision | Verdict | Basis |
|---|---|---|---|
| 1 | **80.40 `None`, never `0.0`, on every degraded path** | **HOLDS — strongest of the four.** Do not soften. | empyrical returns `np.nan` not `0.0` for an unmeasurable series (`stats.py:379-383`); R distinguishes "not available" from "invalid" via NaN payload (Fog §7.4). 80.40 = same semantics, JSON-safe sentinel. |
| 2 | **36.7 non-finite rejection at the boundary** | **HOLDS.** Boundary rejection is stronger than per-comparison checks. | arXiv:2601.14059 §2/§5.4 names the exact bug class and measures it in real code; Fog §7.2–7.3 confirms comparisons and `min`/`max` branches never propagate NaN. |
| 3 | **80.40 negative-percent sign** | **HOLDS.** Claim verified; characterization fair but slightly overstates the ambiguity. | R `invert=TRUE` verified verbatim; empyrical + pyfolio + quantstats all negative. Cockpit-inversion argument is decisive regardless. Optional one-line docstring accuracy fix (§D.2). |
| 4 | **36.7 archive-merge replay + ordered fold** | **HOLDS — but only because of the timestamp sort, which the design description omits.** | "Replay all events" + "order is crucial" + "compensating events" (Microsoft Event Sourcing, 2026-03-27). An order-free extremum merge *would* be unsound; `_read_audit_rows:194` makes it sound. Protect the sort with a mutation-tested guard. |
| 5 | **36.7 `armed: false` fail-loud** | **HOLDS in principle; INCOMPLETE in practice.** | Correct on the resume, auto-resume and UI paths. **Not wired into the order-placing path** — see §A.3. |
| — | **NEW: trading path never reads `armed`, and re-anchors before evaluating** | **DEFECT — recommend a new masterplan step before 36.7 is called fully closed.** | `paper_trader.py:1080/1089-1090` mutate baselines before `evaluate_breach` at `:1092`; `:1097` branches only on `any_breached`. Net effect: silent, un-tokened, unaudited peak re-anchor that reports ARMED and healthy — while the equivalent deliberate act (`reset_peak`) is gated behind `KS-PEAK-RESET: APPROVED`. |
| — | **Residual: no per-row monotonic sequence number** | Low-severity hardening. | Microsoft recommends timestamp **and** "an incremental identifier"; current tie-breaks are positional and lexicographic-`ts`-dependent (§B.3). |

**Disposition recommendation.** 80.40 is fine as shipped (optional docstring accuracy tweak).
36.7's four mechanisms are each individually correct, but the step's stated goal — "a missing
baseline now surfaces an explicit `armed: false` state instead of silently reporting
`any_breached: False`" — is **not achieved on the path that places orders**, which is the path
the goal exists to protect. That is a scope-honesty issue as much as a code issue, so I would
not record 36.7 as fully closed without either (a) the §A.3 fix, or (b) an explicit,
operator-visible disclosure that `armed` is advisory on the trading path and the trading path
self-heals by re-anchoring. Everything else is fine as shipped with the six queued follow-ups.

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 12,
  "snippet_only_sources": 22,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "Validates 3 of 4 shipped decisions against reference implementations and canonical pattern docs, and refines the 4th. 80.40's None-not-0.0 rule is empyrical's own degraded-path semantics (np.nan, never 0.0) with the JSON-serialization bug removed -- do not soften it. The negative-drawdown sign is verified: R's invert=TRUE default is real verbatim, but empyrical, pyfolio and quantstats are unanimously negative, so the split is across the R/Python boundary rather than 50/50. 36.7's isfinite boundary reject prevents a bug class named and measured in arXiv:2601.14059. 36.7's archive merge is the canonical event-sourcing replay, but is sound only because _read_audit_rows sorts by ts and peak_reset assigns -- an order-free max() merge, as the design was described, would be unsound; protect the sort with a mutation-tested guard. NEW DEFECT: paper_trader.check_and_enforce_kill_switch never reads `armed` and re-anchors baselines at :1080/:1089 BEFORE evaluating at :1092, so a lost baseline silently re-anchors the peak to today's NAV and reports ARMED+healthy on the order-placing path -- the same harm 36.7 exists to prevent, and un-tokened where reset_peak requires KS-PEAK-RESET:APPROVED. Recommend a new step before 36.7 is called closed. Knight Capital SEC order fetch FAILED (SEC interstitial); no public NaN-disabled-risk-control postmortem exists.",
  "brief_path": "handoff/current/research_brief_36.7_80.40.md",
  "gate_passed": true
}
```
