# Research Brief — phase-36.13: P0 kill-switch alternate-path bypass in `execute_buy`

**Tier:** moderate (caller-specified). NOT audit-class.
**Date:** 2026-07-26
**Question:** `paper_trader.execute_buy` has no kill-switch gate. Choose between
(a) gate inside `execute_buy` with an audited named bypass, (b) gate at every
caller, (c) guarded wrapper + private unguarded primitive.

---

## Queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| Q1 | `Saltzer Schroeder complete mediation reference monitor every access checked` | **year-less canonical** |
| Q2 | `Martin Fowler FlagArgument boolean parameter anti-pattern refactoring` | **year-less canonical** |
| Q3 | `IEC 61511 bypass override discipline safety instrumented system authorization time limit alarm indication` | **year-less canonical** |
| Q4 | `non-bypassable invariants agent system arXiv 2603.10092` | **current-year frontier (2026)** |
| Q5 | `"test mode" debug flag shipped enabled production outage postmortem 2025 safety interlock bypass` | **last-2-year window (2025)** |

Direct URL fetches (no search step needed — canonical registry pages):
`cwe.mitre.org/data/definitions/{424,638,288}.html`.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://www.cs.virginia.edu/~evans/cs551/saltzer/ | 2026-07-26 | paper (peer-reviewed, Proc. IEEE 1975) | WebFetch | *"Every access to every object must be checked for authority. This principle, when systematically applied, is the primary underpinning of the protection system."* Requires *"a system-wide view of access control, which in addition to normal operation includes initialization, recovery, shutdown, and **maintenance**."* Fail-safe defaults: a restrictive-mechanism mistake *"tend[s] to fail by allowing access, a failure which may go unnoticed."* |
| 2 | https://cwe.mitre.org/data/definitions/424.html | 2026-07-26 | official (MITRE registry) | WebFetch | Verbatim: *"The product does not sufficiently protect all possible paths that a user can take to access restricted functionality or resources."* **Parents: CWE-638 (Not Using Complete Mediation) and CWE-693 (Protection Mechanism Failure).** Mitigation (Architecture & Design): *"Deploy different layers of protection to implement security in depth."* |
| 3 | https://cwe.mitre.org/data/definitions/638.html | 2026-07-26 | official (MITRE registry) | WebFetch | Mitigation, verbatim: *"Identify all possible code paths that might access sensitive resources. If possible, create and use a **single interface that performs the access checks**, and develop **code standards that require use of this interface**."* Demonstrative example 1 = library files reachable by direct request, bypassing the controller's checks. |
| 4 | https://cwe.mitre.org/data/definitions/288.html | 2026-07-26 | official (MITRE registry) | WebFetch | *"The product requires authentication, but the product has an alternate path or channel that does not require authentication."* Mitigation, verbatim: *"**Funnel all access through a single choke point** to simplify how users can access a resource. For every access, perform a check..."* CVE-2003-1035: an **API path brute-forced past a GUI-only lockout** — the same control missing on the programmatic route. |
| 5 | https://martinfowler.com/bliki/FlagArgument.html | 2026-07-26 | authoritative blog (Fowler) | WebFetch | *"My general reaction to flag arguments is to avoid them."* Prefers `regularBook(martin)` / `premiumBook(martin)` over `book(martin, false)`. **Caveat that matters here:** when logic is tangled, *"one option is to retain the method with the flag argument, but keep it **hidden**"* — public named methods delegating to a **private** impl. Also: if the branch is derivable from object state, *the flag should not be a parameter at all*. |
| 6 | https://instrumentationtools.com/iec-61511-standard-requirements-for-safety-bypass-and-override/ | 2026-07-26 | industry (standard summary, IEC 61511-1:2016 clauses) | WebFetch | Bypass discipline = 6 mandatory elements: written procedure (10.3.2, 16.2.4), **authorization** (16.2.6/16.2.7) + key-lock/password (11.7.2.2-3), **indication** that the bypass is active + alarms not disabled (16.2.7, 11.7.3.2), **maximum time in bypass** (16.2.3), **compensating measures** (11.8.5), and **a bypass log** (16.2.7). Bypass facilities must be *deliberately designed into* the SIS, "not improvised ad-hoc workarounds". |
| 7 | https://arxiv.org/html/2603.10092 | 2026-07-26 | preprint (arXiv 2026) | WebFetch (`/html/`, per rules) | Survivability-Aware Execution: *"SAE sits in the last-mile boundary... This placement ensures constraints are **non-bypassable**: even if upstream intent is manipulated, the executor only receives SAE-approved actions."* Treats upstream outputs as *"untrusted intent"*. Empirics vs NoSAE: max drawdown 0.4643 -> 0.0319 (-93.1%), CVaR 4.025e-3 -> ~1.02e-4. Notably **describes no dry-run/bypass mode** — "the implicit assumption is that enforcement remains active in all modes"; auditability comes from a per-decision audit record, not from a bypass. |
| 8 | https://arxiv.org/html/2605.18991 | 2026-07-26 | preprint (arXiv 2026) — **[COUNTER-ARGUMENT]** | WebFetch (`/html/`) | *"Agent security must be approached as a systems problem... security invariants must be enforced at the system level."* Advocates a Reference Monitor + Complete Mediation, but **explicitly argues for layered, heterogeneous enforcement** over one unified checkpoint: *"By deploying mechanisms that operate at different layers of abstraction... [attackers face] multiple, mechanistically distinct barriers."* AND warns that **testing modes that disable security become the attack surface** (Cursor AgentFlayer; "Auto Run / YOLO mode"). Cites the exact defect class: Claude Code *"mistakenly allowed `ping` to execute without human approval"* while gating other shell commands; Devin's `expose_port` lacked restrictions other tools had. |
| 9 | https://featureflip.io/blog/feature-flag-anti-patterns/ | 2026-07-26 | industry blog (tier 4) | WebFetch | Anti-pattern #1: *"Reusing a deprecated flag bit cost Knight Capital $460 million in 45 minutes"* — dead-but-reachable code path triggered by a stale flag on one un-updated server. Anti-pattern #6: kill switches must be **permanent infrastructure with quarterly testing**, not a release flag. Anti-pattern #8: no observability on flag evaluations makes flag incidents undebuggable. |

Tier honesty: 8 of 9 are tier 1-3 (peer-reviewed / official registry / authoritative
blog / arXiv preprint). Source 9 is tier 4 and is used only for the Knight Capital
and kill-switch-testing anecdotes, not for any load-bearing design claim.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.cs.cornell.edu/fbs/publications/chptr.enfMech.refMonitor.pdf | paper (Schneider, Reference Monitors chapter) | Binary PDF; Saltzer primary already read; the "always invoked / tamperproof / small enough to analyze" triad was captured via the search snippet |
| https://shostack.org/blog/the-security-principles-of-saltzer-and-schroeder | blog | Secondary commentary on source 1 |
| https://handwiki.org/wiki/Saltzer_and_Schroeder's_design_principles | encyclopedia | Tertiary |
| https://nocomplexity.com/documents/securityarchitecture/architecture/saltzer_designprinciples.html | reference architecture | Tertiary restatement |
| https://www.opensecurityarchitecture.org/foundations/design-principles/ | industry | Tertiary restatement |
| https://blog.rweisleder.de/posts/flag-parameter-anti-pattern/ | blog | Restates Fowler (source 5) |
| https://luzkan.github.io/smells/flag-argument/ | code-smell catalogue | Restates Fowler |
| https://ardalis.com/are-boolean-flags-on-methods-a-code-smell/ | blog | Restates Fowler |
| https://automationforum.co/iec-61511-safety-bypass-override-sis-maintenance/ | industry | Overlaps source 6 |
| https://instrumentationtools.com/safety-bypass-management-system/ | industry | Overlaps source 6 |
| https://www.controleng.com/when-should-you-bypass-your-safety-system/ | industry | Overlaps source 6 |
| https://arxiv.org/abs/2603.10092 | abstract page | HTML full text read instead (source 7) |
| https://arxiv.org/pdf/2601.17744 (Faramesh: protocol-agnostic execution control plane) | preprint 2026 | Recency-scan corroboration; same thesis as source 7 |
| https://arxiv.org/pdf/2604.17517 (From Admission to Invariants: Measuring Deviation in Delegated Agent Systems) | preprint 2026 | Recency-scan corroboration |
| https://arxiv.org/html/2605.29251v1 (Provably Secure Agent Guardrail) | preprint 2026 | Recency-scan corroboration |
| https://arxiv.org/pdf/2606.04903 (Provably Auditable and Safe LLM Agents from Human-Authored Ontologies) | preprint 2026 | Recency-scan corroboration |
| https://arxiv.org/pdf/2605.15228 (Verifiable Agentic Infrastructure: Proof-Derived Authorization) | preprint 2026 | Recency-scan corroboration |
| https://posthog.com/handbook/company/post-mortems/2025-09-29-flags-is-down | postmortem 2025 | Flag-service availability, not bypass discipline |
| https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming) | encyclopedia | Tangential |

**URLs collected: 28** (9 read in full + 19 snippet-only).

## Recency scan (2024-2026)

**Result: 6 new 2026 findings that COMPLEMENT — and one that materially
QUALIFIES — the canonical 1975/2006-era sources.**

The 1975 complete-mediation principle and the CWE entries are the canonical prior
art and have not been superseded. What the 2024-2026 window adds is an entire
research thread on **where** to place enforcement in *agentic* systems that place
real orders — which is precisely this defect's setting:

1. **arXiv:2603.10092 (2026)** — Survivability-Aware Execution puts hard invariants
   at the **last mile where side effects occur**, explicitly so they are
   non-bypassable when upstream intent is compromised. This is a direct,
   same-domain (agentic crypto trading) endorsement of gating *inside* the
   execution primitive rather than at the strategy/caller layer. It also reports
   the empirical cost of not doing so (93% worse max drawdown vs NoSAE).
2. **arXiv:2605.18991 (2026)** — *qualifies* the above: it argues for **layered,
   heterogeneous** enforcement rather than a single unified checkpoint, and it
   documents that **test/bypass modes become the attack surface**. It names three
   real 2025-2026 incidents of the exact CWE-424 shape (Claude Code's un-gated
   `ping`; Devin's un-gated `expose_port`; Terminal DiLLMa's context-dependent
   sanitization).
3. **arXiv:2601.17744 / 2604.17517 / 2605.29251 / 2605.15228 / 2606.04903 (2026)** —
   a converging cluster (execution control planes, admission-to-invariant
   deviation measurement, provable guardrails, proof-derived authorization) all
   placing the mediation point at the action boundary. Nothing in this cluster
   advocates enforcement at the *caller*.
4. **Feature-flag practice (2025-2026)** — kill switches are now framed as
   *permanent, periodically-exercised infrastructure*, and the canonical
   catastrophe (Knight Capital) is a **dead-but-still-reachable code path**
   re-activated by a stale flag. Direct evidence against leaving an unguarded
   primitive reachable in the same module.

No source in the window argues that a safety control should be replicated at each
call site, and none argues that a public unchecked primitive is safer than a
checked one.

## Key findings

1. **Complete mediation is the named parent of this exact CWE.** CWE-424's parent
   is CWE-638 "Not Using Complete Mediation" — so the defect and the remedy come
   from the same taxonomy node. (Source: CWE-424,
   https://cwe.mitre.org/data/definitions/424.html, accessed 2026-07-26)
2. **The registry prescribes a single checked interface, not per-caller checks.**
   *"Identify all possible code paths... create and use a single interface that
   performs the access checks, and develop code standards that require use of this
   interface."* (CWE-638) and *"Funnel all access through a single choke point"*
   (CWE-288). Both explicitly reject option (b).
3. **Saltzer & Schroeder extend mediation to maintenance, not just normal
   operation** — *"a system-wide view of access control, which in addition to
   normal operation includes initialization, recovery, shutdown, and
   maintenance."* A drill *is* maintenance; the principle says the drill path must
   still be mediated, not exempted. (Source 1)
4. **Fail-safe defaults cut against an unchecked primitive.** *"A design mistake in
   [permissive] mechanisms tends to fail by refusing permission, a safe
   situation,"* whereas the restrictive kind *"tend[s] to fail by allowing access,
   a failure which may go unnoticed."* A primitive that trades unless someone
   remembered to wrap it fails **open**. (Source 1)
5. **Fowler's actual position is subtler than "no flags".** He would split
   `execute_buy(force=True)` into two named methods — but his stated fallback for
   tangled logic is *"retain the method with the flag argument, but keep it
   hidden"*: named public methods over a **private** impl. Applied here: the
   unchecked primitive must be **private**, and the flag (if any) must not be part
   of the public call signature. (Source 5)
6. **Safety engineering does not answer "provide an unguarded primitive" — it
   answers "design an authorized, indicated, time-limited, logged bypass into the
   protected function".** IEC 61511 requires all six of: authorization, indication,
   compensating measures, maximum bypass duration, written procedure, and a bypass
   log — and requires the facility to be designed in rather than improvised.
   (Source 6)
7. **2026 agentic-trading research puts the invariant at the last mile.** *"SAE
   sits in the last-mile boundary... This placement ensures constraints are
   non-bypassable."* (Source 7)
8. **[COUNTER] The strongest 2026 systems paper warns that the test escape hatch
   IS the vulnerability, and prefers layered barriers to one checkpoint.** Cursor
   "YOLO mode"; Claude Code's un-gated `ping` next to gated shell commands.
   (Source 8)
9. **[COUNTER, empirical] Knight Capital = a dead-but-reachable path re-entered by
   accident, $460M in 45 minutes.** The single most expensive instance of "we left
   the old unguarded route in the binary". (Source 9)

## Internal code inventory

### Verbatim grep output (re-derived 2026-07-26, cwd = repo root)

Note: the naive form `grep -rn ... --include=*.py` FAILS under zsh
(`no matches found: --include=*.py`) — the glob must be quoted.

```
$ grep -rn '\.execute_buy(' --include='*.py' backend scripts | grep -v /tests/
backend/agents/mcp_servers/signals_server.py:444:                trade = self.paper_trader.execute_buy(
backend/services/autonomous_loop.py:236:    buy_trade = trader.execute_buy(
scripts/smoketest_stages_5_through_13.py:188:        trade = trader.execute_buy(
scripts/go_live_drills/zero_orders_drill.py:94:    result = trader.execute_buy(
```
**CONFIRMED: exactly FOUR non-test call sites**, as the step claims.

```
$ grep -rn '\.execute_buy(' --include='*.py' backend scripts   # NO test filter
backend/tests/test_phase_61_2_decision_integrity.py:430
backend/tests/test_64_4_multi_market_e2e.py:151
backend/tests/test_price_tolerance_gate.py:83,105,127,158,189
backend/tests/test_phase_70_3_atomic_swap.py:220,228
backend/tests/test_dod4_tier1_coverage_investment.py:614
backend/tests/test_phase_70_4_gate_observability.py:77,93
backend/tests/test_phase_50_2_multicurrency.py:132,190
backend/tests/test_64_3_currency_path.py:70,124
  + the 4 non-test sites above
```
**Test-side surface: 15 call sites across 8 test files** (total 19).
This matters for option (c): a rename of the primitive touches 19 sites,
not 4. (`test_phase_70_3_atomic_swap.py:36,40` also define a *fake*
`execute_buy`/`execute_sell` on a stub trader — a rename must keep the
stub's public name aligned with whatever `autonomous_loop` calls.)

```
$ grep -rn 'is_paused()' --include='*.py' backend scripts | grep -v /tests/
backend/services/paper_trader.py:1197:        if breach["any_breached"] and not state.is_paused():
backend/services/autonomous_loop.py:1316:            halt_reason = cycle_halt_reason(ks_check, _ks_state().is_paused())
```
**CONFIRMED: exactly TWO non-test consumers**, as the step claims.
(`grep -rn 'is_paused'` without the `()` also finds
`backend/agents/mcp_servers/risk_server.py:91,181` — a *dict key* named
`"is_paused"` derived from `snap.get("paused")`, i.e. a THIRD reader of
the same state through a different spelling. See "Third reader" below.)

**Line-number drift vs the step text:** the step cites
`paper_trader.py:1097` and `autonomous_loop.py:1287`; the live lines are
`paper_trader.py:1094` (`def check_and_enforce_kill_switch`) / `:1197`
(the `is_paused()` read) and `autonomous_loop.py:1316`. Use the live
numbers in the contract.

### `execute_buy` read in full (paper_trader.py:148-414)

Signature at `:148-178` (17 kwargs, all keyword-friendly). Returns
`Optional[dict]`: the trade record on success, **`None` on every
refusal**. Existing refusal points, in order:

| Line | Guard | On refusal |
|------|-------|-----------|
| :189-196 | no `stop_loss_price` -> **synthesize** 8% stop (does NOT refuse) | `logger.warning`, continues |
| :209-227 | phase-30.6 price-tolerance gate | `logger.warning` + **`self.buy_rejections.append({...})`** + `return None` |
| :236-241 | insufficient cash (incl. `reserved_cash`) | `logger.warning` + `return None` |
| :246-248 | `paper_max_positions` reached | `logger.warning` + `return None` |
| :256-258 | FX unavailable for `market` | `logger.warning` + `return None` |
| :268-288 | phase-23.1.15 idempotency (30-min duplicate BUY) | `logger.warning` + `return None` |

**Observability hook to REUSE for a kill-switch refusal:
`self.buy_rejections.append({...})` (`:222-226`, phase-70.4 G2-A).** It is
the ONLY structured (non-log-string) refusal channel in `execute_buy`, and
only the price-tolerance gate currently populates it — the other four
refusals are log-only. A kill-switch refusal should append
`{"ticker":..., "reason": "kill_switch_paused"|"kill_switch_disarmed", ...}`
so the phase-70.4 gate-observability consumer sees it.

**Critical ordering note:** the phase-30.6 comment at `:202-205` already
states the project's own precedent — *"Placed BEFORE the ExecutionRouter
call so the non-bypassable-invariants pattern ... holds (gate cannot be
circumvented by routing)."* That is complete-mediation reasoning applied
one layer down (router-level bypass), and it is the in-repo precedent for
putting the kill-switch check **inside** `execute_buy` rather than at the
callers. A kill-switch gate belongs at the TOP of the body (before
`:189`'s stop synthesis and before any BQ read), because everything below
`:229` touches portfolio state.

### The two "deliberate" callers — DECISIVE FINDING

Both callers the step worries about are **already fully stubbed**; neither
touches the live book:

- `scripts/go_live_drills/zero_orders_drill.py:64,92` — builds
  `bq = StubBQ(starting_cash=10000.0)` (a local in-file class, `:58-59`
  `save_paper_trade` just appends to `self.saved_trades`) and injects it:
  `PaperTrader(settings=settings, bq_client=bq)`. It asserts
  `result is None` -> `"FAIL: execute_buy returned None (refused to
  execute)"` at `:106-108`, then asserts `bq.saved_trades` is non-empty.
- `scripts/smoketest_stages_5_through_13.py:174,181,183` — `bq =
  MagicMock()`, `PaperTrader(settings=settings, bq_client=bq)`, and it
  even patches `backend.services.paper_trader.ExecutionRouter`. Uses a
  `SimpleNamespace` settings object (`:166-173`) with only 6 attributes.

So the "escape hatch" question is NOT "let a drill trade through a live
interlock". It is narrower: **would a gate inside `execute_buy` consult
PROCESS-GLOBAL kill-switch state that the drill cannot inject?** If the
gate reads state through the same injected seam the rest of the drill
already controls, no bypass parameter is needed at all — the drills keep
passing with zero API change. See the kill-switch state-source analysis
below.

Second-order note on the smoketest: its settings object is a
`SimpleNamespace` with 6 attributes, so any new gate MUST read its
config via `getattr(self.settings, ..., default)` (the house idiom at
`:190`, `:207`, `:339`, `:356`) or the smoketest dies with
`AttributeError` — a hard constraint on the implementation regardless of
which option is chosen.

### Is the drill breakage REAL? YES — pause is restart-durable (measured)

`kill_switch.py:675` is a **module-level singleton**: `_state = KillSwitchState()`,
returned by `get_state()` (`:713-714`). Its `__init__` (`:167-189`) ends with
`self._load_from_audit()`, and the replay at `:254-268` handles the `pause` event
explicitly:

```python
if event == "pause":
    self._paused = True
    self._pause_reason = row.get("trigger")
    self._paused_at = row.get("ts")
```

So **a pause recorded in `handoff/kill_switch_audit.jsonl` (plus rotated archives,
merged by `_read_audit_rows` `:191`) is restored in EVERY new Python process that
imports `kill_switch`** — including a drill process. A naive
`if get_state().is_paused(): return None` inside `execute_buy` therefore
**genuinely breaks both drills whenever the live book is paused**, even though the
drills use `StubBQ`/`MagicMock` and touch no real data. The concern in the step
prompt is confirmed, not hypothetical.

Corollary: the failure would be **intermittent and state-dependent** — the drills
pass on an unpaused day and fail after a breach. That is the worst possible
failure shape for a verification gate (it goes red exactly when the operator is
already firefighting).

### `zero_orders_drill` is a masterplan VERIFICATION criterion

`handoff/archive/phase-16.19/contract.md:33` runs
`alpaca_shadow_drill.py && zero_orders_drill.py && kill_switch_test.py` as the
step's verification command, with `zero_orders_drill_pass` as a named criterion
(`:38`). Breaking it is not just breaking a script — it invalidates a
harness verification path. `scripts/go_live_drills/` holds **34 drills**; only
`zero_orders_drill.py` and `smoketest_stages_5_through_13.py` call `execute_buy`.

### Third reader of the same state, under a different spelling

`backend/agents/mcp_servers/risk_server.py:91` builds
`"is_paused": bool(snap.get("paused", False))` from a kill-switch snapshot and
gates on it at `:181`. So the live surface is: **2 method-call readers + 1
snapshot-dict reader**. Any contract wording that says "two consumers" should say
"two `is_paused()` callers, plus one snapshot-derived reader in `risk_server`".

### The alternate path has a DIFFERENT, WEAKER control (not none)

`signals_server.py` contains **zero** references to `kill_switch`, `is_paused`, or
`paused`. What it has instead is its own re-implementation:
- `:88-89` an **in-memory** trailing peak-equity high-water mark ("In-memory"
  verbatim in the comment) — reset on every process restart, unlike the
  kill-switch peak which is audit-replayed;
- `:1213-1276` `track_drawdown()` computing `drawdown_pct` from that in-memory peak;
- `:826, :838, :950-953` a `drawdown_circuit_breaker` conflict raised when
  `current_dd <= max_drawdown_pct` (default -15.0), consumed by `risk_check` at
  `:429-436`, which returns `risk_rejected:<conflict>` **before** reaching
  `execute_buy` at `:444`.

This is why the gap was not obvious: the alternate path *looks* guarded. It is
guarded by a **duplicate policy that cannot see a pause, cannot see a disarm, and
loses its peak on restart** — CWE-638's "policy replicated per path then drifts"
failure mode. Worse for verification:
`scripts/go_live_drills/kill_switch_test.py:72-129` exercises exactly this weak
duplicate (`scenario_1_deep_drawdown_blocks_buy(server)` etc. against
`signals_server`), **not** `kill_switch.py`. The drill named "kill_switch_test"
does not test the kill switch. Worth its own queued masterplan step.

### `execute_sell`: the asymmetry is deliberate and should stay

`execute_sell` (`:416-423`) has the same absence of a kill-switch gate, and that
is correct: `check_and_enforce_kill_switch` responds to a breach by calling
`self.flatten_all(reason="kill_switch_auto_flatten")` (`:1199`), which reaches
`self.execute_sell(...)` at `:1029` (also `:736`, `:764` for stop-loss/trailing
exits). **Gating `execute_sell` on `is_paused()` would deadlock the flatten**: the
switch pauses, then refuses the very sells that implement the pause. Sells are the
safe direction. Recommendation: gate BUY only, and put that reasoning in a comment
at the `execute_sell` def so a future reader does not "fix" the asymmetry. (This
mirrors the existing house precedent at `:441-455`, where an FX outage BLOCKS a
sell only because booking at 1.0 would *poison the kill-switch peak* — i.e. sells
are already treated as the privileged direction.)

### Existing refusal / audit vocabulary to MATCH

| Symbol | Location | Shape |
|--------|----------|-------|
| `buy_rejections` | `paper_trader.py:107` (init), `:222-226` (only producer), consumed `autonomous_loop.py:1574-1582` (`buy_rejections_by_reason` Counter into the cycle summary) | `{"ticker":..., "reason": "<snake_case>", ...}` — **the hook to reuse** |
| `block_reason` | `paper_trader.py:1263` (`"kill_switch_disarmed_lost_history"`), read `autonomous_loop.py:162` via `ks_check.get("block_reason")` | string on the `check_and_enforce_kill_switch` result dict |
| `cycle_halt_reason(ks_check, is_paused)` | `autonomous_loop.py:139-163`, called `:1316` | returns `Optional[str]`; `:163` `if is_paused: ...` |
| `record_lost_history_anchor(...)` | `kill_switch.py:607`, called `paper_trader.py:1231` | latching provenance marker + audit row |
| `KillSwitchState._append_audit(event, **fields)` | `kill_switch.py:412-420` | one JSON line per event into the audit stream — the natural place for a `buy_blocked` / `bypass_used` row |
| `raise_cron_alert_sync(source, error_type, severity, title, details)` | `observability/alerting.py:253` (async form `:179`), P1 precedent at `paper_trader.py:1244-1259` with `source="kill_switch"`, `error_type="disarmed_lost_history_block"` | `P0/P1` are in `_CRITICAL_SEVERITIES` (`:54`) and bypass the 3-in-5-min deduper (`:83`) |

A kill-switch refusal in `execute_buy` should: append to `buy_rejections`, write a
`_append_audit("buy_blocked", ...)` row, and (for the DISARMED case, matching
`:1244`) raise a P1. It should **not** invent new vocabulary.

### Other order-placing primitives — is there a fourth path?

```
$ grep -rn 'save_paper_trade\|_safe_save_trade' --include='*.py' backend scripts | grep -v /tests/
backend/db/bigquery_client.py:693:    def save_paper_trade(self, row: dict) -> None:
backend/services/paper_trader.py:324:        self._safe_save_trade(trade)      # <- execute_buy
backend/services/paper_trader.py:510:        self._safe_save_trade(trade)      # <- execute_sell
backend/services/paper_trader.py:1359:    def _safe_save_trade(self, row: dict) -> None:
scripts/go_live_drills/zero_orders_drill.py:58: (StubBQ)
```
**Only two producers of a `paper_trades` row exist: `execute_buy:324` and
`execute_sell:510`.** `portfolio_manager.py` is pure decision logic — it emits
`TradeOrder` dataclasses and never writes a trade (its only `execute_buy`
references are comments at `:33,:42,:45,:50,:53,:451,:457`). So the BUY surface to
mediate is exactly one function. That is the good news: a last-mile gate in
`execute_buy` achieves **complete** mediation of the BUY side, not partial.

`ExecutionRouter.submit_order` is reachable independently
(`scripts/harness/paper_execution_parity.py:69,77`,
`scripts/go_live_drills/alpaca_shadow_drill.py:48,58`,
`scripts/harness/mcp_ab_test.py:162`) — those DO hit Alpaca paper but write no
`paper_trades` row and are out of scope for 36.13. Note them as a residual path.

### The in-repo precedent for the recommended seam

`execution_router.py:268-269`:
```python
def __init__(self, mode: BackendMode | None = None) -> None:
    self.mode: BackendMode = mode or _current_mode()
```
Injectable dependency, defaulting to the process-wide source. Its module docstring
(`:4`) even cites *Fowler "ops toggle"*. `PaperTrader.__init__`
(`paper_trader.py:94-107`) already follows the same shape for `bq_client` and
`trade_notifier`. This is the house idiom, and it is exactly what the drills need.

## Consensus vs debate

**Consensus (5 of 5 tier-1/2 sources):** the check belongs at ONE mediated
interface at the point of effect, and every path must reach it.
- Saltzer & Schroeder: *every* access, including maintenance paths.
- CWE-638: *"a single interface that performs the access checks."*
- CWE-288: *"Funnel all access through a single choke point."*
- arXiv:2603.10092: *"non-bypassable... at the last mile."*
- arXiv:2605.18991: a Reference Monitor with Complete Mediation is the goal.

**Nobody recommends option (b).** Replicating the check at each of four callers is
not proposed by any source read; CWE-638 lists exactly that situation (checks in
the controller, absent on direct paths) as the *weakness*.

**Genuine debate — two live tensions:**

1. **Single choke point vs defense in depth.** CWE-424's own mitigation says
   *"Deploy different layers of protection to implement security in depth"*, and
   arXiv:2605.18991 argues for *"mechanisms that operate at different layers of
   abstraction"* rather than one checkpoint. This does **not** rehabilitate (b) —
   layered means *different kinds* of barrier at *different abstraction levels*
   (e.g. the signals_server drawdown breaker AND the paper_trader kill-switch gate
   AND a broker-level limit), not the same check copy-pasted at N call sites. The
   correct reading: **add the last-mile gate, and keep the existing upstream gates
   rather than consolidating them away.**
2. **Is a designed bypass safer than a naked primitive?** IEC 61511 says bypasses
   must be *designed in* with authorization + indication + time limit + log +
   compensating measures — i.e. safety engineering assumes the hatch exists and
   regulates it. arXiv:2605.18991 says test hatches *become the attack surface*
   (Cursor YOLO mode). These are reconcilable only one way: **the hatch must be
   loud, logged, and impossible to use by accident.** A `force=True` kwarg fails
   that test; so does a same-name-different-module primitive.

## Pitfalls (from literature)

- **Fail-open by omission.** The dangerous class fails *"by allowing access, a
  failure which may go unnoticed"* (Saltzer). A future 5th caller that forgets the
  gate produces no error — it produces a trade. This is the argument that decides
  between (a/c) and (b).
- **Dead-but-reachable paths.** Knight Capital: an unused code path left in the
  binary, re-entered by accident, $460M/45min. Directly applicable to option (c)'s
  `_execute_buy_unchecked`.
- **Policy replicated then drifted.** CWE-638's core scenario, and already
  instantiated in this repo by `signals_server`'s in-memory drawdown breaker.
- **Test modes that ship enabled.** Cursor AgentFlayer / "Auto Run (YOLO) mode"
  (arXiv:2605.18991). Any bypass must be un-settable from config/env.
- **Kill switches that are never exercised.** Feature-flag anti-pattern #6: a kill
  switch first used during an incident is a kill switch you are debugging during an
  incident. Argues *for* keeping the drills working, deliberately.
- **Caching an authority decision.** Saltzer warns that remembered auth results
  must be invalidated on change. Do not snapshot `is_paused()` once per cycle and
  reuse it across many BUYs; read it per call.
- **Flag arguments hide intent at the call site** (Fowler): `book(martin, false)`.

## Application to pyfinagent

| Literature claim | pyfinagent anchor |
|---|---|
| Alternate path reaching the same effect (CWE-424) | `signals_server.py:444` -> `paper_trader.py:148` with no gate; the gated route is `autonomous_loop.py:1316` + `paper_trader.py:1094-1270` |
| "Single interface that performs the access checks" (CWE-638) | Only two `paper_trades` producers exist (`:324`, `:510`) — a single BUY interface is achievable |
| "Last mile where side effects occur" (arXiv:2603.10092) | `_safe_save_trade` at `:324` and `_update_portfolio_cash` at `:408`; the gate must precede `:229` (`get_or_create_portfolio`) |
| Non-bypassable-by-routing (SAE) | The repo already argues this at `:202-205` for the price-tolerance gate: *"Placed BEFORE the ExecutionRouter call so the non-bypassable-invariants pattern ... holds"* — same paper, same file |
| Designed bypass w/ indication + log (IEC 61511) | `KillSwitchState._append_audit` (`:412`) + `buy_rejections` (`:107`) + P1 via `alerting.py` |
| Layered, not consolidated (arXiv:2605.18991) | Keep `signals_server` `drawdown_circuit_breaker` (`:950`) AND add the paper_trader gate |
| Injectable dependency instead of a flag | `ExecutionRouter.__init__(mode=None) -> mode or _current_mode()` (`execution_router.py:268-269`) |

## RECOMMENDATION

### Verdict: **(a), with the bypass implemented as dependency injection rather than a parameter — call it (a′). Do NOT do (c). Do NOT do (b).**

This is a recommendation *against* the instinct stated in the prompt.

**The gate goes at the top of `execute_buy` (before `:189`), unconditionally:**

```
execute_buy(...):
    ks = self._ks_state()            # per-call read; no cached authority (Saltzer)
    if ks.is_paused() -> record buy_rejections{reason:"kill_switch_paused"}
                         + _append_audit("buy_blocked", ...) ; return None
    if disarmed-with-history (mirror paper_trader.py:1230's predicate)
                      -> same, reason "kill_switch_disarmed_lost_history", + P1
```

**The "bypass" is not a parameter. It is the constructor seam the drills already
use:**

```
PaperTrader.__init__(self, settings, bq_client, trade_notifier=None,
                     kill_switch_state=None)      # <- new, defaults to None
    self._injected_ks = kill_switch_state
def _ks_state(self):                              # mirrors ExecutionRouter:269
    return self._injected_ks or kill_switch.get_state()
```

The two drills change ONE line each — they already construct
`PaperTrader(settings=settings, bq_client=bq)` with a stub BQ; they add
`kill_switch_state=UnpausedStubState()` (or an in-file 3-line stub, exactly like
`zero_orders_drill.StubBQ` at `:44-59`). Nothing in production changes.

**Why this beats (c), my reading of the evidence against the prompt's instinct:**

1. **(c) leaves a public-by-convention unguarded route in the same module.**
   `_execute_buy_unchecked` is protected by a leading underscore — a *style*
   convention with zero enforcement in Python. CWE-638 is explicit that the single
   checked interface must be paired with *"code standards that require use of this
   interface"* — i.e. the registry itself concedes that (c) is only as strong as a
   convention nobody can execute. Knight Capital is what a reachable-but-nominally-
   retired path costs.
2. **(c) makes the safe call the longer one.** Saltzer's psychological
   acceptability: *"the human interface [must] be designed for ease of use, so that
   users routinely and automatically apply the protection mechanisms correctly."*
   Under (c) the shortest, most obvious name (`execute_buy`) must be the guarded
   one — fine — but a future author debugging a refusal will find
   `_execute_buy_unchecked` and reach for it, because it is *documented as the way
   to bypass*. Under (a′) there is no such affordance to find.
3. **(c) costs 19 call-site edits, not 4.** The full grep (above) shows 15 test
   call sites. A rename churns 8 test files, including
   `test_phase_70_3_atomic_swap.py:36,40` which defines a *stub* `execute_buy`
   whose name must stay aligned with `autonomous_loop`'s call. Large diffs on a
   P0 safety fix are their own risk.
4. **The escape hatch (c) was designed to serve does not need to exist.** Both
   "deliberate" callers use `StubBQ`/`MagicMock` and never touch the live book.
   They do not need to bypass a safety control; they need to *supply their own
   state*, which is the same thing they already do for BigQuery. Building a bypass
   API for a need that dependency injection already satisfies is exactly the
   "test mode that later ships enabled" pattern arXiv:2605.18991 warns about.
5. **(b) is recommended by nobody** and fails on the 5th caller. CWE-638 and
   CWE-288 both name the funnel as the mitigation.

**IEC 61511 compliance of (a′):** authorization = injection is only possible at
construction, in source, visible in code review (vs a runtime `force=True` from any
caller); indication + log = every refusal writes `buy_rejections` + an audit row,
and the drills should print that they are running with an injected state;
compensating measures = the drills use stub BQ so no real order can result; time
limit = N/A because the hatch cannot be activated in a production process at all
(there is no config or env path to it) — which is strictly stronger than a time
limit; bypass log = `_append_audit`.

### The strongest counter-argument to my recommendation

**Constructor injection is still a bypass, and it is a *silent* one.** Under (a′),
any code that constructs `PaperTrader(..., kill_switch_state=X)` gets whatever
policy `X` implements, with no marker in the trade row and no audit event
distinguishing "traded with the real switch" from "traded with an injected stub".
That is *weaker* than (c) on one axis: with (c), a bypass is visible at the **call
site** as a differently-named method (`_execute_buy_unchecked(...)`), greppable in
one pass; with (a′) the bypass is visible only at the **construction site**, which
may be many frames away from the trade. Fowler's own caveat cuts here too — if the
behaviour is derived from injected object state rather than an explicit argument,
the reader of `execute_buy(...)` cannot tell which policy is in force. And
arXiv:2605.18991's central warning is precisely that a testing seam becomes the
production hole; a defaulted constructor kwarg is a broad seam.

**Mitigations that make the counter-argument survivable** (fold these into the
contract, otherwise the counter-argument wins):
- The injected state object must be **required to self-identify**, and
  `execute_buy` must stamp the trade row / audit row with which state answered
  (e.g. `ks_source: "live" | "<class name>"`). Then a bypassed trade is visible
  *after the fact* in `paper_trades`, satisfying IEC 61511's "bypass log".
- `PaperTrader.__init__` should **log at WARNING** when a non-`None`
  `kill_switch_state` is passed, naming the class. Loud, once, per construction.
- A test must assert that **no `backend/` module** passes `kill_switch_state=`
  (a source-scan guard over `backend/`, with the drills as the only allowed
  `scripts/` users). This is the executable form of CWE-638's "code standards
  that require use of this interface" — and unlike a naming convention, it fails
  CI. Note the standing project rule: that guard must be **mutation-tested** (add
  a `kill_switch_state=` in a scratch copy of a backend module and prove the guard
  goes red), or it does not count.
- Keep the `signals_server` drawdown breaker in place (layered defence, per
  arXiv:2605.18991), and queue a separate step for the fact that
  `kill_switch_test.py` tests the weak duplicate rather than `kill_switch.py`.

If the operator rejects the injected-state seam, the fallback ranking is
**(c) second, (b) last** — (c) at least keeps one guarded public name; (b) fails
open on the next caller.

### Residual defects found while researching (queue as their own steps)

1. `scripts/go_live_drills/kill_switch_test.py` exercises `signals_server`'s
   in-memory `drawdown_circuit_breaker`, not `backend/services/kill_switch.py`.
   The drill named for the kill switch does not test the kill switch.
2. `signals_server`'s peak-equity high-water mark is **in-memory** (`:88-89`) and
   resets on every restart, whereas `kill_switch`'s peak is audit-replayed. Two
   drawdown policies, different durability.
3. `risk_server.py:91,181` is a third reader of pause state via a snapshot dict
   key rather than `is_paused()` — a spelling that greps for `is_paused()` miss.
4. `ExecutionRouter.submit_order` is reachable directly from three
   `scripts/harness` + `scripts/go_live_drills` entry points and can hit Alpaca
   paper without writing a `paper_trades` row. Out of 36.13's scope; not nothing.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **9**
- [x] 10+ unique URLs total (incl. snippet-only) — **28**
- [x] Recency scan (last 2 years) performed + reported — 6 findings, incl. one
      that qualifies the consensus
- [x] Full papers / pages read (not abstracts); arXiv via `/html/` per
      `.claude/rules/research-gate.md`, never `/pdf/`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered `paper_trader.py`, `kill_switch.py`,
      `autonomous_loop.py`, `signals_server.py`, `risk_server.py`,
      `execution_router.py`, `portfolio_manager.py`, both drills, and the
      `go_live_drills/` inventory (**11 files inspected**)
- [x] Contradictions / consensus noted (single-choke-point vs defense-in-depth;
      designed-bypass vs no-bypass)
- [x] All claims cited per-claim with URL + access date or file:line
- [ ] **Gap:** I did not read the IEC 61511-1:2016 standard text itself (paywalled
      at iteh.ai); clause numbers come from an industry summary (source 6). Treat
      clause citations as indicative, not verbatim standard text.
- [ ] **Gap:** I did not verify at runtime that a live pause is currently present
      in `handoff/kill_switch_audit.jsonl` (constraint: do not touch that file).
      The durability claim is derived from `kill_switch.py:254-268` by reading,
      not by execution.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 19,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "CWE-424's parent is CWE-638 'Not Using Complete Mediation', so defect and remedy share a taxonomy node. CWE-638 ('a single interface that performs the access checks'), CWE-288 ('funnel all access through a single choke point'), Saltzer & Schroeder ('every access to every object', extended explicitly to maintenance paths), and arXiv:2603.10092 ('non-bypassable invariants at the last mile') all converge on gating inside execute_buy; NO source recommends option (b) per-caller checks. Recommendation is (a'), a gate at the top of execute_buy with NO bypass parameter: make the kill-switch state an injectable constructor dependency defaulting to get_state(), mirroring ExecutionRouter.__init__(mode=None) at execution_router.py:268. Decisive internal finding: both 'deliberate' callers already inject StubBQ/MagicMock, so they need to supply state, not bypass a control; and pause IS restart-durable (kill_switch.py:254-268 replays the pause event), so a naive gate really would break them. Against (c): Python's underscore is convention-only, Knight Capital is the cost of a reachable retired path, and a rename churns 19 call sites not 4. Strongest counter-argument: injection is a silent bypass visible only at the construction site -- mitigate by stamping ks_source on the trade row, WARN-logging non-default injection, and a mutation-tested source-scan guard that no backend/ module passes kill_switch_state=.",
  "brief_path": "handoff/current/research_brief_36.13.md",
  "gate_passed": true
}
```
