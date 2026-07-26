# Research Brief — masterplan step 36.12

**Tier:** moderate | **audit_class:** false | **Researcher:** Layer-3 (Opus 5, 1M)
**Started:** 2026-07-26 | **Status:** IN PROGRESS (write-first; appended incrementally)

**Question:** The kill-switch gate `check_and_enforce_kill_switch` mutates its own
baselines (`update_peak`, `update_sod_nav`) BEFORE evaluating a breach, and branches
only on `any_breached`, never on `armed`. On a post-rotation cycle with unrecoverable
history both baselines silently re-anchor to today's NAV, the real drawdown is
forgiven, and the switch reports ARMED AND HEALTHY.

---

## Search queries run (3-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | circuit breaker safety interlock "read before write" evaluate before updating baseline anti-pattern |
| 2 | year-less canonical | SEC Rule 15c3-5 market access pre-trade risk controls block order when risk check unavailable |
| 3 | year-less canonical | "fail-safe defaults" deny when authorization decision cannot be made policy decision point unavailable NIST |
| 4 | year-less canonical | distinguish cold start from lost state initialization sentinel "trust on first use" event sourcing first event marker |
| 5 | year-less canonical | Leveson STPA process model inconsistent with controlled process unsafe control action safe state when measurement unavailable |
| 6 | **current-year 2026** | trading system kill switch disarmed missing baseline pre-trade control fail-open 2026 risk limit re-anchor |
| 7 | **last-2-year 2025** | distinguish fresh install from lost state bootstrap marker control file "first boot" versus corrupted state database 2025 |
| 8 | **current-year 2026** | "self-healing" monitoring anti-pattern alert threshold resets itself baseline auto-adjusts hides regression 2026 |
| 9 | **last-2-year 2025** | arXiv 2025 safety monitor invariant checked against state the monitor itself updated stale baseline runtime verification |

All three variants exercised. As in the 36.7 gate, the **year-less** queries were the
decisive ones (CFR text, CWE entries, the exchange risk-control specs); the year-locked
queries mostly returned SEO content but did surface the FINRA 2026 report, the NYSE Pillar
v4.7 (2026-03-19) spec, and two 2026 arXiv preprints.

---

## Read in full (10; gate floor is 5)

| # | URL | Accessed | Kind / tier | Fetched how | Key finding |
|---|-----|----------|-------------|-------------|-------------|
| 1 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-07-26 | Primary regulation (tier 1) | WebFetch, full | (c)(1) controls must "Prevent the entry of orders that exceed appropriate pre-set credit or capital thresholds ... **by rejecting orders** if such orders would exceed". (c)(2) "Prevent the entry of orders **unless there has been compliance** with all regulatory requirements". (d) controls "shall be under the direct and exclusive control of the broker or dealer". **Measured negative result, same as the 36.7 gate: there is NO text mandating a block when a control cannot be EVALUATED.** The rule's grammar is prohibitive ("prevent ... unless"), which reads as deny-by-default, but it is never made explicit. |
| 2 | https://cwe.mitre.org/data/definitions/367.html | 2026-07-26 | Official taxonomy (MITRE) | WebFetch, full | "The product checks the state of a resource before using that resource, but the resource's state can change between the check and the use in a way that invalidates the results of the check." Mitigations include "Ensure that locking occurs before the check, as opposed to afterwards, such that the resource, as checked, is the same as it is when in use." **Explicit negative finding, recorded verbatim from the fetch: "CWE-367 does not explicitly define a weakness for programs that themselves modify state before checking it -- only for external state changes between check and use."** So 36.12's defect is TOCTOU-*adjacent* but is NOT CWE-367; there is no named CWE for a guard that mutates its own datum. |
| 3 | https://quality.arc42.org/approaches/fail-safe-defaults | 2026-07-26 | Architecture reference (arc42 quality model) | WebFetch, full | Systems "transition to a predefined safe state rather than continuing in an undefined or hazardous mode". Origin: "formalized by Saltzer and Schroeder in 1975 as one of eight design principles for information protection, where it meant basing access decisions on permission rather than exclusion." Prescription when validity cannot be confirmed: "deny access, reject commands, disable features, halt motion." Listed drawbacks: "Reduced availability, abrupt rather than graceful degradation, and increased recovery time." |
| 4 | https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/market-access-rule | 2026-07-26 | Regulator report, **2026** | WebFetch, full | Exam findings include "Not establishing **reasonable** pre-trade order limits, preset capital **and credit** thresholds"; "**Excluding** certain orders from a firm's pre-trade erroneous controls based on order types"; "not maintaining direct and exclusive control over controls by allowing the ATS **or exchange** to unilaterally set financial thresholds". **The recurring 2026 finding is scope holes and un-owned thresholds -- an alternate path that escapes the control -- which is exactly 36.12's shape.** |
| 5 | https://arxiv.org/html/2601.17744v1 | 2026-07-26 | Preprint, **Jan 25 2026** (arXiv:2601.17744, Fatmi, *Faramesh*) | WebFetch, full HTML | §5.4 "**Fail-Closed Semantics. Any failure in the authorization process results in denial or deferral of the proposed action**". §5.1: execution proceeds "if and only if authorization explicitly permits it". §7.1: "**An absent decision is not treated as implicit permission but mapped to DENY by default.**" §7.4: DEFER is a first-class third state requiring external approval. §9.1: "Append-Only and Tamper-Evident Semantics. Records are immutable and ordered"; decision records are first-class so the log "distinguishes explicit decisions from absences". |
| 6 | https://cwe.mitre.org/data/definitions/424.html | 2026-07-26 | Official taxonomy (MITRE) | WebFetch, full | **CWE-424 Improper Protection of Alternate Path**: "The product does not sufficiently protect all possible paths that a user can take to access restricted functionality or resources." Consequence: "Bypass Protection Mechanism; Gain Privileges". Related: CWE-638 Not Using Complete Mediation, CWE-693 Protection Mechanism Failure. **This is the named prior art for question 4** (deliberate path token-gated, accidental path un-gated). |
| 7 | https://cwe.mitre.org/data/definitions/223.html | 2026-07-26 | Official taxonomy (MITRE) | WebFetch, full | **CWE-223 Omission of Security-relevant Information**: "The product does not record or display information that would be important for identifying the source or nature of an attack, **or determining if an action is safe**." Consequence "Hide Activities" / Non-Repudiation. ParentOf CWE-778 Insufficient Logging. **This is the named prior art for the audit-trail-indistinguishability half of question 4**: a bare `peak_update` row omits exactly the information needed to determine whether the action was safe. |
| 8 | https://www.nasdaqtrader.com/content/EquityKillSwitch.pdf | 2026-07-26 | Exchange operator spec (Nasdaq) | curl + **pdfplumber**, all 2 pp. | "the system disables order entry ports and cancels open orders for the firm." **"Kill Switch is a post trade, best efforts process."** And the reset asymmetry, verbatim: "Once a Kill Switch is triggered, **a call must be made to the Nasdaq Trade desk** ... in order for the Risk Exposure limit(s) for an MPID to be reset." Re-arming is a human, out-of-band act -- never something the trading system does to itself. |
| 9 | https://www.nyse.com/publicdocs/nyse/NYSE_Pillar_Risk_Controls.pdf | 2026-07-26 | Exchange operator spec (NYSE Pillar v4.7, **March 19 2026**), 40 pp. | curl + **pdfplumber**, all 40 pp. extracted + keyword-indexed (`Kill Switch` x30, `breach` x67, `reject` x31, `unavailable` **x0**) | Two decisive findings. (a) **Un-block is bound to the actor who blocked**: "Following a Kill Switch - Block action, the affected Risk Entity may be unblocked **only by the same Risk User (firm) that applied the block**" (p.22) and FAQ 16 repeats it. (b) **Every refusal carries a distinguishable reason code** -- p.35 lists 22 of them (`R225: Risk - Gross Credit Breach`, `R226: Risk - Kill Switch`, `R081: Price Too Far Outside`, ...) so a rejected order says WHICH control fired. (c) Explicit no-data default, p.34: "If no consolidated last sale is available, Pillar will default the Risk reference price to the leg's NBO. **If no NBO is available, Pillar will default the Risk reference price to $0 for that leg (i.e., no check).**" -- a real-world exchange that DOES fail-open on missing reference data, and says so out loud. **Adversarial data point against a naive "everyone fails closed" claim; note that it fails open on a PRICE input, never on the kill switch itself.** |
| 10 | https://mariadb.com/kb/en/getting-started-with-mariadb-galera-cluster/ | 2026-07-26 | Official docs (MariaDB) | WebFetch, full | The `grastate.dat` provenance pattern: "You can determine which node is the most advanced by checking `grastate.dat` on each node and looking for the node with the highest `seqno`. **If the node crashed and `seqno=-1`**, then you can find the most advanced node by recovering the `seqno` ... with the `wsrep_recover` option." "You can set `safe_to_bootstrap=1` on the most advanced node." "**In some cases Galera will refuse to bootstrap a node if it detects that it might not be the most advanced node in the cluster.**" -- i.e. an explicit on-disk marker distinguishes trustworthy state from state-of-unknown-provenance, and the system REFUSES rather than guessing. (Page is thinner than galeracluster.com, which returned HTTP 403.) |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0 | SEC staff FAQ | SEC pages served an interstitial to the 36.7 gate on the same day; Cornell LII carries the operative rule text and was used instead |
| https://www.sec.gov/files/rules/final/2010/34-63241-secg.htm | SEC small-entity guide | superseded by #1 for the operative text |
| https://www.nasdaqtrader.com/content/productsservices/trading/ften/sec_mar.pdf | Nasdaq 15c3-5 explainer | duplicate of #1 + #8 coverage |
| https://arxiv.org/abs/2506.01782 | preprint, STPA for Frontier AI (Mylius, Jun 2 2025) | abstract page only; the abstract does NOT contain the process-model-flaw taxonomy I needed, and the HTML body was not reachable in budget. **Explicitly NOT counted as read-in-full** |
| https://arxiv.org/abs/2606.15980 | preprint, "Do Safety Monitors Stay Reliable After an Update?" (2026) | recency-scan hit; concerns ML activation probes, mechanism does not transfer |
| https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii | consultancy | MiFID II RTS 6 adjacent, no text on unevaluable controls |
| https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box | industry commentary | opinion piece, tier 5 |
| https://quality.arc42.org/ (index), https://nhimg.org/glossary/fail-closed/, https://nhimg.org/glossary/policy-decision-point/ | glossary | tier-5 restatements of #3 |
| https://softwaremill.com/things-i-wish-i-knew-when-i-started-with-event-sourcing-part-1/ | industry blog | superseded by the 36.7 gate's Microsoft event-sourcing + Kurrent snapshot reads |
| https://galeracluster.com/library/documentation/crash-recovery.html | vendor docs | **HTTP 403** -- substituted MariaDB KB (#10) |
| https://blog.langchain.com/production-agents-self-heal/ | vendor blog | "self-heal" in the agent-retry sense, not the guard-resets-its-own-baseline sense |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | official docs | read in full during the 36.7 gate; carried over, not re-fetched |
| https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | industry standard-setter | read in full during the 36.7 gate; carried over (see Carry-over section) |

**URLs collected (unique, incl. search-result hits not tabled above): 34.**

---

## Recency scan (2024-2026) -- PERFORMED

Four year-scoped passes (queries 6-9 above) plus two 2026-dated primary documents read in
full.

**Findings in the window:**

1. **FINRA 2026 Annual Regulatory Oversight Report, Market Access Rule section** (source #4,
   published Jan 2026). The recurring 2026 exam finding is *scope holes*: order types
   excluded from pre-trade controls, thresholds set by a party other than the firm, and
   undocumented threshold reasonableness. This **complements rather than supersedes** the
   15c3-5 text: the regulator's 2026 concern is precisely "a path that escapes the control",
   which is the 36.12 shape (CWE-424).
2. **NYSE Pillar Risk Controls v4.7, dated 2026-03-19** (source #9). New in the 2024-2026
   revisions: Order Rate Threshold across all symbols (4.1, Jun 2024), Gross Credit
   Executed/Open split (4.0, Mar 2024), Floor Broker Firm risk controls (4.7, Mar 2026).
   The unblock-by-same-actor rule and the per-control reject reason codes are current
   2026 practice, not legacy.
3. **arXiv:2601.17744 Faramesh, Jan 2026** (source #5) -- the strongest *explicit* statement
   of the doctrine the step needs: an absent authorization decision maps to DENY, and the
   decision itself is a logged first-class artifact. This is 2026 work on agent execution
   control planes, and it transfers cleanly because pyfinagent's kill switch IS an execution
   gate in front of an autonomous agent.
4. **arXiv:2606.15980 (2026), "Do Safety Monitors Stay Reliable After an Update?"** --
   surfaced by the 2025/2026 runtime-verification query. Concerns *activation-probe*
   staleness after model updates. Mechanism does not transfer (no baseline-mutation issue);
   recorded so the absence is visible, snippet-only.

**No finding in the window supersedes the canonical sources.** Saltzer & Schroeder 1975
(via arc42) remains the operative principle; the 2026 material tightens it (Faramesh's
"absent decision != permission") and evidences it (FINRA's alternate-path findings).

---

## Key findings

**F1. "Measure before mutate" is NOT a named canonical pattern for safety interlocks, and
the closest named weakness explicitly excludes this case.** CWE-367 TOCTOU covers *external*
state change between check and use; the fetch's own summary states CWE "does not explicitly
define a weakness for programs that themselves modify state before checking it" (source #2).
Do not write the contract as if an established rule is being applied -- there is no
"self-healing gate" or "observer mutates the observable" entry in CWE, and the 2026
self-healing-threshold search returned only vendor content, not an engineering-literature
anti-pattern. **The defensible framing is CWE-424 (alternate path) + fail-safe defaults
(Saltzer & Schroeder), not TOCTOU.**

**F2. When a control cannot be evaluated, the doctrine is DENY -- but the trading-regulation
text never says so; the software-safety literature does.** 15c3-5(c)(2) is phrased
prohibitively ("Prevent the entry of orders **unless** there has been compliance"), which
is deny-by-default in grammar (source #1), and FINRA's 2026 findings punish scope holes
(source #4) -- but neither says "block when the control is unevaluable". The explicit
statements come from arc42/Saltzer-Schroeder ("deny access, reject commands ... when
validity cannot be confirmed", source #3) and Faramesh §7.1 ("An absent decision is not
treated as implicit permission but mapped to DENY by default", source #5). **Cite the
software-safety sources for the DENY doctrine and the regulation only for the
prevent-the-entry framing. Do not overstate the regulation.**

**F3. ADVERSARIAL: a real exchange does fail-OPEN on missing input, and documents it.**
NYSE Pillar (source #9, p.34): "If no NBO is available, Pillar will default the Risk
reference price to $0 for that leg (**i.e., no check**)." So "everyone fails closed" is
false as a blanket claim. The distinction that survives: Pillar fails open on a *price
reference input* to one check while every other control and the Kill Switch itself keep
operating -- it degrades one leg, loudly and by written spec. That is very close to
`evaluate_breach`'s existing per-leg design (criterion 6) and is **evidence FOR keeping
per-leg independence**, and evidence AGAINST an all-or-nothing internal short-circuit.
What Pillar does NOT do is let the missing input silently *manufacture* a passing check --
and that is what 36.12's re-anchor does.

**F4. Re-arming is an out-of-band human act, in every exchange spec read.** Nasdaq: "a call
must be made to the Nasdaq Trade desk ... in order for the Risk Exposure limit(s) ... to be
reset" (source #8). NYSE: "unblocked **only by the same Risk User (firm) that applied the
block**" (source #9 p.22, FAQ 16). **This is direct external corroboration of pyfinagent's
own `KS-PEAK-RESET` token policy, and therefore of criterion 5**: the trading path must not
acquire a route to re-anchor. The industry position is stronger than the step requires --
these systems never let the trading engine re-arm itself at all.

**F5. The audit-indistinguishability half of the policy asymmetry has a name: CWE-223,
Omission of Security-relevant Information** -- "does not record ... information that would
be important for identifying the source or nature of an attack, **or determining if an
action is safe**" (source #6/#7). The alternate-path half is **CWE-424**. Both prescribe the
same two-part remedy shape: (a) protect the alternate path (CWE-424 mitigation: "Deploy
different layers of protection"; parent CWE-638 "Not Using Complete Mediation"), and (b)
make the event distinguishable in the log. Faramesh §9.1 gives the positive form: the
decision is a first-class logged artifact so the log "distinguishes explicit decisions from
absences". **Answer to the caller's question 4: do BOTH -- deny the implicit path AND emit a
distinguishable audited event. The literature does not treat them as alternatives.**

**F6. The provenance-marker pattern for question 3 is real and has a concrete reference
implementation: Galera's `grastate.dat` / `safe_to_bootstrap` / `seqno: -1`** (source #10).
The pattern's three parts: (i) an on-disk marker whose *value* says whether the state is
trustworthy, not merely whether it exists; (ii) `seqno = -1` as an explicit
"state-of-unknown-provenance" sentinel distinct from a real position; (iii) the system
"will refuse to bootstrap a node if it detects that it might not be the most advanced node"
-- refuse, rather than guess. Note what this pattern requires that pyfinagent does not have:
a marker written *in advance*. A brand-new pyfinagent book and a book whose audit history
was lost are today byte-identical (`sod_nav=None, peak_nav=None`) -- so the discriminator
must be derived from *other* evidence (see Application, D1/D2), and an
explicit-sentinel design can only apply to books created after it ships.

**F7. Nothing found on keeping operator-facing remediation text in lockstep with a
behaviour change (question 5). Recording the gap honestly rather than padding.** The nearest
transferable material is NYSE's per-control reject reason codes (source #9 p.35: 22 codes,
one per control) -- i.e. the operator-facing string is *generated from* the control that
fired rather than hand-written prose about what will happen next. That is a design
suggestion, not a citation for the practice of updating docs with code. Criterion 8 should
be justified on first principles (the three strings become false statements) and, if a
mechanism is wanted, on the reason-code shape.

---

## Internal code inventory

**Audit-file integrity:** `handoff/kill_switch_audit.jsonl` md5 at brief start =
`ce8fb93348bb9a3bbe26f2d91b1bc05e` (matches the caller's required value). No test
was executed by this session; re-verified at the end of the brief.

### A. The defect site, verbatim

`backend/services/paper_trader.py:1069-1116` `check_and_enforce_kill_switch`. The
order is exactly as the step describes:

| Line | Statement | Effect |
|---|---|---|
| :1077 | `nav = float(portfolio.get("total_nav") or portfolio.get("starting_capital") or 0.0)` | NAV, `or 0.0` on failure |
| :1080 | `state.update_peak(nav)` | **MUTATION 1** — anchors peak when `_peak_nav is None` |
| :1087-:1090 | `snap = state.snapshot()`; `if snap.get("sod_nav") is None or snap.get("sod_date") != today: state.update_sod_nav(nav, date=today)` | **MUTATION 2** — anchors SOD when `sod_nav is None` |
| :1092-:1096 | `breach = evaluate_breach(current_nav=nav, ...)` | **MEASUREMENT — happens AFTER both mutations** |
| :1097 | `if breach["any_breached"] and not state.is_paused():` | **only `any_breached`; `armed` never read** |
| :1105-:1110 | `check_auto_resume(...)` | reads `armed` internally (kill_switch.py:601) |
| :1116 | `return {"triggered": False, "breach": breach, "auto_resume": auto_resume}` | `breach` (incl. `armed`) IS returned — the caller just never branches on it |

There are **no early returns** in the function. Exactly two exits: `:1101`
(triggered) and `:1116` (not triggered). The `armed` key IS present in the returned
`breach` dict on both paths (kill_switch.py:493 and :518) — the information reaches
the caller and is discarded.

**Read-then-write violation is exact:** `update_peak` (kill_switch.py:376-381)
anchors on `self._peak_nav is None`; `evaluate_breach` (kill_switch.py:476) computes
`trailing_baseline_missing = not (peak is not None and peak > 0)` from
`_state.snapshot()`. Because :1080 ran first, `peak` is never None at :1092. Same
for the SOD leg (:1089-:1090 vs kill_switch.py:475). Net: `armed` is
**structurally always True** on this code path — the flag 36.7 added cannot fire
here, by construction.

### B. Call-site enumeration (grep, not hand-count)

`grep -rn "check_and_enforce_kill_switch" --include="*.py"`:

| Caller | Line | Kind |
|---|---|---|
| `backend/services/autonomous_loop.py` | **:1285** `ks_check = await asyncio.to_thread(trader.check_and_enforce_kill_switch)` | **the ONLY production caller** |
| `backend/tests/test_dod4_tier1_coverage_investment.py` | :964, :993 | test |
| `backend/tests/test_phase_38_1_kill_switch_auto_resume.py` | :197 | test |
| `backend/slack_bot/scheduler.py` | :846 | **docstring only** (prose in `notify_kill_switch`), not a call |
| `backend/services/kill_switch.py` | :449 | docstring cross-reference |

The loop's use of the return value (`autonomous_loop.py:1286-1292`):

```python
summary["kill_switch"] = ks_check
if ks_check.get("triggered") or _ks_state().is_paused():
    logger.warning("Paper trading: kill-switch active -- skipping decide/execute")
    summary["steps"].append("kill_switch_halted"); summary["halted"] = True
    ...
    return summary
```

So the halt condition is `triggered OR is_paused()`. **`armed` is not consulted**,
and `triggered` is itself derived only from `any_breached`. This is the exact
integration point for criterion 4 — a `armed is False` branch here would block
`decide/execute` with the existing halt machinery, no new control flow needed.

### C. Every consumer of `armed` — 36.10's claim VERIFIED-WITH-A-CORRECTION

`grep -rnw "armed"` across `*.py *.ts *.tsx *.js` (node_modules + .next excluded).
Non-test, non-comment production reads:

| Consumer | Line | What it does |
|---|---|---|
| `backend/api/paper_trading.py` | **:593** `if not breach.get("armed", True):` | POST /resume → HTTP 409 refusal |
| `frontend/src/components/KillSwitchPanel.tsx` | **:137** `const disarmed = breach.armed === false;` | DISARMED badge + Resume disabled |
| `frontend/src/components/OpsStatusBar.tsx` | **:318** `const disarmed = kill.breach.armed === false;` | Kill segment badge |
| `backend/services/kill_switch.py` | **:601** `if not breach.get("armed", True):` inside `check_auto_resume` | blocks the T+2h auto-resume |

Type declarations only: `KillSwitchPanel.tsx:25`, `OpsStatusBar.tsx:39`.
Tests: `test_phase_36_7_kill_switch_rotation_rearm.py` (:223,:259,:277,:347,:628,
:866,:885), `test_phase_23_2_5_kill_switch_no_false_fires.py:234`,
`KillSwitchPanel.disarmed.test.tsx` (:60,:76,:132,:181,:193).

**Verdict on 36.10's "nothing outside UI/API reads armed":** essentially correct but
**one counter-example**: `kill_switch.py:601` is a backend *service*, not UI/API. It
is still not the *trading* path. The accurate restatement, which the 36.12 contract
should use: **no order-placing path reads `armed`.** All four consumers gate
*resume* or *display*; none gates *order entry*. That is precisely the hole.

`grep -rn "baseline_missing"` adds no production consumers beyond the same four
sites plus display formatting (`OpsStatusBar.tsx:320,:323`;
`KillSwitchPanel.tsx:186,:192`).

`backend/agents/mcp_servers/risk_server.py:73-91` (`kill_switch` MCP tool) calls
`evaluate_breach` and returns the whole `breach` dict verbatim — so `armed`
propagates to the MAS layer, but the tool itself does not branch on it and is
read-only by design ("Does NOT flip state").

### D. Blast-radius bound is CONFIRMED (per-leg independence)

`evaluate_breach` (kill_switch.py:455 docstring point 2, code :498 and :504) gates
each leg on its OWN marker:

```python
if not daily_baseline_missing:  # :498
    daily_loss_pct = (sod - current_nav) / sod * 100.0 ...
if not trailing_baseline_missing:  # :504
    trailing_dd_pct = (peak - current_nav) / peak * 100.0 ...
```

No wholesale early return. Criterion 6 ("losing one baseline must still leave the
other leg enforcing") is a **preserve-this-property** criterion, not a build-this
one — the fix must not introduce an `if not armed: return` short-circuit inside
`evaluate_breach`. The natural fix shape (branch in the caller) leaves this
untouched.

### E. Policy asymmetry — measured

| Path | Gate | Audit row |
|---|---|---|
| `reset_peak` (kill_switch.py:383-414) | `settings.kill_switch_peak_reset_enabled` — DARK, returns `None` at :407 until `KS-PEAK-RESET: APPROVED` (settings.py:39) | `peak_reset` with `old_peak`, `new_peak`, `trigger`, `operator` (:411-413) |
| `update_peak` via `check_and_enforce_kill_switch:1080` | **none** | `peak_update` with only `nav` (:381) — **indistinguishable from a legitimate ratchet** |

Confirmed: a re-anchor-from-None writes a bare `peak_update` row carrying no
`old_peak`, no `trigger`, no marker that it was an anchor rather than a ratchet.
There is no forensic way, from the audit trail alone, to tell "the peak ratcheted up
to 24666" from "the peak was lost and re-anchored to 12000". This is criterion 5's
motivation and also suggests a cheap, non-behavioural half-fix (see Application).

### F. Operator-facing strings — quoted verbatim, line numbers RE-VERIFIED today

All three line numbers in the step are **still accurate** (grep-confirmed
2026-07-26):

1. `backend/api/paper_trading.py:600` (inside the 409 raised at :593-:602):
   > `"The next paper-trading cycle re-anchors both baselines; retry after "` /
   > `"it runs, or check handoff/kill_switch_audit.jsonl for sod_snapshot/"` /
   > `"peak_update rows."`
2. `frontend/src/components/KillSwitchPanel.tsx:172` (badge `title`):
   > `"DISARMED: the loss baselines could not be restored, so neither breach leg can fire. Resume is blocked until the next cycle re-anchors them."`
3. `frontend/src/components/KillSwitchPanel.tsx:221` (Resume-button `title`):
   > `"Cannot resume: kill switch DISARMED (loss baselines unrestorable). The next cycle re-anchors them."`

All three tell the operator to WAIT FOR THE DEFECT TO FIRE. They are accurate
descriptions of current behaviour and become false the moment the fix lands —
criterion 8 is a correctness requirement, not cosmetics.

**A FOURTH string the step does not list**, same class, must be checked by the
contract: `backend/services/kill_switch.py:527-528` — the `_log_disarmed_once`
docstring enumerates the operator-visible surfaces as "the `armed` flag on the API
response, the UNKNOWN badge in the UI, the /resume 409, and this log line". If the
fix adds a trading-path block, that enumeration is incomplete. Also
`OpsStatusBar.tsx:318-325` renders a DISARMED badge with no re-anchor promise (safe
as-is; verify at contract time).

---

### G. Test-fixture inventory (criterion 1 + the isolation mandate)

Two reusable fixtures exist in
`backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py`; **a new 36.12 test MUST use
one of them.**

| Fixture | Line | What it isolates | Use when |
|---|---|---|---|
| `ks_tmp_audit` | :108-131 | monkeypatches `ks._AUDIT_PATH` to a tmp tree; **the archive dir is DERIVED**, so `_audit_archive_dir()` follows. Yields `(ks_module, live_path, archive_dir)`. Leaves module-global `_state` untouched | boot-replay / `_load_from_audit` tests |
| `isolated_state` | :133-174 | monkeypatches `_AUDIT_PATH` **AND** installs a detached `_state` built with `object.__new__` **AND** resets `_disarmed_logged` | anything that calls `evaluate_breach`, `check_auto_resume`, or mutates state — **this is the one a 36.12 test wants** |
| `_live_audit_file_is_write_protected` | :176-195 | `autouse=True` byte-comparison of the REAL `handoff/kill_switch_audit.jsonl` before/after every test in the module | free if the new test lives in that module; **must be replicated if a new module is created** |

The docstring at :133-174 records the exact prior incident the caller warned about: "This
happened once during development (12 rows written, then removed)". If 36.12 gets its own
file (`backend/tests/test_phase_36_12_*.py`), **port the autouse write-protect fixture into
it** — otherwise the brace exists only in the 36.7 module.

`test_phase_23_2_5_kill_switch_no_false_fires.py:105` has a third,
`isolated_kill_switch_state`; older and narrower — prefer `isolated_state`.

**Existing tests that exercise `check_and_enforce_kill_switch` and therefore constrain the
fix** (all three are in the `-k 'kill_switch or paper_trader'` verification command's path):

| Test | Line | Pre-state | Why it constrains 36.12 |
|---|---|---|---|
| `test_paper_trader_check_and_enforce_kill_switch_no_breach` | `test_dod4_tier1_coverage_investment.py:952` | fresh `KillSwitchState()`, empty tmp audit → **both baselines None**; portfolio `total_nav == starting_capital == 100_000` | asserts `triggered is False`. Under a naive "block whenever pre-mutation `armed` is False", this book — which is indistinguishable from a first-ever boot — must still trade. **This test IS criterion 3.** |
| `test_paper_trader_check_and_enforce_kill_switch_breach_triggers_flatten` | `:968` | `sod=100_000` pre-set, **`peak` still None**, nav `90_000` | asserts `triggered is True` + flatten + pause. `armed` is False here (trailing leg missing) while a REAL daily breach exists. **A disarmed-block placed BEFORE the breach branch would suppress a real flatten and break this test.** Ordering is load-bearing — see the recommendation. |
| `test_phase_38_1_1_check_and_enforce_kill_switch_invokes_auto_resume` | `test_phase_38_1_kill_switch_auto_resume.py:178` | fresh state, empty audit, `total_nav == starting_capital == 100_000` | asserts `result["auto_resume"]["action"] == "no_op"`; an early return on the disarmed path would drop the `auto_resume` key entirely |

---

## Application to pyfinagent

### Recommendation for criteria 2 + 4 — take this position

**Do BOTH, but in a specific order, and do NOT blanket-reorder the mutations.**

```
pre = evaluate_breach(nav, ...)            # (1) MEASURE FIRST, pre-mutation
if pre["any_breached"] and not paused:     # (2) a REAL breach still wins, unchanged
    flatten + pause; return
if not pre["armed"] and not first_ever_boot(...):   # (3) NEW: unknown != healthy
    anchor_with_distinguishable_audit_event()
    P1 alert
    return {"triggered": False, "armed": False, "blocked": True, "breach": pre, ...}
# (4) otherwise: existing ratchet / SOD roll / post-mutation evaluate, byte-identical
```

Five reasons, each load-bearing:

1. **Blanket "measure before mutate" is WRONG here and would create a new false-positive
   flatten.** The SOD daily roll at `:1089-1090` is a *legitimate* pre-measurement mutation:
   a daily-loss limit is by definition measured from today's open. Evaluating before the
   roll computes `(yesterday_sod - today_nav) / yesterday_sod` — a multi-day move misread as
   a same-day loss. That is exactly step **36.9 finding (1)** with the sign flipped, and at
   the live book's numbers 36.9 measured it as `daily_loss_pct = 4.0` *exactly* — i.e. it
   would fire `flatten_all` on the first cycle after a restart. The `update_peak` ratchet is
   likewise legitimate and provably harmless to reorder (the ratchet only fires when
   `nav > peak`, at which moment `trailing_dd_pct` is negative either way). **Only the
   `None → anchor` branch is the defect.** The contract must say this explicitly or an
   executor will "fix" the roll and ship a live-money regression.
2. **Measure-before-mutate ALONE is insufficient** — criterion 2's first option cannot stand
   on its own. With both baselines `None`, the pre-mutation `evaluate_breach` returns
   `any_breached=False` **and** `armed=False`: the drawdown is *unmeasurable*, not
   measurable-and-forgiven. Reordering changes no observable unless something branches on
   `armed`. So criterion 2 is only satisfiable via its second option (decline to place
   orders), with the reorder as the enabling mechanism.
3. **The breach branch must keep precedence over the disarmed branch.** With one baseline
   present and one missing, a REAL breach on the surviving leg must still flatten. Putting
   the disarmed-block first suppresses it — and `test_dod4...breach_triggers_flatten` (`:968`)
   is exactly that state, so the wrong order is caught by an existing test. This also
   satisfies criterion 6 for free: `evaluate_breach` internals stay byte-untouched and the
   surviving leg keeps enforcing.
4. **BLOCK, do not PAUSE — and this is a trap, not a preference.** `state.pause()` is a
   latching, audited transition requiring an operator resume, and
   `paper_trading.py:593` **409s the resume while `armed` is False**. A trading path that
   pauses on disarmed creates a circular wedge: resume needs armed baselines, and the
   anchor that would produce them lives on the path that just refused to run. That is
   36.9 finding (3)'s wedge reached by a new route. The correct shape is a **non-latching,
   per-cycle refusal to place NEW orders**, with existing positions untouched — which is
   also the module's own documented semantics (`kill_switch.py:13`: "Pause = halt new
   entries; existing positions kept").
5. **Do not flatten on absence.** `evaluate_breach`'s docstring already argues this
   (kill_switch.py:448-452): flattening a healthy book because a housekeeping sweep moved a
   file is a *new destructive behaviour*, not a conservative one. External corroboration is
   thin in the opposite direction too — NYSE Pillar degrades one leg on missing input rather
   than halting (F3). Block new entries; keep the book.

**Honest statement of what this does NOT do, which the contract should pre-empt so Q/A does
not score it as a miss:** the true historical peak is still lost. 36.12 converts a *silent*
forgiveness into a *loud, audited, order-blocking* forgiveness. Recovering the real
high-water mark from the archives is **36.8's** job; deliberately re-anchoring it downward
is the operator's, behind `KS-PEAK-RESET`.

### First-ever-boot discriminator (criterion 3) — the crux

`armed is False` is TRUE on a genuinely new book too, so the block needs a provenance test
or it deadlocks a new book (and breaks `test_dod4...no_breach:952`). Galera's pattern (F6)
needs a marker written in advance, which today's book does not have. Ranked options:

| | Discriminator | Verdict |
|---|---|---|
| **D1** | **Audit-stream provenance**: did `_read_audit_rows()` ever yield a `sod_snapshot` / `peak_update` / `peak_reset` row across ALL sources? Zero ⇒ never anchored ⇒ possibly new. | **Recommended.** Reuses existing machinery, no schema change, no BQ round-trip. **Note the 36.8 collision** — 36.8 rewrites `_audit_source_paths` / the merge, so a helper built on `_read_audit_rows` shares its surface. |
| **D2** | **Book provenance**: `total_nav != starting_capital` ⇒ the book has traded ⇒ NOT new. Free (both fields are already in the `portfolio` dict at `:1077`). | **Recommended as an AND-companion to D1.** Verified compatible with all three existing tests, which all use `total_nav == starting_capital == 100_000`. Weak alone (a book can sit exactly at starting capital); strong in conjunction. On the LIVE book (`nav ≈ 23838`, `starting_capital` differs) it correctly reads "has history". |
| **D3** | **Explicit initialization sentinel** — a one-time `book_initialized` / `baseline_epoch` audit event, the Galera `safe_to_bootstrap` shape. | **Reject for 36.12.** The live book has no such row, so it would be misclassified as new — the exact failure the step is fixing. File as a follow-up for books created after it ships. |

Proposed rule: `first_ever_boot = (D1: no baseline row has EVER existed) AND (D2: nav ==
starting_capital)`. Ambiguity therefore resolves to *lost-history → block*, which is the
conservative direction. State the D2 residual (a book coincidentally at exactly starting
capital with a wiped audit trail trades one cycle unprotected) in the contract rather than
hiding it.

### Criterion 5 — assert no route to `reset_peak`

Two independent assertions, both cheap:
1. After the new path runs against `ks_tmp_audit`/`isolated_state`, read the tmp audit file
   and assert **zero rows with `event == "peak_reset"`**.
2. `monkeypatch.setattr(ks.KillSwitchState, "reset_peak", _raises)` for the duration of the
   test, so an added call fails loudly rather than being a no-op (it is DARK today —
   `settings.kill_switch_peak_reset_enabled=False` — so a call would silently return `None`
   and assertion 1 alone would pass. **Assertion 1 without assertion 2 is a guard that
   cannot fail** — see `feedback_mutation_test_guards_and_fixtures`).

The new anchor event must also be replay-safe: if a new event name is introduced,
`_load_from_audit` must handle it as a **ratchet** (`max`), never as an authoritative
downward move — `peak_reset` is the only assignment-semantics event (kill_switch.py:247-255)
and that must stay true.

### Criterion 7 — the mutation that must fail

Reverting the ordering (moving `evaluate_breach` back below `update_peak`/`update_sod_nav`)
makes `pre["armed"]` structurally `True`, the block never fires, and the new test must go
red. That is a genuine mutation, not a tautology. **Also mutate the discriminator**: force
`first_ever_boot` to always-True and assert the lost-history test goes red; force it to
always-False and assert the first-boot test goes red. Both directions, per
`feedback_mutation_test_guards_and_fixtures`.

### Criterion 8 — the strings, plus a fourth site

The three listed strings (§F) plus **`kill_switch.py:527-528`**, whose docstring enumerates
the operator-visible surfaces and will be incomplete once a trading-path block exists. Check
`OpsStatusBar.tsx:318-325` too (currently makes no re-anchor promise — likely fine as-is,
confirm at contract time). Suggested grep guard for the "no test asserts a promise remains"
half: assert the strings `re-anchors them`, `re-anchors both baselines`, and
`next cycle re-anchor` appear **zero** times across `backend/api/paper_trading.py` and
`frontend/src/components/KillSwitchPanel.tsx`. A grep-based guard is weak on its own (it is
a source-scan, the class `feedback_mutation_test_guards_and_fixtures` warns about) — pair it
with a behavioural test that the 409 body names the NEW behaviour.

### Scope-adjacent defect found while auditing — DO NOT silently absorb

`grep -rn "\.execute_buy("` shows **`backend/agents/mcp_servers/signals_server.py:444`**
calls `paper_trader.execute_buy(...)` directly, and `is_paused()` is consulted in exactly two
places repo-wide (`paper_trader.py:1097`, `autonomous_loop.py:1287`). **`execute_buy` itself
has no kill-switch gate.** So an order placed via the signals MCP path bypasses the kill
switch entirely, paused or not — a textbook CWE-424 alternate path, and it means a block
implemented in `autonomous_loop.py` alone is incomplete. This is **out of 36.12's stated
scope**; per `feedback_queue_discovered_defects_in_masterplan` it should get its own
research-gated step rather than a prose mention. Flagging it here so the contract can
(a) place 36.12's block in `check_and_enforce_kill_switch`'s return + the loop's existing
halt branch, and (b) state plainly that the MCP path remains ungated and why.

### Overlap map with the other open 36.x steps (mandatory check)

| Step | Status | Lines it intends to change | Collision with 36.12 |
|---|---|---|---|
| **36.7** | `pending` **but the code is already committed** (`b0abb061 wip(36.7,80.40): ... NO Q/A VERDICT YET (3x API 529)`; working tree clean for all kill-switch files) | `kill_switch.py` (shipped) | **Sequencing, not code.** 36.12 is built entirely on 36.7's `armed` flag. If 36.7's Q/A ever returns FAIL and the commit is reverted, 36.12 evaporates. Contract should state the dependency. |
| **36.8** (P0) | pending | `kill_switch.py:98-107` `_audit_source_paths`, the boot merge, peak authority; proposes "an explicit re-anchor audit event that the merge must respect as authoritative" | **REAL COLLISION, two ways.** (a) 36.12's D1 discriminator reads `_read_audit_rows()`, whose sources 36.8 rewrites. (b) **Both steps want to introduce a distinguishable re-anchor audit event.** Agree ONE event schema across both, or the second step will find the first one's event and have to reconcile. Recommend 36.12 define the event and 36.8 consume it. |
| **36.9** (P0) | pending | `evaluate_breach` (stale `sod_date`; `armed` computed before the `nav_invalid` early return; `sod_nav=0.0`) **and explicitly `paper_trader.py`'s `sod_nav is None` re-anchor test at `:1089`** | **REAL COLLISION on `paper_trader.py:1089`.** 36.9 finding (3) wants that condition widened from `is None` to `is None or <= 0`; 36.12 restructures the same statement. Also: 36.9 finding (1) currently asserts "The destructive cycle path re-anchors before evaluating (paper_trader.py:1080/1089) and is unaffected" — **36.12 makes that sentence false**, so whichever lands second must re-scope. Recommend 36.12 first (it owns the restructure) and 36.9 rebased onto it. |
| **36.10** (P1) | pending | away-ops paging scripts, `paper_trading.py`'s second audit-path constant, `_disarmed_logged` reset | **Soft.** 36.12 adds a new operator-facing signal (blocked cycle + P1). Coordinate so the paging surface 36.10 builds also carries "cycle blocked, disarmed", not just `paused`. 36.12's new path will also hit `_log_disarmed_once`, whose one-shot flag 36.10 is fixing. |
| **36.11** (P2) | pending | fail-CLOSED tests for `breach.get('armed', True)` at `paper_trading.py:593` and `kill_switch.py:601`; unrelated cockpit-helpers threshold conflict | **Soft but decide now.** If 36.12 adds a third `armed` consumer, pick the default deliberately: the call is same-process so the key is always present — use **`breach["armed"]` (direct index)**, not `.get(..., True)`, so no third fail-open default is created for 36.11 to litigate. Say so in the contract. |

### Carry-over from the 36.7/80.40 gate (`handoff/current/research_brief_36.7_80.40.md`)

**Carries over (do not re-derive):**
- §A.3 is the origin of this step; its mechanism reading matches what I re-verified at
  source today, line for line.
- §A.3b's per-leg-independence bound — re-verified at `kill_switch.py:475-476`, `:498`,
  `:504`. Now additionally corroborated externally by NYSE Pillar's per-leg degradation (F3).
- Source #5 of that brief (15c3-5 via Cornell) and its measured negative — *"No text
  prohibiting a control from silently self-disabling"*. I re-fetched it independently today
  and reached the same conclusion; F2 above refines it.
- Source #4 (FIA July 2024 whitepaper): "the automated trader should not be able to override
  a kill switch invoked by the broker" and "this functionality should serve in addition to
  and as a final backstop". Directly supports criterion 5. Not re-fetched (carried).
- The repo's own catalogued anti-pattern `project_fabricated_safe_80_36` —
  "discriminate on PRESENCE, never on VALUE" — is the same family as this defect and the
  frontend already implements it correctly (`KillSwitchPanel.tsx:134-137`).

**Does NOT carry over:**
- That brief's Microsoft circuit-breaker source concluded breaker state is *derived* and
  auto-resets periodically. That is a fine model for an RPC breaker and an actively
  misleading one here: F4 shows financial kill switches re-arm only by human action. Do not
  cite the circuit-breaker pattern as support for any auto-re-anchor.
- Its arXiv:2601.14059 NaN material (80.40 scope) is irrelevant to 36.12.
- Its 12-source read-in-full table is that step's gate evidence, not this one's — this brief
  re-cleared the floor independently (10 sources, 2 shared with it: Cornell 15c3-5, and the
  FIA paper carried as a citation rather than a re-fetch).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (or curl+pdfplumber) — **10**
- [x] 10+ unique URLs total (incl. snippet-only) — **34**
- [x] Recency scan (last 2 years) performed + reported — 4 year-scoped passes, 3 findings + 1 non-transferring
- [x] Full pages/papers read, not abstracts — the one abstract-only hit (arXiv:2506.01782) is explicitly excluded from the count
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (`paper_trader`, `kill_switch`, `autonomous_loop`, `paper_trading` API, `risk_server` MCP, both frontend components, 5 test modules, masterplan 36.7-36.12)
- [x] Contradictions / consensus noted — F3 is an explicit counter-example to the fail-closed consensus; F1 records that the expected named anti-pattern does **not** exist
- [x] All claims cited per-claim
- [ ] **Gap, disclosed:** question 5 (operator text in lockstep with behaviour) returned no usable literature — F7 records the null result rather than padding
- [ ] **Gap, disclosed:** no source read in full establishes the *first-boot vs lost-history* discriminator for a system with no pre-existing marker; Galera (F6) needs a marker written in advance, so D1/D2 are reasoned from first principles, not cited
- [ ] **Not verified:** I did not run any test (per the caller's constraint), so the claim that the three existing tests behave as tabled in §G is read from source, not executed. The contract's cycle 1 must actually run them.

**Read-only compliance:** no production code or state was written. Only writes this session
were this brief and two PDFs into the session scratchpad. No test executed. No POST to
`:8000`, no `:3000` traffic, no masterplan edit, no commit.
`handoff/kill_switch_audit.jsonl` md5 verified **unchanged**: `ce8fb93348bb9a3bbe26f2d91b1bc05e`
(checked at brief start and at brief end).

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 13,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "Gate cleared. Defect re-verified at source: paper_trader.py:1080/:1089-1090 mutate both baselines before evaluate_breach at :1092, so `armed` is structurally always True on the order-placing path; :1097 branches only on any_breached. Only production caller is autonomous_loop.py:1285. All four `armed` consumers gate resume or display -- 36.10's claim is right in substance (one counter-example: kill_switch.py:601 is a service, not UI/API); the accurate restatement is `no order-placing path reads armed`. Externally: no named CWE covers a guard that mutates its own datum (CWE-367 explicitly excludes it); the defensible framing is CWE-424 alternate path + CWE-223 omitted audit information + Saltzer-Schroeder fail-safe defaults, with Faramesh 2026 the crispest `absent decision maps to DENY`. 15c3-5 never mandates blocking an unevaluable control -- do not overstate it. ADVERSARIAL: NYSE Pillar documents failing OPEN on a missing price reference, which supports keeping per-leg independence (criterion 6). Nasdaq + NYSE both require human out-of-band re-arm, corroborating criterion 5. RECOMMENDATION: block, do not pause (pausing wedges against the /resume 409); keep the breach branch ahead of the disarmed branch; and do NOT blanket-reorder -- reordering the SOD daily roll would misread a multi-day move as a 4% daily loss and fire a false flatten. Real collisions with 36.8 (shared re-anchor event) and 36.9 (same line :1089).",
  "brief_path": "handoff/current/research_brief_36.12.md",
  "gate_passed": true
}
```
