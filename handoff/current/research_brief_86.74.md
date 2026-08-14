# Research Brief -- step 86.74

**Topic:** Correct enforcement of a risk-gate veto at the position-sizing boundary
(falsy-zero REJECT inverted into a max-size allocation).
**Tier:** moderate (caller-stated). **Audit-class:** NO (coverage reported for information only).
**Researcher:** Layer-3 combined external-literature + internal-code explorer.
**Started:** 2026-08-14.

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 19,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

**Sources counted toward the gate (all 7 via WebFetch, accessed 2026-08-14):**

1. https://www.law.cornell.edu/cfr/text/17/240.15c3-5
2. https://web.mit.edu/Saltzer/www/publications/protection/Basic.html
3. https://python-patterns.guide/python/sentinel-object/
4. https://arxiv.org/html/2604.01483v1
5. https://arxiv.org/html/2605.14744v1
6. https://arxiv.org/html/2503.18666v3
7. https://www.finra.org/rules-guidance/key-topics/market-access

*Note on tier length:* this brief exceeds the nominal `moderate` word budget.
The overrun is entirely in the internal-inventory half (sections 1.5-1.7),
which surfaced two defects beyond the one the caller described -- a three-flag
conjunction with an ordering hazard, and three unguarded `or 10.0` sites on the
swap path. Truncating those to hit a word count would have hidden findings that
change the shape of the fix. External analysis is held to tier depth.

## Status log (append-only)

- 2026-08-14 -- brief created, envelope born INCOMPLETE. Starting internal
  exploration + external search in parallel.
- 2026-08-14 -- round 1: internal seams read (portfolio_manager.py :296-365,
  :460-579, :939-999; incident memo). External round 1: 3 searches, 3 fetches
  attempted, 2 landed (Cornell CFR 15c3-5, python-patterns sentinel).
  federalregister.gov 302-redirects to an unblock interstitial -- retry via CFTC.

---

# 1. Internal code inventory (the Explore half)

All anchors verified by direct read on 2026-08-14 against the working tree.

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/portfolio_manager.py` | 939-955 | `_extract_position_pct` -- the defect | LIVE, defective |
| `backend/services/portfolio_manager.py` | 499-510 | sizing seam (`or 10.0` vs `is not None`) | LIVE, flag-gated fix present |
| `backend/services/portfolio_manager.py` | 307-331 | judge-view resolution + explicit-0.0 recovery | LIVE, flag-gated |
| `backend/services/portfolio_manager.py` | 350-372 | BINDING REJECT gate (`paper_risk_judge_reject_binding`) | LIVE, flag default-OFF |
| `backend/services/portfolio_manager.py` | 556-579 | main-path `TradeOrder` emit | LIVE |
| `backend/services/portfolio_manager.py` | 901-919 | swap-path `TradeOrder` emit (2nd BUY seam) | LIVE |
| `backend/config/settings.py` | 342, 350 | both flags defined, both default `False` | LIVE |
| `backend/api/settings_api.py` | 283 | `_FIELD_TO_ENV` mapping for the shape-fix flag | LIVE |
| `.claude/masterplan.json` | 19419-19423 | step 79.1 OPERATOR ACTION -- promote the flags | PENDING |

## 1.1 The defect, verbatim

`backend/services/portfolio_manager.py:939-955`:

```python
def _extract_position_pct(risk_assessment: dict, analysis: dict) -> Optional[float]:
    """Extract recommended position % from risk assessment."""
    # Try risk_judge output
    pct = risk_assessment.get("recommended_position_pct")
    if pct:                       # <-- 0.0 is FALSY
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    # Fall back to analysis-level field
    pct = analysis.get("risk_judge_position_pct")
    if pct:                       # <-- same
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    return None                   # REJECT/0% now == "no judge ran"
```

`backend/services/portfolio_manager.py:504-507`:

```python
if getattr(settings, "paper_risk_judge_shape_fix_enabled", False):
    position_pct = cand["position_pct"] if cand["position_pct"] is not None else 10.0
else:
    position_pct = cand["position_pct"] or 10.0  # Default 10% if Risk Judge didn't specify
```

**The type signature is the bug.** `Optional[float]` has exactly one channel
(`None`) for what the domain needs three of: *judge said 0% (REJECT)*, *judge
said N%*, *no judge ran*. The function collapses state 1 into state 3, and the
consumer's default for state 3 is the **maximum** position. Note the collapse
happens even before `or 10.0`: `_extract_position_pct` itself is lossy, so
**fixing only line 507 is insufficient** -- which is exactly why the phase-66.2
fix needed a *second* patch at :324-330 to re-read the raw value the extractor
had already destroyed.

## 1.2 The recovery patch is a workaround, not a repair

`portfolio_manager.py:321-330` calls `_extract_position_pct`, then immediately
re-reads the same raw key to undo the extractor's information loss:

```python
position_pct = _extract_position_pct(_rj_view, analysis)
if getattr(settings, "paper_risk_judge_shape_fix_enabled", False):
    _raw_pct = _rj_view.get("recommended_position_pct")
    if _raw_pct is not None:
        try:
            position_pct = float(_raw_pct)
        except (ValueError, TypeError):
            pass
```

Two observations for PLAN:
1. The recovery only covers `_rj_view["recommended_position_pct"]`. The
   **second** source `analysis["risk_judge_position_pct"]` (:949) keeps its
   falsy-zero check under every flag setting -- an explicit 0.0 arriving on
   that path still becomes `None` -> `10.0`. That is a residual hole in the
   approved fix.
2. `except (ValueError, TypeError): pass` at :329-330 silently leaves
   `position_pct` at whatever the lossy extractor returned. A malformed judge
   pct therefore also lands on the 10% default. Fail-**open** on parse error.

## 1.3 Two BUY seams, one gate

`decide_trades` emits BUY orders from **two** places: the main loop
(:556-579) and the swap path (:901-919). The phase-57.1 comment at :350-357
states the binding gate was deliberately placed at the candidate-build
chokepoint because "the away week executed 3 REJECT BUYs -- all via the swap
path", and explicitly cites "SEC 15c3-5(d) non-bypassable placement". So the
codebase has **already adopted the regulatory reasoning** this brief is asked
to research; the open question is whether one chokepoint is sufficient.

Note the swap path reads `cand["position_pct"]` too but its sizing is computed
earlier in that function -- PLAN should confirm whether the swap seam's sizing
also passes through the `or 10.0` idiom.

## 1.4 Configuration state

Both flags ship default-OFF (`settings.py:342, :350`;
`test_phase_66_2_risk_judge_shape.py:141` asserts the default is `False`).
Masterplan **79.1** is an `[OPERATOR ACTION, pending]` to append
`PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true` to `backend/.env` and restart. The
step text records the measured state: the two lines are **absent from
`backend/.env`**, approval was granted 2026-07-09 and never consumed.

**Design consequence:** the fix for a safety inversion is currently behind a
default-OFF flag whose promotion requires a human action that has not happened
in 5 weeks. Section 4 below argues from the literature that this is itself the
wrong shape -- fail-closed behaviour should not be opt-in.

## 1.5 THE HEADLINE INTERNAL FINDING -- three default-OFF flags, and one of them makes things WORSE alone

There is a **third** flag, not named in the caller's scope:
`paper_risk_judge_parse_fail_reject` (`backend/config/settings.py:346`,
phase-75.14). Its own description states the failure mode verbatim:

> "OFF (default) = byte-identical legacy fallback: a garbled/empty judge
> response silently becomes APPROVE_REDUCED at 3% NAV. ON = the fallback
> verdict is REJECT with recommended_position_pct 0 -- fail-safe: an
> unparseable risk gate should not approve. **NOTE the True-path REJECT only
> actually blocks the BUY when shape_fix (full path) or reject_binding (lite
> path) is ALSO on; on the all-OFF default even a REJECT verdict may not
> bind.**"

`backend/agents/risk_debate.py:127-167` implements that fallback and returns
`"recommended_position_pct": 0` (`:152`) on the fail-safe branch.
`risk_debate.py:345` nests the judge verdict under the `"judge"` key.

**Trace the fail-safe value through the extractor with `parse_fail_reject=ON`
but `shape_fix=OFF` (a legal, one-flag-at-a-time promotion):**

1. `risk_debate` writes the fail-safe `0` into `risk_assessment["judge"]`.
2. `portfolio_manager.py:316` -- flag OFF, so `_rj_view` stays *top-level*; the
   nested `0` is never seen. `_extract_position_pct` returns `None`.
3. `portfolio_manager.py:507` -- `None or 10.0` -> **10.0**.

So promoting the flag *named* "fail-safe" **alone** turns an unparseable risk
gate from a 3%-NAV position into a **10%-NAV position**. The safety flag makes
the failure strictly worse. Even with `shape_fix=ON` the value only survives
because of the :324-330 workaround, not because the extractor is correct.

**The safety property is the conjunction of three independently-defaulted-OFF
flags** (`shape_fix`, `reject_binding`, `parse_fail_reject`), with a non-obvious
ordering constraint between them. This is the single most important thing for
PLAN to absorb: a partial promotion is not a partial fix, it is a regression.

## 1.6 There is no second line of defence at the execution seam

`backend/services/paper_trader.py` accepts the judge fields as parameters --
`risk_judge_decision` (`:243`) and `risk_judge_position_pct` (`:245`) -- and
writes them to storage (`:432`, `:489`, `:513`, `:677`). Grep for `REJECT` in
that file returns **nothing**. `execute_buy` therefore **records** the verdict
and **never enforces** it. The order-submission seam is a recorder, not a gate.

Answering the caller's question (d) empirically: today the veto is enforced at
**one** seam (candidate-build, and only under flags), and at **zero** seams on
the submission path.

## 1.7 SECOND HEADLINE FINDING -- the `or 10.0` idiom occurs at FOUR sites; the approved fix guards ONE

`grep -n "or 10.0" backend/services/portfolio_manager.py` (run 2026-08-14):

| Line | Context | Guarded by `shape_fix`? |
|---|---|---|
| `:507` | main BUY loop sizing | **YES** (`:504-505` branch) |
| `:800` | cross-sector rotation, `buy_amount` passed to `_cross_rotation_safe` | **NO** |
| `:853` | swap path, sector-NAV-cap projection | **NO** |
| `:878` | swap path, **actual `buy_amount` sizing** | **NO** |

Verbatim, `portfolio_manager.py:878-879` -- the swap path's real sizing:

```python
position_pct = cand.get("position_pct") or 10.0
buy_amount = nav * (float(position_pct) / 100.0)
```

**Consequence:** even with `paper_risk_judge_shape_fix_enabled=True`, a candidate
carrying an explicit `position_pct == 0.0` that reaches the **swap path** is
still sized at **10% of NAV**. The approved, operator-blessed, 5-weeks-pending
fix closes 1 of 4 sites.

This is not a hypothetical path. The phase-57.1 comment at `:350-357` records:

> "the away week executed 3 REJECT BUYs -- **all via the swap path**"

So the three unguarded sites are precisely the ones on the path with a measured
history of executing REJECTs. `:853` additionally means the sector-NAV cap
*projects* the wrong (10x too large) exposure for a 0% candidate, so the cap
itself is being evaluated against a fabricated number.

**This resolves open question 1 in the affirmative and escalates it: the swap
path is not merely unverified, it is unfixed.** Any contract that promotes the
66.2 flag without also closing `:800`, `:853`, `:878` will ship a fix that
demonstrably does not bind on the path where the failure was first observed.

Note this also strengthens the section-3 recommendation for a check at the
**submission** seam: four call sites already exist and a fifth can be added by
any future feature, which is exactly the condition Saltzer's *complete
mediation* ("Every access to every object must be checked for authority")
exists to rule out. A single chokepoint at `paper_trader.execute_buy` is
invariant to the number of upstream sizing sites.

---

# 2. External research

## 2.1 Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| Year-less canonical | `SEC Rule 15c3-5 market access rule pre-trade risk controls credit capital thresholds` |
| Year-less canonical | `FIA pre-trade risk controls best practices order gating kill switch exchange` |
| Year-less canonical | `null object pattern sentinel value versus absent Optional None zero fail-safe defaults software safety` |
| Year-less canonical | `XACML Indeterminate versus Deny deny-overrides combining algorithm policy decision point` |
| Last-2-year | `algorithmic trading risk control bypass incident 2025 automated order gate fail-open position limit` |
| Current-year | `LLM agent guardrails financial trading veto enforcement 2026 arxiv risk gate` |

## 2.2 Read in full (>=5 required; counts toward the gate)

All accessed 2026-08-14.

| # | URL | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| 1 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | regulation (peer of official) | WebFetch | Operative text of the Market Access Rule: controls must "prevent the entry of orders ... by rejecting orders"; "direct and exclusive control". |
| 2 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | peer-reviewed (Proc. IEEE 1975) | WebFetch | **Fail-safe defaults**: "Base access decisions on permission rather than exclusion." A default-permit mistake "fails dangerously by granting access (potentially unnoticed)". |
| 3 | https://python-patterns.guide/python/sentinel-object/ | authoritative blog (Brandon Rhodes) | WebFetch | Sentinel identity, not value, must carry "absent": "it is the object's identity -- *not* its value -- that lets the surrounding code recognize its significance." |
| 4 | https://arxiv.org/html/2604.01483v1 | preprint (2026) | WebFetch | Ties 15c3-5 to fail-closed agent architecture; "The Orchestrator node intercepts this API call before it reaches the execution environment"; absence of proof = rejection. |
| 5 | https://arxiv.org/html/2605.14744v1 | preprint (2026) | WebFetch | **Governance-task decoupling**, measured: text-only governance CDL 0.273 vs mechanical 0.074 (-73%); "When the model that must comply with a policy also interprets what compliance means, the policy becomes a proxy target." |
| 6 | https://arxiv.org/html/2503.18666v3 | preprint, ICSE'26 | WebFetch | Runtime enforcement > pre-execution assessment; hard `stop` action; >90% unsafe executions prevented; ~2.83ms overhead. **[ADVERSARIAL -- see 2.4]** |
| 7 | https://www.finra.org/rules-guidance/key-topics/market-access | official regulator | WebFetch | "'unfiltered' or 'naked' sponsored access has effectively been prohibited, since the rule requires that a broker-dealer apply these controls on a pre-trade basis." |

**Supplementary verified read (NOT counted toward the gate):**
`https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-17?chapter=II&part=240&section=240.15c3-5`
-- fetched by `curl` + tag-strip to obtain the complete authoritative rule text,
because the WebFetch summary of source #1 asserted the rule lacks certain
language. The eCFR full text **confirms** the summary on that point (the phrase
"automated, pre-trade basis" is in the adopting release, not the rule text; the
rule says "on an order-by-order basis or over a short period of time"). Counted
as verification, not as a gate source.

## 2.3 Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0 | official | **HTTP 403** to WebFetch |
| https://www.federalregister.gov/documents/2020/07/15/2020-14381/electronic-trading-risk-principles | official | **302** to an `unblock.federalregister.gov` interstitial |
| https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | industry | binary PDF; floor already cleared, not worth the pypdf round-trip at moderate tier |
| https://www.fia.org/fia/articles/fia-releases-best-practices-automated-trading-risk-controls-and-system-safeguards | industry | landing page for the above |
| https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access | official | adopting release; rule text already obtained via eCFR |
| https://www.nasdaqtrader.com/content/productsservices/trading/ften/sec_mar.pdf | industry | PDF |
| https://www.finra.org/rules-guidance/guidance/reports/2022-finras-examination-and-risk-monitoring-program/market-access-rule | official | duplicate of #7's scope |
| https://arxiv.org/html/2605.19337v1 | preprint | "Agentic Trading: When LLM Agents Meet Financial Markets" -- survey, adjacent |
| https://arxiv.org/pdf/2605.27333 | preprint | FinHarness -- inline lifecycle safety harness; PDF |
| https://arxiv.org/abs/2606.05805 | preprint | TRIAD -- proceed/refuse/update tri-state guardrail |
| https://arxiv.org/pdf/2502.11448 | preprint | AGrail -- adaptive guardrail |
| https://arxiv.org/pdf/2509.23614 | preprint | PSG-Agent -- personality-aware guardrail, off-topic |
| https://www.sciencedirect.com/science/article/pii/S0167642313001238 | peer-reviewed | "The Logic of XACML" -- paywalled |
| https://arxiv.org/pdf/1110.3706 | preprint | "The Logic of XACML - Extended" -- PDF |
| https://arxiv.org/pdf/1206.5327 | preprint | XACML 3.0 in Answer Set Programming |
| https://insights.wisdomchain.com/agentic-trading-evidence-ledger/ | community | low tier |
| https://questdb.com/glossary/algorithmic-risk-controls/ | community | glossary |
| https://www.nasdaq.com/articles/fintech/regulatory-roundup-september-2025 | industry | 2025 enforcement roundup |
| https://www.wilmerhale.com/en/insights/client-alerts/sec-staff-issues-first-set-of-faqs-on-rule-15c3-5-risk-management-controls-for-brokers-or-dealers-with-market-access | industry (law firm) | secondary to the SEC FAQ |

**urls_collected = 7 read-in-full + 19 snippet-only + 1 curl-verified = 27.**

## 2.4 Recency scan (last 2 years, 2024-2026)

Performed. **Result: 4 new findings that COMPLEMENT and one that QUALIFIES the
canonical sources.** The 2024-2026 window has produced a distinct literature on
*enforcing* LLM-agent verdicts mechanically, which did not exist when
Saltzer & Schroeder (1975) or Rule 15c3-5 (2010) were written:

1. **arXiv:2605.14744 (2026)** supplies the measured vocabulary for this exact
   defect: *governance-task decoupling*. Text-only governance leaves 27.3% of
   deferrals "cosmetic" (vacuous); mechanical enforcement cuts that to 7.4%.
   pyfinagent's judge produced a correct REJECT that the executing code did not
   honour -- a textbook instance.
2. **arXiv:2604.01483 (2026)** is the first source found that explicitly derives
   an agent-architecture requirement *from* Rule 15c3-5, arguing a probabilistic
   guardrail cannot satisfy "direct and exclusive control".
3. **arXiv:2503.18666 (ICSE 2026)** establishes that pre-execution risk
   *assessment* without runtime *enforcement* is the dominant industry gap --
   which is precisely pyfinagent's shape (the judge assesses; nothing enforces).
4. **arXiv:2606.05805 (TRIAD)** moves guardrails from binary allow/deny to a
   tri-state `proceed / refuse / update`, corroborating from a different angle
   that a two-valued channel (here: `Optional[float]`) is under-specified.
5. **[ADVERSARIAL / qualifying]** arXiv:2503.18666 also *exhibits the very bug
   class under study*: its rules fire only when "every predicate evaluates to
   true", so "If predicates fail to evaluate, the rule does not trigger --
   effectively defaulting to permissive behavior rather than rejection." A
   state-of-the-art 2026 runtime-enforcement framework is itself fail-open on an
   indeterminate check. **Finding: adopting an enforcement framework does not
   by itself buy fail-closed semantics; the indeterminate case must be
   specified separately.** This is the strongest caution against a fix that
   only adds a gate without defining absent-verdict behaviour.

No source found in the window *contradicts* fail-closed defaulting at a risk
boundary; the consensus is uniform. The qualification above is about
implementation completeness, not direction.

## 2.5 Key findings, cited per claim

**F1 -- Deny-by-default is the founding principle, and the asymmetry of failure
is the stated reason.** "Base access decisions on permission rather than
exclusion." A mistake in a permission-based mechanism "fails safely by denying
access (quickly detected), whereas a mistake in an exclusion-based mechanism
fails dangerously by granting access (potentially unnoticed)"
(Saltzer & Schroeder, *The Protection of Information in Computer Systems*,
https://web.mit.edu/Saltzer/www/publications/protection/Basic.html, accessed
2026-08-14). pyfinagent's `or 10.0` is exclusion-based: it grants the maximum
unless something affirmatively objects, and it went unnoticed for weeks --
exactly the predicted failure signature.

**F2 -- The regulation is written in permission form, not exclusion form.**
17 CFR 240.15c3-5(c)(2)(i) requires controls reasonably designed to "Prevent the
entry of orders **unless** there has been compliance with all regulatory
requirements that must be satisfied on a pre-order entry basis"
(https://www.law.cornell.edu/cfr/text/17/240.15c3-5, accessed 2026-08-14;
full text verified against eCFR). The order is blocked *unless* affirmative
compliance exists -- absence does not permit. (c)(1)(i) is the sizing analogue:
prevent orders exceeding thresholds "by **rejecting** orders if such orders
would exceed the applicable credit or capital thresholds."

**F3 -- Controls must be non-bypassable and applied pre-trade to all flow.**
15c3-5(d): controls "shall be under the direct and exclusive control of the
broker or dealer". FINRA: "'unfiltered' or 'naked' sponsored access has
effectively been prohibited, since the rule requires that a broker-dealer apply
these controls on a pre-trade basis"
(https://www.finra.org/rules-guidance/key-topics/market-access, accessed
2026-08-14). A gate reachable on only one of two BUY paths, or defeatable by a
default-OFF flag, is the software analogue of unfiltered access.

**F4 -- "Absent" must be carried by identity, not by a value in the value
domain.** "it is the object's identity -- *not* its value -- that lets the
surrounding code recognize its significance"; a store "doesn't have the option
of using `None` for missing data if users might themselves try to store the
`None` object" (Rhodes, *The Sentinel Object Pattern*,
https://python-patterns.guide/python/sentinel-object/, accessed 2026-08-14).
The generalisation that bites here: **whenever one channel means two things,
one of them needs its own name.** `Optional[float]` gives `None` two jobs
(judge-said-zero, judge-absent) and the consumer resolves the ambiguity in the
unsafe direction.

**F5 -- Advisory governance measurably fails to bind; mechanical enforcement
is the fix.** "When the model that must comply with a policy also interprets
what compliance means, the policy becomes a proxy target"; mechanical
enforcement "reduces CDL to 0.074 (-73%), raises DIU from 0.298 to 0.766"
(arXiv:2605.14744, https://arxiv.org/html/2605.14744v1, accessed 2026-08-14).
Note the paper's own design puts gates at **two** stages -- "Pre-LLM are
evaluated before the model call" and a post-LLM gate that "overrides the model's
decision ... when information completeness is insufficient" -- i.e. layered.

**F6 -- The enforcement point belongs between the decision and the venue, and
absence of an approval is a rejection.** "The Orchestrator node intercepts this
API call before it reaches the execution environment"; permitted "if and only
if" the checker proves compliance, and when it "cannot verify the proof ... The
action is definitively blocked" (arXiv:2604.01483,
https://arxiv.org/html/2604.01483v1, accessed 2026-08-14). This is a direct
answer to question (d): the *authoritative* seam is the one adjacent to
execution, because that is the one nothing can route around.

**F7 -- Pre-execution assessment is not enforcement.** "most existing solutions
lack explicit safety enforcement mechanisms, focusing instead on pre-execution
risk assessments", leaving agents "vulnerable to runtime deviations ... as there
are no active constraints to prevent unsafe actions during execution"
(AgentSpec, https://arxiv.org/html/2503.18666v3, accessed 2026-08-14). Overhead
for real enforcement is ~2.83ms -- cost is not a defence.

**F8 -- Access-control systems already solve "reject vs no verdict" with a
three-valued decision and a deny-biased combiner.** XACML distinguishes `Deny`
from `Indeterminate`, and the deny-overrides algorithm resolves
`IndeterminateD` toward denial rather than permission (snippet-level, from the
XACML literature search; not read in full -- see 2.3). This is the canonical
prior art for question (c) and directly names what pyfinagent lacks: a
verdict type with a third state and a combination rule that is deny-biased.

## 2.6 Consensus vs debate

**Consensus (unanimous across regulator, 1975 security theory, and 2026 agent
literature):** at a safety boundary the default must be denial; absence of an
affirmative approval must not be read as approval; and enforcement must be
mechanical and non-bypassable rather than advisory.

**Genuine debate, relevant to PLAN:** *where* to place the gate.
- arXiv:2604.01483 argues for a **single authoritative chokepoint** adjacent to
  execution ("intercepts ... before it reaches the execution environment") --
  simplest to prove non-bypassable, and consonant with Saltzer's *economy of
  mechanism* ("Keep the design as simple and small as possible") and *complete
  mediation* ("Every access to every object must be checked for authority").
- arXiv:2605.14744 and the FIA/exchange practice surveyed above favour
  **layered** controls (pre-LLM gate + post-LLM override; firm-level +
  exchange-level + kill switch).
These are reconcilable: complete mediation demands *one* seam that nothing can
bypass; defence-in-depth adds *earlier* seams for better behaviour (budget
reallocation, cleaner logs), not as a substitute for the last one.

## 2.7 Pitfalls from the literature

1. **A guardrail framework does not imply fail-closed.** AgentSpec's rules
   don't fire when a predicate cannot be evaluated -- permissive by omission
   (arXiv:2503.18666). Specify the indeterminate case explicitly or inherit
   the bug.
2. **Self-interpreted policy becomes a proxy target** (arXiv:2605.14744) --
   don't let the sized/executing code re-derive what the judge "meant".
3. **A control that is too permissive is not a control.** 15c3-5's standard is
   "reasonably designed"; a threshold defaulting to the *maximum* fails it.
4. **Recording is not enforcing.** Persisting `risk_judge_decision` (as
   `paper_trader.py` does) satisfies 15c3-5(c)(2)(iv)'s post-trade surveillance
   limb, not the (c)(1)/(c)(2)(i) prevention limbs.

---

# 3. Application to pyfinagent

| Finding | Anchor | Implication for PLAN |
|---|---|---|
| F1, F4 | `portfolio_manager.py:939-955` | The repair is a **type change**, not a truthiness change. `Optional[float]` cannot carry three states. Options: a small enum/dataclass (`REJECT` / `Size(pct)` / `NoVerdict`), or a module-level sentinel `_ABSENT = object()` tested with `is`. Fixing `if pct:` -> `if pct is not None:` in place is necessary but *still* leaves `None` overloaded for the caller. |
| F2 | `portfolio_manager.py:504-507` | Invert the default's polarity: a BUY should require an affirmative size, not survive its absence. If PLAN keeps a default at all, `10.0` is the worst possible choice -- the regulation's shape is "prevent ... unless". |
| F1.2 residual | `portfolio_manager.py:949` | The second source `analysis["risk_judge_position_pct"]` keeps the falsy-zero check under **every** flag setting. The approved fix does not cover it. |
| F1.2 residual | `portfolio_manager.py:329-330` | `except (ValueError, TypeError): pass` leaves a malformed pct falling through to the 10% default -- fail-open on parse error, the same class as the parse-fail fallback issue in 1.5. |
| **1.5** | `settings.py:342/346/350`, `risk_debate.py:127-167,345` | **Blocker for PLAN:** three default-OFF flags whose conjunction is the safety property, and promoting `parse_fail_reject` alone escalates a parse failure from 3% to 10% NAV. Any contract must treat promotion as atomic and say so, or collapse the flags. |
| F3, F6, F7 | `portfolio_manager.py:350-372` vs `paper_trader.py:243-245,432` | Question (d): the existing gate sits at candidate-build; `execute_buy` only records. Recommend **both**, with the authoritative non-bypassable check at the submission seam (complete mediation -- it is the common ancestor of main path, swap path, and any future path) and the candidate-build gate retained as the early/efficient layer. The 57.1 comment already reasons this way for the two BUY paths; the argument extends one seam further down. |
| F5, F8 | `risk_debate.py:345` -> `portfolio_manager.py:321` | The judge already emits a structured verdict with a `decision` field. The gate should key on **`decision == "REJECT"`** (an explicit state) rather than inferring rejection from a numeric 0 -- and treat a missing/unparseable verdict as a third state that also blocks. That is XACML's Deny / Indeterminate split. |
| F3 | `settings.py:350` default `False` | A fail-closed behaviour behind a default-OFF flag is opt-in safety. Consider whether the *safe* branch should become the default with the flag inverted to an explicit, logged escape hatch. |

## 3.1 Open questions PLAN must resolve (not settled by this research)

1. ~~Does the swap path's sizing also traverse `or 10.0`?~~ **RESOLVED --
   YES, at `:878`, and it is UNGUARDED. See section 1.7.** Superseded by the
   new question: are there other unguarded `or <default>` risk-defaulting
   idioms outside `portfolio_manager.py`? Only this one file was swept for the
   pattern.
2. What is the **live** value of all three flags in the running process? The
   incident memo measured only `shape_fix` (null/OFF) on 2026-08-13. Per
   `feedback_committed_is_not_in_force`, the running process must be read, not
   the file. **Attempted here and NOT resolved:** `GET
   http://127.0.0.1:8000/api/settings/` returned a payload in which all three
   keys were **absent** (not `false` -- absent), so this endpoint does not
   expose them and the live state is UNMEASURED by this brief. Do not read the
   absence as OFF; measure it properly (e.g. the 79.1 verification command,
   which imports settings in-process) before relying on it.
3. Whether any **historical** position was opened under the same inversion --
   the incident explicitly notes `paper_trades` history was NOT swept.
4. Whether the fix can ship without an operator `.env` promotion (79.1 has been
   pending 5 weeks), e.g. by inverting the default in code.

---

# 4. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total -- **27**
- [x] Recency scan (last 2 years) performed + reported -- section 2.4
- [x] Full pages read (not abstracts) for the read-in-full set; arXiv accessed
      via `/html/` per the fetch chain; no `arxiv.org/pdf/` WebFetch attempted
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
      (`portfolio_manager.py`, `paper_trader.py`, `risk_debate.py`,
      `settings.py`, the incident memo) plus one out-of-scope discovery
      (`settings.py:346`, the third flag)
- [x] Contradictions / consensus noted -- section 2.6; one adversarial
      qualification recorded in 2.4
- [x] All claims cited per-claim with URL + access date or file:line

**Known gaps (declared, not padded):** the SEC staff FAQ (403) and the CFTC
Electronic Trading Risk Principles (302 redirect) could not be fetched; the FIA
best-practices PDF was left unread once the floor was cleared. None of these
would change the direction of F1-F8, which are unanimous; they would add
enforcement-practice detail. XACML (F8) is snippet-level only and is flagged as
such -- if PLAN leans on the Deny/Indeterminate model as a design template
rather than as corroboration, it should be read in full first.
