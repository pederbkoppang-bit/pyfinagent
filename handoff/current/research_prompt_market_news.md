# Research prompt — leveraging market news for pyfinagent

**Status:** durable artifact. Drives the `researcher` spawn for masterplan step
**83.1**. Authored 2026-08-04 during phase-83 planning. Not swept by
`archive-handoff.sh` (filename matches neither its explicit list nor the
`<sid>-*.md` / `phase-<sid>-*.md` patterns) — it must survive until 83.1 runs.

**Launch:** Workflow rail first (`.claude/workflows/` researcher script) per the
standing operator rule that BOTH dev-MAS agents go through Workflows; Agent-tool
`researcher` subagent is the documented fallback. Main transcribes the returned
envelope verbatim and never authors the brief itself.

**Why this exists:** operator goal set 2026-08-04 — leverage market news for
money. The four reference cases below are the operator's own words: COVID lifting
medical names, the AI rush lifting datacenter and therefore memory/AI names, the
Ukraine war lifting weapons makers, and the Iran/US conflict lifting crude and the
oil complex. Every one is a slow, cross-sector, months-long theme. That is the
target. Fast headline drift is explicitly out of scope.

---

## Spawn prompt (copy below this line)

```markdown
# Researcher spawn — market-news leverage for pyfinagent

tier: complex
coverage.audit_class: true   # unknown-denominator; loop-until-dry, K=2

## Objective
Determine how pyfinagent should convert market news into a DURABLE, CROSS-SECTOR
thematic signal that survives our promotion gates (DSR>=0.95, PBO<=0.5, beat
incumbent OOS, clears costs) and adds NET, COST-ADJUSTED P&L.

Target the SLOW channel: themes persisting weeks-to-months that propagate to
second-order beneficiaries. Fast headline drift is OUT OF SCOPE — operator
decision, on the grounds that its turnover makes it cost-negative for us, and
that published evidence shows headline momentum dissipating quickly in large
caps (Chen/Kelly/Xiu, SSRN 4416687).

Reference cases the design must explain, all four:
  COVID -> pharma/medical          | AI buildout -> datacenter -> memory/power
  Ukraine war -> defense           | Iran-US conflict -> crude -> oil complex

## Questions that must be answered

1. THEME REPRESENTATION. What is the best point-in-time representation of a theme
   (intensity series, membership set, birth/death dates)? Compare: LDA/topic-share
   (Bybee/Kelly/Manela/Xiu, JF 2024), embedding clusters, LLM-labelled taxonomy,
   and keyword+confirmation gates (our own defense_signal.py precedent). Give the
   EVIDENCE for each option, not a preference.

2. BENEFICIARY MAPPING. How do we get from "AI datacenter buildout" to memory
   makers and power suppliers WITHOUT a paid supply-chain dataset? Evaluate at
   least: 10-K text similarity (Hoberg-Phillips), news co-mention graphs,
   revenue-segment disclosure, ETF-holdings overlap as a free proxy. Report
   published recall/precision figures where they exist; say so where they do not.

3. TIMING AND CROWDING. Where in a theme's life is the tradable edge — birth,
   acceleration, or confirmation? Quantify the decay. Separate FIRST-order
   beneficiaries (priced fast) from SECOND-order (slow). Address the
   thematic-ETF-launch-as-top-signal problem directly.

4. COST AND TURNOVER. For each candidate design, estimate rebalance frequency and
   holding period, and state whether the edge survives OUR costs. A design that
   only works gross is a REJECTED design — say so explicitly rather than leaving
   it to the reader.

5. FREE-SOURCE FEASIBILITY. Strictly $0 — this is a hard operator constraint, not
   a preference. For GDELT, SEC EDGAR, Alpaca news, and the Alpha Vantage free
   tier, report each on: history depth, latency, rate limit, point-in-time
   integrity (does it revise published records?), and licence terms for derived
   signals. Address GDELT's ~55% field accuracy and ~20% redundancy explicitly —
   do not wave it through.

6. LOOKAHEAD DEFENCE. Specify the concrete protocol: publication-vs-ingestion
   timestamps, embargo length, the LAP (Lookahead Propensity) interaction test
   from arXiv 2512.23847, and knowledge-cutoff-aware evaluation windows. Name what
   would FALSIFY a theme backtest.

7. NEGATIVE EVIDENCE (mandatory). Where has news-driven thematic trading FAILED?
   Crowding, cost drag, sector-cap interaction, regime dependence, post-publication
   decay. A brief without a substantive negative-evidence section does not clear
   the gate.

## Internal audit (same session — you own internal code exploration too)

Read and report on:
- backend/services/defense_signal.py — the Ukraine reference case, ALREADY BUILT
  as a single hard-coded theme ("reference-case signal", GPR-Acts AND XAR-momentum
  gate, default OFF). How would it generalise to N themes? What breaks?
- backend/services/call_transcript_gpr.py — its docstring reports CONTEMPORANEOUS
  relationship only, NO forward return predictability, and is therefore scoped as
  a risk filter not an alpha source. This is the exact failure mode we must not
  repeat. What distinguishes a theme signal from this?
- backend/services/peer_leadlag_screen.py — intra-sector lead-lag (Hou 2007). It
  groups BY SECTOR. What breaks when the propagation is cross-sector, as in
  AI -> memory?
- backend/services/news_screen.py + backend/news/* — what exists, what is live.
- backend/backtest/backtest_engine.py:49-62 (_NUMERIC_FEATURES) — 37 features,
  zero news-derived. Confirm and state what a news feature must look like to be
  admissible here.
- backend/services/overlay_math.py::sign_safe_mult — the seam any 15th overlay
  must route through.

## Hard blockers (from .claude/rules/research-gate.md)

- >=5 external sources read IN FULL via WebFetch. Search snippets do NOT count.
- Three query variants per topic: current-year 2026, last-2-year, and YEAR-LESS
  canonical. A single year-locked query is a protocol breach.
- arXiv: try /html/<id> first, then ar5iv, then pdfplumber. NEVER WebFetch
  /pdf/<id> as the primary attempt and then skip the paper as unreadable.
- A dedicated "Recency scan (last 2 years)" section, present even when empty.
- audit_class = true: keep running rounds until 2 consecutive DRY rounds (zero new
  read-in-full findings after de-dup), then set coverage.dry = true.
- WRITE-FIRST: create the brief early and append as you read. Never a single
  end-of-session flush. A session that cannot clear the gate must still leave a
  partial brief plus an honest gate_passed: false envelope.
- Emit the JSON envelope. gate_passed requires external_sources_read_in_full >= 5
  AND recency_scan_performed == true AND coverage.dry == true.

## Output

handoff/current/research_brief_phase83.md

Must contain a ranked recommendation, and for EACH of the four reference cases a
walkthrough of: how the proposed design would have detected the theme, what it
would have bought, when, and what it would have COST to hold.
```

---

## Cross-references

- `.claude/rules/research-gate.md` — gate mechanics (source floor, recency scan,
  query-variant discipline, adaptive coverage, envelope shape).
- `.claude/agents/researcher.md` — the agent-facing system prompt.
- `docs/runbooks/per-step-protocol.md` — where the research gate sits in the cycle.
- Plan of record: `/Users/ford/.claude/plans/i-would-like-you-compressed-newell.md`.
