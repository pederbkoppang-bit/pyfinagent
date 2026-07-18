# Research Brief — phase-72.0 P0 SCORING-RAIL RESTORATION AUDIT

**Step:** phase-72.0 · **Tier:** complex · **Audit-class:** false
**Researcher spawn:** 2026-07-18
**Brief path:** handoff/current/research_brief.md

> WRITE-FIRST: this file is created before any source is read and grown
> incrementally. Sections fill in as sources are fetched.

## Objective

Root-cause the degraded LLM scoring rail that has held the paper-trading book
at ~97% cash / 0-trade cycles since early July 2026, and surface the exact
config/code seams a later RESTORATION masterplan step would touch. THIS session
does not edit product code — it produces evidence + seams for a contract.

Three external topics + one recency scan + an internal code audit:
1. Anthropic API failure semantics for "credit exhaustion" (429 vs 400/402,
   retry-after, official client-side fallback/retry design).
2. Claude Code subscription rail vs metered API for programmatic/headless use.
3. Design patterns for LLM-dependent decision/trading systems under model
   outage / degraded signal (fail-closed HOLD vs graceful degradation vs
   rule-based fallback).
4. Recency scan (2025-2026) on all three.

---

## Queries run (3-variant discipline: current-year / last-2-year / year-less)

_(filled incrementally)_

---

## Read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|

---

## Recency scan (2025-2026)

_(filled after reads)_

---

## Key findings (external)

_(filled incrementally, per-claim cited)_

---

## Internal code inventory

_(filled from Grep/Read of the repo)_

---

## Restoration seams (file:line)

_(the exact config/code seams a restoration step would touch)_

---

## Application to pyfinagent

_(mapping external findings to internal file:line anchors)_

---

## Research Gate Checklist

- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Full papers / pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim

---

## JSON envelope

_(filled at end)_
