# Phase-24 Application-Wide Audit — Findings Directory

**Created:** 2026-05-12.
**Plan:** `/Users/ford/.claude/plans/sunny-jingling-deer.md` (approved).
**Masterplan steps:** `phase-24.0` through `phase-24.14` (15 buckets).
**Master prompt:** `scripts/audit/phase_24_audit_prompt.md`.

This directory is the destination for the 15 audit findings documents
produced by the phase-24 audit. Each bucket's findings doc is written
by Main during its harness cycle (research-gate -> contract -> Q/A ->
log -> flip status). Phase-24 is **READ-ONLY** — no code changes —
the audit produces ranked phase-25.x candidate code-change steps.

## Expected files (one per bucket)

| Bucket | Filename | Status | Cycle |
|---|---|---|---|
| 24.0  | `24.0-charter-findings.md`              | pending | — |
| 24.1  | `24.1-execution-trading-findings.md`    | pending | — |
| 24.2  | `24.2-pipeline-routing-findings.md`     | pending | — |
| 24.3  | `24.3-autoresearch-wiring-findings.md`  | pending | — |
| 24.4  | `24.4-agent-rationale-findings.md`      | pending | — |
| 24.5  | `24.5-slack-notifications-findings.md`  | pending | — |
| 24.6  | `24.6-backtest-engine-findings.md`      | pending | — |
| 24.7  | `24.7-data-quality-findings.md`         | pending | — |
| 24.8  | `24.8-observability-findings.md`        | pending | — |
| 24.9  | `24.9-llm-conformance-findings.md`      | pending | — |
| 24.10 | `24.10-mcp-security-findings.md`        | pending | — |
| 24.11 | `24.11-frontend-data-wiring-findings.md`| pending | — |
| 24.12 | `24.12-ui-ux-presentation-findings.md`  | pending | — |
| 24.13 | `24.13-redline-synthesis-findings.md`   | pending | — |
| 24.14 | `24.14-final-synthesis-findings.md`     | pending | — |

`screenshots/` — visual regression baseline screenshots of each
frontend page, written during bucket 24.12.

## Findings doc structure (each bucket)

Every findings doc MUST have these sections in order:

1. **Frontmatter** — bucket id, cycle, date, researcher gate envelope.
2. **Executive summary** — 1-paragraph TL;DR.
3. **Code-grounded findings** — file:line anchors + grep evidence for
   every claim.
4. **External-research summary** — citations to canonical Anthropic /
   Google / MCP / academic docs the researcher fetched in full.
5. **Recency scan (2024-2026)** — last-two-year window findings.
6. **Proposed phase-25.x candidate steps** — at least 3 per bucket,
   each with: name, files list (absolute paths), draft verification
   command, priority (P0/P1/P2), one-paragraph rationale.
7. **Open questions** — anything the bucket surfaced but couldn't
   answer without code changes or live data.
8. **References** — full URL list (canonical + supplementary).

## Verifier-driven enforcement

Each bucket has a `tests/verify_phase_24_<n>.py` verifier with 10-16
immutable claims. The verifier checks:
- Findings doc exists at the expected path.
- Research-gate envelope present with `gate_passed: true`.
- ≥5 external sources cited.
- Canonical URL substring cited verbatim.
- "Recency scan" section present.
- ≥3 phase-25 candidates proposed.
- Each candidate has files list + verification command draft.
- Bucket-specific anchors (e.g., `paper_trader.py:414-423` for 24.1,
  `Trader.*RiskJudge` for 24.4, `5x SNDK` for 24.5).
- harness_log has the cycle entry for this bucket.

A bucket cannot flip `status=done` until its verifier returns full PASS.
The `verification.live_check` field on each masterplan step also
requires the findings doc to exist on disk (per the phase-23.8.1 R-1
gate).

## After all 15 buckets complete

Bucket 24.14 (final synthesis) produces a ranked list of phase-25.x
candidate code-change steps. Operator pastes those into
`.claude/masterplan.json` (likely under a new `phase-25` top-level
entry) and runs them through the standard harness in priority order
(P0 first — trading-execution, agent-rationale, Slack).
