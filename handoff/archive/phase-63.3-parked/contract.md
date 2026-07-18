# Contract — Step 63.3 (Verified defect register published)

**Step id:** 63.3
**Name (verbatim):** "Verified defect register published -- handoff/away_ops/defect_register.md consolidating 63.1+63.2 findings, P0/P1/P2 triage, digest summary; operator screenshot areas all covered or explicitly cleared."
**depends_on:** null (63.1 + 63.2 both `done`)
**Tier:** moderate | **audit_class:** false

## Research-gate summary
`researcher` spawned BEFORE this contract (gate_passed: **true**, moderate, 6 external sources read in full, recency scan performed, all hard-blockers satisfied). Brief: `handoff/current/research_brief_63.3.md`. Sources: Rootly P-levels, BrowserStack + TestRail (2025-05-08) defect fields, softwaretestinghelp sev≠pri, auditfindings lifecycle=no-silent-drops, DefectDojo merge-duplicates-as-link-to-original. Recency scan: classic severity/priority + no-silent-drops model STABLE 2024–2026, not superseded. Internal anchors: `defect_register.md` (existing 63.2 DEF-001, EXTEND not re-number), `route_walk_2026-07-17/walk_summary.json` (63.1 contributes exactly ONE finding: `console_error_routes=['/agent-map']`; failed_request/page_error empty), `frontend/src/components/AgentMap.tsx` L258-276 (both edge branches omit `sourceHandle` → React Flow error#008 → root cause of DEF-002; fix belongs to 63.4), `backend/slack_bot/formatters.py:49` + `scripts/away_ops/send_away_digest.py:80,85` (`chat_postMessage`+`chat_getPermalink` = the outward-facing criterion-3 poster).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json 63.3 verification.success_criteria)
1. "every console-error route, failed-request route, and number mismatch from 63.1/63.2 appears as exactly one DEF- row (no silent drops; duplicates merged with cross-references)"
2. "all four operator-reported screenshot areas map to register rows or an explicit ALL-CLEAR entry with evidence"
3. "the register summary appeared in a Slack digest (sections wired in 62.8)"

**Immutable verification command (verbatim):**
`cd /Users/ford/.openclaw/workspace/pyfinagent && grep -cE '^\| DEF-[0-9]+ \|' handoff/away_ops/defect_register.md && grep -c 'SCREENSHOT-AREA' handoff/away_ops/defect_register.md`

**live_check (verbatim):** "live_check_63.3.md with the register header, row count, and digest permalink"

## Hypothesis
The register (63.2) already exists with DEF-001 + 24 matching triples. 63.3 EXTENDS it to consolidate the 63.1 route-walk finding, add screenshot-area coverage, and triage — closing criteria 1 + 2 at $0. **Criterion 3 (the register summary "appeared in a Slack digest" + the live_check "digest permalink") is an OUTWARD-FACING action** (a `chat_postMessage` to Slack requiring the bot token) that this unattended $0/paper drain will NOT auto-perform → **criteria 1+2 are built DARK; the step is PARKED on criterion 3 with the operator Slack-post token + permalink owed.**

## Plan steps (buildable-DARK; criterion 3 parked)
1. **DEF rows — no silent drops, no double-count (crit 1):**
   - DEF-001 (63.2) KEEP → triage **P1** (reporting-broken).
   - DEF-002 (63.1) ADD → `/agent-map` React Flow error#008 "source handle null", **120 warnings across ~24 edges merged to ONE row** (instance count noted), root cause `AgentMap.tsx` L258-276 (edges omit `sourceHandle`), fix → 63.4. Triage **P2** (cosmetic/console; graph edges drop; no money/risk impact).
   - Explicitly record `failed_request_routes=[]` and `page_error_routes=[]` → **0 rows** (no silent drop).
2. **P0/P1/P2 triage rubric + table (crit 1):** P0=money/risk-affecting (none); P1=reporting/gate-feeding broken (DEF-001); P2=cosmetic/console (DEF-002). Priority defaults to severity; escalate only on a money/risk reason.
3. **4 SCREENSHOT-AREA rows (crit 2)** — literal `SCREENSHOT-AREA` token so `grep -c 'SCREENSHOT-AREA' >= 4`:
   - reports → **DEF-001** (all-0; outcome_tracking 404-absent).
   - positions/currency → **ALL-CLEAR** (63.2: AMD qty/avg/cost/sector MATCH BQ; currency + 64.3 currency-path tests).
   - dashboard numbers → **ALL-CLEAR** (63.2: every API==BQ exact — NAV 23874.56, cash 23214.43, P&L 19.37, benchmark 5.18, counts).
   - new pages → **DEF-002** (the 63.1 walk covered all 22 routes; only `/agent-map` warned) else ALL-CLEAR.
4. **Digest summary draft (crit 3, DARK):** write the register-summary digest text into the register + a `live_check_63.3.md` stub recording the register header + row count, with the **digest permalink field left OPEN (owed operator token)**. Do NOT call `chat_postMessage`.
5. Run the immutable verification command; confirm `grep DEF- >= 2` and `grep SCREENSHOT-AREA >= 4`.

## Boundaries honored
$0 (no metered LLM; no Slack post); paper-only; do-no-harm (no kill-switch/stops/sector-cap/DSR/PBO byte touched); no production code (documentation-only consolidation); historical_macro FROZEN; operator `:3000` untouched; harness = exactly 3 agents. This step FLIPS NOTHING to done — it builds the DARK part and PARKS on criterion 3 (Slack post) with the token owed.

## References
- `handoff/current/research_brief_63.3.md` (research gate)
- `handoff/away_ops/defect_register.md` (63.2 base being extended)
- `handoff/away_ops/route_walk_2026-07-17/walk_summary.json` (63.1 finding)
- `frontend/src/components/AgentMap.tsx` L258-276 (DEF-002 root cause)
- `backend/slack_bot/formatters.py:49`, `scripts/away_ops/send_away_digest.py:80,85` (criterion-3 outward poster)
