# Research Brief — Step 63.3 (Verified Defect Register)

Tier: **moderate** | Research gate for phase-63 step 63.3
Date: 2026-07-17/18 | Status: COMPLETE | gate_passed: **true**

## Objective (verbatim, masterplan 63.3)
"Verified defect register published -- handoff/away_ops/defect_register.md
consolidating 63.1+63.2 findings, P0/P1/P2 triage, digest summary; operator
screenshot areas all covered or explicitly cleared."

## Immutable success criteria (VERBATIM from `.claude/masterplan.json` 63.3 → verification.success_criteria)
1. "every console-error route, failed-request route, and number mismatch from 63.1/63.2 appears as exactly one DEF- row (no silent drops; duplicates merged with cross-references)"
2. "all four operator-reported screenshot areas map to register rows or an explicit ALL-CLEAR entry with evidence"
3. "the register summary appeared in a Slack digest (sections wired in 62.8)"

Immutable verification command:
`cd /Users/ford/.openclaw/workspace/pyfinagent && grep -cE '^\| DEF-[0-9]+ \|' handoff/away_ops/defect_register.md && grep -c 'SCREENSHOT-AREA' handoff/away_ops/defect_register.md`

---

## Read in full (6; gate floor is 5 — CLEARED)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://rootly.com/incident-response/support-levels | 2026-07-17 | vendor doc | WebFetch full | P1=critical/revenue/outage (mins→1-4h); P2=degraded/partial (4-8h→24h); P3=cosmetic UI/single-user (days). Distinguishing criterion = revenue-loss + reputational damage vs cosmetic. |
| 2 | https://www.browserstack.com/guide/how-to-write-a-good-defect-report | 2026-07-17 | vendor doc | WebFetch full | Core defect-row fields: **Defect ID (unique, e.g. DEF-0012), Title, Description, Steps-to-Repro, Expected, Actual, Environment, Severity, Priority, Module, Attachments/evidence, Status, Assigned-To**. Best practice: **one defect per report**, consistent ID format for auditability. |
| 3 | https://www.softwaretestinghelp.com/how-to-set-defect-priority-and-severity-with-defect-triage-process/ | 2026-07-17 | practitioner | WebFetch full | Severity=technical impact (tester sets); Priority=urgency (business sets). High-sev/low-pri = rare beta bug; low-sev/high-pri = misspelled logo (brand risk). No hard auto-default, but severity strongly influences priority. |
| 4 | https://www.auditfindings.com/audit-findings-lifecycle/ | 2026-07-17 | practitioner | WebFetch full | 6-stage lifecycle (identify→prioritize→assign owner→remediate→response→validate/close). **Centralize into single source of truth; no silent drops — every finding stays visible until closure**; consistent fields incl. description/root-cause/risk/evidence. |
| 5 | https://docs.defectdojo.com/triage_findings/finding_deduplication/about_deduplication/ | 2026-07-17 | official docs | WebFetch full | Duplicates are **linked to an original, never silently deleted** ("Finding B marked as duplicate of Finding A"); original never auto-deleted. Cross-tool/multi-source dedup preserves audit trail — the exact "merge with cross-reference" pattern criterion-1 wants. |
| 6 | https://www.testrail.com/blog/defect-tracking-guide/ | 2026-07-17 | vendor doc (May 8 2025) | WebFetch full | Defect record fields (Title, ID, Environment, Severity, Description w/ expected-vs-actual, Steps, Logs/evidence); severity≠priority; **centralized tracking beats spreadsheets**. Dated 2025-05-08 = recency anchor. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.betterbugs.io/blog/bug-triage ("[2026]") | 2026 blog | WebFetch returned empty body twice (JS-rendered/bot-blocked); recency covered by TestRail + snippet |
| https://www.issuelinker.com/blog/defect-management-process | blog | bug lifecycle; snippet sufficient |
| https://upstat.io/blog/priority-vs-severity | vendor | P1 severity-vs-priority; snippet |
| https://www.kualitee.com/blog/bug-management/severity-levels-vs-priority-levels-bug-tracking/ | vendor | severity vs priority; snippet |
| https://plane.so/blog/bug-severity-vs-priority-in-testing-key-differences | vendor | snippet |
| https://crosscheck.cloud/blogs/bug-triage-process-how-to-prioritize-bugs-in-a-sprint | vendor | triage cadence; snippet |
| https://www.sec.gov/rules-regulations/2026/04/s7-2026-12 | regulator (2026) | consolidated-audit-trail dedup; snippet (recency) |
| https://simplerqms.com/audit-findings/ | vendor | audit-finding fields; snippet |
| https://www.techtarget.com/searchsoftwarequality/tip/What-details-to-include-on-a-software-defect-report | media | fields; snippet |
| https://keploy.io/blog/community/defect-management-in-software-testing | blog | process; snippet |
| https://www.tmap.net/node/142 | methodology | minimum defect-report fields; snippet |
| https://www.stacklesson.com/qa-engineering/defect-management/ch05-lesson-05-defect-triage-metrics-and-best-practices/ | course | triage cadence/metrics; snippet |

Unique URLs collected: **25+** (3 searches × ~10 each + DefectDojo redirect) — well over the 10 floor.

## Recency scan (2024-2026)
Ran the 3-variant discipline: current-year frontier (`...P0 P1 P2 2026`), last-2-year (`...deduplication cross-reference 2025`), and year-less canonical (`defect register software testing best practice fields`). **Result: the classic model is STABLE and NOT superseded.** 2025-2026 sources (TestRail 2025-05-08 read-in-full; betterbugs "[2026]", crosscheck, SEC 2026 CAT concept release — snippets) all reaffirm: (a) severity≠priority, (b) P0/P1/P2 keyed to business/money impact, (c) centralize into a single source of truth with no silent drops, (d) merge duplicates as links to an original rather than deleting. No new finding overturns the canonical guidance; the register design below is safe to build on it.

## Key findings (external)
1. **No silent drops = every finding stays visible to closure** — auditfindings lifecycle; DefectDojo "original never auto-deleted." Directly satisfies criterion-1 "no silent drops."
2. **Merge duplicates as a link/cross-reference to ONE original**, not N rows — DefectDojo "Finding B marked as duplicate of Finding A." The 120 /agent-map warnings collapse to ONE DEF row with an instance-count note.
3. **Severity (technical) ≠ Priority (urgency/business)** — STH, TestRail, Rootly. Our P0/P1/P2 is a PRIORITY axis keyed to money/risk; severity is a secondary note. Default priority to severity, escalate only for a money/risk reason.
4. **One defect per row, unique stable ID (DEF-NNNN), consistent fields** — BrowserStack/TestRail. The existing register's DEF-001 format already complies; extend it.
5. **Single source of truth beats scattered spreadsheets** — auditfindings/TestRail. `defect_register.md` IS that SoT for the 63.4 fix queue (matches masterplan `audit_basis`).

## Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| handoff/away_ops/defect_register.md | 1-104 | EXISTING 63.2 register; has DEF-001 (/performance+/learnings outcome_tracking absent) + 24 API-vs-BQ triples | Extend (add DEF-002, SCREENSHOT-AREA rows, triage) |
| handoff/away_ops/route_walk_2026-07-17/walk_summary.json | 1-765 | 63.1 walk: console_error_routes=['/agent-map']; failed_request_routes=[]; page_error_routes=[]; route_list_delta empty | 1 finding → DEF-002 |
| frontend/src/components/AgentMap.tsx | ~258-276 (edge-builder `.map`) | Both edge branches build `{id, source, target}` with **no `sourceHandle`** → React Flow error#008 "Couldn't create edge for source handle id null" (120 warning instances across ~24 edges, repeated on re-render) | Root cause of DEF-002; fix is a 63.4 item, not 63.3 |
| backend/slack_bot/formatters.py | 49 (`format_away_digest_sections`) | 62.8 away-digest 6 sections | DONE (masterplan 62.8=done) |
| scripts/away_ops/send_away_digest.py | 80 (`chat_postMessage`), 85 (`chat_getPermalink`) | POSTS digest to Slack + returns permalink | **Outward-facing** — the criterion-3 gate |
| backend/slack_bot/scheduler.py | 620-627 | Evening digest injects `format_away_digest_sections` then `chat_postMessage` | Outward-facing (needs bot running) |

## Application to pyfinagent (the exact rows)

### (a) DEF rows — 2 total, no double-count
- **DEF-001** (KEEP, from 63.2): `/performance` (+`/learnings`) — outcome_tracking table absent → all-0. Severity MEDIUM. **Priority P1** (reporting-feeding broken; no money/risk mis-statement). Source: `backend/services/autonomous_loop.py:2948` (flag-off writer) + missing migration.
- **DEF-002** (ADD, from 63.1): `/agent-map` — React Flow error#008 "Couldn't create edge for source handle id: null", **120 console-warning instances across ~24 distinct edges (main-researcher, main-qa, …), repeated on re-render — merged to ONE row per dedup best-practice**. Severity LOW/cosmetic. **Priority P2** (graph edges drop from the visualization only; no money/risk/data-correctness impact). Suspected file: `frontend/src/components/AgentMap.tsx` edge-builder (~L258-276) — edges omit `sourceHandle`.
- failed_request_routes=[] and page_error_routes=[] → **0 rows** (nothing to drop; note the empties explicitly in a "no-finding" line so the audit shows they were considered, not silently omitted).

### (b) 4 SCREENSHOT-AREA rows (each MUST contain the literal token `SCREENSHOT-AREA` so `grep -c 'SCREENSHOT-AREA'` ≥ 4)
Suggested row format (pipe table): `| SCREENSHOT-AREA | <area> | <maps-to DEF-xxx / ALL-CLEAR> | <evidence> |`
1. `| SCREENSHOT-AREA | reports | DEF-001 | /performance+/learnings render all-0; root cause pyfinagent_data.outcome_tracking table absent (63.2 Q5 BQ 404) |`
2. `| SCREENSHOT-AREA | positions/currency | ALL-CLEAR | 63.2: AMD qty 1.319955 / avg_entry 545.42 / cost_basis 719.93 / sector Technology all MATCH BQ; identities hold; currency correct + 64.3 currency tests green |`
3. `| SCREENSHOT-AREA | dashboard numbers | ALL-CLEAR | 63.2: every API-vs-BQ stored number MATCHES (NAV 23874.56, cash 23214.43, P&L% 19.37, benchmark 5.18, pos-count 1, trades 61) |`
4. `| SCREENSHOT-AREA | new pages | DEF-002 (else ALL-CLEAR) | 63.1 walk visited all 22 routes incl. new pages; only /agent-map raised console warnings (→DEF-002); all others 200 / 0 console-errors / 0 failed-requests |`

### (c) P0/P1/P2 triage rubric (PRIORITY axis, keyed to money/risk)
- **P0** = money- or risk-affecting: wrong trade, a displayed money/position number that could mislead a live trading decision, broken kill-switch/risk gate. → **none in this register.**
- **P1** = gate-feeding / reporting broken: a decision- or gate-feeding page cannot render real data, but no money/risk mis-statement. → **DEF-001.**
- **P2** = cosmetic / console warnings / no money-risk-data-correctness impact. → **DEF-002.**
Rubric note (research-grounded): priority defaults to severity, escalates only for a named money/risk reason. DEF-001 sev=MEDIUM→P1; DEF-002 sev=LOW→P2. No P0.

### (d) Criterion 3 = outward-facing operator gate → PARK
62.8 formatter is DONE, so the digest-summary CONTENT is $0-buildable (Main can draft the summary text into the register). But "the register summary **appeared in a Slack digest**" + live_check "digest **permalink**" require an actual `chat_postMessage` via `scripts/away_ops/send_away_digest.py:80` (or the running bot scheduler:620-627) → an OUTWARD-FACING, side-effecting Slack post needing the bot token / a live bot. An unattended $0 step must NOT auto-post. **Criterion 3 is operator-gated: build criteria 1+2 DARK, draft the digest summary, and PARK the Slack post (owe the operator a "post away digest" token + the permalink for live_check_63.3.md).**

## Research Gate Checklist
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (25+)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module (register, walk json, AgentMap, formatters, digest poster, scheduler)
- [x] Contradictions/consensus noted (severity vs priority; classic model stable)
- [x] Per-claim citation

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "63.3 extends the existing 63.2 defect_register.md. Add exactly 2 DEF rows: DEF-001 (KEEP, /performance+/learnings outcome_tracking absent, P1) + DEF-002 (NEW from 63.1, /agent-map React Flow error#008 — 120 warnings/~24 edges merged to ONE row, P2). failed_request/page_error routes empty → 0 rows (note explicitly). Add 4 SCREENSHOT-AREA rows: reports→DEF-001; positions/currency→ALL-CLEAR (63.2 AMD+currency match); dashboard numbers→ALL-CLEAR (all API==BQ); new pages→DEF-002-else-ALL-CLEAR. Triage rubric P0=money/risk, P1=reporting/gate-broken, P2=cosmetic. Criterion 3 (register summary appears in a Slack digest + permalink) = outward-facing operator gate; 62.8 formatter done, but the chat_postMessage post is side-effecting → PARK, owe operator a post token. Research: no-silent-drops + merge-duplicates-as-links-to-original (DefectDojo/auditfindings), severity≠priority, one-defect-per-row; classic model stable 2024-2026.",
  "brief_path": "handoff/current/research_brief_63.3.md",
  "gate_passed": true
}
```

## Sources
- https://rootly.com/incident-response/support-levels
- https://www.browserstack.com/guide/how-to-write-a-good-defect-report
- https://www.softwaretestinghelp.com/how-to-set-defect-priority-and-severity-with-defect-triage-process/
- https://www.auditfindings.com/audit-findings-lifecycle/
- https://docs.defectdojo.com/triage_findings/finding_deduplication/about_deduplication/
- https://www.testrail.com/blog/defect-tracking-guide/
