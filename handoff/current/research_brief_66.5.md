# Research Brief — phase-66.5 Away-backlog triage (planning-only)

Tier: SIMPLE (caller-specified; caller-mandated scope exceeds the simple tool budget — floor rules honored).
Date: 2026-07-07. Agent: researcher (Layer-3). Status: COMPLETE.

## Question

Disposition every phase-63/64/65 step (14 steps) keep/merge/drop with rationale for ATTENDED
operation. These phases were designed 2026-06-12 for a 3-week UNATTENDED window
(handoff/away_ops/approved_plan_2026-06-12.md) and executed 0% (credential death,
34 dead sessions 06-20..07-06). Output feeds handoff/current/triage_phase63-65.md;
masterplan edits ONLY after operator sign-off (criteria immutable; dropped steps get
status=dropped, never deleted — masterplan.json:15200-15206).

---

## Internal audit

### 1. Away-mode assumptions per phase (what no longer holds vs what remains valid)

Away-design cadence assumptions (approved_plan_2026-06-12.md:112-132, :149 rail 8, :157-166
calendar): AM dev slot 07:30 / PM evidence slot 22:00, one step per AM session, digest+token
round-trip as the ONLY operator channel, wall-clock week-1/2/3 windows. Operator now present,
BUT the away calendar still fires (handoff/away_ops/session_pm_20260706T200008Z.json exists;
sessions ran through 07-06) — triage should surface "keep or disarm the away plists" as an
operator question, it is not automatically dead.

- **Phase-63** (masterplan.json:14768-14877): method (Playwright walk → BQ cross-check →
  register → fix queue → re-walk) is cadence-independent and fully attended-valid. Away-only
  residue: "one fix per AM slot" (63.4, :14839), "week-2 end" (63.5, :14859), digest-summary
  criteria (63.3 c3, :14830 — satisfiable: 62.8 IS done, digest sections wired).
- **Phase-64** (:14879-14988): suite design (functional project, no screenshots, Mac-runnable)
  attended-valid. Away residue: "PM session runs it nightly" (64.2 name :14910; 64.5
  verification greps scripts/away_ops/prompt_pm.md :14977).
- **Phase-65** (:14990-15079): EU-zero-trades problem is still real, but the phase presumes a
  RUNNING engine. Engine has been dead: zero BUYs since 06-10, cc_rail 100% fail 06-15..07-06,
  100% cash since 07-03 (phase-66 header :15083, 66.2 audit_basis :15138). Token machinery
  (65.2/65.4) presumed the 62.2 inbound bot handler — 62.2 is still PENDING (masterplan
  phase-62), so tokens never had an ingestion path; attended, in-session operator approval
  replaces the token entirely.

### 2. Overlap analysis vs phase-66

- **65.1 vs 66.2**: 66.2 criterion (b) (:15142) mandates a per-stage funnel diagnosis with
  candidate counts at every gate (signals → scorer → risk judge → execution) for ALL markets —
  but only triggers after >=5 healthy-rail zero-BUY days. 65.1 (:15001) is EU-specific at
  screener granularity (universe → screener → price_quality → calendar → rank → order), with
  per-ticker threshold-exclusion evidence for >=5 DAX names + counters shipped as PERMANENT
  structured-log lines + replay method. Verdict: **substantial overlap, not subsumption** —
  different funnel granularity and 65.1 has a code deliverable (permanent counters) 66.2 lacks.
  Merge the diagnosis leg into 66.2; keep the counter-shipping + per-ticker EU evidence as the
  residue (inside 66.2 execution or a slimmed 65.1). If 66.2 closes via route (a) (a BUY
  happens), EU-specific diagnosis is still owed. NOTE: 64.4 `depends_on_step: "65.1"`
  (:14954) — any merge must repoint that dependency.
- **65.3 window contamination**: "since 2026-06-01" spans ~25 US trading days (06-01..07-07).
  Last BUY 06-10; cc_rail dead 06-15..07-06 (~15 trading days); 100% cash since 07-03. Healthy
  BUY-capable segment = ~7-8 trading days = **~30% of the window; ~70% measures the outage,
  not market health**. Baseline as-written is unfalsifiable-in-reverse: keep the method,
  re-scope to segmented windows (pre-freeze 06-01..06-12 + fresh post-66.2 window).
- **63.3 seed defects** (known, unfiled — register file does NOT exist, ls confirms):
  1. `_resolve_claude_binary` which-vs-env-override docstring mismatch
     (backend/agents/claude_code_client.py:56; 66.1 disclosure).
  2. .env-bleed test isolation (61.1 finding — tests read real backend/.env).
  3. Auth-latch paged:false no-retry (66.4 Q/A note; caller latch never re-pages).
  4. auto-commit-and-push hook silent stalls — recurred: handoff/logs/auto-push.log shows
     12 INVOKED lines 2026-07-06T22:55..23:19Z with zero commit/push lines following.
  5. Changelog trailing-commit race (chore auto-changelog commits landing after the push).
  6. historical_macro ~103d stale (goal_post_away_review.md:124; cycle_block_summary.md:82).
  7. Alpaca short_market_value -13842.89 long-only anomaly (folded into 66.2 c3 :15144).
  8. paper_portfolio single-US-row despite KR trades (folded into 66.2 c4 :15145).
  Items 7-8 are already owned by 66.2 — the register must cross-reference, not duplicate.

### 3. Ground truth (verified 2026-07-07)

| Claim in step text | Check | Result |
|---|---|---|
| 22 routes (63.1/64.2) | `find frontend/src/app -name page.tsx \| wc -l` | **22 — still true** |
| tests/e2e-functional (64.1) | ls | **absent** (64.1 premise intact, nothing built) |
| playwright.config.ts projects | read | **ONE project** (`chromium`, testDir `./tests/visual-regression`, frontend/playwright.config.ts:38-39,:67-82) — "second project" premise intact |
| defect_register.md (63.2/63.3) | ls handoff/away_ops/ | **absent** (63.x executed 0%) |
| approved_plan_2026-06-12.md | ls | present (13,124 B) |
| goal_away_ops.md | ls | present (41,713 B; payload duplicated in masterplan) |
| 62.8 digest sections (63.3 c3) | masterplan phase-62 | 62.8 **done**; 62.1/62.2/62.6/62.7 pending |

### 4. Dependency sanity

- **Blocked on phase-66 outcomes**: 65.4 (needs the engine trading — 66.1 rail + 66.2
  redeploy — plus 65.2 flag ON + 65.3 thresholds); 65.3 (needs a fresh post-restart window);
  63.2 (cheaper AFTER 66.3 fixes phantom $0.50 cost rows and the macro refresh lands — else
  displayed-vs-BQ triples generate spurious DEF rows on any cost/macro-bearing page); 64.4
  (needs funnel counters from the 65.1/66.2 merge).
- **Runnable independently now**: 63.1 (walk), 63.3 (seedable from §2 list even before
  63.2 completes), 64.1 (scaffold), 64.3 (`depends_on_step: null`, backend-only).
- **Strict-order finding**: one P0 chain at a time — 66.1 → 66.2 first; no 63/64/65 build
  work should start until the decision path is restored (WIP-limit literature below).

---

## External research

Queries run (three-variant discipline): year-less canonical — "postmortem action item
prioritization SRE incident review backlog"; "when to invest in end-to-end test suite
stabilize core first test pyramid"; "kanban WIP limits finish work in progress before starting
new theory of constraints". Current-year — "post-incident action items triage
re-prioritization 2026". Last-2-year — "E2E test suite investment timing flaky maintenance
2025"; "WIP limits one P0 at a time incident recovery focus 2025".

### Read in full (6; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://sre.google/workbook/postmortem-culture/ | 2026-07-07 | Official doc (Google SRE workbook) | WebFetch full | "All action items have both an owner and a tracking number... assigned a priority level"; "Without a formal tracking process, action items from postmortems are often forgotten, resulting in outages" |
| https://incident.io/blog/sre-incident-postmortem-best-practices | 2026-07-07 | Industry (authoritative) | WebFetch full | Mitigative vs preventative split; "Three months later the same incident recurs because those items lived in a Google Doc nobody reopened"; <50% completion = "postmortems are theater"; move items into the real tracker immediately |
| https://martinfowler.com/articles/practical-test-pyramid.html | 2026-07-07 | Authoritative blog (Vocke/Fowler) | WebFetch full | "Due to their high maintenance cost you should aim to reduce the number of end-to-end tests to a bare minimum"; pick "user journeys that define the core value of your product" |
| https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html | 2026-07-07 | Official (Google Testing Blog, canonical 2015) | WebFetch full | 70/20/10 unit/integration/E2E; E2E = narrow safety net for critical flows, not primary QA; slow feedback + flake cost dominate |
| https://www.rainforestqa.com/blog/when-to-run-e2e-tests | 2026-07-07 | Industry | WebFetch full | E2E = "last line of defense"; only flows "you wouldn't fix right away if they broke"; unstable product → false-positive failures → maintenance spiral |
| https://www.planview.com/resources/articles/wip-limits/ | 2026-07-07 | Industry (LeanKit heritage) | WebFetch full | "Stop starting, start finishing"; 23min15s refocus cost per interruption; "by doing less at a time, we can actually get more done" |

### Identified but snippet-only (12 of 46; context, not gate-counting)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.atlassian.com/agile/kanban/wip-limits | Industry | Fetch attempted — page returned nav-only markup, no body extracted |
| https://sre.google/sre-book/postmortem-culture/ | Official doc | Workbook chapter (fetched) is the action-item-focused successor |
| https://rootly.com/incident-postmortems/meeting-guide | Industry 2026 | Recency corroboration only (severity-vs-effort matrix) |
| https://www.atlassian.com/incident-management/handbook/postmortems | Industry | Duplicates SRE-workbook guidance |
| https://uplevelteam.com/blog/wip-limits | Industry 2025 | Snippet sufficed: urgent items bypass WIP ONLY with explicit deprioritization of something else |
| https://www.pmi.org/disciplined-agile/controlling-work-in-process-wip | Industry | Duplicative of Planview |
| https://getautonoma.com/blog/testing-pyramid | Industry 2025-26 | Contrarian "E2E-first" take noted (see debate) |
| https://diffie.ai/blog/state-of-e2e-testing-2026 | Industry 2026 | Recency stats (flake-share 10%→26%, Bitrise 2025) via snippet |
| https://tryzerocheck.com/guides/e2e-testing-cost/ | Industry 2026 | Cost stats (30-40% of testing effort = maintenance, Capgemini WQR 2024-25) via snippet |
| https://aiopsschool.com/blog/incident-triage/ | Community 2026 | Continuous re-triage framing; low weight |
| https://panther.com/blog/ai-agents-incident-triage-prioritization | Industry 2025-26 | AI-SOC triage — off-domain for a single-operator local system |
| https://kanbantool.com/kanban-guide/kanban-fundamentals/limit-work-in-progress | Industry | Duplicative of Planview |

(46 total snippet-only URLs collected across 6 searches; 52 unique URLs overall.)

### Recency scan (2024-2026)

Performed (dedicated 2026- and 2025-scoped query passes). Findings: (1) 2025-2026 incident
literature adds AI-assisted triage tooling and "prioritization is continuous, not a one-time
event" framing — the latter directly endorses re-triaging a stale backlog after context
change (what 66.5 is). (2) 2025 testing data hardens the canonical position: teams with flaky
tests rose 10%→26% (Bitrise Mobile Insights 2025); 30-40% of testing effort is maintenance
(Capgemini World Quality Report 2024-25); ~3.7 eng-hours per flaky-test fix — reinforcing
64.1's no-screenshot functional-assertion design and a minimal suite. (3) One contrarian
2025-26 thread ("Testing Pyramid Is Upside Down", Autonoma — an E2E-tooling vendor) argues
E2E-first for fast-shipping startups; conflicts with Google/Fowler canon and is
vendor-motivated; noted, not adopted. No finding supersedes the canonical sources.

## Key findings

1. **Stale action items must be re-triaged into the live tracker with owner+priority, or they
   cause repeat incidents** — "Without a formal tracking process, action items... are often
   forgotten, resulting in outages" (Google SRE Workbook, postmortem-culture); "items lived in
   a Google Doc nobody reopened" (incident.io). 66.5's keep/merge/drop into masterplan status
   IS the prescribed mechanism; silent resumption is the anti-pattern.
2. **Prioritize by risk-reduction, severity-vs-effort**: high-sev/low-effort first (63.3
   seeding, 64.1 scaffold), high-sev/high-effort to the roadmap (65.4 proof) (incident.io;
   rootly 2026 corroborates).
3. **Stabilize before you pave**: E2E suites belong on a stable core; unstable systems make
   E2E false-positive machines with 30-40% maintenance drag (Rainforest; Google Testing Blog
   70/20/10; Fowler "bare minimum"; Capgemini WQR 24-25). → Phase-64 build-out sequences
   AFTER 66.1/66.2 restore the engine; keep 64.1's cheap change-insensitive smoke assertions.
4. **WIP limits / one-P0-chain**: "Stop starting, start finishing" (Planview); urgent work
   enters only by explicitly deprioritizing something else (uplevelteam 2025). Grounds the
   strict order 66.1→66.2 before any 63/64/65 work, and grounds DROPPING/deferring rather
   than keeping all 14 steps nominally open.

## Per-step disposition findings (14 rows — researcher recommendation; operator decides)

| Step | One-line finding → suggested disposition |
|---|---|
| 63.1 route walk | Attended-valid as-is; 22 routes confirmed; only away residue is PM-slot cadence → **KEEP** (runnable now) |
| 63.2 BQ cross-check | Method proven, but truth-side is polluted until 66.3 (phantom $0.50 rows) + macro refresh → **KEEP, resequence after 66.3** |
| 63.3 defect register | File absent; 8 known unfiled defects (§2) seed it today; digest criterion satisfiable (62.8 done) → **KEEP** (seed immediately; cross-ref 66.2 for items 7-8) |
| 63.4 fix queue | "One fix per AM slot" = dead cadence; fix discipline (failing→passing test + Q/A + live re-capture) is cadence-free → **KEEP, re-worded cadence at sign-off** (criteria untouched) |
| 63.5 regression re-walk | "Week-2 end" wall-clock dead; diff-vs-baseline method valid → **KEEP, re-anchor to fix-queue drain** |
| 64.1 functional-E2E project | Nothing built; single-project config confirmed; premise intact; Mac-runnable, no screenshots → **KEEP** (post-66.2 start per stabilize-first) |
| 64.2 22-route specs | Valid; "PM session runs it nightly" away-flavored; <15min budget sound → **KEEP, MERGE the nightly-runner clause with 64.5** |
| 64.3 backend gap tests | Kill-switch machine/currency/screener/learnings still untested; no away coupling; depends_on null → **KEEP** (runnable now) |
| 64.4 multi-market e2e | Sound, but depends_on 65.1 → **KEEP, repoint dependency to the 65.1/66.2 merged diagnosis** |
| 64.5 CI + nightly runner | CI leg attended-valid; nightly-runner leg greps an away prompt file → **MERGE into 64.2 (CI leg), drop the PM-runner leg or re-home to cron** |
| 65.1 EU funnel diagnosis | Substantially overlaps 66.2(b) but adds per-ticker EU evidence + permanent counters → **MERGE into 66.2; keep counter/per-ticker residue** |
| 65.2 per-market screener dark | Still the likely EU fix; dark+flag-OFF discipline right; token → in-session approval (62.2 never shipped) → **KEEP, gated on merged diagnosis** |
| 65.3 US+KR baseline | ~70% of the since-06-01 window is the freeze → **KEEP-MODIFIED: segment pre-freeze (06-01..06-12) + fresh post-66.2 window** |
| 65.4 three-market proof | North-star, falsifiable; hard-blocked on 66.1+66.2+65.2-ON+65.3; wall-clock week-3 dead → **KEEP, re-anchor window to post-65.2 approval** |

Net: 0 outright drops recommended — the away design was sound; what died was the cadence,
token machinery, wall-clock anchors, and the presumption of a running engine. 2 merges
(65.1→66.2, 64.5→64.2), 3 re-anchors, all sequenced behind the 66.1→66.2 P0 chain.

## Consensus vs debate

Consensus: action items need owner/priority/tracking and re-triage on context change; E2E
minimal + post-stabilization; WIP constrained. Debate: E2E-first contrarians (Autonoma 2025-26,
vendor-motivated) vs Google/Fowler canon — canon holds for a solo-operator system where
maintenance hours are the scarcest resource.

## Pitfalls (from literature → applied)

- Backlog theater: re-listing all 14 steps unchanged = "<50% completion... theater"
  (incident.io). Dispositions must change sequencing/status, not just relabel.
- Building 64.2's full suite against a not-yet-redeployed engine → false-positive burn
  (Rainforest; Capgemini 30-40% maintenance figure).
- Fixing before registering (63.4 before 63.3) — order is load-bearing in both the away
  design and SRE practice.
- Baselining on the contaminated window (65.3) makes 65.4 judge against outage statistics.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (52)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered masterplan 63/64/65/66 + phase-62 statuses + approved plan + config/dir ground truth
- [x] Contradictions/consensus noted (E2E-first contrarian flagged)
- [x] Per-claim citations

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 46,
  "urls_collected": 52,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief_66.5.md",
  "gate_passed": true
}
```
