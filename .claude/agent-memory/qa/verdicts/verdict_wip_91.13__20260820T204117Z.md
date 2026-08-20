STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 91.13
WRITTEN: 2026-08-20T20:41:17Z

## Scope
Immutable criteria (91.13):
1. `grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx` returns 0 after the fix
   (glow prop removed from Total Cost card), OR all 4 cards made visually consistent by
   deliberate design with a documented reason.
2. A Playwright screenshot of the Cost & Token Usage panel shows a uniform border/background
   across all 4 stat boxes.

## A. HARNESS COMPLIANCE (5 items)
- research-gate-before-contract: PASS. research_brief_91.13.md envelope:
  brief_status=COMPLETE, tier=simple, external_sources_read_in_full=7 (floor 5),
  snippet_only=25, urls_collected=32, recency_scan_performed=true, internal_files_inspected=8,
  gate_passed=true. Independently counted 32 unique http(s) URLs literally present in the brief.
  Spot-verified 3 contract citations against the brief and all reproduce with verbatim quotes:
  activecampaign.design KPI-card doc (brief:61), Smashing 2025 "200 to 400 milliseconds"
  (brief:59), W3C WCAG22 SC 2.2.2 "The five-second threshold would not exempt indefinite
  animations" (brief:57). Contract cites the researcher, not decoration.
- contract-before-generate: PASS. mtime chain (UTC):
    research_brief 20:32:30 < contract 20:39:32 < CostDashboard.tsx 20:39:39
    < capture 20:40:46 < experiment_results 20:41:05.
- experiment_results present: PASS.
- log-last: PASS. `grep -Fc "91.13" handoff/harness_log.md` -> 0. masterplan 91.13 = "pending".
- no-verdict-shopping: N/A -- first spawn.
    qa_wip.py 91.13 --spawned-at 2026-08-20T20:41:17Z -> source_present=true,
      attempt_number_status="ok", attempt_number=1, prior_attempts=0,
      records_retained=1 (gauge, = my own record), prior_records=[].
    verdict_history_86_21.py --step 91.13 --evidence-only -> status=no_rows_for_step,
      verdicts=(none). Cross-check: prior_attempts (0) NOT > ledger rows (0) -> no staleness
      signal. sequence: no prior verdicts recorded for this step.
- ** GAP FOUND (6th item, this step's own declared gate): live_check_91.13.md is ABSENT. **
    masterplan 91.13 sets verification.live_check. Executed the real gate:
      live_check_gate.gate_decision(".claude/masterplan.json","91.13","handoff/current")
        -> "skip"      (= live_check set AND artifact MISSING -> hook logs WARN, skips push)
      same call for "91.9"  -> "passed"   (POSITIVE CONTROL: my invocation discriminates)
    Docstring :14-15 + CLAUDE.md: a "skip" exit-0s at auto-commit-and-push.sh:155/181/206,
    BEFORE `git add -A` at :239 -> the commit AND changelog AND push are all held.
    108 live_check_*.md files exist in handoff/current/, incl. live_check_91.9.md written by
    the sibling step ~20 min ago. This step has none.

## B. DETERMINISTIC
- IMMUTABLE CMD: `grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx`
    stdout = 0   (process exit 1 = grep's standard no-match code). Criterion 1 reads on the
    COUNT ("returns 0"), which is 0. MET.
- Evasion check: `grep -n -i 'glow' <file>` -> exit 1, ZERO hits anywhere in the file. No
  `glow={true}`, no multi-line prop, no `className="alpha-score-glow"` back-door.
- Diff is EXACTLY `-<BentoCard glow>` / `+<BentoCard>`; numstat 1 1. Matches the claim.
- Source-path exhaustion: `.alpha-score-glow` has exactly ONE application site repo-wide
  (BentoCard.tsx:19 `glow && "alpha-score-glow"`) and one CSS rule (globals.css:66-67).
  No positional CSS (`:first-child`/`nth-child`) could re-add it -- the only nth-child rules
  are `.gemini-bar:nth-child(2..4)` (globals.css:97-99), scoped elsewhere. All 4 cards pass
  neither `glow` nor `className`, so clsx yields an identical class string for all four.
- Consumer contract: `glow` is NOT orphaned (still used at performance/page.tsx:178 and
  GlassBoxCards.tsx:31), so no interface break. No test references CostDashboard/BentoCard.
- tsc --noEmit: exit 0.
- eslint . : exit 1, 26 errors -- ALL in gitignored build dirs (.next-audit-36-12/,
  .next-functional/ webpack chunks); ZERO in src/. CostDashboard.tsx: 0 errors, 0 warnings.
- Python lint gate: N/A (`git diff --name-only HEAD -- '*.py'` empty). Backend smoke: N/A.

## C. MUTATION MATRIX -- criterion 1's grep guard (run on copies, tree untouched)
  CELL                             grep -c   RESULT
  CONTROL (as shipped)             0         baseline
  M1 restore `<BentoCard glow>`    1         KILLED
  M2 `glow={true}`                 1         KILLED (literal is a substring -- I predicted
                                             SURVIVE; the execution corrected me)
  M3 `className="alpha-score-glow"` 0        SURVIVED grep
  M4 multi-line prop on own line   0         SURVIVED grep
  M5 glow moved to Total Tokens    1         KILLED
  => criterion 1's guard is NARROW but NOT vacuous (kills the filed defect shape and 2 more).
     M3/M4 restore the exact visual defect while evading it -- they are covered by
     criterion 2, so the pair is complementary, not co-vacuous. NOTE-level, not blocking.

## D. CRITERION 2 -- MEASURED, WITH A POSITIVE CONTROL
- WHO TOOK THE CAPTURE (phase-75.20 disclosure): I could NOT re-take it. `browser_click` is
  deliberately absent from the Q/A tool surface, and the Cost tab is pure `useState`
  (ReportTabs.tsx:19) with no URL addressability -- so the panel is unreachable from
  navigate/snapshot/screenshot alone. This is the documented degraded fallback ("tools absent
  from your surface"). I DID drive the live app myself: navigated
  http://localhost:3000/reports and /reports/NTAP behind a live authenticated NextAuth
  session, URL-confirmed both (NOT /login), and waited for async data (the reports table was
  empty on first snapshot, populated on the second).
- INDEPENDENT CORROBORATION of Main's capture, from the playwright server's OWN artifact (not
  Main's prose): .playwright-mcp/page-2026-08-20T20-40-11-561Z.yml -- MRVL / "Marvell
  Technology, Inc." report, "Cost & Token Usage" heading, Total Cost $8.21 / Total Tokens
  154.8K / LLM Calls 31 / Deep Think Calls 4. Values match the PNG and experiment_results
  EXACTLY. Timeline: edit 20:39:39 -> snapshot 20:40:11 -> PNG 20:40:46. Post-fix, live, right
  page. (An earlier "no artifacts in Main's window" observation of mine was WRONG -- a broken
  locale-formatted ls|awk filter. Re-derived with `stat -f %Sm -t` and retracted.)
- QUANTITATIVE MEASUREMENT of the shipped PNG (1120x140, PIL/numpy), not eyeballing:
    4 cards detected, each x-width 271px, y-height 140px
    border RGB left/right/top/bottom = (30,41,59) on ALL FOUR (identical)
    interior background mode        = (12,18,36) on ALL FOUR (identical)
    inter-card gaps gap0_1 / gap1_2 / gap2_3 = (2,6,23) with max==min==mean (zero variance)
- POSITIVE CONTROL (executed, not reasoned): navigated a data: URL rendering 4 cards with the
  verbatim globals.css pulse-glow keyframes + alpha-score-glow on card 0, screenshotted at the
  same 1120x140 geometry:
    gap0_1 mean=(10.55, 34.85, 58.42) max=(15,50,77)   <- halo spills, clearly detectable
    gap1_2 / gap2_3 = (2,6,23)                          <- unchanged
  So the oracle DISCRIMINATES. Same instrument returns pure background on the shipped capture.
- TEMPORAL BLIND-SPOT CLOSED: pulse-glow's trough is NOT zero --
  `0%,100% { box-shadow: 0 0 20px rgba(56,189,248,0.3) }` (globals.css:62). There is no
  animation frame at which a glow would be invisible, so a single screenshot is a sound oracle.
=> Criterion 2 MET.

## E. SCOPE HONESTY -- disclosed out-of-scope claims RE-DERIVED
- "second glow misuse at app/performance/page.tsx:178"  -> REPRODUCES at that exact line.
- "AlphaScoreCard has 0 render sites"                   -> REPRODUCES (only its own definition
  at GlassBoxCards.tsx:25).
- "4 of 5 infinite animations lack prefers-reduced-motion" -> globals.css has 4 `infinite`
  animations (shimmer :56, pulse-glow :67, spin-slow :76, gemini-bounce :94), ALL 4 unguarded,
  and exactly one prefers-reduced-motion block (:178) covering the NumberFlow tint -- which is
  900ms one-shot, NOT infinite. So the phrasing "4 of 5 infinite" is imprecise; the true figure
  is 4 of 4 infinite unguarded. It UNDERSTATES the gap and sits in an explicitly out-of-scope
  follow-up note. NOTE only.
- 3 further modified frontend files (backtest/page.tsx, observability/page.tsx, page.tsx) are
  attributable to the CONCURRENT step 91.9 and disclosed in ITS experiment_results. None touch
  CostDashboard.tsx / BentoCard.tsx / globals.css / the Cost tab. Not a 91.13 scope violation.

## F. TREE NOT FROZEN + MY OWN ARTIFACT
- Sibling 91.9 edited observability/page.tsx again at ~20:49, mid-evaluation. Re-checked at the
  end: CostDashboard.tsx md5 81af7fe96db1b93d0ea4de30f87e8e5b, numstat still 1/1, immutable cmd
  still 0. HEAD moved 53e3f20c -> b81c2b38 during the session but no commit touched
  CostDashboard.tsx or any 91.13 artifact; masterplan 91.13 still "pending".
- I CREATED /Users/ford/.openclaw/workspace/pyfinagent/qa9113_positive_control_glow.png at the
  repo ROOT (the playwright screenshot tool's cwd). It is MY evaluation artifact, not Main's
  work. I am read-only and cannot delete it; auto-commit does `git add -A`, so Main should
  remove it before committing.

## G. CRITERION MAP
  1. grep returns 0 ................. MET  (stdout 0; evasion-checked; M1/M2/M5 killed)
  2. uniform border/background ...... MET  (measured identical border+bg+gaps; positive control)
  harness compliance ................ NOT CLEAN: live_check_91.13.md absent; gate = "skip"

COMPLETED: 2026-08-20T21:00:34Z
