# Evaluator Critique -- phase-91.22
Step: Recharts chart tooltips illegible -- itemStyle never set

Q/A verdict, transcribed VERBATIM from the Workflow qa-verdict rail return value (no editorial
edits, no paraphrase), per CLAUDE.md's Layer-3 harness protocol.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria MET. (1) I independently derived the population with `grep -rn '<Tooltip' frontend/src/` -- 16 files / 21 instances, of which 15 render default content and ALL 15 now set `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` from one shared constant (frontend/src/lib/chart-tooltip-style.ts:15, `{color:'#e2e8f0'}`); the 6 custom-`content` instances are mechanically proven inert for itemStyle (none reads the prop; DefaultTooltipContent never runs there) and are disclosed + queued. (2) Immutable command exit=0, non-zero for all 7 named files (1,2,1,1,1,1,2 = the 9 audit_basis instances), re-run at start and end of eval; all 9 matches are real JSX props, zero comment tokens. (3)+(4) Three live post-fix Playwright captures measured by me at the pixel level: brightest tooltip pixel is exactly (226,232,240)=#e2e8f0 at 14.48:1 in all three, with ~0 black-glyph pixels; Sector Rotation shows both \"Healthcare\" and \"Return : 16.9%\" legible. I recomputed every quantified claim (recharts 2.15.4; es6/DefaultTooltipContent.js:58 `color: entry.color || '#000'` exact; 14.48:1 / 11.87:1 / 1.18:1 -- all reproduce) and ran a BEHAVIOURAL mutation matrix against the installed library via react-dom/server: itemStyle removed -> `color:#000` (defect reproduced), itemStyle set -> `color:#e2e8f0`, itemStyle beats a series `entry.color`, and contentStyle.color alone leaves the item row at #000 (so the 3 removals are behaviourally justified). Root cause independently reproduced: SectorDashboard.tsx:132 `<Bar>` has no `fill`, so entry.color is undefined. tsc --noEmit exit=0; eslint on the 11 changed files exit=0 (6 pre-existing hook warnings at untouched lines). Harness compliance clean: gate_passed=true / 7 sources / 26 URLs / recency scan, research 22:32:42 < contract 22:42:24, log-last verified (0 harness_log rows, status pending, not in any commit), attempt 1 with no prior verdicts. NOTE-level flags only (see notes): a disclosed 32s contract-ordering blur, an off-by-one in the research brief's SUMMARY counts (its anchor table is complete), imprecise \"dead no-op\" phrasing, and four sibling-step edits sitting in the same unfrozen working tree.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "verification_command_exit_code",
    "independent_scope_derivation_git_grep",
    "known_member_recall_test",
    "comment_token_trap_check",
    "behavioural_mutation_matrix_recharts_2.15.4",
    "in_memory_mutation_cells_on_grep_guard",
    "guard_vacuity_4c",
    "claim_reproduction_4b",
    "wcag_contrast_recomputation",
    "recharts_library_source_verification",
    "regression_probe_contentStyle_color_removal",
    "custom_content_exclusion_mechanism_verification",
    "tsc_noEmit",
    "eslint_changed_files",
    "live_playwright_navigate_signals",
    "live_playwright_navigate_home",
    "browser_console_messages",
    "pixel_measurement_of_captures",
    "capture_freshness_mtime",
    "qa_wip_prior_attempts",
    "verdict_ledger_evidence_only",
    "code_review_heuristics",
    "worst_of_n_lenses"
  ],
  "harness_compliance_ok": true,
  "notes": "CAPTURE AUTHORSHIP -- EXPLICITLY-DEGRADED FALLBACK (qa.md 1c, mandatory disclosure): the three hover-state captures were produced by MAIN, not by me. My granted surface is navigate/snapshot/screenshot/console only -- no hover, click or type tool -- and frontend/src/app/signals/page.tsx:15 holds the ticker in local useState with no URL param, so I can neither load the Sector deep-dive nor trigger a Recharts hover. I compensated with: (a) pixel measurement of all three PNGs (brightest px exactly #e2e8f0, 14.48:1, black-glyph pixels 2/105/1 where the 105 is crop-edge chart area, not glyphs); (b) freshness proof -- captures 22:46:50/22:47:20/22:47:44 all postdate every source edit (22:41:52..22:45:30); (c) my OWN live captures at 1440x900 of http://localhost:3000/signals and http://localhost:3000/ -- URLs confirmed (NOT /login), authenticated as pytest@localhost, 0 console errors, Red Line Monitor rendering under current source, page confirmed SETTLED (GATE/KILL/CYCLE/LAST populated, not em-dashes); (d) the executed library-level mutation matrix. I started and killed no server.\n\nSEQUENCE: attempt_number=1, attempt_number_status=\"ok\", attempt_number_is_lower_bound=false, prior_attempts=0, prior_records=[], source_present=true, records_retained=1 (this spawn's own write-first record -- a gauge, not a counter). verdict_history_86_21.py --step 91.22 --evidence-only -> status=\"no_rows_for_step\", verdicts=(none). prior_attempts(0) == ledger rows(0), so no staleness signal. No prior verdict exists for this step; the simultaneous-presentation / sycophancy rules do not apply.\n\nNOTE-LEVEL FINDINGS (none degrades the verdict; all named per severity-dispatch NOTE = PASS-with-flag):\nN1 contract-before-generate blur: frontend/src/lib/chart-tooltip-style.ts mtime 22:41:52 predates contract_91.22.md 22:42:24 by 32s. The contract DISCLOSES it at :32 (\"done during research write-up\") and the other 13 edits all follow it, so the chain is RESEARCH -> research artifact -> CONTRACT -> GENERATE, not a contract fitted to finished code.\nN2 CLAIM DEFECT in the research brief (not in the fix): research_brief_91.22.md:199 says \"16 files, 20 <Tooltip> instances\" and :227 says \"10 files / 14 instances\" render default content. My derivation is 21 and 15. The brief's file:line ANCHOR TABLE is complete -- all 21 anchors are present -- and GENERATE applied 15, so there is NO coverage gap; only the two prose summary numbers are one low. experiment_results' \"~15\" matches reality. Worth correcting in the brief so a future reader does not re-derive from the wrong denominator.\nN3 \"Removed the 3 dead contentStyle.color no-ops\" is imprecise. contentStyle.color never reached ITEM ROWS (behaviourally confirmed: mutation cell M5 leaves the row at color:#000), but it DID reach the LABEL by inheritance from the wrapper div, and none of RedLineMonitor/StrategyDetail/TransformerForecastPanel sets labelStyle. The removal is behaviour-preserving only because globals.css:12-14 sets `body { color:#e2e8f0 }` -- the same value. Confirmed live: the RedLineMonitor capture's label \"2026-08-04\" is legible post-removal. A latent coupling to the body token, not a defect.\nN4 TREE NOT FROZEN DURING EVALUATE. git diff --name-only HEAD carries 4 files with non-91.22 edits: app/observability/page.tsx (phase-91.9), components/CostDashboard.tsx (phase-91.13 BentoCard glow removal), and two comment-only \"phase-X.Y\" -> \"phase X.Y\" rewordings at app/page.tsx:465 and app/backtest/page.tsx:1519 that I could NOT attribute to 91.9/91.13/91.18/91.22 (the audit stream had no matching rows; I do not guess the author). All are comment/UI-copy only with zero runtime and zero tooltip effect, so no criterion is affected -- but the backtest one sits INSIDE a file 91.22 edits, so a per-step auto-commit for 91.22 will sweep in a foreign comment edit unless Main stages selectively.\nN5 KNOWN LIMIT of the immutable command (vacuity shape #3, literal-kept-behaviour-stripped): in-memory mutation `itemStyle={{ color: \"#000\" }}` still yields count 1, so criterion 2's grep cannot distinguish a readable colour from black. It is NOT vacuous overall -- stripping the prop yields 0 -- and criterion 1's \"readable colour\" half rests on three independent behavioural guards (the measured constant, the three measured captures, and mutation cells M1/M2), so this is a named limit rather than a blocking finding.\nN6 Code-review heuristics: no BLOCK or WARN fired. Diff is presentation-only TSX/TS -- no secrets, no subprocess/eval, no kill-switch / stop-loss / perf-metrics / execution path touched, no dependency pin changed, no LLM-output sink. anti-rubber-stamp satisfied by the executed library-level mutation matrix rather than by a source scan.\n\nFOLLOW-UPS the author queued and I agree are genuinely separate (do not treat as gaps in this step): contrast audit of the 6 custom-content tooltip renderers (their own JSX; all currently use readable slate/amber tokens, so no instance of the reported black-text defect survives), and WCAG SC 1.4.13 hoverable/dismissible tooltip content.\n\nWIP record: .claude/agent-memory/qa/verdicts/verdict_wip_91.22__20260820T205120Z.md (COMPLETE; evidence for a future spawn, never a verdict). Q/A's own screenshots landed at repo-root qa_91_22_signals_live.png and qa_91_22_home_redline_live.png (no handoff/ write access) -- Main relocated/removed them; they are not part of this step's evidence set.",
  "escalation": {
    "sequence_supplied": null,
    "sequence_status": "not_supplied",
    "consecutive_conditionals": null,
    "would_auto_fail": null,
    "attempt_number": null,
    "budget_exhausted": null,
    "max_attempts": 5,
    "burden_on": "the party departing from the computed escalation",
    "override": null,
    "override_reason": null,
    "judge_was_told_consequence": false,
    "judge_was_told_consequence_evidence": null
  },
  "research_routing": {
    "research_needed": null,
    "research_brief_spec": null,
    "next_action_on_research_needed": null
  },
  "verdict_unmodified": true
}
```
