# Evaluator Critique — Step 70.1 (S1: make the setting changeable)

**Evaluator:** fresh, independent Q/A via the Workflow structured-output path (Opus 4.8, `effort: max`,
$0 Max rail, stall-immune). Verdicts transcribed VERBATIM by Main (no-self-eval guardrail).

## Cycle 1 — verdict: CONDITIONAL (run wf_98dd8a92-b0b)

**Harness compliance 5/5 PASS** (audited first). **Deterministic:** verification exit 0, `npm run build`
green, no backend change (frontend/ + handoff/ only), no emoji. **do_no_harm_ok: true.** No criterion
VIOLATED in the implementation — all 5 met in code, backend untouched, no risk-limit threshold moved.

**Why CONDITIONAL (evidentiary, not a code defect):** the binding UI live-capture gate (qa.md §1c) was only
partially satisfied — `70.1-risk-limits-panel.png` captured the page TOP, not the risk panel (it is below the
fold), so criteria 3 & 4 (the risk panel — the step's largest new UI surface) and criterion 2 (the
out-of-range inline error + disabled Save) had no VISUAL live evidence, only prose. Q/A remediation guidance:
re-capture (a) the RiskLimitsPanel with an active override (amber badge + warning banner + configured-vs-
effective + enabled Clear) and (b) the out-of-range error with Save disabled; update the live_check; fresh Q/A.

**Non-blocking notes (carry-forward, do not block):** RiskLimitsPanel mount-fetch effect + PaperSettingNum
re-seed/onValidity effects trigger React-Compiler "setState in effect" advisories — but this is a PERVASIVE
pre-existing project pattern (66 instances across 39 files), consistent with the codebase, non-fatal, not a
regression. The re-seed effect could theoretically clobber a cleared-but-unsaved field if `stored` changes
via a background context refresh (narrow edge case). Neither blocks.

## Cycle-2 remediation applied by Main (evidence only — NO code change)

Per the canonical cycle-2 flow (fix the flagged gap + update the handoff files + fresh Q/A on updated
evidence — NOT verdict-shopping, the evidence changed):
- Added `70.1-out-of-range-error.png` — element screenshot of the Trading-settings card showing the inline
  "Must be between 0 and 20." error, the "Fix the highlighted fields before saving." summary, and disabled Save.
- Added `70.1-risk-panel-override-active.png` — element screenshot of the RiskLimitsPanel with an ACTIVE
  override on Max NAV % per sector: amber warning banner, "OVERRIDE ACTIVE" badge, configured 30 vs effective
  25, enabled Clear. (Override set for the capture then CLEARED — `active_overrides: []`, `max_per_sector = 2`.)
- Updated `live_check_70.1.md` to map each capture to the criterion it evidences.
- No production code changed between cycle 1 and cycle 2 (git diff on frontend/ is identical); only the
  screenshots + live_check were added.

## Cycle 2 — verdict: CONDITIONAL (run wf_a21250f3-a3a)

Harness compliance 5/5, verification exit 0, build green, no backend change, no emoji, do_no_harm true, all 5
criteria met in CODE, no criterion violated. **Block was again purely evidentiary:** the criteria-3&4 risk-panel
capture (`70.1-risk-panel-override-active.png`, 1109×498) was now correct (amber banner + OVERRIDE ACTIVE badge
+ configured 30 vs effective 25 + enabled Clear), BUT `70.1-out-of-range-error.png` was a 398×48 crop showing
only the card HEADER — the element selector grabbed the header ref, not the card body — so criterion 2's
out-of-range error + disabled Save still had no valid visual evidence, and the live_check mapped a capture whose
content contradicted the description. Non-blocking carry-forwards unchanged. 2nd consecutive CONDITIONAL (below
the 3rd-consecutive auto-FAIL threshold).

## Cycle-3 remediation applied by Main (evidence only — NO code change)

- Re-captured `70.1-out-of-range-error.png` as an element screenshot of the FULL Trading-settings card
  (1109×859). **Main visually confirmed** (Read the PNG) it shows the "MAX POSITIONS PER SECTOR" field = `25`
  with a rose border, the inline "Must be between 0 and 20." error directly below it, the rose "Fix the
  highlighted fields before saving." summary banner, and the disabled Save button.
- Root cause of the two bad captures: this page's shell is `h-screen overflow-hidden` with an inner
  `overflow-y-auto` scroll container, so `fullPage` only yields the 1440×900 viewport (fields below the fold) and
  a `:has-text` div selector matched the small header — an element screenshot of the card `div.rounded-xl`
  (filtered by the "Trading settings" heading) is the correct tool and captures the full card.
- Updated `live_check_70.1.md` with the cycle-3 correction note. No production code changed between cycles 1–3
  (git diff on `frontend/` unchanged since 20:14); only the one screenshot + live_check text were updated.

## Cycle 3 — verdict: PASS (run wf_cd856131-e18)

Fresh independent Q/A (Workflow structured-output, Opus 4.8) that VISUALLY INSPECTED the capture PNGs (Read
tool). **verdict: PASS** | violated_criteria: [] | do_no_harm_ok: true | live_capture_gate.satisfied: true.

Visual inspection (what the Q/A SAW, verbatim):
- `70.1-out-of-range-error.png` (full Trading-settings card, 1109×859): the MAX POSITIONS PER SECTOR field
  contains 25 with a highlighted rose border and spinner arrows; a rose inline error "Must be between 0 and 20."
  sits directly below the field; a rose summary banner near the card top reads "Fix the highlighted fields
  before saving."; the top-right Save button is rendered muted/grayed (disabled) versus the tinted enabled Save
  in the saved-state capture — all four criterion-2 elements visible.
- `70.1-risk-panel-override-active.png`: the "Risk limits (live overrides)" panel shows an amber warning banner
  with a warning-triangle icon ("1 active override shadowing your saved settings: Max NAV % per sector..."); the
  Max NAV % per sector row shows CONFIGURED 30 and EFFECTIVE 25 (25 in amber), an "OVERRIDE ACTIVE" badge, and an
  enabled Clear button while non-overridden rows have muted Clear buttons — criterion-3 elements present, and the
  nav_pct Set input on that row is its editor (criterion 4).
- `70.1-manage-fixed-fullpage.png`: green "Settings saved." banner; MAX POSITIONS PER SECTOR = 5 — criterion 1.

Harness compliance 5/5 (research-gate-before-contract, contract-before-generate, results present, log-last, no
verdict-shopping — the only on-disk change since cycles 1/2 is the re-captured PNG + live_check text; frontend
code diff unchanged since cycle 1, so this is the documented file-based cycle-N flow, not shopping). Deterministic:
verification exit 0, no backend change (git clean of backend), build green (trusted — no code changed since
cycle 1), no emoji (the only regex hit is a pre-existing `->` arrow in an api.ts comment, not an added line, not
pictographic). Code spot-verified independently (manage/page.tsx Save disabled via hasFieldError; cockpit-helpers
useState<string> fix + "Must be between" message + onValidity + aria-invalid; api.ts get/set/clear with
SET_RISK_LIMIT token; positions/page.tsx false comment removed; RiskLimitsPanel Phosphor IconWarning + states).
Auto-FAIL rule acknowledged (3rd cycle after 2 CONDITIONALs): the Q/A judged PASS on genuine visual + code
evidence, not reflexively. do_no_harm: UI + api.ts only, no backend route change, no live-loop behavior change, no
risk-limit threshold moved; capture override set then CLEARED (active_overrides=[], max_per_sector=2 restored).
Non-blocking carry-forwards (React-Compiler setState-in-effect advisories matching a pervasive pre-existing
project pattern; the re-seed effect's narrow clobber edge case) recorded, do not block.

**Cycle summary:** 3 Q/A cycles — CONDITIONAL (risk panel + out-of-range not visually captured) → CONDITIONAL
(out-of-range capture grabbed the card header) → PASS (out-of-range re-captured as full-card element screenshot,
visually confirmed). No production code changed across the three cycles; the harness rigor was entirely about
producing valid live-UI evidence for the binding qa.md §1c gate.
