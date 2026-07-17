# Evaluator Critique — Step 64.1 (Functional-E2E Playwright project + smoke)

**Evaluator:** fresh, independent Q/A via `.claude/workflows/qa-verdict.js` (Workflow structured-output, Opus 4.8,
`effort:max`, `model:opus`, $0 Max rail). Verdicts transcribed VERBATIM by Main + persisted to
`handoff/current/evaluator_critique.json`.

## Cycle 1 — verdict (transcribed VERBATIM; run wf_7f1eca97-1bd)

**verdict: CONDITIONAL** | ok: false | harness_compliance_ok: true | certified_fallback: false

**violated_criteria:** ["Unjustified_Inference: visual-regression CI decoupling claim -- functional project runs
under bare `npx playwright test` (visual-regression.yml lines 62/75, CI, no LIGHTHOUSE_SKIP_AUTH) and fails against a
:3100 that path never starts"]

**reason (verbatim):** All 3 immutable criteria MET and do-no-harm to operator :3000 is fully clean, but the
experiment_results' load-bearing claim "no CI regression to visual-regression.yml (runs `npx playwright test` without
--project)" is materially INACCURATE, so PASS would rubber-stamp a falsified overclaim. Criteria evidence: (1)
functional project exists with testDir ./tests/e2e-functional and ZERO screenshot assertions MET; (2) immutable
command exited 0 with "1 passed" reproduced 4x by me plus author's 4x MET; (3) NEXT_PUBLIC_E2E_TESTING injected in the
:3100 webServer env MET. Do-no-harm verified independently: :3000 /login=200 and /=302 before AND after every run,
next-env.d.ts/tsconfig.json git-clean after (teardown restored to .next), next.config.js has NO distDir key when
PLAYWRIGHT_DIST_DIR unset (byte-identical prod/CI), :3100 torn down. THE GAP: `npx playwright test --list` (CI path,
no LIGHTHOUSE_SKIP_AUTH) enumerates 9 tests across BOTH projects incl. [functional] smoke.spec.ts; that path's
webServer ternary starts ONLY :3000, so the functional smoke (baseURL :3100) runs against a down :3100 and FAILS.
visual-regression.yml lines 62 & 75 run bare `npx playwright test` (CI=true) and trigger on push/PR to main with paths
frontend/** -> that CI step would go red. The webServer ternary decouples the SERVER but not TEST SELECTION (functional
project is unconditionally in projects[]). Fix: gate the functional project behind LIGHTHOUSE_SKIP_AUTH (mirror the
webServer ternary) OR scope visual-regression.yml to --project=chromium, and correct the experiment_results claim; then
re-spawn a fresh Q/A on changed evidence.

**notes (verbatim, abridged):** Harness-compliance 5/5 clean (research-before-contract gate_passed; mtime ordering;
results present incl. the required incident disclosure; log-last; cycle 1 no verdict-shopping). The honest INCIDENT
disclosure (intermediate global webServer array ran predev rm -rf .next against :3000 -> 500/404, recovered +
permanently isolated) is correct scope-honesty and the FINAL state is verified do-no-harm-clean; that disclosure is
NOT the basis for the CONDITIONAL. The CONDITIONAL rests SOLELY on the falsified 'no CI regression' claim + the
concrete latent breakage of visual-regression.yml's bare `npx playwright test`. Recommended fix is a ~3-line change
(conditional projects[] gated on LIGHTHOUSE_SKIP_AUTH) that preserves all 3 criteria (the immutable command sets
LIGHTHOUSE_SKIP_AUTH=1 so functional stays present) while eliminating the regression.

## Cycle 2 — Main's disposition (record; the fix), fresh Q/A being spawned

The CONDITIONAL is a CORRECT, valuable catch (a real latent CI regression + a falsified claim). Fix applied (the
Q/A's recommended ~3-line change):
- **`playwright.config.ts`**: the functional PROJECT is now also gated on `LIGHTHOUSE_SKIP_AUTH`
  (`...(process.env.LIGHTHOUSE_SKIP_AUTH ? [functionalProject] : [])`), mirroring the webServer ternary. VERIFIED: bare
  `npx playwright test --list` → **[functional] count = 0** (8 chromium tests only); `LIGHTHOUSE_SKIP_AUTH=1 ... --list`
  → functional smoke present. So the visual-regression CI never enumerates/runs the functional smoke → no regression.
  (Also fixed a tsc widening error the gating exposed: `reducedMotion: "reduce" as const`.)
- **`experiment_results.md`**: corrected the decoupling claim to the accurate `--list`-verified statement (both the
  project AND its webServer are gated; bare run = 0 functional tests).
- Re-verified: immutable cmd exit 0 ("1 passed"); :3000 /login 200 before+after (untouched); next-env.d.ts/tsconfig
  git-clean; tsc exit 0.

This is the canonical cycle-2 flow: Main fixed the blocker + updated the handoff files; a FRESH independent Q/A now
judges the CHANGED evidence (NOT verdict-shopping — the config + claim genuinely changed). The new verdict is
transcribed VERBATIM below.

## Cycle 2 — fresh Q/A verdict (transcribed VERBATIM; run wf_4f659b10-c51)

**verdict: PASS** | ok: true | harness_compliance_ok: true | certified_fallback: false | violated_criteria: []

**reason (verbatim):** All 3 immutable criteria MET on CHANGED (cycle-2) evidence, and the Cycle-1 CONDITIONAL's sole
basis -- the falsified 'no CI regression' claim -- is now ELIMINATED and independently reverified. (1) The functional
project exists in playwright.config.ts (name:'functional', testDir './tests/e2e-functional') with ZERO screenshot
assertions. (2) The immutable command exited 0 with '1 passed (6.3s)', reproduced by me against a freshly-started
:3100 auth-bypass server (:3100 was down pre-run, Playwright started+tore it down). (3) NEXT_PUBLIC_E2E_TESTING is
injected in the :3100 webServer env matching the existing config note; honestly disclosed to have NO app consumer
today so the <60s smoke cannot flake. CI-REGRESSION FIX VERIFIED: bare `npx playwright test --list` (the exact
visual-regression.yml path, CI=true, no env) enumerates 8 chromium tests and [functional] count = 0 -- the corrected
experiment_results claim matches exactly. DO-NO-HARM clean: :3000 login=200/root=302 before AND after; next-env.d.ts +
tsconfig.json git-clean post-run (teardown restored); :3100 torn down. NO unintended production change (git diff:
next.config.js conditional distDir UNSET by default -> byte-identical, and playwright.config.ts functional project +
webServer both gated on LIGHTHOUSE_SKIP_AUTH; new tests/e2e-functional/ + .gitignore + handoff docs). tsc exit=0,
eslint exit=0.

**notes (verbatim):** CYCLE-2 fresh-respawn on CHANGED evidence (NOT verdict-shopping): projects[] now conditionally
spreads the functional project on LIGHTHOUSE_SKIP_AUTH + experiment_results corrected the decoupling claim. 3rd-
CONDITIONAL auto-FAIL NOT triggered. CONCUR WITH CYCLE-1 on the disclosed do-no-harm INCIDENT (predev rm -rf .next
against :3000 -> 500/404, recovered + permanently isolated) -- honest scope-disclosure, does NOT cap the verdict;
FINAL state independently reverified clean. ONE NON-BLOCKING observation (register-only): global-teardown.ts restores
next-env.d.ts/tsconfig.json to HEAD when they contain '.next-functional'; in the narrow case of UNRELATED uncommitted
tsconfig edits + a functional-suite run, the restore would also discard those edits -- acceptable for best-effort
local test-infra (try/catch, never throws, inert in CI). Live book untouched; historical_macro FROZEN;
kill-switch/stops/caps/DSR/PBO untouched.

Full machine-readable verdict persisted to handoff/current/evaluator_critique.json (step_id=64.1, cycle_num=2).

## Main's disposition
PASS, violated_criteria=[]. The Cycle-1 CONDITIONAL (a real latent CI regression + a falsified claim) was correctly
caught and fixed via the ~3-line project-gating change; the fresh Q/A independently reverified the regression is gone
(bare `--list` → 0 functional). The 1 non-blocking teardown observation is accepted (register-only; best-effort local
test-infra; inert in CI). Proceeding to LOG (Cycle 105, one entry: result=PASS with the cycle-1 CONDITIONAL noted
inline) then flip 64.1 -> done.
