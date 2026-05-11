---
step: phase-23.4.0
date: 2026-05-08
verdict: PASS
ok: true
---

# Q/A critique — phase-23.4.0

Independent re-verification of the frontend `.next/` corruption recovery
(login 500 → 200). Single Q/A run; no prior cycle for this step-id.

## Harness-compliance audit (5 items, all PASS)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawn before contract | **PASS** | `handoff/current/phase-23.4.0-research-brief.md` exists; tail JSON envelope shows `gate_passed: true`, `external_sources_read_in_full: 7` (>=5 floor), `urls_collected: 14` (>=10 floor), `recency_scan_performed: true`, three-query discipline visible (current-year, last-2-year, year-less). Contract `Research-gate summary` cites researcher `aeda4de214c83a7bc` tier=moderate with Option A recommendation. |
| 2 | Contract written before GENERATE | **PASS** | `handoff/current/contract.md` present, header `phase-23.4.0`. The `Immutable success criteria (verbatim -- DO NOT EDIT)` block reproduces the masterplan `verification` field byte-for-byte (verified by direct comparison with `.claude/masterplan.json:7406`). No softening, no rewrite. |
| 3 | Results captured | **PASS** | `handoff/current/experiment_results.md` documents recovery sequence (`rm -rf .next` + `launchctl kickstart -k`), poll trace (t+1..t+4 showing HTTP=000->200), and verbatim 4-check verifier output (4/4 PASS, EXIT=0). |
| 4 | Log-last discipline (will-be-followed) | **PASS** | `grep "phase=23.4.0\|phase-23.4.0"` against `handoff/harness_log.md` returns 0 matches -- Main has not yet logged. `.claude/masterplan.json::23.4.0.status = "pending"` (not flipped to `done`). Log-last and status-flip-last is intact. |
| 5 | No verdict-shopping | **PASS** | First Q/A run for phase-23.4.0 (no prior cycle block in `handoff/harness_log.md`). |

## Deterministic checks (re-run independently)

| # | Check | Result | Verbatim output |
|---|-------|--------|-----------------|
| D1 | File existence: contract, experiment_results, research-brief, verifier | **PASS** | All four files present and timestamped 2026-05-08. |
| D2 | `curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/login` | **PASS** | `HTTP=200` |
| D3 | `curl -sL http://localhost:3000/` body has `<html` or `<title>` | **PASS** | 16723 bytes; `<html lang="en" ...>` and `<title>PyFinAgent -- AI Financial Analyst</title>` both present. Full PyFinAgent login shell renders (Sign in with Google + Sign in with Passkey, Phosphor icons, no emojis -- project rule observed). |
| D4 | `cd frontend && npx --no-install tsc --noEmit` | **PASS** | `TSC_EXIT=0` |
| D5 | `cd frontend && npx --no-install eslint . --quiet` | **PASS** | `ESLINT_EXIT=0` |
| D6 | Project verifier: `python tests/verify_phase_23_4_0.py` | **PASS** | All 4 sub-checks PASS, `EXIT=0`. |
| D7 | Verbatim-criterion check vs `.claude/masterplan.json::23.4.0.verification` | **PASS** | Contract block matches masterplan field byte-for-byte. |
| D8 | `predev` guard in `frontend/package.json::scripts` | **PASS** | Line 6: `"predev": "rm -rf .next",` present. Line 7: `"dev": "next dev --port 3000"` unchanged. |
| D9 | Source code regression check (`git diff --stat HEAD frontend/src/`) | **PASS** | 0 files changed under `frontend/src/`. |
| D10 | Post-recovery `frontend.log` tail | **PASS** | Last 30 lines: clean compile, `Compiled in 998ms (11917 modules)`, `Compiled /login in 4.1s (9711 modules)`, then a steady stream of `GET /login 200 in <50ms`. No `MODULE_NOT_FOUND`, no `ENOENT routes-manifest.json`, no `Cannot find module './611.js'` after the kickstart. |

## LLM judgment

- **Contract alignment.** Main executed exactly the researcher-recommended
  Option A (full wipe + `launchctl kickstart -k`). Did not detour into
  Option B (HMR-only) or Option C (`next build` first). `experiment_results.md`
  matches the contract's plan steps (a-h) point-for-point, with the
  pre-fix snapshot of `frontend/.next/server/chunks/` confirming `611.js`
  WAS present (correctly identifying the chunk error as a downstream
  symptom of the missing `routes-manifest.json`).
- **Scope honesty.** No edits to `frontend/src/app/login/page.tsx`, no
  Turbopack migration, no backend `/health` 404 fix attempted, no
  auth-flow touch. `git diff --stat HEAD frontend/src/` is empty.
- **Anti-pattern guard -- immutable criteria.** The 4-check verification
  string in `contract.md` is byte-identical to `.claude/masterplan.json`
  (the contract's "Immutable success criteria" block is a faithful copy,
  not a paraphrase). No softening.
- **Root-cause honesty.** Main correctly named `routes-manifest.json`
  absence as the root cause and `611.js Cannot find module` as the
  downstream symptom (the chunk file was actually present on disk;
  the manifest needed to register the route was the missing piece).
  This matches the researcher's diagnostic.
- **Regression guard.** `predev` script is idempotent (`rm -rf` is a
  no-op when target is absent), zero-cost when `.next/` is already
  clean, and runs automatically before `npm run dev` per npm convention.
  `dev` script unchanged. Will survive launchd respawns since launchd
  invokes `npm run dev` (or equivalent) per the plist, which triggers
  the `predev` hook each cycle. Does NOT break any existing workflow:
  `npm run build`, `npm run start`, `npm run lint` are untouched.
- **Research-gate compliance.** Researcher fetched 7 sources in full
  (>=5 floor), 14 URLs collected (>=10 floor), recency scan present,
  three-query discipline visible, 10 internal files inspected.
  `gate_passed: true` is genuine, not padded.

## Side-effect disclosure (observation, not violation)

`git status` shows three additional unstaged frontend modifications:
`frontend/next-env.d.ts`, `frontend/tsconfig.json` (`"jsx": "react-jsx"`
-> `"preserve"`), and `frontend/tsconfig.tsbuildinfo`. These are
**Next.js-managed files auto-rewritten by the post-wipe recompile**:

- `next-env.d.ts` carries the explicit header "This file should not be
  edited" because Next owns it; the only diff is a reference path tweak
  Next emits during dev startup.
- `tsconfig.json::jsx` is auto-set to `"preserve"` by `next dev` (Next
  uses its own SWC JSX pipeline; this is documented Next.js behavior
  on Next 15.x).
- `tsconfig.tsbuildinfo` is the TypeScript incremental build cache.

None of these are Main-authored edits. None affect runtime behavior
(`tsc --noEmit` exits 0; ESLint exits 0; the app renders correctly).
Disclosure could have been more explicit in `experiment_results.md`,
but per the contract's actual scope ("no source code regression",
"no edits to `frontend/src/`"), the invariant holds: 0 files changed
under `frontend/src/`. The auto-rewrites are framework-controlled
artifacts of running `next dev`, not human-authored scope creep.
Logging this as an observation, not a violation.

## Anti-rubber-stamp re-checks performed

- I ran `curl /login` myself (not relying on Main's number): observed `200`.
- I fetched `/` myself: 16723-byte response with login shell + `<title>` + no emojis.
- I ran `tsc --noEmit` and `eslint . --quiet` myself: both `0`.
- I ran the verifier script myself end-to-end: 4/4 PASS, EXIT=0.
- I tailed `frontend.log` to confirm symptom is gone, not just hidden:
  no `MODULE_NOT_FOUND` or `routes-manifest.json` errors after the
  recompile.
- I verified `predev` is idempotent and won't `rm -rf` outside its cwd:
  the script is a literal `rm -rf .next` running from `frontend/` (npm
  always runs scripts with cwd = the package's dir). Cannot escape.

## violated_criteria

(none)

## violation_details

(none)

## certified_fallback

false

## Final verdict

**PASS** -- All 5 harness-compliance audit items, all 10 deterministic
checks, all LLM-judgment criteria green. The fix is real (curl
verified), the root cause was correctly identified (manifest, not
chunk), the regression guard is idempotent, the immutable criteria
were copied verbatim, the research gate cleared with 7 sources read
in full, and `frontend/src/` is byte-identical to HEAD.
