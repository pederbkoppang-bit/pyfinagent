---
step: phase-23.2.24
title: Fix React Rules-of-Hooks bug + harden Q/A with frontend lint coverage
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_24.py'
research_brief: handoff/current/phase-23.2.24-external-research.md (also see phase-23.2.24-internal-codebase-audit.md)
---

# Contract — phase-23.2.24

## Hypothesis

User opened `/cron` after phase-23.2.23 shipped and saw a runtime
React error: "React has detected a change in the order of Hooks
called by JobsTab. ... 9. undefined / useMemo." The Q/A for
phase-23.2.23 returned PASS without catching this.

User feedback (verbatim 2026-05-07): "the code didnt work as you see
it should also check everything like copilot review on github with a
workflow that everything works. if not it is sent back to main for
fixing. research how copilot reviw on github is setup."

**Two distinct issues:**

A) **Bug location**: `frontend/src/app/cron/page.tsx::JobsTab` line
   218 calls `useMemo` AFTER three conditional early returns at lines
   171, 180, 205. Classic Rules-of-Hooks violation. React detects
   hook-count mismatch between renders and errors at runtime in dev
   mode. Production build behavior is undefined.

B) **Q/A coverage gap**: `.claude/agents/qa.md:41-54` deterministic
   checks block lists `python -c "import ast"`, `test -f`, the
   immutable verification command, and `pytest`. It does NOT include
   a frontend `eslint` invocation. The bug WOULD have been caught by
   `npx eslint . --max-warnings=0` because
   `frontend/eslint.config.mjs:34` already sets
   `react-hooks/rules-of-hooks: "error"`. The gap is that Q/A never
   runs the project's own lint script.

**TypeScript can't catch this.** Hook-call ordering is a runtime
execution-order constraint with no model in the type system. ESLint
react-hooks plugin performs AST-level control-flow analysis and IS
the canonical guard.

**Anthropic harness pattern is structurally sound.** The existing
FAIL→Main→re-Q/A loop is already what the user wants ("send back to
main for fixing"). What's missing is check coverage — Q/A needs to
run the right commands. GitHub Copilot review (per researcher's
audit) is advisory-only and cannot block merges; the pyfinagent
harness is structurally STRONGER once the check rubric is fixed.

## Research-gate summary

Researcher (ac0bca7dac3b163da) returned `gate_passed: true`:
- 10 sources read in full via WebFetch (GitHub Copilot review configure
  + use docs, React rules-of-hooks docs, Next.js ESLint config,
  ESLint configure-rules, Anthropic harness-design + multi-agent
  research, KAIRI Copilot 2025 analysis, dev.to Copilot guide 2026,
  eslint-plugin-react-hooks rules index)
- 20 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 with 3-variant query discipline; ESLint 9
  flat config + Copilot review GA (Apr 2025) + agentic update Mar 2026
  noted
- 7 internal files inspected with file:line anchors
- Concrete recommendation: literal commands + literal diff snippet
  for `qa.md`

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `frontend/src/app/cron/page.tsx::JobsTab` is restructured so EVERY
   hook (`useState`, `useRef`, `useCallback`, `useEffect`, `useMemo`)
   is called BEFORE any conditional `return`. The `useMemo` call must
   move above the loading/error/empty early-return branches OR be
   replaced with a different idiom that respects Rules-of-Hooks.
2. `cd frontend && npx eslint .` exits 0 (errors-only). The
   hook-order class of bug surfaces as an ERROR (`rules-of-hooks` is
   already set to `"error"` severity in
   `frontend/eslint.config.mjs:34`) so this catches the actual user
   complaint. The codebase contains 37 pre-existing WARNINGS
   (set-state-in-effect, exhaustive-deps, an unused eslint-disable
   directive) that this phase intentionally does NOT fix because the
   user asked for the cron crash + Q/A hardening, not a 37-warning
   cleanup. Those warnings are documented in the Honest Disclosures
   section as a phase-2 deferral with their current count baseline.
3. `cd frontend && npx tsc --noEmit` exits 0.
4. `.claude/agents/qa.md` is updated to include a new mandatory
   deterministic-check section "1b. Frontend lint" containing the
   literal commands above. Q/A must run them for ANY phase whose
   diff touches `frontend/**` or `.claude/agents/qa.md` itself. The
   section must explicitly state that `tsc --noEmit` does NOT catch
   hook-order violations and that ESLint with
   `react-hooks/rules-of-hooks: "error"` is the canonical guard.
5. `tests/verify_phase_23_2_24.py` runs `npx eslint . --max-warnings=0`
   from `frontend/` and asserts exit 0. This becomes a regression
   net so a future hook-order violation in any frontend file fails
   the verifier deterministically. (Subsequent phase verifiers can
   reuse the helper.)
6. `frontend/package.json` lint script is unchanged at `"eslint ."`
   (errors-only). Tightening to `--max-warnings=N` with a baseline
   is deferred — see criterion 2 honest-disclosures discussion.
7. `docs/runbooks/per-step-protocol.md` (or a clearly cross-linked
   section) documents the explicit retry-on-FAIL loop: when Q/A
   returns FAIL, Main reads `violated_criteria`, fixes the blockers,
   updates `experiment_results.md`, then spawns a fresh Q/A. Max-3
   retries before `certified_fallback`. Cite Anthropic's
   harness-design doc and the existing 3rd-CONDITIONAL auto-FAIL
   rule so this is unified, not contradicting.
8. The phase's experiment_results.md "Honest disclosures" section
   names the phase-23.2.23 Q/A miss as a real-world example — what
   was missed, why, and what the new rubric would have done.
9. `python tests/verify_phase_23_2_24.py` exits 0 with all checks
   green, including the live ESLint run.
10. The `/cron` page renders without console errors when opened in
    a browser. Best-effort verification via the existing axe / build
    tooling; if that's not feasible inside the harness, document the
    manual reproduction steps the user can run.

## Plan steps

1. **Fix `frontend/src/app/cron/page.tsx::JobsTab`** by moving
   `useMemo` above the early returns. The `useMemo` callback must
   tolerate `jobs === null` (return `{}`); the rendering code below
   the early returns is unaffected because `jobs` is non-null when
   it runs.
2. **Update `frontend/package.json`** lint script to add
   `--max-warnings=0`.
3. **Update `.claude/agents/qa.md`** — insert the literal "1b.
   Frontend lint" section per researcher's diff snippet, with the
   `cd frontend && npx eslint . --max-warnings=0` and
   `cd frontend && npx tsc --noEmit` commands and the explanation
   that TypeScript does NOT catch hook-order violations.
4. **Update `docs/runbooks/per-step-protocol.md`** — add (or
   tighten) the §4-EVALUATE retry-on-FAIL section. Quote Anthropic
   harness-design. Specify the max-3 ceiling already in
   `feedback_harness_rigor.md` / 3rd-CONDITIONAL auto-FAIL.
5. **New `tests/verify_phase_23_2_24.py`** — runs `npx eslint .
   --max-warnings=0` AND `npx tsc --noEmit` from `frontend/` via
   subprocess. Captures verbatim output on failure. Also greps
   qa.md for the new section header.
6. **Run full regression**: `verify_phase_23_2_23.py` should still
   pass (cron page exists + tabs + no emoji); new
   `verify_phase_23_2_24.py` should pass; backend pytest unaffected.
7. **Live browser check**: best-effort instructions in
   experiment_results.md for the user (open localhost:3000/cron in
   the existing dev server, confirm no console errors).
8. **Append `harness_log.md`** AFTER Q/A PASS, BEFORE any
   masterplan flip.

## Out of scope

- Adding ESLint to the harness's pre-commit hook (a separate phase
  with its own deny-list audit).
- Migrating other React components to verify they don't have similar
  hook-order violations beyond what `eslint . --max-warnings=0`
  catches automatically across the entire repo.
- Building a GitHub Actions workflow that mirrors Copilot review
  semantics (the researcher's analysis confirms our existing harness
  is structurally stronger; replicating Copilot's advisory-only
  pattern would be a downgrade).
- Restructuring qa.md to add per-domain check matrices (e.g., "if
  diff touches `backend/services/cycle_health.py`, run X"). The
  generic "if frontend touched, run lint" is sufficient for now.
- Adding a Playwright / browser-runtime smoke test (heavyweight; out
  of scope for a check that's already statically caught).

## Backwards compatibility

- The hook-order fix preserves `useMemo` semantics — `grouped` is
  still recomputed only when `jobs` changes; the only difference is
  that the empty case `jobs === null` produces `{}` and the empty
  Object.entries map is rendered through the same JSX path.
- `--max-warnings=0` strictly tightens lint; any existing warnings
  in the codebase that would surface need to be cleaned in this
  phase OR explicitly suppressed with a per-line `// eslint-disable`
  comment + reason.
- `qa.md` change is purely additive to the deterministic-check list;
  existing checks unaffected. Future Q/A runs that don't touch
  `frontend/**` will skip the new section.
- `per-step-protocol.md` change is doc-only.
- `package.json` change tightens local `npm run lint` only; CI is
  already separate.

## References

- Researcher: `handoff/current/phase-23.2.24-external-research.md`,
  `handoff/current/phase-23.2.24-internal-codebase-audit.md`
- Bug: `frontend/src/app/cron/page.tsx:218`
- Existing config: `frontend/eslint.config.mjs:34` (rule already
  enabled but never run by Q/A)
- Q/A definition: `.claude/agents/qa.md:41-54`
- React official docs: rules-of-hooks
- Anthropic harness-design (file-based handoff, retry-on-FAIL)
- GitHub Copilot review docs (confirms advisory-only — pyfinagent
  harness already stronger)
