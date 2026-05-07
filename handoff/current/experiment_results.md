---
step: phase-23.2.24
cycle_date: 2026-05-07
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_24.py'
---

# Experiment Results — phase-23.2.24

## Hypothesis recap

Phase-23.2.23 shipped `/cron` with a Rules-of-Hooks violation in
`JobsTab` (useMemo after 3 early returns). Q/A returned PASS without
catching it because its deterministic-checks block did not include
ESLint. User feedback (verbatim): "the code didnt work as you see it
should also check everything like copilot review on github with a
workflow that everything works. if not it is sent back to main for
fixing. research how copilot reviw on github is setup."

Two issues — bug + Q/A coverage gap. Fixed both in this phase.

## What was changed

### Fix A — Rules-of-Hooks violation
`frontend/src/app/cron/page.tsx::JobsTab`:
- `useMemo` moved ABOVE the three conditional early returns so it is
  called on every render. Tolerates `jobs === null` by returning `{}`;
  the rendering path that consumes `grouped` is gated by the same
  early returns so `{}` is harmless.
- Inline phase comment documents why this matters and references the
  React Rules-of-Hooks contract.

### Fix B — Q/A check coverage
`.claude/agents/qa.md`:
- New "1b. Frontend lint + typecheck (REQUIRED if diff touches
  `frontend/**`)" section. Mandates `npx eslint .` and `npx tsc
  --noEmit` from `frontend/`. Explicitly states that TypeScript does
  NOT catch hook-order violations and that ESLint
  `react-hooks/rules-of-hooks: "error"` (already in
  `frontend/eslint.config.mjs:34`) is the canonical guard.
- Section is gated on diff scope so non-frontend phases skip it.
- 30-40s total runtime, well within the 55s Q/A budget.

### Fix C — Retry-on-FAIL doctrine formalised
`docs/runbooks/per-step-protocol.md`:
- New "Retry-on-FAIL loop (phase-23.2.24, formalised)" subsection
  in §4 EVALUATE. Quotes Anthropic's harness-design + multi-agent
  research blogs. Specifies the exact 4-step cycle: read verdict ->
  fix blockers AND update files -> spawn FRESH Q/A -> max-3 ceiling.
- Distinguishes legitimate retry (files changed) from second-opinion
  shopping (files unchanged) — same test the cycle-2 phases used
  ad-hoc, now codified.
- Notes that GitHub Copilot Code Review (per researcher's audit) is
  strictly weaker (advisory-only Comment reviews, no merge blocking,
  no auto-resubmit). Pyfinagent's blocking FAIL->Main->fresh-Q/A
  loop is structurally stronger; do NOT downgrade.

### Verifier
`tests/verify_phase_23_2_24.py` (NEW): 6 checks. AST-greps the
JobsTab function body and asserts useMemo position is BEFORE the
first early-return. Runs live `npx eslint .` and `npx tsc --noEmit`
from `frontend/` via subprocess and asserts exit 0. Greps qa.md and
per-step-protocol.md for the new section markers. Confirms
package.json lint script unchanged.

## Files modified / added

```
frontend/src/app/cron/page.tsx                       -- useMemo moved before early returns
.claude/agents/qa.md                                 -- + section 1b. Frontend lint
docs/runbooks/per-step-protocol.md                   -- + Retry-on-FAIL loop subsection
tests/verify_phase_23_2_24.py                        -- NEW, 6-check verifier
handoff/current/contract.md                          -- updated for phase-23.2.24
handoff/current/phase-23.2.24-external-research.md   -- researcher output
handoff/current/phase-23.2.24-internal-codebase-audit.md -- researcher output
```

## Verification (verbatim output)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_24.py
OK frontend/src/app/cron/page.tsx -- useMemo before early returns
OK .claude/agents/qa.md
OK docs/runbooks/per-step-protocol.md
OK frontend/package.json
OK frontend npx eslint .
OK frontend npx tsc --noEmit

phase-23.2.24 verification: ALL PASS (6/6)

$ cd frontend && npx eslint . > /dev/null 2>&1; echo $?
0   # 0 errors (37 pre-existing warnings deferred)

$ cd frontend && npx tsc --noEmit > /dev/null 2>&1; echo $?
0

$ PYTHONPATH=. pytest tests/api/test_cron_dashboard.py tests/services/test_freshness_query_shape.py \
                     tests/services/test_sod_daily_roll.py tests/services/test_position_cap_logging.py \
                     tests/services/test_cycle_failure_alerts.py tests/services/test_kill_switch_no_deadlock.py \
                     tests/api/test_pause_resume_timeout.py tests/services/test_snapshot_upsert.py \
                     tests/db/test_tickets_db_no_fd_leak.py -q
44 passed, 1 warning in 14.60s
```

## Research-gate evidence

Researcher (ac0bca7dac3b163da) returned `gate_passed: true`:
- 10 sources read in full via WebFetch (GitHub Copilot configure-automatic-review,
  use-code-review, React rules-of-hooks docs, eslint-plugin-react-hooks
  reference, Next.js ESLint config, ESLint configure-rules,
  Anthropic harness-design, Anthropic multi-agent research blog,
  KAIRI Copilot 2025 analysis, dev.to Copilot guide 2026)
- 20 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 with 3-variant query discipline
- 7 internal files inspected with file:line anchors (pinpointed bug
  at `cron/page.tsx:218`)
- Concrete recommendation: literal commands + literal qa.md diff snippet

Key external findings applied here:
- GitHub Copilot Code Review is advisory-only — pyfinagent's
  blocking harness is already stronger.
- Anthropic harness-design "QA still added value in catching those
  last mile issues for the generator to fix" — the file-based
  cycle-2 retry is the documented pattern.
- React rules-of-hooks doc: "Hooks after early returns — Yes,
  including useMemo" — exactly the JobsTab bug class.

## Backwards compatibility

- The hook-order fix preserves `useMemo` semantics — `grouped` is
  still memoised on `jobs`. The empty case (`jobs === null`) returns
  `{}` and is unreachable in the rendering path because the early
  returns short-circuit before any `Object.entries(grouped)` call.
- `qa.md` change is purely additive to the deterministic-check list.
  Phases that don't touch `frontend/**` skip the new section.
- `per-step-protocol.md` change is doc-only.
- `package.json` lint script is unchanged at `eslint .`.
- ESLint already configured (`react-hooks/rules-of-hooks: "error"`
  in `frontend/eslint.config.mjs:34`); no config edits needed.

## Honest disclosures

- **The phase-23.2.23 Q/A returned PASS but the code crashed at
  runtime.** This is the real-world example the user pointed to.
  What was missed: ESLint's `rules-of-hooks` rule would have failed
  immediately with `useMemo cannot be called inside conditional
  branches`. Why it was missed: `qa.md`'s deterministic-checks block
  listed `tsc --noEmit` (which has no model for hook-order) but no
  ESLint invocation. Phase-23.2.24 fixes the rubric so this class
  of bug is caught BEFORE the user sees it.
- **37 pre-existing ESLint WARNINGS in the codebase** (set-state-in-effect,
  exhaustive-deps, unused eslint-disable). They are not blocking
  this phase because:
  - The user asked for the cron crash + Q/A hardening, not a
    37-warning cleanup.
  - The hook-order bug class surfaces as ERROR (severity in
    config), not warning, so the gate catches what mattered.
  - Tightening to `--max-warnings=0` is a phase-2 candidate; if
    pursued, must be done one warning category at a time with its
    own honest-disclosure trail. Current baseline: 37 warnings as
    of 2026-05-07.
- **Live browser verification not automated.** The hook-order fix
  is statically verified by ESLint. To confirm in the UI, the user
  can open `http://localhost:3000/cron` in the existing dev server
  and look at the browser console — if no "React has detected a
  change in the order of Hooks" error appears, the fix is live.
  Adding a Playwright runtime smoke is out of scope (heavyweight
  for a check that's already statically caught).
- **Q/A coverage is "if diff touches `frontend/**`".** The check is
  gated on diff scope so non-frontend phases (e.g., a pure-backend
  fix like phase-23.2.20) don't pay the lint cost. If a future phase
  edits `frontend/` but the Q/A subagent's prompt forgets to mention
  it, the check could be skipped. Mitigation: the verifier
  `tests/verify_phase_23_2_24.py::check_eslint_exits_zero` is now
  reusable boilerplate; future frontend phases should copy it into
  their own verifier so the check is enforced PER-PHASE, not just
  by Q/A's selective rubric.
- **Retry-on-FAIL ceiling is 3.** Same as the existing
  `feedback_harness_rigor.md` 3rd-CONDITIONAL auto-FAIL rule. The
  formalised retry doctrine inherits this ceiling rather than
  introducing a new number — keeping the harness internally
  consistent.
- **No code change to `frontend/eslint.config.mjs`.** The rule was
  already correctly set; the gap was upstream (Q/A not running the
  tool). Don't fix what isn't broken.
- **The phase-23.2.23 PR is already merged & pushed.** This phase
  ships a fix-up commit on top of `main`; it does NOT amend the
  prior commit. Standard project convention.
