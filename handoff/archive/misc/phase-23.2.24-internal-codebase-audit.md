# Phase-23.2.24 Internal Codebase Audit

## 1. Q/A Agent Check Rubric (`.claude/agents/qa.md`)

### Current deterministic checks (from qa.md lines 41-54)

```bash
# Syntax
python -c "import ast; ast.parse(open('file.py').read())"

# File existence
test -f expected/output/file.py

# Immutable verification command from masterplan.json
source .venv/bin/activate && <step.verification.command>

# Test suite if present
python -m pytest tests/ -v --timeout=30
```

### Frontend lint gap — CONFIRMED MISSING

The Q/A agent definition at `.claude/agents/qa.md` lines 41-54 does NOT include any frontend-specific checks. Specifically:

- No `npm run lint` invocation
- No `npx eslint .` invocation
- No `--max-warnings=0` enforcement
- No `npm run build` (Next.js production build)
- No `cd frontend` before any of the above

The only frontend check implicitly available is the masterplan step's own `verification.command`, which for phase-23.2.23 was the Python verifier script `tests/verify_phase_23_2_23.py`. That script does NOT run ESLint (confirmed below, item 6).

### Q/A retry-on-FAIL loop

`qa.md` lines 163-165 describe the second-opinion-shopping prohibition:

> "Never second-opinion-shop. If the first spawn returned CONDITIONAL, the orchestrator must fix the blockers then SendMessage back to the SAME agent, not spawn a new one."

This is a constraint on Q/A, not a prescription for what Q/A itself checks. The retry-on-FAIL pattern (Main fixes blockers, spawns fresh Q/A) is documented in `docs/runbooks/per-step-protocol.md` §4 EVALUATE (see section 2 below).

---

## 2. Per-Step-Protocol Cycle-2 Retry Pattern

From `docs/runbooks/per-step-protocol.md` §4 EVALUATE (lines 111-138):

> "Q/A runs deterministic-first:
> 1. Syntax / file-existence / verification.command exit code
> 2. Reads existing handoff/current/evaluator_critique.md + experiment_results.md
> 3. Optional harness dry-run (under 55s budget)
> 4. LLM judgment on contract alignment, scope honesty, mutation-resistance, and research-gate compliance"

The CONDITIONAL escalation clause (lines 140-160) documents:

> "A CONDITIONAL verdict is appropriate when: (a) underlying functionality is intact, (b) production code paths are unaffected, and (c) the step was designed to discover a gap rather than deliver a fix. CONDITIONAL is NOT an indefinite soft-pass."

> "If a single masterplan step-id accumulates 3 or more consecutive CONDITIONAL verdicts without an intervening PASS or FAIL, the next Q/A pass MUST return FAIL."

The canonical cycle-2 flow from CLAUDE.md (Harness Protocol section):

> "1. Main reads the critique's violated_criteria + violation_details.
> 2. Main fixes the blockers and updates the handoff files.
> 3. Main spawns a fresh Q/A. The fresh Q/A reads the updated files."

This is the documented retry-on-FAIL pattern: Main fixes, re-spawns Q/A. Retry ceiling is 3 consecutive CONDITIONALs (auto-FAIL on 4th), or `max_retries` from masterplan.json (defaulting to 3) for FAIL→certified_fallback.

---

## 3. Hook-Order Bug in `frontend/src/app/cron/page.tsx::JobsTab`

### Bug location: line 218

```tsx
// Lines 171-215: Three early returns
if (jobs === null && error === null) {
  return (...)        // line 171 — Loading state
}

if (error && jobs === null) {
  return (...)        // line 180 — Error state
}

if (jobs && jobs.length === 0) {
  return (...)        // line 205 — Empty state
}

// Line 218: useMemo called AFTER three early returns
const grouped = useMemo(() => {
  const out: Record<string, JobInfo[]> = {};
  for (const j of jobs ?? []) {
    (out[j.source] ??= []).push(j);
  }
  return out;
}, [jobs]);
```

### Why this violates the Rules of Hooks

When `jobs === null && error === null` on first render, React executes hooks 1-8 (useState x4, useRef x2, useCallback x1, useEffect x1) and returns early. On a subsequent render where `jobs.length > 0`, React now encounters `useMemo` as hook #9 — which it never saw on the previous render. React's reconciler detects the mismatch and throws:

```
React has detected a change in the order of Hooks called by JobsTab.
   Previous render            Next render
1. useState                   useState
...
8. useEffect                  useEffect
9. undefined                  useMemo
```

### TypeScript does NOT catch this

TypeScript's type system checks value types and signatures. It has no model of React's hook-call-order constraint. `useMemo<Record<string, JobInfo[]>>(..., [jobs])` is fully type-correct — there is no type error here.

### ESLint `react-hooks/rules-of-hooks` DOES catch this

The rule performs static control-flow analysis. It detects that `useMemo` at line 218 sits in a code path reachable only after three conditional early-return guards (lines 171, 180, 205). This is a classic "hook after early return" violation.

---

## 4. ESLint Config Inventory

### Config file: `frontend/eslint.config.mjs`

The flat config (ESLint 9+) is present. Key findings:

- **`react-hooks/rules-of-hooks`** is set to `"error"` (line 34) — the correct severity. This WILL cause `eslint .` to exit non-zero on the `JobsTab` violation.
- **`react-hooks/exhaustive-deps`** is set to `"warn"` (line 23) — warning-only. This will NOT cause exit code 1.
- **`eslint-config-next/core-web-vitals`** is spread in (line 7) — includes the recommended react-hooks rules from `eslint-plugin-react-hooks`.
- New React Compiler rules (`set-state-in-effect`, `purity`, `immutability`) are set to `"warn"` — per the inline comment, intentionally deferred.
- **`no-restricted-imports`** for `@phosphor-icons/react` is set to `"error"` — correct.

**Conclusion:** The config is correct. `react-hooks/rules-of-hooks: "error"` is already set. Running `npx eslint .` from the `frontend/` directory WOULD have caught the hook-order bug in `cron/page.tsx` before the Q/A passed phase-23.2.23. The Q/A never ran it.

### package.json lint script

```json
"lint": "eslint ."
```

This is a bare `eslint .` — it will fail with exit code 1 on `rules-of-hooks: "error"` violations. However, it does NOT use `--max-warnings=0`, so warning-level violations (exhaustive-deps, React Compiler rules) would silently pass. For the specific hook-order bug, the existing `"error"` severity is sufficient.

---

## 5. Lint Script Analysis

`frontend/package.json` line 9: `"lint": "eslint ."`

Issues:
1. No `--max-warnings=0` flag — warnings do not fail CI. The React Compiler rules (`purity`, `immutability`, `set-state-in-effect`) are all `"warn"`. They will not block.
2. No `--max-warnings=0` means a file with 50 warnings passes cleanly. For the pyfinagent harness, all rules that matter should be promoted to `"error"` or the script should use `--max-warnings=0`.
3. The Q/A must `cd frontend` first. Running `eslint .` from the repo root will fail with "no files matched" or use the wrong config, because `eslint.config.mjs` lives in `frontend/`.

**Correct invocation for Q/A:**
```bash
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && npx eslint . --max-warnings=0
```

---

## 6. verify_phase_23_2_23.py::check_frontend_page — Confirmed No Lint Step

`tests/verify_phase_23_2_23.py` lines 109-128 (`check_frontend_page`):

The function:
1. Reads `frontend/src/app/cron/page.tsx` as text
2. Asserts presence of CSS class strings (`flex h-screen overflow-hidden`, `scrollable-thin`, etc.)
3. Asserts presence of component names (`JobsTab`, `LogsTab`)
4. Asserts Phosphor import pattern (`from "@/lib/icons"`)
5. Runs a regex scan for pictographic emoji Unicode ranges

It does NOT:
- Call `subprocess.run` for ESLint
- Call `npm run lint`
- Do any static analysis of hook call order
- Run `tsc --noEmit` (the docstring at line 15 says "tsc --noEmit passes (caller runs separately for speed)" — meaning it's deferred to the caller, not enforced in the verifier)

---

## 7. Lint References Across All Tests

Grep result: `grep -r "npm run lint\|eslint\|subprocess.*lint" tests/` — returns empty. Confirmed: **no existing phase verifier in `tests/` runs ESLint or `npm run lint`.** This is a systemic gap, not just a phase-23.2.23 gap.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/agents/qa.md` | 174 | Q/A agent definition | Missing frontend lint step |
| `docs/runbooks/per-step-protocol.md` | 235 | Harness operator runbook | Has cycle-2 retry pattern documented |
| `frontend/src/app/cron/page.tsx` | 441 | Cron / Logs UI page | Hook-order bug at line 218 |
| `frontend/eslint.config.mjs` | 60 | ESLint 9 flat config | rules-of-hooks=error already set |
| `frontend/package.json` | 57 | Frontend package config | lint script: "eslint ." (no --max-warnings=0) |
| `tests/verify_phase_23_2_23.py` | 208 | Phase-23.2.23 verifier | No ESLint invocation |
| `tests/` (all files) | ~2000 | Test suite | No file runs eslint anywhere |
