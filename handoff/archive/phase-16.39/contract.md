---
step: phase-16.39
title: 22-file phosphor cleanup sweep + promote ESLint rule to error (#50)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/lib/icons.ts (extend with 12 missing icons + Icon type re-export)
  - 22 frontend files: rewrite imports to use @/lib/icons
  - frontend/eslint.config.mjs (warn -> error)
---

# Sprint Contract -- phase-16.39

## Research-gate summary

`handoff/current/phase-16.39-research-brief.md`. tier=simple, gate_passed=true.
Researcher confirmed actual count is 22 (not 21 as task title says); 12
icons missing from lib/icons.ts; `Icon` type also needs re-export.

## Concrete plan

1. **Extend `frontend/src/lib/icons.ts`:**
   - Add `export type { Icon } from "@phosphor-icons/react"` standalone line
   - Add identity re-exports for ~30 icons used by violators
2. **Bulk-rewrite 22 files:** `from "@phosphor-icons/react"` -> `from "@/lib/icons"`
3. **Promote ESLint rule** in `frontend/eslint.config.mjs` from `"warn"` to `"error"`
4. **Verify:** zero `@phosphor-icons/react` lint warnings; zero new errors;
   `tsc --noEmit` clean

## Success Criteria (verbatim, immutable)

```
cd frontend && \
test -z "$(grep -rln '@phosphor-icons/react' src/ | grep -v 'lib/icons.ts')" && \
npx tsc --noEmit && \
npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
```

Plus:
- `lib_icons_extended`: Icon type re-exported + ~12+ identity re-exports added
- `eslint_rule_at_error`: rule level is `"error"` not `"warn"`
- `zero_violators`: 0 files import from `@phosphor-icons/react` outside lib/icons.ts
- `tsc_clean`: tsc --noEmit exits 0
- `lint_clean_for_phosphor`: 0 lines in `npm run lint` output match `@phosphor-icons/react`
- `no_new_errors`: pre-sweep error count == post-sweep error count
  (pre-existing react-hooks warnings unchanged; not introduced by this cycle)

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. lib/icons.ts has the new `export type { Icon }` line + the identity
   re-exports.
3. eslint.config.mjs rule level is `"error"`.
4. `git status` shows: 22 frontend src files + lib/icons.ts + eslint.config.mjs
   + handoff/* rolling. NO backend changes.
5. tsc --noEmit on the entire frontend tree exits 0.
6. `npm run lint` shows 0 errors total (the rule promotion did NOT
   introduce any new errors).
7. The 34 pre-existing warnings (react-hooks, etc.) are unchanged.
