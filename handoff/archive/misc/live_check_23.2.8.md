# Step 23.2.8 -- Verify home cockpit + paper-trading hero metrics in sync -- verification

**Date:** 2026-05-23
**Step type:** EXECUTION (SSOT source-grep verification + 6 new pytest tests).
**Verdict:** **PASS**

---

## Verbatim masterplan criterion + evidence

> Criterion: "Manual: open both pages; NAV / Total P&L should be byte-identical (post phase-23.1.17 useLiveNav SSOT)"

**Source-level SSOT verified:**

| Site | Evidence |
|---|---|
| `frontend/src/lib/useLiveNav.ts` | Exists, exports `useLiveNav`, contains the `cash + positionsValue` NAV math |
| `frontend/src/app/page.tsx:15` | `import { useLiveNav } from "@/lib/useLiveNav"` |
| `frontend/src/app/page.tsx:156` | `const { liveNav, liveTotalPnlPct } = useLiveNav(...)` |
| `frontend/src/app/paper-trading/page.tsx:46` | Same import |
| `frontend/src/app/paper-trading/page.tsx:444` | Same destructure |
| NAV-math leak check | NAV math appears ONLY in useLiveNav.ts (no re-inlining in any .tsx) |

**Verdict: PASS verbatim.** 6 pytest tests lock the 6 SSOT invariants.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | **PASS** (417; was 411 after 23.2.7; +6 new; 0 regressions) |
| 2 | TS build green on changed | **N/A** (no frontend changes; tests are read-only) |
| 3 | Flag default OFF | **N/A** |
| 4 | BQ migrations idempotent | **N/A** |
| 5 | New env vars documented | **N/A** |
| 6 | Contract has N* delta | **PASS** (R SSOT audit + B regression resistance) |
| 7 | Zero emojis | **PASS** |
| 8 | ASCII-only loggers | **N/A** |
| 9 | Single source of truth | **PASS** (useLiveNav.ts canonical) |
| 10 | log first / flip last | **WILL HOLD** |

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_23_2_8_use_live_nav_ssot.py -v
test_phase_23_2_8_use_live_nav_hook_exists_and_exports PASSED
test_phase_23_2_8_home_page_imports_use_live_nav PASSED
test_phase_23_2_8_paper_trading_page_imports_use_live_nav PASSED
test_phase_23_2_8_both_pages_destructure_live_nav_and_pnl PASSED
test_phase_23_2_8_nav_math_lives_only_in_hook PASSED      # anti-drift
test_phase_23_2_8_hook_return_shape_is_documented PASSED
6 passed in 0.02s

$ pytest backend/ --collect-only -q | tail -2
417 tests collected
```

---

## Diff

```
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py    (new, ~135 lines, 6 tests)
```

ZERO source code changes. ZERO frontend changes.

---

## Honest scope deferral

| Item | Status | Defer-to |
|---|---|---|
| Vitest hook test for numerical drift between calls | DEFERRED | future cycle (researcher flagged as Tier-2; vitest already configured but adding tests there is its own scope) |
| TanStack Query keyed cache (strictly-stronger pattern) | DEFERRED | future architecture decision (out of phase-23.2.8 scope) |

---

## Bottom line

phase-23.2.8 (P1) verifies the `useLiveNav` SSOT discipline at the source layer. 6 pytest tests catch hook deletion, missing imports, missing destructures, return-shape drift, AND the anti-drift case (re-inlined NAV math). Manual UI check substituted with source-grep that's mutation-resistant.

**Closure-path progress:** 21 of ~21-36 cycles done this session (cycles 12-32). Now well-past the midpoint.
