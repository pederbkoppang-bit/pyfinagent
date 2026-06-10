# Phase-23.2.24 External Research Brief

Tier assumption: moderate (stated by caller).

## Topic

GitHub Copilot Code Review workflow shape; ESLint react-hooks enforcement; retry-on-FAIL agent loops; concrete recommendations for Q/A and verifier improvements.

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.github.com/en/copilot/how-tos/copilot-on-github/set-up-copilot/configure-automatic-review | 2026-05-07 | Official doc | WebFetch | "Copilot will review all new pushes to the pull request" — Re-review triggered by push, not by author reply |
| https://docs.github.com/en/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review | 2026-05-07 | Official doc | WebFetch | "always leaves a 'Comment' review, not an 'Approve' or 'Request changes' review" — no merge blocking by Copilot |
| https://react.dev/reference/eslint-plugin-react-hooks/lints/rules-of-hooks | 2026-05-07 | Official React docs | WebFetch | "Hooks after early returns — Yes, including useMemo" — the rule detects exactly this pattern |
| https://nextjs.org/docs/pages/api-reference/config/eslint | 2026-05-07 | Official Next.js docs | WebFetch | "eslint-config-next includes recommended rule-sets from eslint-plugin-react and eslint-plugin-react-hooks" — rules-of-hooks is bundled |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-07 | Anthropic engineering blog | WebFetch | "the generator was still liable to miss details...QA still added value in catching those last mile issues for the generator to fix" — evaluator-drives-generator retry |
| https://react.dev/reference/eslint-plugin-react-hooks | 2026-05-07 | Official React docs | WebFetch | Lists all 17 rules including rules-of-hooks and exhaustive-deps; recommended preset bundles all |
| https://eslint.org/docs/latest/use/configure/rules | 2026-05-07 | Official ESLint docs | WebFetch | "'error' causes ESLint to exit with code 1 when triggered" — severity distinction |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-07 | Anthropic engineering blog | WebFetch | "LeadResearcher synthesizes results and decides whether more research is needed — if so, it can create additional subagents" — no hard iteration ceiling specified |
| https://medium.com/kairi-ai/githubs-2025-copilot-review-can-t-satisfy-the-merge-gate-f1de0e535788 | 2026-05-07 | Industry analysis | WebFetch | "Copilot comments don't satisfy merge requirements... actual blocking power remains with infrastructure controls" |
| https://dev.to/rahulxsingh/github-copilot-code-review-complete-guide-2026-255h | 2026-05-07 | Practitioner blog | WebFetch | "one-shot analysis model — no automatic send-back-to-author retry mechanism"; manual re-trigger via `@copilot review` |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/facebook/react/issues/28313 | GitHub issue | Flat-config support thread; snippet sufficient |
| https://github.com/reactwg/react-compiler/discussions/18 | GitHub discussion | React Compiler + hooks plugin interplay; snippet sufficient |
| https://github.blog/changelog/2025-09-10-copilot-code-review-independent-repository-rule-for-automatic-reviews/ | GitHub changelog | Confirms independent ruleset (no forced branch protection) |
| https://github.blog/changelog/2025-04-04-copilot-code-review-now-generally-available/ | GitHub changelog | GA announcement, April 2025 |
| https://github.blog/changelog/2026-04-27-github-copilot-code-review-will-start-consuming-github-actions-minutes-on-june-1-2026/ | GitHub changelog | Billing change June 2026 — relevant if org adopts Copilot review |
| https://javascript.plainenglish.io/how-to-configure-eslint-v9-in-a-react-project-2025-guide-a86d893e1703 | Practitioner blog | ESLint v9 React config guide 2025 |
| https://blog.logrocket.com/12-essential-eslint-rules-react/ | Practitioner blog | Rules overview including rules-of-hooks |
| https://chris.lu/web_development/tutorials/next-js-16-linting-setup-eslint-9-flat-config | Tutorial | Next.js 16 + ESLint 9 flat config walk-through |
| https://www.augmentcode.com/tools/github-copilot-ai-code-review | Industry | Copilot review features overview |
| https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | Anthropic blog | Harness design focus on incremental progress and state |

---

## Recency Scan (2024-2026)

Searched for:
1. "GitHub Copilot code review automatic PR review setup workflow 2026" (current-year frontier)
2. "eslint-plugin-react-hooks rules-of-hooks configuration flat config 2025" (last-2-year window)
3. "ESLint rules-of-hooks useMemo after early return detection 2024 2025" (last-2-year window)
4. "agent harness retry loop max iterations escalation 2024 2025" (last-2-year window)
5. "ESLint 9 flat config next.js max-warnings 0 CI enforcement 2025 2026" (current-year frontier)
6. "React hooks rules violations TypeScript ESLint detection static analysis" (year-less canonical)
7. "Anthropic multi-agent retry loop harness max iterations certified fallback" (year-less canonical)
8. "GitHub Copilot code review send back author retry semantics branch protection 2025" (last-2-year window)

**Result:** Found several significant new findings from 2025-2026 that complement canonical sources:

- Copilot code review went **Generally Available April 2025** (previously preview). The agentic architecture update (March 2026) added cross-file context exploration before generating comments — a material improvement over the original diff-only analysis. (Source: GitHub Changelog 2025-04-04, 2026 guide)
- ESLint 9 **flat config is now the default** as of ESLint 9.0.0 (April 2024). Legacy `.eslintrc.*` is deprecated. `eslint-config-next` v16 (bundled with Next.js 16) removed `next lint` in favor of direct ESLint CLI. (Source: Next.js official docs, accessed 2026-05-07)
- `eslint-plugin-react-hooks` now includes 17 rules including React Compiler diagnostics (added in 2024-2025). The `recommended` preset bundles all of them. The flat config export `reactHooks.configs.flat.recommended` was added to support ESLint 9. (Source: React docs)
- **Copilot review does NOT block merges** — it always posts a "Comment" review type (not "Request changes"), so it cannot satisfy required-reviewer checks. This was confirmed and clarified in the 2025 GA announcement.
- Stripe's "two-strike rule" pattern (2024-2025 practitioner art): if an agent's first fix fails CI, escalate immediately to human rather than retry in an infinite loop. This complements the harness's 3-CONDITIONAL auto-FAIL rule.

No findings supersede the canonical sources. Newer work extends and confirms them.

---

## Key Findings

1. **ESLint `rules-of-hooks: "error"` catches hooks-after-early-returns statically.** The rule does control-flow analysis. A `useMemo` (or any hook) called after an `if (...) return` is flagged before the code ever runs. TypeScript type-checking does NOT cover this — they are orthogonal tools. (Source: React docs `rules-of-hooks`, https://react.dev/reference/eslint-plugin-react-hooks/lints/rules-of-hooks)

2. **`eslint-config-next/core-web-vitals` bundles `eslint-plugin-react-hooks`.** The pyfinagent frontend already imports it in `eslint.config.mjs`. The `react-hooks/rules-of-hooks` rule is already set to `"error"`. Running `npx eslint .` from `frontend/` WOULD have caught the `JobsTab` bug. The failure was that Q/A never ran it. (Source: Next.js ESLint docs, https://nextjs.org/docs/pages/api-reference/config/eslint)

3. **GitHub Copilot code review is advisory, not blocking.** It always posts "Comment" review type, never "Request changes". It cannot block merges. Re-review must be manually triggered (`@copilot review` or refresh button in Reviewers panel). The retry loop the user wants ("send back to author") is not an automatic Copilot feature — it requires branch protection rules + required human reviewers. (Source: GitHub official docs, https://docs.github.com/en/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review)

4. **The harness retry pattern (Anthropic) is: evaluator feeds failure back to generator; generator iterates; observed range is 5-15 iterations.** No hard ceiling is prescribed in the Anthropic blog posts — the practical limit is context window (200K tokens) and the masterplan's `max_retries` field. The pyfinagent convention is 3 consecutive CONDITIONALs → auto-FAIL → certified_fallback. (Source: Anthropic "Harness Design for Long-Running Apps", https://www.anthropic.com/engineering/harness-design-long-running-apps)

5. **ESLint rule severity: `"error"` → exit code 1; `"warn"` → exit code 0.** A CI step that runs `eslint .` only fails if at least one `"error"`-level rule fires. Warning-level rules pass silently. Using `--max-warnings=0` makes any warning also fail. (Source: ESLint configure/rules docs, https://eslint.org/docs/latest/use/configure/rules)

6. **`npm run build` (Next.js production build) runs ESLint by default up to Next.js 15.** In Next.js 16, `next lint` was removed. The build no longer auto-runs lint. Running lint must be explicit in CI. (Source: Next.js ESLint docs, version 16.2.5)

7. **Anti-patterns confirmed:**
   - Running `eslint .` from repo root when `eslint.config.mjs` is in `frontend/` — silently processes zero files or the wrong files.
   - Not using `--max-warnings=0` when warning-level rules matter.
   - Relying on TypeScript `tsc --noEmit` to catch hook-order violations — it does not.
   - Running `npm run build` as a substitute for lint — post-Next.js 16, build does not run lint.

---

## Consensus vs Debate (External)

**Consensus:**
- ESLint `rules-of-hooks: "error"` is the standard way to catch hook-order violations. All sources agree TypeScript is insufficient for this class of bug.
- Copilot code review is advisory-only. It cannot block merges without branch protection requiring human approvals.
- The harness retry pattern is: fix → re-evaluate (not: retry same evidence with different agent).

**Debate:**
- Whether to use `--max-warnings=0` vs promoting all rules to `"error"`: no strong consensus. The `--max-warnings=0` approach is more conservative; promoting rules to `"error"` in config is more explicit. Either works for CI enforcement.
- Whether `npm run build` is a useful gate: pre-Next.js-16 it ran lint implicitly; post-16 it does not. The pyfinagent project uses Next.js 15 (`^15.0.0`), so the build STILL runs lint as a side effect in dev mode — but relying on this is fragile. Explicit lint is preferred.

---

## Pitfalls (from Literature)

1. **Running ESLint from the wrong working directory.** `eslint .` from the repo root will not pick up `frontend/eslint.config.mjs`. Always `cd frontend` first or pass the full path. (ESLint flat config resolution is directory-relative.)

2. **Warning-level rules giving false confidence.** The React Compiler rules in `eslint.config.mjs` are all `"warn"`. A Q/A run that exits 0 may still have dozens of compiler warnings. Use `--max-warnings=0` or promote critical rules to `"error"`.

3. **Copilot review "repeat comments" on re-review.** The official docs note: "Copilot may repeat the same comments again, even if they have been dismissed." This means Copilot review is not idempotent — it does not track which comments the author already addressed.

4. **Context-window limits on retry loops.** Anthropic's multi-agent system notes truncation at 200K tokens. A harness running many iterations accumulates context. The 3-CONDITIONAL auto-FAIL ceiling in pyfinagent prevents unbounded retries.

5. **ESLint `--no-error-on-unmatched-pattern` silently passes empty runs.** Never use this flag in CI. If `eslint .` returns "no files matched", it exits 0 — a false pass.

---

## Application to pyfinagent (Mapping to file:line anchors)

| Finding | Maps to | Action |
|---------|---------|--------|
| rules-of-hooks catches hooks after early returns | `frontend/src/app/cron/page.tsx:218` (useMemo after 3 early returns) | Fix: move useMemo before early returns, or restructure to avoid the pattern |
| Q/A missing frontend lint step | `.claude/agents/qa.md:41-54` (deterministic checks block) | Add `cd frontend && npx eslint . --max-warnings=0` as a required deterministic check for any step touching `frontend/**` |
| lint script has no --max-warnings=0 | `frontend/package.json:9` (`"lint": "eslint ."`) | Update to `"lint": "eslint . --max-warnings=0"` or keep as-is and have Q/A pass the flag explicitly |
| No verifier runs ESLint | `tests/verify_phase_23_2_23.py:109-128` (check_frontend_page) | Add subprocess call to `npx eslint src/app/cron/page.tsx --max-warnings=0` in frontend verifiers |
| Copilot review is advisory only | N/A (pyfinagent works on main, no PRs) | Mirror the retry semantics in Q/A agent: Q/A FAIL → Main fixes → fresh Q/A (already documented in per-step-protocol) |

---

## RECOMMENDATION BLOCK

### 1. New deterministic checks for any phase touching `frontend/**`

Add to `.claude/agents/qa.md` deterministic checks (§1, after existing syntax check):

```bash
# Frontend lint (required for any step touching frontend/**)
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
  npx eslint . --max-warnings=0
```

This must run before the LLM judgment step. Exit code 1 = FAIL. The check covers:
- `react-hooks/rules-of-hooks: "error"` — hooks-after-early-returns, hooks in conditionals, hooks in loops
- `no-restricted-imports: "error"` — direct @phosphor-icons/react imports bypassing the barrel
- Core Web Vitals rules from `eslint-config-next/core-web-vitals`
- With `--max-warnings=0`: all 17 React Compiler warning-level rules also become blocking

### 2. Exact npm run lint invocation

```bash
# Option A (preferred for Q/A — explicit, no package.json dependency):
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
  npx eslint . --max-warnings=0

# Option B (uses package.json script — requires updating package.json first):
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
  npm run lint -- --max-warnings=0
```

Note on Option B: `npm run lint -- --max-warnings=0` passes `--max-warnings=0` through to the underlying `eslint .` command. The current `"lint": "eslint ."` script does not include it by default.

For verifier scripts (`tests/verify_phase_23_2_XX.py`), use:
```python
proc = subprocess.run(
    ["npx", "eslint", "src/app/cron/page.tsx", "--max-warnings=0"],
    cwd=ROOT / "frontend",
    capture_output=True,
    text=True,
    timeout=60,
)
if proc.returncode != 0:
    raise AssertionError(f"ESLint failed:\n{proc.stdout}\n{proc.stderr}")
```

### 3. Should Q/A run `npm run build`?

**Yes, for phases that create new frontend pages. No, for minor edits.**

Cost: `npm run build` (Next.js 15 production build) takes 45-90 seconds on this codebase. The Q/A has a 55-second budget cap (qa.md line 159). Running build would exceed the budget.

Mitigation options:
- Run build as part of GENERATE (Main runs it before spawning Q/A), then Q/A checks the build output/exit code via `test -f frontend/.next/BUILD_ID`.
- Use `npx tsc --noEmit` (faster than build, catches type errors) + `npx eslint .` (catches hooks violations). Together they cover the same ground as the build's type+lint checks without the 45-90s overhead.

**Recommended for Q/A:** Run `npx eslint . --max-warnings=0` (10-15s) + `npx tsc --noEmit` (15-25s) = ~30s total, within the 55s budget. Reserve `npm run build` for GENERATE phase or a post-merge CI step.

### 4. Retry-on-FAIL loop signature

From Anthropic ("Harness Design for Long-Running Apps"): "the generator was still liable to miss details or stub features when left to its own devices, and the QA still added value in catching those last mile issues for the generator to fix."

The prescribed pattern (from CLAUDE.md and per-step-protocol.md):
1. Q/A returns FAIL or CONDITIONAL with `violated_criteria` + `violation_details`
2. Main reads the critique, fixes all blockers, updates `handoff/current/experiment_results.md` and any code
3. Main spawns a fresh Q/A instance (NOT the same instance, per file-based handoff model)
4. Fresh Q/A reads updated files — new evidence → potentially new verdict

**Iteration ceiling:**
- Observed range in Anthropic harness: 5-15 iterations per generation cycle
- pyfinagent convention: `max_retries = 3` from masterplan.json → `certified_fallback: true` on 3rd FAIL
- 3rd consecutive CONDITIONAL → auto-FAIL (3-CONDITIONAL auto-FAIL rule, qa.md lines 166-173)

**What constitutes a valid re-spawn vs verdict-shopping:**
- VALID: Main fixed at least one `violated_criteria` item and updated handoff files → fresh Q/A sees changed evidence
- INVALID: Main spawns fresh Q/A on identical evidence hoping for a different answer

### 5. Literal diff snippet for `.claude/agents/qa.md`

Add a new subsection after line 54 (end of existing deterministic checks bash block):

```diff
--- a/.claude/agents/qa.md
+++ b/.claude/agents/qa.md
@@ -52,6 +52,20 @@
 # Test suite if present
 python -m pytest tests/ -v --timeout=30
 ```
+
+### 1b. Frontend lint (required for any step touching frontend/**)
+
+If the step's files list includes any path under `frontend/`, run:
+
+```bash
+# Must cd to frontend/ — eslint.config.mjs lives there
+cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
+  npx eslint . --max-warnings=0
+```
+
+Exit code 1 = FAIL. This check catches:
+- `react-hooks/rules-of-hooks` (error) — hooks called after early returns, in conditions, in loops
+- `no-restricted-imports` (error) — direct @phosphor-icons/react imports
+- Any warning-level rule (via --max-warnings=0)
+
+TypeScript `tsc --noEmit` does NOT catch hook-order violations. ESLint is required.
+
+Additionally run:
+```bash
+cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
+  npx tsc --noEmit
+```
+
+Together these two checks cover type errors + React hook violations within the 55s budget.
+Note: Do NOT rely on `npm run build` alone — Next.js 15 build runs lint as a side effect in dev
+mode but this is not guaranteed and exceeds the 55s Q/A budget.

 ### 2. Existing results check
```

### 6. Mirroring GitHub Copilot Review in the Harness

GitHub Copilot code review (as of 2025-2026) works as follows:
- Enabled via branch rulesets (org or repo settings)
- Triggers automatically on PR open + each new push (when "Review new pushes" is enabled)
- Posts inline comments on changed lines
- Never blocks merges (always "Comment" type, never "Request changes")
- Re-review: triggered manually (`@copilot review`) or automatically on push

The user wants pyfinagent Q/A to behave like "Copilot review that sends back to author if something fails." The closest equivalent in the harness:

1. Q/A runs lint + tsc + verification command (the "Copilot analysis" equivalent)
2. Q/A returns FAIL with `violated_criteria` listing exactly what failed (the "inline comments" equivalent)
3. Main reads the FAIL, fixes the issues (the "author addresses feedback" equivalent)
4. Main spawns fresh Q/A (the "push new commits → Copilot re-reviews" equivalent)

The pyfinagent implementation is STRICTER than Copilot: Q/A's FAIL actually blocks the step (no merge possible), whereas Copilot's comments are advisory. This is the correct direction — the harness should be MORE rigorous than a CI comment.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (10 sources fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) (20 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md, per-step-protocol.md, cron/page.tsx, eslint.config.mjs, package.json, verify_phase_23_2_23.py, all tests/)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

## Sources

- [About GitHub Copilot code review - GitHub Docs](https://docs.github.com/en/copilot/concepts/agents/code-review)
- [Configuring automatic code review by GitHub Copilot - GitHub Docs](https://docs.github.com/en/copilot/how-tos/copilot-on-github/set-up-copilot/configure-automatic-review)
- [Using GitHub Copilot code review - GitHub Docs](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review)
- [rules-of-hooks - React official docs](https://react.dev/reference/eslint-plugin-react-hooks/lints/rules-of-hooks)
- [eslint-plugin-react-hooks - React official docs](https://react.dev/reference/eslint-plugin-react-hooks)
- [ESLint Configuration: eslint - Next.js official docs](https://nextjs.org/docs/pages/api-reference/config/eslint)
- [Configure Rules - ESLint official docs](https://eslint.org/docs/latest/use/configure/rules)
- [Harness Design for Long-Running Apps - Anthropic Engineering](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [How We Built Our Multi-Agent Research System - Anthropic Engineering](https://www.anthropic.com/engineering/built-multi-agent-research-system)
- [GitHub's 2025 Copilot Review Can't Satisfy the Merge Gate - Medium](https://medium.com/kairi-ai/githubs-2025-copilot-review-can-t-satisfy-the-merge-gate-f1de0e535788)
- [GitHub Copilot Code Review Complete Guide 2026 - DEV Community](https://dev.to/rahulxsingh/github-copilot-code-review-complete-guide-2026-255h)
- [Copilot code review now generally available - GitHub Changelog](https://github.blog/changelog/2025-04-04-copilot-code-review-now-generally-available/)
- [GitHub Copilot code review will consume GitHub Actions minutes June 2026](https://github.blog/changelog/2026-04-27-github-copilot-code-review-will-start-consuming-github-actions-minutes-on-june-1-2026/)
