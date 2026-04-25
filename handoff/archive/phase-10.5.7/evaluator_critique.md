---
step: phase-10.5.7
cycle_date: 2026-04-24
evaluator: qa (merged qa-evaluator + harness-verifier)
verdict: CONDITIONAL
forward_cycle: true
---

# Q/A Critique -- phase-10.5.7 (Homepage Red Line hero embed, compact variant)

## Harness-compliance audit (5 items, per feedback_qa_harness_compliance_first.md)

1. **Research gate -- PASS.**
   `handoff/current/phase-10.5.7-research-brief.md` (14,593 bytes, mtime
   2026-04-24 23:45:42) ends with the JSON envelope:
   `tier=simple, external_sources_read_in_full=6 (>=5), snippet_only=7,
   urls_collected=13 (>=10), recency_scan_performed=true,
   internal_files_inspected=7, gate_passed=true`. Spot-check of
   `https://nextjs.org/docs/app/guides/lazy-loading` returned `HTTP/2 200`
   -- source is live. Six sources read-in-full clears the 5-source floor.

2. **Contract-before-GENERATE -- PASS.**
   mtime ordering (epoch):
   - research-brief: 1777067142 (23:45:42)
   - contract.md:    1777067222 (23:47:02)
   - page.tsx:       1777067271 (23:47:51)
   - experiment_results: 1777067551 (23:52:31)

   Strict forward order satisfied. Contract frontmatter:
   `step: phase-10.5.7, forward_cycle: true`. Verbatim success criteria
   copied from masterplan.

3. **Experiment results committed -- PASS.**
   `handoff/current/experiment_results.md` (6,884 bytes) includes:
   (a) verbatim broken-command result (`--url` positional bug),
   (b) run-correctly result with lighthouse 0.97 score,
   (c) honest disclosure that measurement is against `/login` (302 from `/`
      when unauthenticated), NOT the authenticated home that carries the
      hero,
   (d) pre-existing RedLineMonitor.tsx:16 `@phosphor-icons/react` direct
      import flagged as out-of-scope.

4. **Log-last ordering -- PARTIAL / ADVISORY.**
   `grep -c "phase-10.5.7" handoff/harness_log.md` = **1** (not 0 as the
   prompt expected). The one entry is from **2026-04-22** and describes a
   prior implementation (`h-[60vh]`, edited RedLineMonitor.tsx prop;
   Researcher skipped). That attempt was apparently rolled back -- current
   masterplan status for 10.5.7 is still `pending`, and the current page.tsx
   uses the 55svh/next-dynamic pattern from this cycle's contract, not the
   2026-04-22 `h-[60vh]` pattern. So the stale log entry is an artifact of
   the prior attempt, not a current-cycle breach. **Advisory:** when Main
   appends the current-cycle block, it MUST be a new `## phase-10.5.7 --
   2026-04-24` header preserving the older one as history (do not
   overwrite), and the append MUST happen AFTER this Q/A and BEFORE the
   masterplan status flip.

5. **No verdict-shopping -- PASS.** No prior evaluator_critique.md for the
   current 10.5.7 cycle (last critique in handoff/current/ at 23:37 was for
   the Alpaca phase, unrelated). This is the first Q/A of the current
   forward cycle.

Harness-compliance subtotal: **5/5 PASS (log-last is advisory, not a
blocker).**

## Deterministic checks

| Check | Command | Result |
|-------|---------|--------|
| TypeScript typecheck | `cd frontend && npx tsc --noEmit` | **exit 0** (clean) |
| RedLineMonitor wiring | `grep -c 'RedLineMonitor' frontend/src/app/page.tsx` | **4** (>= 3 required) |
| 55svh floor | `grep -c 'min-h-\[55svh\]' frontend/src/app/page.tsx` | **3** (>= 2 required: wrapper + skeleton + one more) |
| Dynamic import pattern | `grep 'next/dynamic' frontend/src/app/page.tsx` | `import dynamic from "next/dynamic";` (line 5) |
| SSR exclusion | `grep 'ssr: false' frontend/src/app/page.tsx` | `ssr: false,` (line 25) |
| Component isolation | `git diff --stat frontend/src/components/RedLineMonitor.tsx` | **empty** (no changes this cycle) |
| Frontend liveness | `curl -sI http://127.0.0.1:3000/` | `HTTP/1.1 302 Found` (NextAuth redirect, expected) |
| Lighthouse freshness | `fetchTime` field | `2026-04-24T21:49:53.374Z` (today, not the 2026-04-18 baseline) |
| Lighthouse perf | `categories.performance.score` | **0.97** (>= 0.9 literal criterion) |
| Lighthouse target | `finalDisplayedUrl` | `http://localhost:3000/login` (302 redirect target) |

All deterministic gates pass the literal rule. The `finalDisplayedUrl`
field is the trust-but-verify smoking gun for the LLM-judgment section.

## LLM judgment

### perf_measurement_on_login_not_home -- MATERIAL CONCERN

The immutable verification criterion targets `http://localhost:3000` and
accepts whatever that URL's Lighthouse score reports. Literally compliant:
0.97 >= 0.9. But the **spirit** of `lighthouse_perf_ge_90` is "the new
hero does not regress the homepage's render budget." Measuring `/login`
(a stripped-down NextAuth form with no Recharts, no RedLineMonitor, no
client-side fetches) validates nothing about the hero's real impact. The
0.97 was effectively measuring a different page.

Mitigating factors Main cited are legitimate but incomplete:
- `ssr:false` does keep Recharts out of the initial HTML -- true for
  FCP/LCP/CLS on the *first* paint.
- However, **TBT, TTI, and Speed Index after hydration** are what a
  homepage user actually experiences, and those aren't captured when
  Lighthouse bounces on a 302.
- `isAnimationActive={false}` reduces post-hydration cost but doesn't
  eliminate the Recharts JS parse/eval cost.

**Verdict on this axis: insufficient evidence for the hero's runtime
impact.** The literal rule passes; the spirit of the guardrail is not
demonstrated.

### hero_presence_without_auth_e2e -- STRUCTURAL ONLY

`grep` confirms the hero is mounted in source. `tsc --noEmit` confirms
the types wire up. But there is NO end-to-end evidence that the
authenticated homepage actually renders the chart. A silent runtime
error (e.g., the `getSovereignRedLine` response shape changes, the
`next/dynamic` promise rejects, or `RedLineWindow` type mismatch) would
not be caught by any check run this cycle. **Static compliance: PASS.
Runtime compliance: unverified.**

### vertical_claim_structural_only -- ACCEPTABLE

`min-h-[55svh]` on the wrapper + `h-full min-h-[16rem]` on the compact
branch of RedLineMonitor is structurally sound: even if the data array is
empty, the wrapper clamps the vertical floor to 55% of the small-viewport
height. This is the stronger of the two `svh`/`dvh` options per the
research brief (svh avoids mobile toolbar CLS). No DOM-snapshot test was
run, but one isn't required for a structural min-height; the CSS itself
is the contract. Acceptable without end-to-end verification.

### preexisting_phosphor_violation -- OUT OF SCOPE

RedLineMonitor.tsx:16 imports `TrendDown` from `@phosphor-icons/react`
directly, bypassing `@/lib/icons`. This violates `.claude/rules/frontend.md`.
Main flagged it but did not fix. Correct scope-honesty call: 10.5.7's
deliverable is the page.tsx embed, not a refactor of a 10.5.3-era
component, and unilaterally editing RedLineMonitor.tsx would have
contradicted the contract's Plan Step 2 ("Do NOT modify
RedLineMonitor.tsx"). **Flag as follow-up ticket, do not block 10.5.7.**

### broken_command_pattern -- SYSTEMIC

Third broken verification command this session:
- phase-10.5.0: `pytest` working-directory bug
- phase-10.5.2: missing audit script
- phase-10.5.7: `lighthouse --url` flag not recognized (positional URL)

A remediation ticket covering all three is warranted. The current Q/A
does not block 10.5.7 on this (the command's intent is clear and the
run-correctly form passes), but **note for the harness log** that this
is the third instance in one week -- the next forward cycle should take
a systemic fix.

## Verdict

**CONDITIONAL PASS.**

All three immutable success criteria pass the literal rule:
- `red_line_hero_present_on_home`: structural PASS via grep + tsc
- `takes_at_least_55pct_vertical`: structural PASS via CSS min-height
- `lighthouse_perf_ge_90`: literal PASS (0.97 >= 0.9)

But the lighthouse measurement was against `/login` due to the 302
redirect, not the authenticated home where the hero actually renders.
This erodes the SPIRIT of the perf guardrail. Main disclosed this
honestly -- that disclosure is what keeps the verdict at CONDITIONAL
rather than FAIL.

### Remediation path (do one of these, in a future cycle, not 10.5.7)

1. **Preferred:** Stand up an authenticated Lighthouse harness --
   NextAuth credentials provider + `lighthouse --chrome-flags` with a
   persisted session cookie, or Playwright + Lighthouse-CI. Scope this
   as its own ticket (say **phase-10.5.9 or 10.5.10 "authenticated perf
   harness"**) and re-run against the authenticated `/`.

2. **Cheaper:** Add a route-group `/dev/home-preview` that renders the
   authenticated homepage shell without NextAuth middleware (dev-only,
   feature-flagged OFF in prod). Lighthouse can hit that without
   auth. Documented as a dev tool, not a prod surface.

3. **Weakest:** Manually run Lighthouse from an authenticated Chrome
   profile on Peder's Mac and paste the result into a follow-up
   `experiment_results_addendum.md`. Non-reproducible but closes the
   gap for this specific build.

### Follow-up tickets

1. **[new]** Authenticated-home Lighthouse harness (remediation path
   above, preferred option).
2. **[new]** Cleanup ticket -- amend masterplan verification commands
   for phase-10.5.0 (pytest cwd), phase-10.5.2 (audit script path), and
   phase-10.5.7 (lighthouse `--url` positional). Suggested:
   `scripts/housekeeping/fix_verification_commands.py`.
3. **[new]** RedLineMonitor.tsx:16 -- replace direct
   `@phosphor-icons/react` import with `@/lib/icons` alias. Out of
   scope for 10.5.7; file its own micro-ticket.

### Machine-readable envelope

```json
{
  "ok": true,
  "verdict": "CONDITIONAL",
  "violated_criteria": [],
  "violation_details": [
    {
      "violation_type": "Unjustified_Inference",
      "action": "lighthouse ran against finalDisplayedUrl=http://localhost:3000/login (302 from /)",
      "state": "perf=0.97, fetchTime=2026-04-24T21:49:53Z, measured page is the NextAuth login form, not the authenticated home",
      "constraint": "spirit of lighthouse_perf_ge_90 is 'hero does not regress homepage perf'; literal rule passed but spirit unverified"
    }
  ],
  "follow_up_tickets": [
    "authenticated-home-lighthouse-harness",
    "fix-three-broken-verification-commands",
    "redlinemonitor-phosphor-import-via-lib-icons"
  ],
  "certified_fallback": null,
  "checks_run": [
    "harness_compliance_5_item_audit",
    "research_gate_envelope",
    "research_source_http_spotcheck",
    "contract_before_generate_mtime",
    "experiment_results_caveat_disclosure",
    "log_last_ordering",
    "no_verdict_shopping",
    "tsc_noemit",
    "grep_redlinemonitor",
    "grep_55svh",
    "grep_next_dynamic",
    "grep_ssr_false",
    "git_diff_redlinemonitor_unchanged",
    "curl_frontend_liveness",
    "lighthouse_json_fresh_fetchtime",
    "lighthouse_perf_score",
    "lighthouse_finaldisplayedurl"
  ]
}
```

### What Main should do next

1. **Do NOT mark 10.5.7 `done` yet.** Append the current-cycle block to
   `handoff/harness_log.md` under a new `## phase-10.5.7 -- 2026-04-24`
   header (preserve the stale 2026-04-22 entry as history). Include
   CONDITIONAL verdict + the three follow-up tickets above.
2. **File the three follow-up tickets** in the masterplan (phase-10.5.9
   or later). The authenticated-home Lighthouse harness is the one that
   actually closes this CONDITIONAL.
3. **After logging + ticketing**, flip 10.5.7 to `done` in
   `.claude/masterplan.json`. The CONDITIONAL verdict is accepted
   because (a) all three literal criteria pass, (b) the caveat is
   honestly disclosed in experiment_results.md, (c) a concrete
   remediation path is funded via a follow-up ticket. A FAIL would
   require re-running Lighthouse *today* against the authenticated
   home, which is infrastructurally out of reach this cycle.

This is NOT a rubber-stamp: the CONDITIONAL verdict + explicit
remediation ticket is the documented pattern for
"literally compliant, spiritually insufficient" outcomes. The ticket
is the teeth -- without it being filed, the next Q/A on phase-10.5.9
should escalate to FAIL.
