---
name: tightened-guard-opens-false-negative
description: When a guard is TIGHTENED to kill a false positive, mutate for the false negative it opens — especially in the file's own idiom (80.3 c3, `!hidden` vs `hidden`)
metadata:
  type: feedback
---

**A remediation that narrows a matcher is a coverage change in BOTH directions.
Re-run the mutation that the loose form used to kill.**

**Why:** phase-80.3. Cycle 1's mutation M7 hid the React Flow handles with
`className="… !hidden"` and the guard's `.not.toContain("hidden")` KILLED it.
Cycle 2 flagged the mirror-image fragility (a benign `overflow-hidden` would
false-fail), so cycle 3 tightened it to `/(^|\s)hidden(\s|$)/`. That fixed the
false positive and silently opened a false negative: `!hidden` is no longer
matched, so the mutant that died at cycle 1 now SURVIVES. Compiling the project's
own Tailwind 3.4.19 settles the severity — `.\!hidden { display: none !important }`
vs `.hidden { display: none }` — the surviving form is the *stronger* one, and
`!`-prefixed utilities are the idiom in that very className
(`!h-1.5 !w-1.5 !border-0 !bg-slate-500`). The author verified "BOTH directions"
(benign passes, bare `hidden` dies); both statements reproduced, and neither was
the harmful direction.

**How to apply:** whenever a diff replaces a broad matcher with a narrow one
(`includes` → word boundary, `startsWith` → exact, a widened `except`), enumerate
the strings the OLD form caught and the new one does not, then execute the one
that matches the surrounding code's conventions. "Verified both directions" is
two data points, not a proof — ask which third string the codebase would actually
produce. For CSS-class guards, compile the utility rather than reasoning about it
(`npx tailwindcss -c <cfg> -i <in> -o <out>` on a probe HTML file). Related:
[[killed-mutant-needs-differential-too]],
[[survivor-needs-behavioural-differential]].

Corollary from the same session: **verify the mutant is the mutation you
intended.** My swap-the-handles mutant emitted `position={Bottom}` instead of
`position={Position.Bottom}`, so it died on `ReferenceError: Bottom is not
defined` — a kill by my own harness bug, not by the assertion under test (vacuity
shape #11 turned on the evaluator). Print the mutated region and read the failure
REASON, never just the exit code.
