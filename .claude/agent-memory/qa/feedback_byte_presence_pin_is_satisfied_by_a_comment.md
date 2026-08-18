---
name: byte-presence-pin-is-satisfied-by-a-comment
description: A coupling pin that asserts "literal X still appears in file Y" passes when X survives only in a `//` comment while the emitted string moved -- and quote-then-replace discipline GUARANTEES that comment exists
metadata:
  type: feedback
---

When a fix pins a cross-file coupling by asserting a literal still appears in the
other file, **mutate the emitting site while KEEPING the literal in a comment** --
not just delete it. Deletion is the mutation the author will have tried; comment
retention is the one that survives.

**Why:** 86.84 cycle 10. The cycle-9 finding was a silent zero: an orphan-role
classifier matched prompt literals emitted by `qa-verdict.js` / `research-gate.js`,
nothing pinned the coupling, and drifting the marker one word zeroed the channel
with verify still green. Cycle 10 added `verify()`: `if marker not in wf_src:
problems.append(...)`. My three executed mutations on a temp mirror:

- MUT-B, literal removed outright -> `verify_ok=False`, named problem. The pin is real.
- MUT-A, header renamed to `'BINDING ACCEPTANCE TESTS for ' + stepId` with the retired
  literal kept in two `//` comment lines -> `verify_ok=True, problems=[]`. **SURVIVOR.**
  Measured: literal on a non-comment line = 0, literal in file bytes = True.

The evasion is not exotic, it is the **house style**: `qa-verdict.js:106` already
carries `// 'EVIDENCE / FILES TO READ: ' + evidence` -- a comment reproducing a
sibling prompt literal verbatim -- and `:609`, `:628`, `research-gate.js:880` all
quote retired text. Step 86.84's own **criterion 5 MANDATES** "quoted before/after,
the correction REPLACES the stale claim". So the discipline that makes corrections
auditable is the same discipline that defeats a byte-presence pin.

Sharpest tell: the author had **already applied this insight to the sibling pin** --
cell M7b pins `maxTurns: 30  # restored` because "a `#` comment ... is the most
likely way it would recur" (live_check:307) -- and did not carry it to the new one.
When a codebase reasons about comment-satisfiability for guard A, ask whether guard
B got the same treatment.

**THE FIX HAS ITS OWN RESIDUAL, and one shape is never all of them (86.84 cycle 11).**
The repair was `[ln for ln in src.splitlines() if not ln.lstrip().startswith(("//", "*", "/*"))]`
-- a PREFIX test, so it only excludes comment shapes whose every line starts with a
comment token. Two mutants still survived my run: (a) the residual the code itself
states, a literal in a TRAILING same-line comment after code; (b) one it does NOT
state, a literal inside `/* ... */` whose continuation line has no leading `*` --
first non-whitespace is the literal, so the line reads as CODE. When a guard excludes
comments by prefix, **enumerate the language's comment shapes and mutate each**;
"we disclosed a residual" is not "we disclosed the residual". Mitigating context I
checked before grading severity: all 4 block comments in that file are JSDoc with
` * ` continuations, so (b) is deliberate evasion rather than drift -- measure the
house style before deciding whether a survivor is realistic.

**SECOND INDEPENDENT REPO, SAME PREFIX BUG (86.79 cycle 5).** A different step,
a different author and a different file shipped the same repair and it failed the
same way: `qav = "\n".join(ln for ln in re.sub(r"/\*[\s\S]*?\*/", "", raw).splitlines()
if not ln.strip().startswith("//"))`. The `/* */` half is a SPAN strip and is
sound; the `//` half is a prefix test. My mutant deleted both pinned payload
strings from the prompt array and appended
`const _pin = 1; // null is NEVER 0 -- a gauge, not a counter` -- gate **60/60 ALL
CHECKS PASS, exit 0**, while the whole-line-comment form the author had driven died
(exit 1) and plain deletion died (exit 1). The lesson generalises past one codebase:
**when a comment-stripping repair mixes a span strip with a prefix strip, the prefix
half is the survivor every time.** Correct shape is strip-to-end-of-line
(`re.sub(r"//.*$", "", ln)`), matching what the `/* */` half already does.

**How to apply:** for any `literal in file` assertion, run TWO mutants -- delete
(expect KILL, proves non-inertness) and comment-retain (expect the survivor). Then
check whether the counter the pin protects is even *rendered*: at 86.84 the
companion counter `erased_unclassified` was computed into a dict and read by
nothing, so hardcoding it to 0 left verify green and `render()` byte-identical.
Related: [[feedback_the_guard_carries_the_defect_it_guards]],
[[feedback_assert_the_output_not_its_feed]],
[[feedback_the_instrument_that_closes_a_channel_opens_one]].
