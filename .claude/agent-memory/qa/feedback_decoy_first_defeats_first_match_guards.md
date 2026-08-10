---
name: decoy-first-defeats-first-match-guards
description: A guard built on re.search (first match) over a whole document is order-dependent -- mutate by ADDING a correct decoy EARLY while making the authoritative line wrong; also sweep retired figures into other agents' memory files
metadata:
  type: feedback
---

Two probes that only fire if you construct the mutant a specific way.

**1. Decoy-first mutation on `re.search` guards.** Any guard shaped
`re.search(rf"{name}\s*=\s*([0-9.]+)", whole_document)` reads the FIRST
occurrence. The usual mutation (corrupt the authoritative line) dies correctly
and looks like proof. The mutation that survives is: leave a CORRECT value
early in the document AND make the authoritative line wrong. Measured 83.1
cycle 3 -- prelude `max_pbo = 0.20` + section-2 `max_pbo = 0.50` passed the C2
test, i.e. the exact 2.5x calibration error the step existed to prevent would
have shipped undetected. Before grading such a guard, `grep -c` the token in
the artifact: single occurrence => criterion genuinely met today; more than one
=> re-derive which one the guard actually reads.

**Why:** vacuity shape #2 (source-scan defeated by moving/adding the scanned
text). Cycles 1 and 2 both mutated the authoritative line only and both scored
the guard as fully behavioral.

**How to apply:** pair every "wrong value" mutant with a "wrong value + early
correct decoy" mutant. Same trick for hash/token guards -- 83.1's
`PREREGISTRATION_SHA256` guard is also first-match, but there the stale-first
decoy correctly FAILS (it compares to the real file), so first-match is only
dangerous when the guard compares text-to-runtime, not text-to-text.

**1b. The same defect in JS `indexOf` ORDERING guards (86.28, 2026-08-10).**
An "X happens before Y" guard written as
`src.indexOf('if (tierUnsupported) {') < src.indexOf('const envelope = await agent(')`
is first-match on BOTH operands, so a `//` COMMENT containing the token
satisfies the left side. Measured mutant: leave
`// harmless note: if (tierUnsupported) { we would refuse here }` before the
spawn and move the REAL refusal block to AFTER the spawn -- checker still
printed `ALL GREEN: 61 passed, 0 failed`. This is vacuity shapes #2 + #8
combined, and it is the shape source-position guards take whenever a driver is
module-level code that cannot be executed outside its runtime. Named fix:
strip comment lines before indexing, or match a comment-insensitive regex on
the block body (`/if \(tierUnsupported\) \{[\s\S]*?return \{/`). Grade it WARN
not BLOCK when a live run supplies the behavioural half AND the downstream
guard still fails closed (there the cost of defeat is wasted tokens, not a
false certification) -- but say plainly that the ordering has no re-runnable
regression guard.

**2. A retired figure's population includes OTHER agents' memory files.** When
grading "did the author sweep the whole population of a corrected number",
derive the scope repo-wide, not over the handoff artifacts. 83.1 cycle 3 swept
six handoff files (and self-found a fifth site no Q/A had named) but the
retired `85` survived in
`.claude/agent-memory/researcher/project_phase83_design_pack_83_1.md`, under a
heading literally reading "Corpus inventory (so it needn't be re-derived)" --
the single most forward-looking consumer in the tree, and part of the step's
own untracked change set.

**Why:** agent-memory is designed to be re-read by a FUTURE spawn, so a stale
number there re-enters the system after every handoff artifact is archived.

**How to apply:** add `.claude/agent-memory/**` to the grep scope for every
retired-figure recall test. Note the remediation asymmetry: Main editing the
Researcher's memory is cross-agent tampering, so the right fix is a queued
step or a researcher-directed note, not another annotation cycle -- which is
why this stays WARN and does not by itself justify blocking a step whose
immutable criteria are all met. See [[feedback_recheck_prior_remediation_list]]
and [[feedback_stale_figure_in_gate_artifact]].
