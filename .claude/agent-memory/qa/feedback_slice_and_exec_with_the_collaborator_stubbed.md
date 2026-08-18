---
name: slice-and-exec-with-the-collaborator-stubbed
description: To tell a genuine DRIVE from a re-implementation, slice the shipped block between two source anchors, exec it with its collaborator stubbed to CAPTURE the argv it assembles, then check it fails CLOSED when the anchors move -- and check the None-return is not scored as a kill
metadata:
  type: feedback
---

The strongest guard shape I have graded for "does this test the SHIPPED code or a
copy of it": slice the shipped source between two anchor lines, `exec` it with the
one collaborator stubbed so the stub CAPTURES the arguments the shipped code
assembles, then run the real tool with that captured argv. The probe never
re-derives the inputs, so a mutation of the shipped line is visible to it.

**Why:** 86.91's corpus pin. v1 was a substring scan and the mutant survived. v2
*re-implemented* the argv assembly and the cell still scored SURVIVED -- the
author's disclosure, and the reason it was caught was only that the cell went red.
v3 slices `CORPUS_SINCE =` .. `rc, out = sh(*_log_args)`, execs it with `sh`
stubbed to append its args, and requires the newest selected commit to equal the
resolved pin. I verified it discriminates: control head `8dc70502...` == the pin,
mutant head `821f2569...` (HEAD). Real behavioural differential, not an artifact.

**Two checks that decide whether the shape is trustworthy, both cheap:**
- **Does it fail CLOSED when the code is refactored out from between the
  anchors?** I ran four shapes -- helper hoisted above the start anchor, start
  anchor reworded (`CORPUS_SINCE: str =`), end anchor reworded
  (`sh(*list(_log_args))`), sliced block made to raise NameError. All returned
  `None`, which turns the CONTROL assertion RED. Good: silent loss of coverage is
  the failure mode a slice invites, and this one cannot have it.
- **Is that same `None` scored as a KILL on the mutant side?** Here, yes:
  `DETECTED if (mh is None or mh != pin)`, and the slicer swallows its own
  exceptions (`except Exception: return None`), so the UNSCORABLE branch the same
  cycle added for the *other* mutant path is unreachable on this one. A mutant that
  cannot build scores DETECTED. Harmless while the real cell discriminates; it is
  the shape to name, because it is the fix applied to one branch and not its
  sibling.

**How to apply:** when an artifact claims a guard "DRIVES the shipped X", find the
anchors, and mutate (a) the shipped line, expecting a differing VALUE not just a
`None`, and (b) the anchors themselves, expecting RED. Related:
[[feedback_a_mutant_that_cannot_build_scores_as_a_kill]],
[[feedback_mutation_probe_must_discriminate]],
[[feedback_mutate_without_touching_the_tree]].
