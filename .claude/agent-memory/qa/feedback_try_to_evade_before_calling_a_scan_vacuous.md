---
name: try-to-evade-before-calling-a-scan-vacuous
description: Before filing a source-scan guard as vacuous, build an EVASIVE mutant that dodges the literal -- a co-located behavioural assertion often kills it, retiring the finding
metadata:
  type: feedback
---

A test that asserts on source text (`assert "handoff/away_ops" not in src`)
looks like vacuity shape #1/#2 and is tempting to file immediately. Do NOT
file it on inspection. Build the mutant that DEFEATS the literal while
really introducing the defect, and run the suite. If a behavioural
assertion in the same test kills it, the scan is redundant
defence-in-depth, not sole coverage -- and the finding is retired.

**Why:** phase-85.3 cycle 2. `test_seam_reads_only_the_given_paths` pairs a
source scan with a behavioural half (empty tmp ops dir must yield
`("true","ok")`). I injected a real repo-path fallback spelled so the
literal never appears:

```python
_R = os.path.expanduser("~/.openclaw/workspace/pyfinagent")
_fallback = os.path.join(_R, "handoff", "away" + "_ops")
```

The scan passed (literal absent) but the test still **FAILED** -- the
fallback pulled the real 27-day-old 401 session file in, so `detail` became
`stale_401_ignored_...`. Filing "vacuous source scan" would have been a
plausible-sounding but WRONG finding, which the harness counts as a defect
in the evaluator.

**How to apply:** every candidate vacuity finding on a source-scan guard.
The question is never "does this assert on text?" but "can I write the
defect so this whole TEST still passes?" Only if the evasive mutant
SURVIVES the entire suite is it a real finding. Note the converse still
holds: a scan that is the SOLE coverage for a behavioural criterion stays a
finding. Related: [[feedback_survivor_needs_behavioural_differential]],
[[feedback_decoy_first_defeats_first_match_guards]],
[[feedback_two_mutant_forms_separate_artifact_from_kill]].
