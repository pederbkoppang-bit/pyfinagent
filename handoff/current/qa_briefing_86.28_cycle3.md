# Q/A briefing -- step 86.28, cycle 3

**Why this file exists.** The cycle-3 Q/A spawn (`wf_01c83c86-09d`) DROPPED
at 197,091 tokens / 40 tool uses without returning a verdict -- the known
long-context failure on this rail. The evidence set is now ~1,300 lines
across three documents; reading all of it is what exhausted the budget.
This is a NAVIGATION aid to cut mandatory reading. It is written by Main,
who is the author under evaluation. **Do not trust it. Verify anything you
rely on.** Every claim here is checkable with the commands given.

## The change under evaluation, in one paragraph

`.claude/workflows/research-gate.js` silently downgraded an unsupported
`tier` (e.g. `deep`) to `moderate`, so the gate certified at a standard the
caller never requested, and the substitution reached only the agent PROMPT,
never the return value. It also trusted `recency_scan_performed` and
`urls_collected` as bare self-report inside a gate whose stated thesis is
recompute-never-trust. And the docs claimed `agentType:'general-purpose'`
while the code pins `'researcher'`. Fixed: ABSENT vs UNSUPPORTED tier are
now distinct, UNSUPPORTED refuses to spawn and is reported in the return,
both self-reports are corroborated against the brief by the existing
stage-2 verifier, docs corrected. `'deep'` was deliberately NOT added to
`VALID_TIERS`.

## Verdict history -- verify with: `grep -n "^## Verdict\|^# CYCLE" handoff/current/evaluator_critique_86.28.md`

| Cycle | Run | Verdict | Findings |
|---|---|---|---|
| 1 | `wf_10c6cbd2-cad` | CONDITIONAL | W1 illusory ordering guard; W2 three stale line citations |
| 2 | `wf_d0934c91-70b` | CONDITIONAL | W3 block-comment defeat of the hardened guard; W4 class audit excluded the production file |
| 3 | `wf_01c83c86-09d` | **NO VERDICT** (rail dropped, no StructuredOutput) | -- |

**TWO CONDITIONALs precede you.** Per CLAUDE.md failure-discipline F1, a
third CONDITIONAL is not available: return PASS or FAIL. Confirm the count
yourself; a dropped run is not a verdict and must not be counted as one.

## What changed in CYCLE 3 only (the delta you are grading)

Commit `294a9a09`. Diff it: `git show --stat 294a9a09`.

1. **W3 fix -- behavioural, not another regex.** New `[6d]` section in the
   checker: `loadDriver()` wraps the whole script in an async function so
   the driver runs outside the Workflow runtime; `driveRecording()` runs it
   with a recording `agent()` stub and COUNTS spawns. Reads no source, so
   comment/string/template tricks cannot reach it. A known-positive check
   leads the section (a SUPPORTED tier really does spawn) so that a reading
   of zero is not vacuous.
2. **Source scan demoted.** It now strips block comments too, and is
   labelled cheap-secondary; `[6d]` is the authority. Reason: under the
   prior Q/A's B1 mutant the scan printed `ok` during a real breach.
3. **W4 fix -- three stale claims in `research-gate.js`**, the file the
   cycle-2 audit missed: two `researcher.md` line citations replaced with
   symbols; "fourth requirement" corrected to "fourth LISTED ELEMENT ...
   CONDITIONAL" in the SOURCE (cycle 2 fixed it only in the artifacts); and
   `` `grep -c deep` returns 0 `` -- which measured 8 -- replaced by two
   ENFORCED checks.
4. **Cycle-2 false sentence corrected**: "a comment cannot stand in for
   code" -> "`//` comment".
5. `research-gate.js` cycle-3 edits are **comment-only**.

## Commands that settle most of it

```
node scripts/qa/verify_research_gate_workflow.mjs          # expect ALL GREEN: 73 passed, 0 failed
git show --stat 294a9a09                                   # the cycle-3 delta
git diff d638a3ec..294a9a09 -- .claude/workflows/research-gate.js | grep -E "^[+-]" | grep -vE "^[+-]{3}|^[+-]\s*//"
                                                           # expect EMPTY = comment-only
grep -n "VALID_TIERS = " .claude/workflows/research-gate.js  # expect no 'deep'
grep -nE "researcher\.md:[0-9]" .claude/workflows/research-gate.js  # expect none
```

## Known gaps, disclosed rather than defended

- **Criterion 9**: the FULL stage-1 + stage-2 live path has NOT been re-run
  after the change. Both post-change live runs take the refusal branch and
  spawn nothing. The cycle-2 Q/A measured the failure direction as safe
  (omitted stage-2 fields -> gate fails CLOSED, never a false pass) and
  judged criterion 9 MET on its literal verb. Re-judge independently.
- **Stale-but-labelled numbers**: `experiment_results` §"Immutable command,
  AFTER" still shows `61`, explicitly labelled CYCLE-1 MEASUREMENT with a
  pointer to the current 73. `live_check` §2/§5 likewise. I labelled rather
  than rewrote; judge whether that is honest.
- **The two prior verdicts are transcribed verbatim** in
  `evaluator_critique_86.28.md` and were not edited. Main's follow-ups are
  appended below each, clearly marked.
- **Attribution**: the peer session's `git add -A` swept part of this step's
  work into its commit `cad38647`. Content intact; attribution is queued
  separately as 86.15 and is out of scope here.

## Full artifacts, if you need them

`contract_86.28.md` (PLAN) · `research_brief_86.28.md` (research gate,
PASSED) · `experiment_results_86.28.md` (GENERATE + 3 follow-ups) ·
`live_check_86.28.md` (§1-8 cycle 1, §9-13 cycle 2, §14-17 cycle 3) ·
`evaluator_critique_86.28.md` (both verbatim verdicts + follow-ups).

Budget note: reading all five in full is what dropped the last spawn.
Prefer targeted `grep`/`sed` over full reads where a claim is checkable.
