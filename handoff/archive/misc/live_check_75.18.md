# live_check -- Step 75.18 (anti-vacuous-guard doctrine codification)

Date: 2026-07-24. Verbatim captures; rc=$? discipline.

## 1. Immutable verification command (exit 0 -- SMOKE CHECK ONLY, disclosed)

```
immutable_exit=0
```
Per the research gate this command is partly PRE-SATISFIED on the
unmodified files and its skill clause is an OR-escape-hatch -- three of
the very anti-patterns the step codifies. The load-bearing evidence is
the KNOWN-MEMBER RECALL TEST in experiment_results_75.18.md (both known
phase-75 vacuous guards -- instance #3 pop-key and #8 seed OR-hatch --
are flagged by the codification as-written) + the Q/A's prose-read.

## 2. Change surface + C5 boundary

```
$ git diff --stat HEAD -- .claude/agents/qa.md docs/runbooks/per-step-protocol.md .claude/skills/
 .claude/agents/qa.md                               | 64 ++++++++++++++++
 .claude/skills/code-review-trading-domain/SKILL.md |  4 ++
 docs/runbooks/per-step-protocol.md                 | 21 +++++++
 3 files changed, 89 insertions(+)
$ git diff --name-only HEAD | grep -E '^(backend|frontend)/' | wc -l
0
```
Pure additions; qa.md 4b/4a untouched; zero product code.

## 3. Separation-of-duties disposition (BINDING)

The STATUS FLIP IS HELD: 75.18 remains `pending` post-Q/A. Review
surface for the operator: `git diff HEAD -- .claude/agents/qa.md`.
Flipping 75.18 to done (one masterplan edit) is the operator's approval
act and triggers the auto-commit/push of these edits. No session step
after this one depends on the new 4c. Next-session owed check:
scripts/qa/verify_qa_roster_live.sh + a fresh-spawn 4c self-disclosure
probe.

## 4. UI / live-loop
None (docs/harness only).
