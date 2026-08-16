# Research Brief -- step 86.97

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` is not required).

**Topic:** Guarding that a function is actually CALLED, not merely defined --
(a) why AST/extraction-based test harnesses go blind to bare call expressions,
(b) end-to-end driving of embedded heredoc scripts, (c) classifying early-exit
paths in shell hooks as must-log vs legitimately-silent (recursion guards).

## STATUS ENVELOPE (born inert at creation -- phase-86.37 -- flipped as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.97.md",
  "gate_passed": true
}
```

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://docs.python.org/3/library/ast.html | 2026-08-16 | official docs (T2) | WebFetch | `ast.Expr` is a distinct statement class: "When an expression, such as a function call, appears as a statement by itself with its return value not used or stored, it is wrapped in this container." `Module.body` "is a list of the module's Statements" -- so a bare call is a top-level `Expr`, NOT an `Assign`/`FunctionDef`. |
| https://www.shellcheck.net/wiki/SC2317 | 2026-08-16 | official tool docs (T2) | WebFetch | ShellCheck does reachability by control-flow analysis and explicitly disclaims dynamic invocation: code "invoked by variable name or in a trap" is flagged wrongly; "defined functions are assumed to be reachable when the script ends (not exits)". A definition-reachability analysis cannot decide invocation. |
| https://ar5iv.labs.arxiv.org/html/2103.08480 | 2026-08-16 | peer-reviewed, ICSE-SEIP'21 (T1) | WebFetch (ar5iv per the arXiv chain) | Extreme mutation = deleting the whole method body. **291/2041 methods (14%) were pseudo-tested** -- "their whole functionality can be removed, and still, no test will fail" -- **despite coverage**; recounting pseudo-tested lines as uncovered moved line coverage 89% -> 82%. Of 25 manually analysed, **14 were side-effect methods "not subject to testing"**. |
| https://ar5iv.labs.arxiv.org/html/1807.05030 | 2026-08-16 | peer-reviewed, EMSE (T1) | WebFetch (ar5iv per the arXiv chain) | Vera-Perez et al., §2.1: "A method is said to be pseudo-tested with respect to a test suite, if the test suite covers the method and does not assess any of its effects." §4.1: 2,540 across 21 projects, prevalence **1%-46%**. §4.4 dissent: only **30 of 101 (30%)** judged worth acting on; "it is not reasonable to prescribe the absolute absence (zero)". |
| https://bats-core.readthedocs.io/en/stable/writing-tests.html | 2026-08-16 | official docs (T2) | WebFetch | `run` "invokes its arguments as a command, saves the exit status and output into special global variables"; `run -N` asserts exit status N; `run --separate-stderr` splits streams. Caveat: `run` "executes its argument(s) in a subshell", so env side-effects do not persist -- assert on the file. |
| https://martinfowler.com/articles/harness-engineering.html | 2026-08-16 | authoritative blog, Bockeler 2026-04-02 (T3) | WebFetch | **Recency.** "If sensors never fire, is that a sign of high quality or inadequate detection mechanisms? We need a way to evaluate harness coverage and quality similar to what code coverage and mutation testing do for tests." Also: feed-forward-only gives "an agent that encodes rules but never finds out whether they worked". |
| https://git-scm.com/docs/githooks | 2026-08-16 | official docs (T2) | WebFetch | `post-commit` "is meant primarily for notification, and cannot affect the outcome of git commit" -- its exit status is ignored. stdout/stderr are forwarded to the user only for the server-side receive hooks. Establishes that git's output guarantee does NOT cover this hook. |
| https://arxiv.org/html/2602.10133v1 | 2026-08-16 | preprint, submitted 2026-02-07 (T1) | WebFetch (arXiv native HTML) | **Recency.** AgentTrace mandates logging, on its Operational Surface, "All explicit agent method calls, argument structures, return values, and execution timing". **Caveat recorded:** the paper "contains no experimental evaluation, benchmarks, or performance metrics" -- design reference, not evidence. |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/koalaman/shellcheck/issues/2966 | issue tracker (T5) | Corroborates that SC2317 conflates unreachable-code with never-invoked-function; the wiki page (read in full) is the authoritative statement |
| https://github.com/koalaman/shellcheck/issues/3137 | issue tracker (T5) | Same class: v0.10.0 miscategorises an uncalled function as unreachable |
| https://github.com/koalaman/shellcheck/issues/2542 | issue tracker (T5) | SC2317 false positive with `trap`/EXIT -- the indirect-invocation blind spot |
| https://github.com/koalaman/shellcheck/issues/2613 | issue tracker (T5) | "SC2317 findings are confusing and difficult to act on" -- noise argument, secondary |
| https://arxiv.org/pdf/2210.17215 | paper (T1) | Clang mutation-testing optimisation; adjacent, not about call-site deletion |
| https://arxiv.org/pdf/1707.01123 | paper (T1) | LittleDarwin framework; tool paper, superseded for this question by 2103.08480 |
| https://circleci.com/blog/what-is-mutation-testing/ | vendor blog (T4) | Introductory; no new claim beyond the read-in-full papers |
| https://oneuptime.com/blog/post/2026-01-30-mutation-testing-strategies/view | blog (T5) | 2026 recency hit; secondary summary of the same surviving-mutant framing |
| https://javapro.io/2026/01/21/test-your-tests-mutation-testing-in-java-with-pit/ | blog (T5) | 2026 recency hit; PIT tutorial, Java-specific, no new claim |
| https://www.diffblue.com/resources/what-is-mutation-testing-java/ | vendor (T4) | Vendor explainer; "delete call operator" definition only |
| https://opensource.com/article/19/2/testing-bash-bats | blog (T3) | Superseded by the official bats-core docs |
| https://github.com/bats-core/bats-core | official repo (T2) | README duplicates the readthedocs page read in full |
| https://earthly.dev/blog/python-ast/ | blog (T3) | Superseded by the official `ast` docs |
| https://en.wikipedia.org/wiki/Test_harness | encyclopedia (T5) | Definitional only |
| https://betterstack.com/community/guides/logging/structured-logging/ | vendor guide (T4) | Structured-logging best practice; no must-log/silent taxonomy, which is the part this step needed |

## Recency scan (2024-2026)

Searched 2024-2026 explicitly (queries listed below). **Two new findings that
complement, and one that sharpens, the canonical sources:**

1. **Böckeler, "Harness engineering for coding agent users", martinfowler.com,
   published 2026-04-02** (read in full) poses this step's exact open question:
   *"If sensors never fire, is that a sign of high quality or inadequate
   detection mechanisms? We need a way to evaluate harness coverage and quality
   similar to what code coverage and mutation testing do for tests."* It frames
   `_log_decision` as a **sensor** and the 86.97 defect as *a sensor whose
   firing was never verified*. It also names the failure mode of a
   feed-forward-only harness: *"an agent that encodes rules but never finds out
   whether they worked."*
2. **AgentTrace (arXiv:2602.10133v1, submitted 2026-02-07)** (read in full)
   mandates, on its "Operational Surface", logging *"All explicit agent method
   calls, argument structures, return values, and execution timing"* -- external
   support for the must-log classification of decision points. **Weight caveat,
   stated because it matters:** the paper *"contains no experimental evaluation,
   benchmarks, or performance metrics"* -- treat as a design reference, not
   evidence.
3. **No 2024-2026 work supersedes the pseudo-tested-method result.** The 2024-26
   mutation-testing hits (JavaPRO 2026-01-21, oneuptime 2026-01-30, ISSTA'24
   equivalent-mutant work) restate the surviving-mutant/equivalent-mutant
   framing already encoded in this repo's own harness. The canonical sources
   (Vera-Pérez EMSE, Niedermayr ICSE-SEIP'21) remain the operative ones.

**Search queries run (three-variant discipline):**
- *year-less canonical*: "Python ast module walk ... ast.Expr bare call expression missed static analysis dead code detection"; "bats-core testing bash scripts end-to-end heredoc embedded python early exit paths"; "shellcheck SC2317 unreachable command detect uncalled function shell script"
- *last-2-year*: "test harness blind spot function defined but never called call site coverage **2025 2026** research"; "structured logging silent failure early return observability must-log versus intentionally silent guard clause **2025**"
- *current-year frontier*: "mutation testing equivalent mutants deleted call statement survives test suite **2026**"

## Key findings

**F1 -- The blindness is a CATEGORY ERROR, not an oversight.** Python's own docs
are explicit: *"When an expression, such as a function call, appears as a
statement by itself with its return value not used or stored, it is wrapped in
this container [`ast.Expr`]"*, and `Module.body` *"is a list of the module's
Statements"* (https://docs.python.org/3/library/ast.html, accessed 2026-08-16).
`detector_source` at `verify_changelog_flip_86_91.py:88-92` keys its filter on
`ast.FunctionDef` / `ast.Assign` / `ast.AnnAssign` -- three **definition** node
classes. A call site is never a definition, so no `ast.Expr` can ever match. The
`NEEDED` tuple at `:78` cannot rescue it either: `NEEDED` is matched against
*names bound by the node*, and a bare call binds no name. **Adding
`"_log_decision"` to `NEEDED` does not help -- it is already there.**

**F2 -- MEASURED (this session, read-only, in-memory): the mutant is not
"surviving", the checker's INPUT IS UNCHANGED.** Deleting only
`_log_decision(bump_type)` (hook `:262`) and leaving the body byte-intact:

```
anchor occurrences in hook: 1
# measured at 52358053; at the phase-86.97 HEAD both are 8617 / 072056e58af2befa
# (the phase-86.97 criterion-5 docstring edit moved the count; the IDENTITY holds at both)
SHIPPED (control) bytes: 7597 sha1: f7458a6ab1f5fe96
SHIPPED (mutant)  bytes: 7597 sha1: f7458a6ab1f5fe96
BYTE-IDENTICAL: True
'_log_decision(bump_type)' appears in extracted SHIPPED? False
'bump_type = _flip_magnitude()' appears in extracted SHIPPED? False
hook source changed by mutation: True | delta bytes: 26
```

Positive control included: the anchor was unique (count=1) and the hook source
*did* change (+26 bytes), so the mutation is real and merely invisible. **The
consequence is stronger than "42/42 stays green": every assertion in that file
runs against `SHIPPED`, so no assertion added to it in the future can ever kill
this mutant.** This is the "a probe cannot detect what it did not extract"
shape -- the same lesson the file already teaches itself at `:378-393` (*"a
re-implementation cannot detect a mutation of the original"*) and then
reproduced one level up.

**F2b -- The blast radius is TWO call sites, not one.** The same run shows
`bump_type = _flip_magnitude()` (hook `:214`, inside an `ast.If` at `:213`) is
equally absent from `SHIPPED`. `run_detector` at `:132` calls
`ns["_flip_magnitude"]()` itself, so the checker **manufactures the call the
hook is never proven to make**. Deleting `:213-216` would leave `bump_type` at
`classify_commit`'s value -- i.e. silently restore the exact attempt-counting
bug phase-86.68 existed to remove -- with the checker green.

**F3 -- The literature name for this is `pseudo-tested`, and this case is a
degree worse.** Vera-Pérez et al.: *"A method is said to be pseudo-tested with
respect to a test suite, if the test suite covers the method and does not assess
any of its effects"* -- 2,540 such methods across 21 projects, prevalence
**1%-46%** (https://ar5iv.labs.arxiv.org/html/1807.05030, §4.1). Niedermayr et
al. measured **291 of 2,041 methods (14%)** pseudo-tested in an industrial
codebase, and recounting those lines as uncovered moved line coverage **89% ->
82%** (https://ar5iv.labs.arxiv.org/html/2103.08480, §IV-A). `_log_decision` is
not merely unasserted: the *production* path to it is untested, and the checker
supplies the invocation itself at `:534`.

**F4 -- The literature also predicts the exact sub-class.** Of 25 pseudo-tested
methods manually analysed, **14 were "side-effect methods not subject to
testing"** (2103.08480, §IV-A). `_log_decision` is a pure-side-effect function:
writes a file, returns `None`. Its correctness has two independent halves --
**(a) the body writes the right line**, guarded at `:537-573` including a
delete-the-write mutation cell; **(b) the body is invoked at all**, unguarded.
phase-86.91 cycle 4 closed (a) and left (b), and (b)'s production effect is
*identical* to (a)'s: zero lines in the decision log.

**F5 -- The endorsed fix is to DRIVE THE PROCESS, and it closes both defects at
once.** bats-core's `run` *"invokes its arguments as a command, saves the exit
status and output into special global variables"*; `run -N` asserts an exact
exit status and `run --separate-stderr` splits streams
(https://bats-core.readthedocs.io/en/stable/writing-tests.html). An end-to-end
drive of the hook as a **process** is the only instrument that reaches the three
bash `exit 0` paths at `:28`/`:33`/`:37` **at all** -- they live outside the
heredoc, so *no* Python-extraction approach can ever touch them. Caveat from the
same docs: `run` executes in a subshell, so environment side-effects do not
persist -- assert on the **log file**, not on variables.

**F6 -- A cheap static complement (strictly weaker, worth having anyway).**
Inside the existing checker, assert against the heredoc AST that a top-level
`ast.Expr(value=ast.Call(func=ast.Name(id="_log_decision")))` exists -- changing
the question from *"is it defined"* to *"is it called"*. It is weaker than a
drive: it cannot see the call being wrapped in `if False:`, moved below an early
`sys.exit(0)`, or shadowed. Recommend **both**: static call-site presence (fast,
runs in the existing 42-assertion file) + one end-to-end drive (authoritative).

**F7 -- Classifying the three early exits: 2 must-log, 1 silent-but-counted.**
The `post-commit` git guarantee does **not** apply: git docs say the hook *"is
meant primarily for notification, and cannot affect the outcome of git commit"*
and forward stdout/stderr to the user only for the *server-side* receive hooks
(https://git-scm.com/docs/githooks). This is not a git hook at all -- it is a
Claude Code PostToolUse entry (`.claude/settings.json:74`) plus a manual
invocation (`auto-commit-and-push.sh:43`) whose stderr lands in a **gitignored**
log. That is the same argument `_log_decision`'s own docstring makes at `:226-232`.

| Path | Condition | Class | Why |
|---|---|---|---|
| `:27-29` | subject matches `^chore: (auto-changelog\|changelog drift)` -- the **recursion guard** | **legitimately silent, but MUST BE COUNTED** | It is the hook declining to act on *its own* commit. MEASURED: **28 of 56** commits on 2026-08-16 match it, so full-verbosity logging roughly doubles the log with self-traffic. But with *no* line at all you cannot distinguish "guard fired" from "hook never ran" -- an unknowable denominator, which is criterion 4's own defect one level up. Minimum viable: a distinct cheap `reason=recursion_guard` line. |
| `:32-34` | `CHANGELOG.md` absent | **MUST-LOG** | Silently and permanently disables the entire feature. |
| `:36-38` | `### Recent Activity` heading missing/renamed | **MUST-LOG** | A heading rename is a plausible, total, silent kill. |

**F8 -- Where the bash-side line goes.** These exits are *pre-heredoc*, so
`_log_decision` (Python) does not exist yet. Two options: **(a)** a one-line
`printf >> handoff/logs/changelog-decisions.log` before each `exit 0`; **(b)**
restructure so the heredoc always runs and the three conditions become Python
`reason=` branches. (b) unifies the closed reason set but moves the recursion
guard, which must stay ahead of any git-writing action (`:380-381`). **Recommend
(a)** -- minimal, preserves the guard's position, and keeps the reason
vocabulary in one closed set.

**F9 -- No shell linter will catch this.** ShellCheck's SC2317 does
*reachability*, not *invocation*, and explicitly disclaims dynamic dispatch:
code *"invoked by variable name or in a trap"* is misflagged, and *"defined
functions are assumed to be reachable when the script ends (not exits) since
another file may source and invoke them"* (https://www.shellcheck.net/wiki/SC2317).
Its sibling SC2329 ("function never invoked") is documented as conflated and
noisy (koalaman issues #2966, #3137, #2542). Decisively: ShellCheck does not
parse the heredoc body, so a deleted **Python** call inside `<< 'PYEOF'` is
outside its language entirely. The step's own immutable command
(`bash -n ... && echo parses`, `.claude/masterplan.json:25979`) is a syntax
check and reaches neither defect.

## Consensus vs debate (external)

**Consensus:** coverage-shaped evidence over-reports; deleting a whole
side-effecting unit and seeing green is the canonical detector (Vera-Pérez;
Niedermayr); driving the real process is the way to test a script's exit paths
(bats-core); reachability analysis cannot decide invocation (ShellCheck).

**Debate / dissent worth recording:** Vera-Pérez et al. explicitly refuse a
zero-tolerance rule -- *"it is not reasonable to prescribe the absolute absence
(zero) of pseudo-tested methods"* -- and found developers judged only **30 of
101 (30%)** pseudo-tested methods worth acting on, rejecting the rest as
generated, debug-only, or trivial. Niedermayr et al. likewise concede extreme
mutation *"trades accuracy for speed gains"* and warn their developer sample was
two people. **Applied here this argues for scope discipline, not inaction:** the
`:262` call is a genuine positive (its deletion is production-silent and
indistinguishable from the bug 86.91 was opened to fix), whereas `:355`
(`lines.insert(...)`, same `ast.Expr` class, also unguarded) is *not* worth a
cell -- its effect is directly visible in `CHANGELOG.md`.

## Pitfalls (from the literature AND this repo's own cycle history)

- **P1. A probe cannot detect a mutation to what it did not extract.** Already
  learned at `verify_changelog_flip_86_91.py:378-393` and re-committed one level
  up. Any new guard must be scored against a *positive control* that proves the
  mutation is real (F2 does this via the `+26 bytes` delta).
- **P2. Adding exemplars cannot close a class bound.** The file states it itself
  at `:328-340` -- *"an N-id fixture is defeated by an N-id whitelist"*. Here the
  analogous dead end is enlarging `NEEDED`: the call site binds no name, so
  there is nothing to add.
- **P3. A mutant that cannot build is UNSCORABLE, not KILLED.** Handled at
  `:493-519` / `:502-507`; a new call-site cell must keep that three-outcome
  discipline. Equivalence is undecidable -- 4-39% of real-world mutants are
  equivalent (ISSTA'24, cited at `:24-26`).
- **P4. Do not over-instrument the silent paths.** 28 of 56 commits today hit the
  recursion guard; a verbose line per hit makes the decision log majority
  self-traffic and degrades the signal criterion 4 was created to produce.
- **P5. `set -euo pipefail` at `:7`** means a bash-side logging `printf` that
  fails (unwritable `handoff/logs/`) would abort the hook. Any added write must
  be `|| true`-guarded, matching the never-raise discipline the Python halves
  already follow at `:206-210` and `:257-259`.

## Application to pyfinagent (external findings -> file:line)

| Finding | Anchor | Implication for the contract |
|---|---|---|
| F1/F2 structural blindness | `verify_changelog_flip_86_91.py:81-95`, `:78` | The fix is **not** a new assertion in this file; its input is invariant. Either add a call-site AST check (F6) or drive the process (F5). |
| F2b second unguarded call | hook `:213-216` | Scope decision for Main: guard **both** call sites or state explicitly why only `:262`. Deleting `:214` silently restores the 86.68 attempt-counting bug. |
| F3/F4 pseudo-tested side-effect | hook `:219-262`, checker `:528-534`, `:559-573` | Body guarded, invocation not. Name this split in the contract so cycle 5 does not close the same half again. |
| F5 end-to-end drive | hook `:27-38` + `:262` | One process-level drive covers the bash exits *and* the call. `run -N` / assert on the log file, not on env vars. |
| F7 must-log classification | hook `:27-29` / `:32-34` / `:36-38` | 2 must-log + 1 silent-but-counted; extends criterion 4's closed reason set with `recursion_guard`, `changelog_absent`, `heading_absent`. |
| F7 bound never stated | `.claude/masterplan.json:25968` (86.97 audit_basis) | 86.91 criterion 4 holds only for invocations that *reach* the detector. That bound belongs in the contract explicitly. |
| Recency (Böckeler 2026) | whole step | "If sensors never fire, is that high quality or inadequate detection?" -- the decision log **is** the sensor; 26 lines vs 51 commits is the measurement that answers it. |
| P5 never-raise | hook `:7`, `:206-210`, `:257-259` | Any bash-side log write must not be able to abort the hook. |

## Internal code inventory (all claims file:line anchored)

### A. `.claude/hooks/post-commit-changelog.sh` (382 lines)

| Anchor | Role | Status |
|---|---|---|
| `:7` | `set -euo pipefail` | live |
| `:27-29` | **Early exit 1/3 -- RECURSION GUARD.** `grep -qiE "^chore: (auto-changelog\|changelog drift)"` -> `exit 0` | live; **structurally silent** (pre-heredoc) |
| `:32-34` | **Early exit 2/3** -- `[ ! -f "$CHANGELOG" ]` -> `exit 0` | live; silent |
| `:36-38` | **Early exit 3/3** -- `! grep -q "### Recent Activity"` -> `exit 0` | live; silent |
| `:43` / `:371` | `python3 - ... << 'PYEOF'` heredoc boundaries | the detector + logger live INSIDE this |
| `:59-92` | `classify_commit` (subject/body -> major/minor/patch/none) | live; **not in `NEEDED`, so never extracted/driven** |
| `:100` | `_ABSENT` identity sentinel (PEP 661) | live; extracted (`Assign`) |
| `:103` | `_FLIP_DECISION: dict = {}` | live; extracted (`AnnAssign`) |
| `:106-210` | `_flip_magnitude()` -- 4 `return "none"` branches, never raises | live; extracted (`FunctionDef`) |
| `:213-216` | `if bump_type != "major": bump_type = _flip_magnitude()` | **`ast.If` -> NOT extracted.** The only production call site of `_flip_magnitude` is invisible to the checker |
| `:219-259` | `_log_decision(bump)` -- writes `handoff/logs/changelog-decisions.log`; never raises | live; extracted (`FunctionDef`) |
| **`:262`** | **`_log_decision(bump_type)` -- the production call** | **`ast.Expr` wrapping `ast.Call` -> NOT extracted, NOT driven, UNGUARDED** |
| `:346`, `:352` | `sys.exit(0)` (no table anchor / duplicate hash) | POST-logger, so they do not suppress the decision line |
| `:355` | `lines.insert(...)` -- the actual Recent-Activity row write | also a bare `ast.Expr`; same structural class |
| `:376-381` | post-write `exit 0` paths + the hook's own `chore: auto-changelog` commit | this commit is what `:27` guards against |

### B. `scripts/qa/verify_changelog_flip_86_91.py` (582 lines, 42 assertions)

| Anchor | Role | Status |
|---|---|---|
| `:65-69` | `heredoc_python()` -- slices between `<< 'PYEOF'` and `\nPYEOF` | correct |
| `:78` | `NEEDED = ("_ABSENT", "_FLIP_DECISION", "_flip_magnitude", "_log_decision")` -- **four DEFINITION names; no call site is nameable here** | the blind spot's proximate cause |
| `:81-95` | `detector_source()` -- iterates `tree.body`, keeps only `ast.FunctionDef`, `ast.Assign`, `ast.AnnAssign` | **STRUCTURAL BLINDNESS: an `ast.Expr` can never match, so `:262` can never enter `SHIPPED`** |
| `:98-106` | `load_detector()` -- `missing = [n for n in NEEDED if n not in ns]` | asserts DEFINEDNESS only; a namespace membership test cannot see invocation |
| `:113-133` | `run_detector()` -- calls `ns["_flip_magnitude"]()` **directly** | the checker supplies the call the hook is never proven to make |
| `:528-534` | `drive_log()` -- calls `ns_l["_log_decision"](bump)` **directly** | same shape one level down |
| `:559-573` | mutation cell "delete-the-decision-log-write" -- deletes the `open(...)/write(...)` INSIDE the body | kills body deletion; **cannot kill call deletion** |
| `:304-320` | `replay_predicate()` -- same `FunctionDef`/`Assign`-only filter on the sibling replay | **the blindness is DUPLICATED, not localised** |

### C. `handoff/logs/changelog-decisions.log` (gitignored)

MEASURED 2026-08-16 (this session):
- File exists, 2,355 bytes, **26 lines**, first `2026-08-16T08:23:33Z 8dc70502`, last `2026-08-16T19:37:52Z e45c1bf6 bump=patch reason=flip_transitioned ... transitioned_done=86.92`.
- **The `:262` call IS firing in production today** -- so this step guards a live, working path, not a dead one.
- `git rev-list --count 8dc70502..HEAD` = **51 commits** in the same window vs **26** decision lines.
- Of 56 commits dated 2026-08-16, **28** match the `:27` recursion-guard regex `^chore: (auto-changelog|changelog drift)`.
- So the silent population is real and roughly half of all commits, and it is dominated by the recursion guard.

### D. Callers / consumers

- `.claude/settings.json:74` -- PostToolUse entry `bash "${CLAUDE_PROJECT_DIR}/.claude/hooks/post-commit-changelog.sh"`.
- `.claude/hooks/auto-commit-and-push.sh:43` -- `CHANGELOG_HOOK="$PROJECT_ROOT/.claude/hooks/post-commit-changelog.sh"` (second invocation path; `:9` comment "invoke post-commit-changelog.sh manually").
- `tests/verify_phase_23_8_4.py:291` -- existence check only.
- `.claude/masterplan.json:25979` -- step 86.97's own immutable command is `bash -n ... && echo parses`, i.e. **a syntax check that cannot reach either defect**.

## Application to pyfinagent

_(pending)_

## Research Gate Checklist

Hard blockers -- all satisfied:
- [x] **8** authoritative external sources READ IN FULL via WebFetch (floor 5). Mix: 2 peer-reviewed (T1) + 1 preprint (T1) + 3 official docs (T2) + 1 authoritative blog (T3). No community-tier source is in the read-in-full set.
- [x] **23** unique URLs total (8 read in full + 15 snippet-only); floor 10.
- [x] Recency scan (2024-2026) performed AND reported, with 2 new findings and 1 explicit "does not supersede" result.
- [x] Full pages read, not abstracts. Both arXiv papers went via the `ar5iv` chain (`1807.05030`, `2103.08480` -- both pre-Dec-2023) and the 2026 preprint via native `arxiv.org/html/2602.10133v1`. **No `arxiv.org/pdf/` URL was WebFetched.**
- [x] file:line anchors for every internal claim (see Internal code inventory, sections A-D).

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE, plus the two invocation paths (`.claude/settings.json:74`, `.claude/hooks/auto-commit-and-push.sh:43`) and the step's own masterplan entry (`.claude/masterplan.json:25968`, `:25979`).
- [x] Contradictions noted -- see "Consensus vs debate": Vera-Perez et al. explicitly refuse a zero-tolerance rule and report 70% of pseudo-tested methods judged **not** worth acting on. Applied as scope discipline (guard `:262`/`:214`; do **not** guard `:355`).
- [x] All claims cited per-claim with URL + access date, not only in a footer.

Coverage (informational; step is NOT audit-class): 2 rounds, 0 dry rounds,
K_required 2, 0 new read-in-full findings in the last round, `dry=false`.
`coverage.dry` is not a gate condition for this step.

## Bounds on this brief (stated, not hidden)

1. **The three bash `exit 0` paths were classified from measurement + reasoning,
   not driven.** Nothing in this session executed the hook with a missing
   `CHANGELOG.md` or a renamed heading. The recursion-guard rate (28/56) IS
   measured; the other two are argued from consequence.
2. **The 51-commits-vs-26-lines figure is a window count, not a per-invocation
   audit.** 56 commits are dated 2026-08-16 and 28 match the `:27` regex,
   leaving 28 non-guard commits against 26 log lines -- a residual of 2 that
   this session did not resolve (candidates: commits made outside the
   PostToolUse path, or lines predating the log's 08:23:33Z start). Do not
   quote "51 vs 26" as an exact silent-invocation count.
3. **F2 proves the checker's input is invariant; it does not prove the whole
   checker stays exit-0.** The byte-identity of `SHIPPED` is the mechanism and
   is decisive for every `SHIPPED`-derived assertion; the "42/42 ALL GREEN"
   figure is carried from the masterplan audit_basis at `:25968`, not
   re-executed here (the scratch-mirror run was denied by the sandbox).

## STATUS ENVELOPE -- FINAL

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.97.md",
  "gate_passed": true
}
```
