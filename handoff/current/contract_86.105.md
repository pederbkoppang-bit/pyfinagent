# Contract -- step 86.105 (PLAN banked; GENERATE deferred to the post-core harness)

**Step:** 86.105 -- the handoff layout invariant is red.

## Research-gate summary (what changes the plan)

Gate PASSED (`wf_5eacb773-aa5`; 7 sources in full, 25 URLs; brief
`research_brief_86.105.md`, 29,514 chars). The spawn itself was CAUSED by a
research_routing signal through `scripts/harness/research_router.py` (round
1 of 2) -- the phase-86.72 end-to-end drive; this contract is the F2 leg's
first productive output.

**The step's audit basis does not reproduce, and the researcher measured
why:**

1. Today's red run is **667 findings, not six**: 664 are ONE class -- every
   `name_<sid>.md` artifact in current/ flagged "no step-id prefix" because
   `verify_handoff_layout.py:51 STEP_ID_RE` expects a LEADING sid and
   matches 0/664 -- plus the 3 real root-level misplacements (2 logs + 1
   audit). The filed "three stray current/ files" was a sample of the
   664-member class recorded as a census.
2. The dead regex means the done-step arm (:120-125) has ZERO reachable
   cells; commit c3286524 already root-caused this and assigned the regex
   to PENDING step **75.11.4** (boundary: scripts/housekeeping/** + tests),
   whose criteria also require status-aware refusal, a pending-vs-done
   fixture pair, a mutation, and idempotency.
3. **Traps measured**: all three move destinations ALREADY EXIST and are
   stale -- `handoff/audit/prompt_leak_redteam_audit.jsonl` holds 2,035
   bytes the root copy lacks, so a bare `git mv` silently DELETES
   append-only history (merge per the phase-36.8 precedent);
   `backfill_handoff_archive.py` carries the same dead regex and its
   documented "safe to re-run" is FALSE (it would sweep 666/666 files
   including the research brief); `autoresearch.launchd.log`'s writer is
   operator-owned (launchd) -- repoint is a numbered ask, only the .jsonl
   is git-tracked. Martin Fowler's ParallelChange names the failure shape:
   expand ran, migrate/contract never did.
4. Blast radius measured by Main on the gate's findings: with a CORRECT
   trailing-sid regex, current/ holds **424 done-step files** (should
   archive), 180 open/parked-step files (must stay), 120 no-sid files
   (rolling + day reports + incident notes -- policy needed per file
   class).

## Sequencing decision (recorded for the evaluator and the operator)

**GENERATE is DEFERRED until 75.11.4 lands.** Criterion 1 (checker exits 0)
is unreachable without the regex fix that 75.11.4 owns, and the 424-file
archive storm is mechanical work explicitly suited to the post-core
harness on the cheaper model (operator token-thrift directive,
2026-08-17). Execution order for the working harness: 75.11.4 (scripts,
bounded) -> 86.105 GENERATE (fixed backfill archives the 424; the 3 roots
merged/moved with writers repointed and consumers grepped; checker green
with the corrected 667-finding before-run quoted beside it -- the "six
findings" phrasing in criterion 1 is quoted AS FILED with the corrected
census beside it, per the corrections-replace discipline).

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

1. python3 scripts/housekeeping/verify_handoff_layout.py exits 0 after the fix, output quoted -- and the 2026-08-17 red run (exit 1, six findings) is quoted beside it as the before
2. every misplaced TRACKED file is moved with git mv so history survives; nothing is deleted; handoff/prompt_leak_redteam_audit.jsonl keeps its append-only content byte-identical across the move
3. the WRITER of each root-level log is found and repointed, or shown already dead -- a move without repointing regresses on the writer's next fire; if the writer is operator-owned (launchd plist, crontab), the repoint is recorded as a numbered operator ask instead of edited unilaterally
4. every consumer reading the old paths is found by grep and repointed in the same commit, with the grep command and its output quoted
5. no file under .claude/hooks is modified by this step, and verdict semantics are unchanged

## References

`research_brief_86.105.md` (the 667-census, the root cause, the three
traps, ParallelChange, Anthropic harness-design, the scratch/audit-tree
patterns); step 75.11.4 (the owning regex fix); phase-36.8 merge precedent;
the 86.43 write-first collision record.

---

## OPERATOR DECISION RECORDED -- 2026-08-18 -- "which rule wins" (log placement)

Operator granted general permission to proceed on parked decisions
("you have my promission", 2026-08-18, verbatim). This addresses the
specific sub-question `goal_next_2026-08-18.md` raised about
`handoff/autoresearch.log` vs the layout invariant -- **not** a decision to
execute this step's full GENERATE tonight (still correctly sequenced behind
75.11.4 per this contract's own "Sequencing decision" above; the 667-finding
archival sweep has real data-loss traps -- see the merge/backfill/regex
hazards in `research_brief_86.105.md` -- and is not attempted at 1am
unattended).

**Read from source** (`backend/api/cron_dashboard_api.py:139-160`,
`_log_paths()`, phase-23.3.5): six files are written to `handoff/` root
because macOS **launchd** plists set `StandardOutPath`/`StandardErrorPath`
directly at those paths -- `mas-harness.log`, `autoresearch.log`,
`mas-harness.launchd.log`, `autoresearch.launchd.log`, `ablation.log`,
`ablation.launchd.log`. This is not a repo-code choice the layout invariant
can override by moving files: launchd will recreate the file at its
configured path on the service's next write regardless of any `git mv`
tonight (the exact "move without repointing regresses on the writer's next
fire" trap this contract already names for `autoresearch.launchd.log`
specifically -- it applies identically to all six). Repointing requires
editing each plist and a `launchctl bootout`+`bootstrap` cycle per service,
which CLAUDE.md reserves to the operator (away-ops rail 9) and which this
step's own criterion 3 already requires be recorded as a numbered ask
rather than edited unilaterally for the launchd-owned case.

**Ruling**: the general rule (no `*.log` at `handoff/` root) is correct for
repo-controlled writers and should not be weakened globally. For the SIX
specifically-named, launchd-plist-controlled paths above, `verify_handoff_
layout.py` should carry a narrow, named exemption (matching `_log_paths()`'s
own allowlist) rather than flagging them as violations it cannot actually
fix without an operator-gated plist change. This does not resolve on its
own -- it is scope for 86.105's eventual GENERATE, recorded now so the
next session does not re-derive it. The `backend.log` root-placement case
("repo-root for legacy reasons", `cron_dashboard_api.py:146`) is NOT covered
by this ruling -- its writer justification was not verified this session and
should be checked separately before assuming the same exemption applies.

No file was moved and no plist was touched by this ruling.
