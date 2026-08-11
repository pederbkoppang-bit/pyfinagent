---
name: silent-glob-archive-86-29
description: Step 86.29 -- archive-handoff.sh's step-specific glob has had an EMPTY DOMAIN for the whole life of the current naming convention; the fall-through is to a rolling file declaring a different step; OAIS+17a-4 both PERMIT the repair
metadata:
  type: project
---

Step 86.29 (silent no-match globs + archive provenance). Facts I MEASURED,
2026-08-10 -- do not re-derive from scratch.

**The glob is not "stale", it never had a domain.** `.claude/hooks/archive-handoff.sh:160`
globs `${sid}-*.md` / `phase-${sid}-*.md`. Measured matches: 0 for 86.29, 86.6,
phase-86.6, 4.5.9, 82.54. Repo-wide the suffix convention (`contract_86.6.md`)
has **896** files; the hyphen-prefix convention has **169**, and ALL 169 are
historical -- 44 in `handoff/archive/_quarantine_2026-04-21`, 7 in
`archive/misc`, 5 loose at `handoff/` root, 4 in the `phase-phase-6.N` dirs,
**0 in `handoff/current/`**. So this is not "the pattern drifted recently".

**Why it is silent:** bash leaves an unmatched pattern UNEXPANDED ("If the
pattern is unsuccessful, the word is left unchanged" -- bash manual), and
`shopt nullglob`/`failglob` are both **off** in this repo. The literal pattern
becomes `$f`, `[ -f "$f" ]` is false, loop body skipped, `moved=0`, no output.
**Trap for future me:** testing this in the default shell gives a FALSE
loud-failure -- zsh's NOMATCH aborts the command with "no matches found".
Always re-run glob behaviour under `bash -c` before claiming what the hook does.

**The fall-through target.** Only the rolling COPY branch at `:146-152` fires,
copying the unsuffixed `contract.md` / `experiment_results.md` /
`evaluator_critique.md` / `research_brief.md` (`research.md` is a DEAD name --
absent). Rolling `contract.md` declares `# Contract -- phase-82.54`. So each
newly-closed step archives 82.54's contract under its own id. 818 archive dirs.

**Second latent defect, same loop:** `:128` computes `short_sid="${sid#phase-}"`
for the target dir but `:160` interpolates the RAW `$sid` into both globs -> for
`phase-6.1` the second pattern is `phase-phase-6.1-*.md`. The
`handoff/archive/phase-phase-6.1..6.4/` dirs are the fossil of the same bug.

**Fixity cannot catch this and neither can the layout verifier.**
`verify_handoff_layout.py` checks WHERE files live (`ROLLING_KEEP:25`,
`ROLLING_KEEP_PREFIXES:42`), never whether an archived contract declares the
step whose dir it sits in. The file is intact; it is simply the wrong file.
A green layout check is fully compatible with 100% misattribution.

**External verdict on remediation (both permit it):** OAIS CCSDS 650.0-M-3
(Dec 2024 = ISO 14721:2025) DEFINES `Transformation` as "a Digital Migration in
which there is an alteration to the Content Information or PDI", and requires
Provenance Information to document "any changes that may have taken place since
it was originated, and who has had custody of it". SEC 17a-4(f)'s audit-trail
alternative (2022/23 amendment) permits modification given date/time/identity
and "re-creation of the original record if it is modified or deleted".
**Undocumented alteration is the prohibition, not alteration.**

**Prior art for the fix shape:** Pants ships the graduated policy verbatim --
`unmatched_build_file_globs` default **warn** (author-declared), `unmatched_cli_globs`
default **error** (user-typed). Snakemake is the derive-once model (one pattern +
wildcards drives both ends; outputs deleted pre-run so a stale file can't pass).
`set -e`/`failglob` are the WRONG tools in this hook -- both fight the
deliberate `trap 'exit 0' EXIT` at `:26`, and the hook runs in PARALLEL with
`auto-commit-and-push.sh` under the same PostToolUse matcher.

**Census caveat (`scripts/qa/derive_archive_misattribution_86_29.py`):** recall
is gated 2/2, but both known positives (`phase-86.6`, `phase-86.26`) are the
SAME failure instance, and **precision is unmeasured** -- the script's own
docstring `:50-53` records an earlier `[0-9]+`-only sid pattern that falsely
flagged 46 correct dirs while recall still read 2/2. See
[[guard-from-instance-not-class]].

Brief: `handoff/current/research_brief_86.29_rerun.md`.
