# Research Brief -- step 75.11.4

**Topic:** Status-aware archival of step-scoped artifacts in a file-based agent harness --
filename-convention reconciliation, safe-by-default migration scripts, and archive
provenance integrity.

**Tier:** moderate (caller-stated). **Audit-class:** YES (loop-until-dry, K=2).
**Researcher:** Layer-3 pyfinagent researcher (external literature + internal code inventory).
**Started:** 2026-08-17.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 18,
  "snippet_only_sources": 54,
  "urls_collected": 72,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": true,
    "rounds": 8,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Every mature archival standard (BagIt RFC 8493, OCFL 1.0, Git, SWHID) makes identity EXPLICIT and treats the filename as non-semantic; BagIt states filenames have no given meaning, OCFL binds content path to logical path by digest not filename. Name-based classification is known-fragile (IEEE ISCC08) and has already disarmed pyfinagent kill switch via a *_audit.jsonl rule. Fuzzy name reconciliation is a trap with an admitted precision/recall bind. Fix: ParallelChange expand phase (recognise BOTH conventions from ONE shared regex, rename nothing), dry-run-by-default with --execute, and additive provenance. LIVE RE-MEASUREMENT REFUTES THREE STEP-TEXT FIGURES: current/ is 0 PREFIX and 582 SUFFIX (not 20); archive is 507 prefix-in-phase-dirs and 488 suffix-in-misc (not 13198/428); misattribution is 156 of 842 at precision 0.9936 (not 129 of 747). A bare backfill run would sweep 664 of 668 .md files to archive/misc/; 395 masterplan verification criteria reference handoff/current/ by literal path.",
  "brief_path": "handoff/current/research_brief_75.11.4.md",
  "gate_passed": true
}
```

**brief_status is COMPLETE**: the loop-until-dry critic reached 2 consecutive dry
rounds and every hard blocker is satisfied. The caller RECOMPUTES `gate_passed` and
cross-checks each claimed URL against this file -- all 18 are present (verified).

---

## Sections (filled incrementally)

- [x] Internal code inventory (live re-measurement)
- [x] Read in full (18, floor is 5)
- [x] Snippet-only (54)
- [x] Recency scan (2024-2026)
- [x] Key findings
- [x] Consensus vs debate
- [x] Pitfalls
- [x] Application to pyfinagent
- [x] Research Gate Checklist

---

# PART 1 -- INTERNAL CODE INVENTORY (live re-measurement)

Every number below was RE-MEASURED on 2026-08-17 in this session. The command that
produced each is given. Where my figure disagrees with the step text, **the step
text's figure is stated and the disagreement is called out, not silently replaced.**

## 1.1 Files inspected

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/hooks/archive-handoff.sh` | 336 | PostToolUse hook; populates `handoff/archive/phase-<sid>/` on masterplan `done` flip | LIVE; carries **both** conventions (see 1.3) |
| `scripts/housekeeping/backfill_handoff_archive.py` | 221 | One-time/idempotent sweeper of `handoff/current/` -> archive | LIVE; **destructive by default** (see 1.5) |
| `scripts/housekeeping/verify_handoff_layout.py` | 154 | Invariant checker for the `handoff/` partition | LIVE; shares the same broken regex |
| `scripts/qa/derive_archive_misattribution_86_29.py` | 393 | **Already-shipped** misattribution census w/ recall gate + synthetic controls + precision oracle | LIVE, read-only; **this is the classifier the step asks for -- it exists** |
| `scripts/qa/prove_archive_provenance_86_29.py` | -- | Companion prover for the PROVENANCE.md contract | present |
| `.claude/hooks/lib/verdict_gate.py` | 106+ | Resolves the machine-readable Q/A verdict; **documents the historical damage** | LIVE |
| `.claude/hooks/auto-commit-and-push.sh` | 258+ | Reads `handoff/current/evaluator_critique*.json` + `live_check_<sid>.md` by literal path | LIVE |
| `.claude/hooks/lib/live_check_gate.py` | -- | Requires `handoff/current/live_check_<id>.md` by literal path | LIVE |
| `.claude/rules/research-gate.md` | 338 | "Handoff folder convention" invariants | doc |

`internal_files_inspected = 9`.

## 1.2 The two conventions, stated precisely

The step frames this as "hook = PREFIX-dash vs backfill = SUFFIX-underscore".
**Measured, that framing is not quite right, and the correction matters for the fix:**

- **`archive-handoff.sh` contains BOTH conventions simultaneously.**
  - *Branch 1 (primary, phase-86.29)* at `.claude/hooks/archive-handoff.sh:226-242`
    derives **SUFFIX-underscore** names:
    `for base in contract experiment_results evaluator_critique research_brief live_check; do src="$CURRENT_DIR/${base}_${short_sid}.md"`.
  - *Legacy branch* at `.claude/hooks/archive-handoff.sh:276` globs **PREFIX-dash**:
    `for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md`.
    The hook's own comment at `:220-223` says this branch matches **zero** files
    ("0 matches each, against a positive control proving the count can be 1").
- **`backfill_handoff_archive.py:64` and `verify_handoff_layout.py:51` carry the
  SAME regex, byte-identical**, and it is **PREFIX-only**:
  `STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")`.

So the reconciliation target is: **one live writer convention (SUFFIX-underscore)
against two readers whose only step-id recogniser is PREFIX-dash.** The legacy
PREFIX branch in the hook is *dead code* on the current corpus, not a competing
live convention.

## 1.3 M1 -- `handoff/current/` census (PREFIX vs SUFFIX)

Command (read-only; exact regexes lifted verbatim from the two scripts):
`python3 <scratchpad>/measure.py` -- reproduced inline in 1.8.

| Bucket | Count |
|---|---|
| total files in `handoff/current/` (non-dir, non-dotfile) | **725** |
| total `.md` files | **668** |
| bare rolling-keep names (`contract.md`, `experiment_results.md`, `evaluator_critique.md`, `research_brief.md`) | **4** |
| **PREFIX** convention (`STEP_ID_RE` matches) | **0** |
| **SUFFIX** convention, hook's 5 derived bases (`<base>_<sid>.md`) | **582** |
| **SUFFIX** convention, other stems (`*_<sid>.md`) | **10** |
| neither convention (day reports, incident notes, ad-hoc) | **72** |
| **`.md` files `backfill` would route to `archive/misc/`** | **664** |

- Step text claimed **0 and 20**. **PREFIX=0 CONFIRMED.** **SUFFIX=20 is REFUTED
  by a factor of ~29**: the live figure is 582 (hook bases) / 592 (incl. other
  stems). Do not carry `20` into the contract.
- The consequential number is the last row: **664 of 668 `.md` files in
  `handoff/current/` would be swept to `archive/misc/` by a bare
  `python scripts/housekeeping/backfill_handoff_archive.py` run.** Only the 4
  bare rolling names survive.

## 1.4 M2 -- historical archive population

Commands:
`find handoff/archive -type f | wc -l`;
`ls -1d handoff/archive/phase-* | wc -l`;
plus the regex classification in `measure.py`.

| Bucket | Count |
|---|---|
| `handoff/archive/phase-*` directories | **842** |
| files inside `phase-*` dirs (recursive) | **3,718** |
| `.md` PREFIX-style inside `phase-*` | **507** |
| `.md` SUFFIX-style inside `phase-*` | **22** |
| `.md` bare/other inside `phase-*` (`contract.md`, `PROVENANCE.md`, ...) | **3,187** |
| `handoff/archive/misc/` total files | **748** |
| `handoff/archive/misc/` `.md` files | **728** |
| -- of which PREFIX-style | **62** |
| -- of which SUFFIX-style | **488** |
| -- of which neither | **178** |
| whole `handoff/archive` tree, all files | **48,398** |
| whole `handoff/` tree, all files | **49,750** |

- Step text claimed **13,198 and 428**. **Neither reproduces.** The
  prefix-in-phase-dirs figure is **507**, not 13,198; the suffix-in-misc figure is
  **488**, not 428. 13,198 is not any denominator I can reconstruct from this tree
  (the plausible large denominators are 3,718 inside `phase-*` and 48,398 for the
  whole archive, the latter dominated by non-`phase-*` dirs -- `handoff/archive/slack/`
  and friends). **Treat both step-text figures as unusable.**
- The ratio that matters: **488 SUFFIX-style `.md` files already sit in
  `archive/misc/`** -- i.e. the sweep has already happened, repeatedly, and
  `archive/misc/` is where a step's real artifacts went.

## 1.5 The migration script is NOT safe-by-default

`scripts/housekeeping/backfill_handoff_archive.py:218-221`:

```python
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
raise SystemExit(main(dry_run=args.dry_run))
```

`--dry-run` is **opt-in**. A bare invocation **executes** `shutil.move` on 664
files. This inverts the safe-by-default property the step asks for (dry-run
default + explicit `--execute`). The module docstring at `:12-14` even advertises
the bare form as a normal usage.

Two further safety gaps in the same file:
- `_move()` at `:125-131` calls `shutil.move`, **not** `git mv`. The hook's legacy
  branch prefers `git mv` (`archive-handoff.sh:279`) and falls back to `mv`. The
  backfill has no such preference, so history/rename-tracking is lost.
- Idempotency is real but is a **`-vN` minting** idempotency (`_safe_target`,
  `:114-122`): re-running never clobbers, but it *does* create `-v2`, `-v3`
  duplicates. That is exactly how `kill_switch_audit.jsonl` reached `-v3` and
  `-v4` (comment at `:73-74`). **"Idempotent" here means "non-destructive of prior
  evidence", NOT "converges to a fixed point".**

## 1.6 M3 -- already-mis-filed archive directories

The step asks for a count and warns its own heuristic must be refined. **A refined,
gated classifier already ships**: `scripts/qa/derive_archive_misattribution_86_29.py`.
I ran it live rather than writing a new heuristic.

Command: `python3 scripts/qa/derive_archive_misattribution_86_29.py`

Its matching rule (`:72-80`, `_DECLARE`): read the first 4,000 chars of the dir's
`contract.md` (or first `contract_*.md`), apply 7 ordered declaration patterns
(`# Contract -- step <sid>`, `# Contract -- phase-<sid>`, `**Step ID**:`,
`**Step**:`, `step: <sid>`, `**Step id:**`, `^#.*phase-<sid>`), with
`_SID = [0-9]+(?:\.[0-9A-Za-z]+)*` (alphanumeric segments -- `25.A` is a real id)
and `_DASH = (?:--|—|–)` (en/em-dash tolerated). First hit wins; `mismatch` iff the
declared sid != the directory's sid.

**Live census over 842 dirs:**

| Bucket | Count |
|---|---|
| **mismatch** | **156** |
| agree | 440 |
| unclassified | 222 (206 = harness per-cycle contracts that declare no step *by design*; 16 genuinely opaque) |
| no_contract | 24 |

**Precision, measured by the script's independent second-opinion oracle** (which is
itself control-tested to prove it can answer both ways):
- CONFIRMED mismatches (dir's own sid appears in **no** declaration in the head): **155**
- SUSPECT (possible parser error): **1** (`phase-69`, census says it declares `69.3`)
- **precision = 0.9936**

**False-positive mode (the step's specific ask).** The dominant one is
**batch contracts**, not incidental cross-references: **43 of the 156** mismatched
dirs *do* mention their own sid somewhere in the head, typically because one
contract covers a batch (`step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)`).
Mentioning is not declaring, so the narrow property still holds, but those 43 are
**contestable positives**. A second false-positive mode is legitimate
cross-reference prose (a contract that cites a prior step in its first 4,000 chars
before declaring itself) -- the ordered-first-hit rule makes this possible, and it
is exactly what the `SUSPECT` oracle is there to catch (it caught 1).

**Therefore the defensible statement is a RANGE, not a point:**
- **firm floor 113** mis-filed dirs (156 mismatches - 43 batch-mention contestables),
- **central estimate 155** (confirmed by the precision oracle),
- **ceiling 156 + up to 16 opaque + 24 no-contract = up to 196** if the unclassified
  tail is adversarial.

Step text claimed **129 of 747**. **Both halves are refuted**: the denominator is
**842** (measured `ls -1d handoff/archive/phase-* | wc -l`), and the numerator from
the shipped, control-gated classifier is **156**, not 129. The step text's own
caveat ("a heuristic that must be refined, not trusted") is correct -- and the
refinement it asks for is **already written and passing its own recall+control
gates**; the contract should *use* it, not re-derive it.

**Top declared-by content of the mis-filed dirs** (what actually leaked into them):
`phase-82.54` in 31 dirs, `62.6` in 14, `80.2` in 12, `10.5` in 8, `45.0` in 7,
`76.9.2` in 6, `62.2` in 5, `40.8` in 5. This is the signature of the rolling-file
branch: one step's `contract.md` sat in `handoff/current/` and was copied into every
archive dir minted while it was there.

## 1.7 Consumers that reference handoff paths BY PATH (the relocation hazard)

Command (masterplan):
`python3` walk of `.claude/masterplan.json`, regex `handoff/[A-Za-z0-9_./\-*{}<>]+`
over `verification.command` and `verification.live_check`, 1,283 steps scanned.

| Measure | Count |
|---|---|
| handoff path references in `verification.{command,live_check}` | **557** (372 in `command`, 185 in `live_check`) |
| distinct handoff paths referenced | **373** |
| references to `handoff/current/...` | **395** |
| references to `handoff/archive/...` | **8** |
| files repo-wide (excl. `handoff/`, `.git`, `node_modules`) mentioning `handoff/current` or `handoff/archive` | **373** |

Most-referenced: `handoff/harness_log.md` (20), `handoff/current/experiment_results.md`
(12), `handoff/kill_switch_audit.jsonl` (9), `handoff/current/money_diagnosis_72.md` (9),
`handoff/current/operator_decision_sheet_72.md` (8), `handoff/current/contract.md` (7),
`handoff/current/research_brief.md` (7).

**Live path-coupled readers (hook layer):**
- `.claude/hooks/auto-commit-and-push.sh:154` -- `handoff/current/live_check_${STEP_ID}.md`
- `.claude/hooks/auto-commit-and-push.sh:257` -- `handoff/current/evaluator_critique_${STEP_ID}.json`
- `.claude/hooks/lib/live_check_gate.py:12` -- same live_check path
- `.claude/hooks/lib/verdict_gate.py:114-119` -- an **ordered 5-location chain**
  (`current:per-step`, `current:rolling`, `archive:step`, `archive:step-rolling`,
  `archive:misc`) that *reports which source answered*
- `.claude/workflows/research-gate.js:423` -- `handoff/current/research_brief_${stepId}.md`
- `scripts/add_phase_27.py:70-72`, `scripts/add_phase_27_6_sub.py:106-112` --
  masterplan verification commands hardcoding handoff paths

**395 masterplan `verification` references into `handoff/current/` is the single
strongest argument against any bulk relocation.** Every one of those is an immutable
criterion the project forbids editing.

## 1.8 The damage this class has ALREADY done (documented in-tree)

`.claude/hooks/lib/verdict_gate.py:87-92`, verbatim:

> `STEP_ID_RE in scripts/housekeeping/*.py matches 0 of the 127 .md files in
> handoff/current/, so backfill_handoff_archive.py classifies every step artifact
> as unclassifiable and moves it to handoff/archive/misc/. Commit fa9aaf8e
> (2026-07-24) swept 315 files that way and took
> handoff/current/evaluator_critique.json with them; the gate then ran dark for 13
> consecutive step closes with no signal.`

Two things to note. (a) The comment's own denominator (**127 `.md` files**) is now
**668** -- the exposure has grown ~5.3x since it was written, and the router that
sweeps them is unchanged. (b) The durable fix chosen there was **not** to relocate
files but to **stop depending on one location** (`resolve_verdict_source`, an
ordered chain that reports its source). That is direct in-repo prior art for the
design question this step is asking.

Phase-81.0 patched the *symptom* narrowly: `ROLLING_KEEP_PREFIXES =
("evaluator_critique_",)` with `name.endswith(".json")`
(`backfill_handoff_archive.py:55,62`; mirrored `verify_handoff_layout.py:42,49`).
**That allowlist covers `.json` only.** Every `.md` per-step artifact --
`contract_<sid>.md`, `research_brief_<sid>.md`, `experiment_results_<sid>.md`,
`live_check_<sid>.md` -- is still unprotected. The 664-file figure in 1.3 is the
size of that hole.

## 1.9 `handoff/kill_switch_audit.jsonl` -- is the move still correct?

**No. It must never move, under any classifier.** Both scripts already encode this:
`backfill_handoff_archive.py:79-81` and `verify_handoff_layout.py:65-67` carry
byte-identical `HANDOFF_ROOT_KEEP = {"kill_switch_audit.jsonl"}`, kept in sync by an
AST-parsing test
(`backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py::test_phase_36_8_both_housekeeping_scripts_protect_the_audit_archives`).

Rationale recorded at `backfill_handoff_archive.py:66-78`: the file is **not a log**,
it is the kill switch's **only persistence**;
`backend/services/kill_switch.py::_load_from_audit` replays it at every process start
to restore `sod_nav`/`peak_nav`. Moving it left the switch **DISARMED** after
restart -- a 50% drawdown returned `any_breached=False` (measured 2026-07-26). It
shipped twice (`fa9aaf8e` -> `-v3`, `77bc7db5` -> `-v4`), i.e. **self-perpetuating**,
because the *verifier* demanded the move that the *backfill* then performed.

**Design lesson for this step, stated as a rule:** a name-shaped classifier
(`*_audit.jsonl` -> "it's audit output") mis-classified a live safety-critical state
file. That is the canonical demonstration of why **filename-pattern classification is
fragile** -- the very question the objective's part (e) asks about. The allowlist is
a patch on the symptom; the general principle is that **the artifact must declare
what it is, rather than the reader inferring it from the name.**

## 1.10 Prior art already in-tree for provenance

`archive-handoff.sh:197-211` already writes a **`PROVENANCE.md` manifest** into every
archive dir: a table of `| archived as | source | how |` rows, plus an explicit
`## RESULT: FAILED -- nothing was archived` block (`:294-307`) when the dir would
otherwise be empty. And `rolling_declares_step()` (`:146-169`) implements
**content-declared ownership** -- it refuses to copy a rolling file that does not
name the step, fail-closed ("unsure" means "do not copy"). The comment at `:132-145`
records that this grammar **already drifted** from the census's grammar within one
cycle (em-dash separator), and the drift was left stated as a RISK rather than denied.

So pyfinagent has, in-tree: (i) a manifest, (ii) content-declared ownership, (iii)
loud-failure-on-empty. What it does **not** have is a *content-addressed* identity or
a shared, single-definition grammar -- both conventions, duplicated by copy.

---

# PART 2 -- EXTERNAL RESEARCH

## 2.1 Search-query composition (three-variant discipline, `.claude/rules/research-gate.md`)

| Variant | Queries actually run |
|---|---|
| **Year-less canonical** | `BagIt file packaging format manifest checksum RFC 8493`; `RO-Crate research object metadata provenance packaging`; `idempotence is not a medical condition Pat Helland ACM Queue`; `filename extension unreliable file type identification content-based magic bytes`; `Software Heritage archive content-addressable identifiers SWHID intrinsic identifiers`; `PREMIS OAIS preservation metadata provenance digital archive fixity` |
| **Current-year frontier (2026)** | `agent harness artifact provenance file-based state durable handoff 2026`; `safe by default destructive CLI tool dry-run confirmation principle data loss incident postmortem` (returned 2026-dated hits) |
| **Last-2-year window (2025-2026)** | `data migration backfill idempotency safety large scale rename 2025 2026` |

## 2.2 Read in full (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://www.rfc-editor.org/rfc/rfc8493.html | 2026-08-17 | Official standard (IETF RFC) | WebFetch, full | **"such subdirectory structures and filenames have no given meaning"** (§2.1.2). Identity/completeness live in `manifest-<algorithm>.txt`, never in the layout. "Complete" requires *every* payload file be listed in *every* payload manifest; "valid" additionally requires every checksum verify (§3). |
| 2 | https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0309210 | 2026-08-17 | Peer-reviewed (PLOS One) | WebFetch, full | Workflow Run RO-Crate. Splits **prospective provenance** ("the execution plan") from **retrospective provenance** ("what actually happened during an execution"). Outputs bind to runs via `s:CreateAction` + `s:result` + `s:instrument`. Three nested granularities (Process / Workflow / Provenance Run Crate). Mapped to W3C PROV-O. Implemented in 6 workflow systems. |
| 3 | https://www.w3.org/TR/prov-primer/ | 2026-08-17 | Official standard (W3C) | WebFetch, full | Entity / Activity / Agent. `wasGeneratedBy`, `wasAttributedTo`, `wasDerivedFrom`, `wasRevisionOf`. **"The provenance of digital objects represents their origins."** A revised document **"is a new entity"** -- versions are distinct entities, not overwrites. |
| 4 | https://git-scm.com/book/en/v2/Git-Internals-Git-Objects | 2026-08-17 | Official doc | WebFetch, full | **Filenames live in TREE objects, not in blobs.** "A single tree object contains one or more entries, each of which is the SHA-1 hash of a blob or subtree with its associated mode, type, and filename." Content identity is independent of name. Commits carry author + parent = provenance. |
| 5 | https://developer.hashicorp.com/terraform/cli/commands/plan | 2026-08-17 | Official doc (vendor) | WebFetch, full | Preview/mutate separation. **"The `plan` command alone does not actually carry out the proposed changes."** `-detailed-exitcode`: 0 = no changes, 1 = error, **2 = non-empty diff**. Plans can be saved (`-out`) and applied later. "you should always re-check the final non-speculative plan before applying". |
| 6 | https://arxiv.org/abs/2108.06503 | 2026-08-17 | Peer-reviewed preprint (Data Science jnl) | curl + **pdfplumber 0.11.9** (ar5iv 307'd to /abs; 128,701 chars extracted) | RO-Crate. A zip "solves the problem of 'packaging', but it does not guarantee downstream access to all artefacts in a programmatic fashion, **nor describe the role of each file in that particular research**". Notes RO-Crate works "even where **no filename or file extension conventions have emerged**". Explicit **layering**: "BC RO-Crate as a stack: transport-level manifests of files (BagIt)" + semantic layer on top. |
| 7 | https://arxiv.org/abs/1002.3174 | 2026-08-17 | Peer-reviewed (IEEE ISCC'08) | curl + **pdfplumber 0.11.9** (ar5iv 307'd to /abs; 26,530 chars extracted) | **"File type detection methods can be categorized into three kinds: extension-based, magic bytes-based, and content-based methods, each of them has its own strengths and weaknesses, and none of them are comprehensive or foolproof enough to satisfy all the requirements."** On name-based classification: "it has a great vulnerability while it can be **easily spoofed by a simple file renaming**." |
| 8 | https://docs.softwareheritage.org/devel/swh-model/persistent-identifiers.html | 2026-08-17 | Official doc | WebFetch, full | **"an important property of any SWHID is that its core identifier is _intrinsic_: it can be _computed from the object itself_, without having to rely on any third party."** Identity is a Merkle DAG hash; `path` and `anchor` are *qualifiers* layered on top -- location is context, never identity. |
| 9 | https://martinfowler.com/bliki/ParallelChange.html | 2026-08-17 | Authoritative blog | WebFetch, full | Expand -> Migrate -> Contract. Old and new **coexist**; clients migrate incrementally. **"If the contract phase is not executed you might end up in a worse state than you started, therefore you need discipline to finish the transition successfully."** |
| 10 | https://danieljamesglover.com/blog/2026-02-01-dry-run-engineering-practice/ | 2026-08-17 | Practitioner blog (2026) | WebFetch, full | **"I have started making `--dry-run` the default for destructive scripts. You have to explicitly pass `--execute` or `--no-dry-run` to make changes."** Rationale: **"A misconfigured job that runs in dry-run mode by default produces logs instead of damage."** Dry-run must print *how many* rows/files are affected plus a sample. |
| 11 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-17 | Official vendor engineering | WebFetch, full | Canonical project reference. **"Communication was handled via files: one agent would write a file, another agent would read it..."** Hard-threshold gates: "Each criterion had a hard threshold, and if any one fell below it, the sprint failed". Self-eval: "agents tend to respond by confidently praising the work". Stress-test doctrine: "Every component in a harness encodes an assumption about what the model can't do on its own". |
| 12 | https://www.infoq.com/articles/shadow-table-strategy-data-migration/ | 2026-08-17 | Industry practitioner | WebFetch, full | Six-phase migration: create shadow -> backfill in chunks -> sync -> **verify** -> cutover -> cleanup. **"Only once these validations confirm that the shadow is consistent with the source can the final cutover be executed."** Cleanup keeps the old store "in read-only mode as a backup until you no longer need it". |
| 13 | https://arxiv.org/html/2605.18747v1 | 2026-08-17 | Preprint (2026) | WebFetch, full (native arXiv HTML) | "Code as Agent Harness". Code artifacts are "stateful, meaning the evolving program represents task progress in a **persistent, modifiable form across steps**". **NEGATIVE RESULT (recorded as such):** the survey "does not discuss naming conventions or metadata systems" and does not analyse file-handoff failure modes -- the 2026 agent-harness literature has **not** addressed artifact-ownership metadata. |
| 14 | https://digitalpreservation.gov/series/challenge/premis.html | 2026-08-17 | Official (Library of Congress) | WebFetch, full | Preservation metadata must record "technical metadata about the original files, the older hardware and software that they ran on, and **what actions had been performed on them**". **Thin source** -- the entity model is in the PREMIS Data Dictionary, not this page; recorded honestly rather than padded. |

## 2.3 Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://dl.acm.org/doi/10.1145/2181796.2187821 | Peer-reviewed (Helland, ACM Queue 2012) | Paywalled at dl.acm.org; the canonical idempotency point is already carried by sources 9/12/16 |
| https://www.rfc-editor.org/info/rfc8493/ | Standard metadata page | Duplicate of source 1 |
| https://www.researchobject.org/ro-crate/ | Project site | Superseded by sources 2 + 6 |
| https://www.researchobject.org/ro-crate/profiles | Project site | Profile detail beyond scope |
| https://www.researchobject.org/ro-crate/about_ro_crate | Project site | Duplicate |
| https://eosc.eu/roadmap/ro-crate-research-object-crate | Org page | Non-technical |
| https://casrai.org/guides/ro-crate-packaging-research-data-metadata-fair-reuse | Guide | Secondary |
| https://arkisto-platform.github.io/standards/ro-crate/ | Platform doc | Secondary |
| https://github.com/ResearchObject/2021-packaging-research-artefacts-with-ro-crate/blob/main/ro-crate-metadata.json | Code artifact | Example JSON; structure covered by source 6 |
| https://www.semanticscholar.org/paper/Idempotence-is-not-a-medical-condition-Helland/87fc322b992c1881854c1e2a0e36ef5b6f739071 | Index page | Abstract only |
| https://dl.acm.org/doi/10.1145/2160718.2160734 | Peer-reviewed (CACM reprint) | Paywalled |
| https://filesignature.org/guides/validate-file-type-magic-bytes | Community guide | Community tier; point covered by source 7 |
| https://arxiv.org/pdf/2007.11246 | Preprint (file-fragment classification) | Adjacent (ML classification of fragments), not archival naming |
| https://micsymposium.org/mics_2005/papers/paper7.pdf | Conference paper (2005) | Superseded by source 7 |
| https://www.archives.gov/files/applied-research/papers/unix-file-command.pdf | Gov report | `file(1)`/magic detail beyond scope |
| https://codecut.ai/python-magic-file-type-detection/ | Community | Community tier |
| https://oneuptime.com/blog/post/2026-03-02-how-to-use-the-file-command-to-identify-file-types-on-ubuntu/view | Community (2026) | Recency hit, community tier |
| https://github.com/contentauth/c2pa-rs/issues/2024 | Issue tracker | Community tier; corroborates "detect by magic bytes, not extension" |
| https://wiki.softwareheritage.org/wiki/Software_Heritage_identifiers | Wiki | Superseded by source 8 |
| https://www.softwareheritage.org/2025/06/13/software-hash-identifier-swhid-tutorial/ | Vendor blog (2025) | Recency hit; content covered by source 8 |
| https://arxiv.org/pdf/2310.10295 | Preprint (SWH open-science ecosystem) | Overlaps source 8 |
| https://arxiv.org/pdf/1909.10760 | Preprint (SWH guidelines) | Overlaps source 8 |
| https://www.oclc.org/content/dam/research/activities/pmwg/pm_framework.pdf | Research report (OCLC/RLG) | PDF; OAIS PDI five-component model captured via search snippet + source 14 |
| https://www.dpconline.org/docs/technology-watch-reports/894-dpctw13-03/file | Technology watch report | PDF, secondary to source 14 |
| https://www.dpconline.org/docs/technology-watch-reports/1359-dpctw14-02/file | Technology watch report (OAIS) | PDF, secondary |
| https://thebackenddevelopers.substack.com/p/zero-downtime-database-migrations | Practitioner (2025+) | Expand/contract point already in source 9 |
| https://dataskew.io/blog/data-pipeline-design-patterns/ | Practitioner (2026) | Recency hit; idempotency covered by source 12 |
| https://www.ml4devs.com/what-is/backfilling-data/ | Practitioner | Backfill idempotency, secondary |
| https://gist.github.com/amazingvince/52158d00fb8b3ba1b8476bc62bb562e3 | Gist (2026) | Community tier; "Modern Agent Harness Blueprint 2026" |
| https://www.pingcap.com/blog/ai-agent-harness-state-layer/ | Vendor blog (2026) | "state belongs outside the harness" -- corroborative, vendor-marketing tier |
| https://codewave.com/insights/the-agent-harness/ | Vendor blog | Low signal |
| https://zylos.ai/research/2026-05-15-long-horizon-agent-goal-persistence/ | Vendor research (2026) | Recency hit, non-peer-reviewed |
| https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-at-build-2026-announce/ | Vendor blog (2026) | Product announcement |
| https://arxiv.org/pdf/2606.20631 | Preprint (2026) | Agent-skills architecture; tangential |
| https://devsecopsschool.com/blog/fail-safe-defaults/ | Community (2026) | Fail-safe-defaults restatement (Saltzer-Schroeder) |
| https://hoop.dev/blog/aws-cli-accident-prevention-guardrails-how-to-avoid-costly-mistakes-in-production/ | Vendor blog | Guardrail patterns, marketing tier |
| https://lobste.rs/s/cwi2ly/praise_dry_run | Forum | Community tier, lowest weight |

| https://onlinelibrary.wiley.com/doi/full/10.1002/smr.70028 | Peer-reviewed (J. Software: Evolution & Process, 2025) | **Fetch ATTEMPTED and FAILED: HTTP 402 Payment Required.** Directly on-topic ("metadata to implement Convention over Configuration decoupled from framework logic"); recorded as an attempted-and-blocked source, not counted |
| https://en.wikipedia.org/wiki/Sidecar_file | Reference | Sidecar naming convention (same stem, different extension); covered by 2.4 |
| https://gohugo.io/content-management/front-matter/ | Official doc | Front-matter mechanics; the pattern is already used in-repo |
| https://ocfl.io/1.0/implementation-notes/ | Official doc | Implementation detail beyond scope |
| https://ocfl.io/draft/spec/ | Official doc (draft) | Draft of source 17 |
| https://arkisto-platform.github.io/standards/ocfl/ | Platform doc | Secondary to source 17 |
| https://www.dpconline.org/handbook/technical-solutions-and-tools/fixity-and-checksums | Handbook | Fixity; covered by sources 1 + 17 |
| https://orbiscascadeulc.github.io/digprezsteps/fixity-deep.html | Guide | Fixity; duplicate |
| https://blogs.loc.gov/thesignal/2014/04/protect-your-data-file-fixity-and-data-integrity/ | Official blog (LoC) | Fixity; duplicate |
| https://arxiv.org/pdf/1403.1180 | Preprint | Distributed integrity catalog; tangential |
| https://arslan.io/2019/07/03/how-to-write-idempotent-bash-scripts/ | Practitioner blog | Idempotent bash; covered by source 12 |
| https://www.zero-downtime-schema.com/database-migration-fundamentals-tool-selection/idempotent-script-design/ | Practitioner | Idempotency + version ledger; corroborative |
| https://docs.github.com/en/actions/tutorials/store-and-share-data | Official doc | CI artifacts are bound to a RUN -- corroborates source 2's binding principle, no new content |
| https://deepwiki.com/jenkinsci/workflow-basic-steps-plugin/5.3-artifact-management | Wiki | "artifacts are permanently associated with the build" -- duplicate of the same principle |
| https://learn.microsoft.com/en-us/archive/msdn-magazine/2009/february/patterns-in-practice-convention-over-configuration | Official doc (archive) | CoC patterns; criticism covered by source 16 |
| https://harnez.ai/posts/fix-broken-project-paths/ | Practitioner | Path-coupling migration w/ `--dry-run`; corroborative of source 10 |
| https://medium.com/@sreekanth.parikipandla/when-renaming-a-repo-breaks-everything-except-your-test-cases-a-devops-story-of-lessons-learned-c4d58a26cf59 | Practitioner | "Renaming breaks static references"; corroborative |

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 15 | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-08-17 | Official vendor engineering | WebFetch, full | **"Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator."** This **"prevents information loss during multi-stage processing and reduces token overhead."** Lead "decides whether more research is needed". Recovery: "agents can resume from where the agent was when the errors occurred". |
| 16 | https://en.wikipedia.org/wiki/Convention_over_configuration | 2026-08-17 | Reference (community tier) | WebFetch, full | **[COUNTERPOINT]** CoC conflicts with **"explicit is better than implicit"**; implicit conventions create "hidden complexity by obscuring key decisions". **Honest caveat recorded:** the disadvantages section carries **no citations** -- it is opinion, not evidence. Weighted accordingly. |
| 17 | https://ocfl.io/1.0/spec/ | 2026-08-17 | Official standard | WebFetch, full | **The single most on-point external source.** "the connection between a file's **content path** on physical storage and its **logical path** in a version of the object's content is made with **a digest of its contents, rather than its filename**." `inventory.json` carries `id`, `manifest` (digest -> content paths), `versions` (version -> `state`: digest -> logical paths). Goal: **"the rebuildability of a repository from an OCFL Storage Root without additional information resources"**. Forward-delta versioning; version dirs immutable. |
| 18 | https://patents.google.com/patent/US10831994B2/en | 2026-08-17 | Patent (US10831994B2) | WebFetch, full | **[COUNTERPOINT / the road not to take]** Fuzzy reconciliation of two naming conventions via Levenshtein + semantic analysis. Its own admitted failure mode is the decisive quote: **"If u1 is high, then the algorithm will make a high number of ... translations. The drawback is that the algorithm is prone to translating ... names incorrectly. If u1 is low ... fewer names will be translated."** i.e. similarity-matching trades false positives against false negatives and **cannot have both** -- exactly the precision/recall bind the 86.29 census already measured in-tree (precision 0.9936, 43 contestable positives). |

## 2.4 Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Searched:** `agent harness artifact provenance file-based state durable handoff 2026`;
`data migration backfill idempotency safety large scale rename 2025 2026`;
`safe by default destructive CLI tool dry-run ... postmortem` (2026-dated hits);
plus 2025/2026-dated hits surfaced by the year-less queries.

**Result: 4 findings in the window that COMPLEMENT the canonical sources; 0 that
supersede them.**

1. **(2026) Dry-run-by-default is now stated as an explicit practice, not folklore.**
   `danieljamesglover.com` (2026-02-01, source 10) states the inversion directly:
   "I have started making `--dry-run` the default for destructive scripts. You have
   to explicitly pass `--execute`." This is the newest articulation of a principle
   whose canonical form is Terraform's plan/apply split (source 5) and
   Saltzer-Schroeder fail-safe defaults. **It does not supersede Terraform; it
   generalises it to scripts.**
2. **(2026) The agent-harness literature has NOT solved artifact ownership.**
   `arXiv:2605.18747` "Code as Agent Harness" (source 13) surveys stateful agent
   systems and **does not discuss naming conventions or artifact-ownership
   metadata at all**. This is a **negative finding and it is load-bearing**: there
   is no 2026 agent-specific standard to adopt, so the design must borrow from
   digital preservation (BagIt/OCFL/PREMIS) and workflow provenance (WRROC/PROV).
   Corroborated by 2026 practitioner sources (`pingcap.com` "state belongs outside
   the harness"; the "Modern Agent Harness Blueprint 2026" gist), which discuss
   *externalising* state but not *attributing* it.
3. **(2025) SWHID tutorial** (`softwareheritage.org`, 2025-06-13, snippet-only)
   confirms intrinsic content-addressed identifiers remain the current
   recommendation; no change to the canonical model in source 8.
4. **(2025) A peer-reviewed paper exists on exactly the CoC-vs-explicit-metadata
   tradeoff** -- Gomes et al., *J. Software: Evolution and Process* (2025),
   `10.1002/smr.70028`. **I could not read it: HTTP 402 Payment Required.**
   Recorded as an attempted-and-blocked source. Its abstract-level claim (metadata
   should be explicit and decoupled from name-derived convention) aligns with, and
   would strengthen rather than change, the conclusion below.

**Nothing in the 2024-2026 window supersedes BagIt (RFC 8493, 2018), OCFL 1.0,
W3C PROV, or the Git/SWHID content-addressing model.** Those remain the canonical
prior art.


## 2.5 Audit-class loop-until-dry log (K_required = 2)

| Round | Queries / probes | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | BagIt; RO-Crate; Helland idempotence; filename-vs-magic-bytes | 7 (sources 1-7) | no |
| 2 | dry-run/safe-by-default postmortems; SWHID/content-addressing | 3 (sources 8-10) | no |
| 3 | agent-harness provenance 2026; backfill idempotency 2025-26; PREMIS/OAIS | 4 (sources 11-14) | no |
| 4 | convention-over-configuration criticism; sidecar vs front-matter; Anthropic multi-agent | 2 (sources 15-16) + 1 blocked (402) | no |
| 5 | OCFL; misattribution/chain-of-custody detection | 1 (source 17) | no |
| 6 | fuzzy name reconciliation; idempotent-move convergence; CI artifact scoping | 1 (source 18) | no |
| 7 | filename-classification failures in production; manifest-vs-layout; path-coupled migration refusal | **0** | **DRY** |
| 8 | agent artifact ownership declaration 2026; sidecar/PROVENANCE-per-directory practice | **0** | **DRY** |

`dry_rounds = 2 >= K_required = 2` -> **`coverage.dry = true`**.

Round 7 and 8 returned only (a) restatements of fixity/manifest concepts already
carried by sources 1 and 17, (b) community/marketing-tier material, and (c)
off-topic hits (C2PA AI-disclosure law, dependency confusion, SharePoint/EF Core
migration errors). One mild irony worth recording: the sidecar-file convention that
several round-8 hits recommend is itself **name-coupled** ("same base name, different
extension"), i.e. it reintroduces exactly the fragility this step is trying to remove.

---

# PART 3 -- SYNTHESIS

## 3.1 Key findings

1. **Every mature archival standard makes identity explicit and treats the filename
   as non-semantic.** BagIt is categorical: *"such subdirectory structures and
   filenames have no given meaning"* (RFC 8493 §2.1.2,
   https://www.rfc-editor.org/rfc/rfc8493.html). OCFL is the same point stated
   positively: *"the connection between a file's content path on physical storage
   and its logical path ... is made with a digest of its contents, rather than its
   filename"* (https://ocfl.io/1.0/spec/). Git separates them structurally --
   filenames live in tree objects, content in blobs
   (https://git-scm.com/book/en/v2/Git-Internals-Git-Objects). SWHID makes identity
   *intrinsic*: *"it can be computed from the object itself, without having to rely
   on any third party"*
   (https://docs.softwareheritage.org/devel/swh-model/persistent-identifiers.html).
   **The consensus is total and spans four independent communities.**

2. **Name-based classification is known-fragile in the literature, with a named
   mechanism.** *"File type detection methods can be categorized into three kinds:
   extension-based, magic bytes-based, and content-based methods ... none of them
   are comprehensive or foolproof enough"*; extension-based *"can be easily spoofed
   by a simple file renaming"* (Amirani, Toorani & Beheshti, IEEE ISCC'08,
   https://arxiv.org/abs/1002.3174). pyfinagent has the in-tree proof: a
   `*_audit.jsonl` suffix rule classified the kill switch's only persistence file as
   log output and **disarmed the switch** (`backfill_handoff_archive.py:66-78`).

3. **Fuzzy reconciliation of two naming conventions is a trap with a proven
   precision/recall bind.** The naming-convention-reconciler patent concedes it
   directly: a high threshold *"is prone to translating ... names incorrectly"*, a
   low one means *"fewer names will be translated"*
   (https://patents.google.com/patent/US10831994B2/en). pyfinagent has already
   measured its own instance of this: the 86.29 census runs at precision 0.9936 with
   **43 of 156 hits contestable** (batch contracts). **Do not add a similarity
   matcher; add an explicit declaration.**

4. **Provenance must be recorded, not inferred, and it has a standard shape.**
   W3C PROV: Entity / Activity / Agent with `wasGeneratedBy` + `wasAttributedTo`
   (https://www.w3.org/TR/prov-primer/). Workflow Run RO-Crate binds each output to
   the run that produced it via `s:CreateAction` -> `s:result`, and separates
   **prospective** provenance (the plan) from **retrospective** (what actually
   happened) -- implemented across six workflow engines
   (https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0309210).
   CI systems reach the same answer independently: an artifact is bound to a *run*,
   not to a name.

5. **Safe-by-default means the preview is the default and the mutation is the
   flag.** Terraform: *"The `plan` command alone does not actually carry out the
   proposed changes"*, with `-detailed-exitcode` distinguishing "no changes" (0)
   from "changes pending" (2)
   (https://developer.hashicorp.com/terraform/cli/commands/plan). The 2026 practice
   statement is the inversion pyfinagent needs: *"make `--dry-run` the default ...
   You have to explicitly pass `--execute`"*, because *"a misconfigured job that
   runs in dry-run mode by default produces logs instead of damage"*
   (https://danieljamesglover.com/blog/2026-02-01-dry-run-engineering-practice/).

6. **Migrating between two conventions is a solved pattern: coexist, then
   contract.** ParallelChange: expand -> migrate -> contract, with the warning
   *"If the contract phase is not executed you might end up in a worse state than
   you started"* (https://martinfowler.com/bliki/ParallelChange.html). The shadow-
   table variant adds the gate this step needs: *"Only once these validations
   confirm that the shadow is consistent with the source can the final cutover be
   executed"* (https://www.infoq.com/articles/shadow-table-strategy-data-migration/).
   **Both say: recognise BOTH conventions, verify, and only then retire one --
   never rename in place.**

7. **The agent-harness literature does not yet answer this question, so borrow.**
   arXiv:2605.18747 (2026) surveys stateful agent harnesses and does not discuss
   artifact naming or ownership metadata at all
   (https://arxiv.org/html/2605.18747v1). Anthropic's own guidance is about the
   *channel* (files as durable handoff) and the *gate* (hard thresholds), not about
   attribution: *"Communication was handled via files"*
   (https://www.anthropic.com/engineering/harness-design-long-running-apps);
   *"Subagents ... store their work in external systems, then pass lightweight
   references back to the coordinator"*
   (https://www.anthropic.com/engineering/multi-agent-research-system).

## 3.2 Consensus vs debate

**Consensus (unanimous across preservation, VCS, workflow-provenance and CI):**
identity and ownership belong in explicit metadata; the filename is a label, not a
fact; verification must be possible from the artifact alone.

**Genuine debate -- where the sources disagree:**
- *Embedded vs sidecar vs central manifest.* BagIt and OCFL put the truth in a
  **separate manifest** (`manifest-sha512.txt`, `inventory.json`). RO-Crate puts it
  in a **single JSON-LD file per crate**. Front-matter advocates put it **inside
  each file**. No winner; the tradeoff is "one file to corrupt" vs "N files to keep
  in sync". pyfinagent's existing `PROVENANCE.md` is the per-directory-manifest
  point on this spectrum.
- *Convention vs explicit configuration.* The counterpoint source argues CoC
  conflicts with *"explicit is better than implicit"* and creates hidden complexity
  (https://en.wikipedia.org/wiki/Convention_over_configuration) -- **but that page
  cites nothing**, and the strongest peer-reviewed treatment (Gomes et al. 2025,
  `10.1002/smr.70028`) was **paywalled (HTTP 402)**. So the "conventions are bad"
  side of this debate is, in this brief, **under-evidenced and I am not resting a
  recommendation on it.** The recommendation below rests on findings 1-3, which are
  fully sourced.
- *How hard to fail.* BagIt/OCFL fail-closed on any mismatch. Practitioner migration
  advice tolerates chunked, resumable, partially-complete states. pyfinagent's hook
  already chose fail-closed (`rolling_declares_step` returns false on "unsure").

## 3.3 Pitfalls (from the literature, mapped to what would bite here)

1. **Never finishing the contract phase.** ParallelChange's explicit warning. A
   "recognise both conventions" fix that is never followed by retirement leaves two
   live conventions forever -- which is the *current* state
   (`archive-handoff.sh:226` vs `:276`).
2. **Idempotent != convergent.** `_safe_target` (`backfill_handoff_archive.py:114`)
   mints `-v2/-v3`, so re-running is non-destructive but **does not converge**; it
   accumulates. `kill_switch_audit.jsonl` reached `-v4` this way. A migration must
   be a fixed point, not merely non-clobbering.
3. **A dry-run that diverges from the real run.** Source 10 does not address this;
   the mitigation is that both paths share one code path and differ only at the
   final `shutil.move` -- which `_move()` already does correctly.
4. **Similarity matching's precision/recall bind** (finding 3). Any classifier
   tuned to catch the 156 will over-flag batch contracts.
5. **Relocating a file that other tooling addresses by literal path.** Measured
   here: **395 masterplan `verification` references into `handoff/current/`**, all
   inside immutable criteria. The in-tree precedent
   (`verdict_gate.py:94-99`) already rejected relocation in favour of an ordered
   resolution chain: *"The durable fix is NOT to argue about where the file belongs
   ... It is to stop depending on one location."*
6. **Fixing the reader while leaving the writer** (or vice versa). The `STEP_ID_RE`
   is duplicated **byte-identically** in two files with a comment demanding they stay
   in sync -- a copy, not a shared definition. The same copy-not-share pattern
   already produced a measured drift between `rolling_declares_step()` and the
   census grammar (em-dash), documented at `archive-handoff.sh:132-145`.

## 3.4 Application to pyfinagent (external findings -> file:line anchors)

| Objective part | Finding | Concrete anchor |
|---|---|---|
| (a) reconcile two conventions **without renaming** | ParallelChange expand-phase: recognise BOTH, retire later. OCFL/BagIt: never let the name be the truth | Make `STEP_ID_RE` a **single shared definition** consumed by `backfill_handoff_archive.py:64` **and** `verify_handoff_layout.py:51`, extended to accept `<base>_<sid>.md` **in addition to** the prefix form. Retire the dead PREFIX glob at `archive-handoff.sh:276` only after the expand phase is proven. **No file is renamed.** |
| (b) safe-by-default migration | Terraform plan/apply; dry-run-by-default | Invert `backfill_handoff_archive.py:218-221` to dry-run default + `--execute`. Adopt Terraform's `-detailed-exitcode` idea: exit 0 = nothing to do, 2 = changes pending, so a cron/hook can gate on it. Add an explicit **refusal** to move any path appearing in the 395 masterplan `verification` references. |
| (c) archives filled from a rolling file owned by another step | WRROC binds output -> run via `CreateAction`; PROV `wasGeneratedBy`; CI binds artifact -> build | Already partly built: `rolling_declares_step()` (`archive-handoff.sh:146-169`) is content-declared ownership, fail-closed. The gap is that **the declaration grammar is copy-pasted** into `derive_archive_misattribution_86_29.py:72-80`, and has already drifted once. Share it. |
| (d) detect + remediate at scale | Manifest/fixity; precision-recall bind | **Do not write a new classifier.** `scripts/qa/derive_archive_misattribution_86_29.py` already ships with a recall gate, 4 synthetic controls, and a control-tested precision oracle. Live result: **156 / 842 mismatch, precision 0.9936, 43 contestable, 222 unclassified (206 by design), 24 no-contract.** Remediation should be **additive** (write a corrective `PROVENANCE.md` / manifest into the mis-filed dirs) rather than a re-shuffle -- OCFL's version dirs are immutable for exactly this reason. |
| (e) prior art on provenance/manifests | BagIt, OCFL, RO-Crate, PROV, PREMIS/OAIS | `PROVENANCE.md` (`archive-handoff.sh:197-211`) is already a per-directory manifest. Upgrading it toward an OCFL-style **machine-readable** `inventory.json`-shaped sidecar (id + source paths + digests + how) would make archives verifiable without re-reading prose, and would let a checker answer "does this dir contain what it claims" from the dir alone -- OCFL's *"rebuildability ... without additional information resources"*. |

**On `handoff/kill_switch_audit.jsonl`:** the move to `handoff/audit/` is **NOT
correct under any classifier** and must stay excluded. Both scripts already encode
this (`backfill_handoff_archive.py:79-81`, `verify_handoff_layout.py:65-67`), guarded
by an AST-parsing drift test. Any new classifier proposed by this step must
**inherit** that allowlist, and the step should verify the AST test still binds. This
file is the canonical demonstration of finding 2: a name-shaped rule mis-classified
live safety state and disarmed the kill switch.

## 3.5 Research Gate Checklist

**Hard blockers:**
- [x] >=5 authoritative external sources READ IN FULL (**18**; 16 via WebFetch, 2 via
      the documented curl+pdfplumber fallback after ar5iv 307'd -- mechanism disclosed
      per-row in 2.2)
- [x] 10+ unique URLs total incl. snippet-only (**72** unique URLs present in this
      file; 18 read in full + 54 snippet-only. Verified by regex over the brief on disk.)
- [x] Recency scan (last 2 years) performed + reported (2.4; 4 findings, 0 superseding)
- [x] Full papers / pages read, not abstracts (arXiv `/pdf/` never WebFetched; chain
      `/html/` -> ar5iv -> pdfplumber followed and the failures recorded)
- [x] file:line anchors for every internal claim (Part 1)
- [x] Audit-class: `coverage.dry == true` (2 consecutive dry rounds, 2.5)

**Soft checks:**
- [x] Internal exploration covered every module named in the internal scope, plus 6
      consumers it did not name
- [x] Contradictions / consensus noted (3.2), including where the counterpoint is
      **under-evidenced** and explicitly not relied on
- [x] Claims cited per-claim with URLs inline (3.1) and file:line inline (Part 1)
- [ ] **GAP, disclosed:** the strongest peer-reviewed source on the
      convention-vs-explicit-metadata tradeoff (Gomes et al. 2025,
      `10.1002/smr.70028`) was **paywalled, HTTP 402**. Not counted, not paraphrased.
- [ ] **GAP, disclosed:** three of the step text's own figures (SUFFIX=20;
      13,198; 428; 129 of 747) **do not reproduce**. Part 1 gives the live values and
      the commands. Main must not carry the step-text numbers into `contract.md`.
