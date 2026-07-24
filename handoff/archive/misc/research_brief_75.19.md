# Research Brief — Step 75.19: Recalibrate the masterplan preflight gate, then triage its true residue

**Tier:** moderate | **audit_class:** false | **Researcher:** Layer-3 | **Date:** 2026-07-24
**Status:** COMPLETE — gate_passed=true (7 sources read in full, recency scan done, all mandatory internal reads done, brief written)

## Step summary (from prompt; full text to be transcribed from masterplan.json id 75.19)

`scripts/meta/preflight_verify_masterplan.py` reports "819 steps, 141 broken, 8 unparseable" but is:
- **status-blind** — pending steps whose artifacts legitimately do not exist yet are flagged identically to done steps whose verification is genuinely unrunnable.
- **transient-blind** — handoff/ outputs, gitignored paths, URL fragments, frontend-relative paths all flagged.

Work items:
- (a) status-aware + transient-aware gate
- (b) fix id-extraction so zero steps resolve to "?"
- (c) make the 212-lines-vs-141-steps-vs-8-unparseable summary internally consistent
- (d) triage surviving residue and annotate genuinely-unrunnable done steps via `superseded_record` per the 75.2.1 shape

**MANDATE:** REUSE the importable classifier from `scripts/qa/sweep_absent_verification_paths.py` (built in 75.17).
The `go_live_drills` cluster is owned by 75.17 — check whether 75.17 already annotated it and exclude those rows.

---

## Section 1 — Internal audit (file:line cites)

### 1.0 Numbers have DRIFTED from the step text
The step (drafted 2026-07-20) quotes `"819 steps, 141 broken, 8 unparseable"` /
"212 BROKEN lines" / "28 unresolved ids". The **live** run today reports a
different baseline (masterplan has grown). The step says "the step must re-derive
it, not trust it" — so all numbers below are re-measured, not quoted.

### 1.1 Target: scripts/meta/preflight_verify_masterplan.py (247 lines) — the mis-calibrated checker
Three structural defects, each pinned to code:

- **D1 — Recursive walk over-counts + drags in non-canonical entries.**
  `_walk_steps` (`:165-177`) recurses into **every** dict value
  (`for v in node.values(): yield from _walk_steps(v)`), so it descends into
  `phases[].subphases[]` and `phases[].archived_legacy_steps[]` — arrays that are
  NOT canonical steps. Measured: it yields **863** "steps-with-command" and emits
  BROKEN lines for **13 ids that do not resolve** to any `phases[].steps[]` id:
  `38.10-38.13` (live at `phases[64].subphases[0]`) and `46.0-46.8` (live at
  `phases[71].subphases[3]`). THIS is the "?" / "28 unresolved ids" of work-item (b).
  The canonical denominator is `phases[].steps[]` = **883** steps (see 1.2).
- **D2 — List-shaped verification silently dropped.** `_extract_command` (`:152-162`)
  handles only `str` and `dict.command`; a **list**-shaped `verification` returns
  `None` and the step is skipped. Masterplan has **13 list-shaped** verifications
  (shape census below). The 75.17 classifier's `verif_commands` handles all four
  shapes — this is a concrete reuse win.
- **D3 — No status filter, no annotation filter, no transient filter.** `verify`
  (`:180-230`) flags any step with a missing path regardless of `status` or a
  `superseded_record` sibling, and treats `handoff/` transient outputs as source
  files. `_is_path_token` (`:72-94`) does exclude `-flags`, URLs (`://`), and
  bare `/routes` via `NON_PATH_PATTERNS`/`PROJECT_ROOTS`, but `handoff/` IS in
  `PROJECT_ROOTS` (`:49-52`) so every `handoff/current/live_check_*.md` reference is
  flagged.

- **Summary computation (`:224-228`):** prints `scanned {n_steps}, {n_broken} broken,
  {n_warn} unparseable` where `n_broken` counts **STEPS** (`:211 n_broken += 1` once
  per step) but stderr emits one `[BROKEN]` **line per path/import** (`:213-222`).
  So "broken" (steps) ≠ emitted lines, with no reconciliation. This is work-item (c).

### 1.2 Classifier API: scripts/qa/sweep_absent_verification_paths.py (425 lines) — the REUSE target (75.17)
Importable, argv-free, `sys.exit`-free core. Exported functions:

| Function | Signature | Role |
|---|---|---|
| `flat_steps(masterplan: dict) -> list[dict]` | `:305` | Canonical iteration: `[s for ph in phases for s in ph.get('steps') or []]`. **The correct denominator = 883.** Never descends into subphases/archived_legacy. |
| `verif_commands(verification: object) -> list[str]` | `:65` | Normalizes all FOUR shapes (dict/str/list/None) → list of command strings; never raises. Fixes D2. |
| `shape_census(masterplan) -> dict[str,int]` | `:309` | `{dict,str,list,none}` counts. Live: **{dict:720, str:126, list:13, none:24}** (sum=883). |
| `_extract_candidates(cmd) -> set[str]` | `:157` | Union of 4 extractors (structure-aware, broad-regex, quoted/ws, maximal-recall). |
| `fp_reason(token, cmd, repo_root, repo_basenames=None) -> str\|None` | `:184` | **The adjudicator.** Returns a FP-class string (url/url-route/abs-host-path/runtime-transient/glob-*/absence-asserted/exists-on-disk/basename-exists-elsewhere/bare-word-no-ext/printf-template/...) or `None` if GENUINE. Handles frontend-relative (`_resolves_on_disk` `:172` checks repo, `frontend/`, `frontend/src/`, `backend/`), `handoff/`+`tmp/` transient (`_RUNTIME_TRANSIENT_PREFIXES` `:58`), abs-host-paths (`_ABS_HOST_PREFIXES` `:53`), negative assertions, glob-prefix re-resolution. Spot-checked live: `handoff/current/live_check_56.1.md` → `runtime/transient`; `handoff/away_ops/fixes/` → `runtime/transient`; `lib/icons.ts` → excluded (resolves under frontend/src). |
| `classify(masterplan, repo_root, *, git_classify_fn=…, repo_basenames=None) -> dict` | `:324` | **The whole recalibrated path-gate in one call.** Scans ONLY `status=="done"` (`:357`), EXCLUDES `superseded_record` (`:359-360`). Returns `{genuine:{sid:[{path,class,retired_by_commit}]}, shape_census, steps_scanned}`. |
| `git_classify(path, repo_root) -> (class, commit)` | `:284` | `never-existed` / `retired` / `in-history-absent`. |
| `load_masterplan(path) -> dict` | `:383` | No try/except (a truncated masterplan MUST raise, M1). |

**Coverage gap the classifier does NOT close:** it is **path-only**. The preflight's
import-check leg (`_check_imports`, `_extract_imports`) has no classifier equivalent.
So the recalibrated tool must KEEP its own import leg — but make it status-aware.

### 1.3 Baseline preflight output (verbatim) — the mis-calibrated baseline
```
preflight_verify_masterplan: scanned 863 steps, 151 broken, 8 unparseable
```
Sample BROKEN lines (transient handoff + pending-test + already-annotated go_live_drills):
```
[BROKEN] step=56.1: missing path 'handoff/current/live_check_56.1.md'
[BROKEN] step=63.4: missing path 'handoff/away_ops/fixes/'
[WARN]  step=70.2: unparseable command (No closing quotation)
[BROKEN] step=75.4.1: missing path 'backend/tests/test_phase_75_4_1_critic_patch.py'   # 75.4.1 is PENDING
[BROKEN] step=75.19: missing path 'backend/tests/test_phase_75_19_preflight_calibration.py'  # THIS step, pending
```
Measured breakdown (recursive walk): **151 broken steps / 222 broken lines
(217 path + 5 import) / 8 unparseable.**
- Broken-step status: **done 82, pending 40, deferred 10, dropped 5, superseded 1,
  UNRESOLVED 13** (the 13 = subphase/legacy phantoms).
- The **5 import-broken** lines are ALL **pending** phase-5 steps
  (`5.2/5.8/5.9/5.11/5.12`, modules `backend.markets.*` not yet built) → zero after status filter.
- **12 broken done-steps carry `superseded_record`** already (the go_live_drills
  cluster + `4.14.24/4.14.26`) → excluded by `classify()`.

### 1.3b Fully-recalibrated residue (simulation) — THE headline finding
Canonical `flat_steps` + `status=="done"` + exclude `superseded_record` + path via
classifier `fp_reason` + status-gated import leg:
```
done steps scanned: 731
GENUINE residue rows: 0   (0 path, 0 import)
```
**The genuine residue is EMPTY.** Work-item (d) "annotate genuinely-unrunnable done
steps" resolves to: **nothing new to annotate** — every genuine defect is already
dispositioned (3 by 75.2.1, 11 by 75.17). This is reproducible via
`python scripts/qa/sweep_absent_verification_paths.py --json` (its `test_sweep_over_
live_masterplan_is_clean` asserts exactly this). Criterion 5 is satisfied honestly
(and vacuously) — which is precisely why the test needs a POSITIVE fixture (see §3).

### 1.3c Unparseable reconciliation (work-item c)
8 canonical **done** steps are shlex-untokenizable (nested quotes in `python -c`):
`23.2.7, 23.3.2, 70.2, 70.3, 70.5, 71.2, 71.4, 71.5`. These are a **checker
limitation** (valid shell the static tokenizer can't split), NOT defects. The
recalibrated summary must report them as their own bucket ("unverifiable-by-static-
checker"), never fold them into "broken".

### 1.4 superseded_record shape (75.2.1 exemplar) — masterplan.json `:3908-3917` (step 4.14.4)
```json
"superseded_record": {
  "superseded_at": "2026-07-20",
  "authorized_by": "operator directive 2026-07-20 (...); recorded in harness_log Cycle 130 (step 75.2.1)",
  "reason": "phase-75.2 deleted backend/slack_bot/assistant_handler.py ... re-running raises ImportError. The step's WORK was genuinely done; only the artifact it inspected has been retired.",
  "retired_by_commit": "f55e6973",
  "retired_in_step": "75.2",
  "still_runnable": false,
  "criteria_amended": false,
  "note": "verification.command and verification.success_criteria are deliberately left BYTE-IDENTICAL (CLAUDE.md: immutable). This record annotates, it does not amend. Byte-identity vs commit 256867d3 asserted by test_phase_75_2_1_push_approval.py."
}
```
75.17 variant (`class-I`) adds `already_broken_before_retirement: bool` +
`on_disk_equivalent: "scripts/go_live_drills/smoke_test_4_17_N.py"` (test lines
`:114-123`). **If residue=0, no NEW record is written** — but the shape is documented
here in case triage surfaces one the sweep missed (import-class, which classify()
can't see).

### 1.5 .gitignore transient classes (what "transient" means)
Gitignored: `handoff/logs/`, `handoff/*.log`, `handoff/archive/_quarantine_*/`,
`.venv/`, `node_modules/`, `.next/`, `*.pyc/__pycache__`, `*.db`, `.env*`,
`.playwright-mcp*/`, `cockpit-*.png`, `.coverage*`, `*.tsbuildinfo`,
`handoff/.autonomous_loop.lock`. Note `handoff/current/` and `handoff/archive/`
(non-quarantine) are **tracked**, but `live_check_*.md` / away_ops outputs are
per-step transient outputs archived on step close — the classifier's
`_RUNTIME_TRANSIENT_PREFIXES=("tmp/","handoff/","frontend/handoff/")` treats the
whole `handoff/` prefix as transient, which is the correct call for a
verification-command reference.

### 1.6 75.17 go_live_drills annotation status — ALREADY DONE
75.17 status = **done**; 75.2.1 status = **done**. The masterplan carries **14**
`superseded_record` siblings total: `4.14.4, 4.14.24, 4.17.9` (from **75.2.1**) +
`4.17.2/3/4/5/6/7/8/11/12, 4.14.26` (the **go_live_drills cluster + skill-pair**,
from **75.17**) + `4.14.26, 68.5`. The 75.17 test (`:59-66`) pins
`ALL_HOLDERS` = 14 and asserts **exactly one** superseded_record per step repo-wide
(`test_exactly_one_superseded_record_repo_wide` `:158-183`). **75.19 must exclude all
14** (classify() does this automatically via the `superseded_record` skip) and MUST
NOT re-annotate the go_live_drills cluster (criterion 5 + the 75.17 ownership rule).

### 1.7 Fixture patterns to mirror (from test_phase_75_17_verification_paths.py)
- `BASELINE_COMMIT` pinned SHA (`:57`) for byte-identity proof via `git show`.
- `_masterplan_at(ref)` (`:81`) / `_steps_by_id(mp)` (`:90`) helpers.
- **Shape-BY-SHAPE fixture** (`_build_all_shapes_fixture` `:235-254`): one done step
  per verification shape, all naming the SAME absent path, so the only variable is
  the shape. `test_all_four_..._without_crash` asserts `isinstance` FIRST
  (`:265-268`) — the 75.2.1 lesson: "a fixture that cannot represent a shape can't
  prove the crash-guard".
- `test_sweep_over_live_masterplan_is_clean` (`:203`) — the residue==0 assertion,
  paired with `test_sweep_over_baseline_..._finds_exactly_the_ten` (`:209`) POSITIVE
  case against a pinned pre-annotation snapshot. **This pairing is the anti-vacuous-
  guard template 75.19 must reuse: clean-on-live + genuine-on-a-defect-fixture.**
- Resolver non-flag proofs (`:289-332`): frontend-relative, url-fragment, truncated
  plist, negative-assertion, glob-prefix — reuse `fp_reason` unit-proofs verbatim.

---

## Section 2 — External research

### 2.1 Queries run (3-variant discipline)
| Topic | Current-year (2026) | Last-2-year (2024-25) | Year-less canonical |
|---|---|---|---|
| Alert-fatigue / actionable findings | "static analysis actionable findings false positive rate developers ignore Google 2026" | (Nature 2025, FSE 2025 surfaced) | "alert fatigue cry wolf static analysis warnings developers ignore effective false positives" / "effective false positives static analysis Google Tricorder developer trust actionable" |
| Status-aware / diff-based gating | — | "differential diff-based static analysis warnings only new changed code baseline suppression" | (Tricorder "newly-introduced warnings" + baseline suppression) |
| Mutation testing of fixtures | — | (Betka 2022, industrial extreme MT) | "pseudo-tested methods mutation testing extreme transformation Vera-Perez" |
| Agent-harness verification gates | "LLM agent harness verification gate self-evaluation reliability 2026" | — | (canonical: Anthropic harness-design, cited internally) |

### 2.2 Read in full (7; >=5 required — gate cleared)
| # | Source | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | Google "Static Analysis" — SWE-at-Google ch.20 (abseil) | 2026-07-24 | Official doc | WebFetch | Tricorder enables a check only if it is understandable, actionable/auto-fixable, **"<10% effective false positives"**, high-impact; **project-level not user-level** suppression; "effective FP" = developer took no positive action. |
| 2 | Sadowski et al., "Lessons from Building Static Analysis Tools at Google" (CACM 2018) | 2026-07-24 | Peer-reviewed | WebFetch (dornea.nu mirror; CACM 403'd) | **"If a tool wastes developer time with false positives and low-priority issues, developers will lose faith and ignore results."** FindBugs was **discontinued** because effective-FPs eroded confidence. "Please Fix" >5,000/day vs "Not useful" 250/day = the trust ledger. |
| 3 | Vera-Pérez, Danglot, Monperrus, Baudry, "A Comprehensive Study of Pseudo-Tested Methods" (EMSE 2019, arXiv:1807.05030) | 2026-07-24 | Peer-reviewed | WebFetch (ar5iv) | A method is **pseudo-tested** if its whole body can be replaced by a constant and **no test fails**. Found in **all 21** projects, **1%-46%** of methods even at high coverage. **Coverage ≠ verification**: a passing test that cannot fail proves nothing. Fix = assertions that make the test fail on behavior change. |
| 4 | Hu, Wang, Rubin, Pradel, "An Empirical Study of Suppressed Static Analysis Warnings" (FSE 2025) | 2026-07-24 | Peer-reviewed | pdfplumber (binary PDF) | 7,357 suppressions / 46 Python projects (1 in 6 files); **50.8% of suppressions are "useless"** (suppress zero warnings); primary reason = false positives; **46% of developers self-report suppressing warnings**. Suppression is the mechanism to encode "known / not-applicable-here" so genuine warnings stay visible. |
| 5 | "Reason Less, Verify More: Deterministic Gates Recover a Silent Policy-Violation Failure Mode in Tool-Using LLM Agents" (arXiv:2607.07405, 2026) | 2026-07-24 | Preprint | WebFetch (HTML) | Deterministic **pure-function, read-only, no-LLM** gates beat model self-verification; +12.4pp (budget) / +10.4pp (frontier). BUT gates need **per-policy/per-model auditing**: one gate had **5% precision / 40 false blocks** and caused "genuine gate-related harm" — an over-firing gate is discarded. |
| 6 | "From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws" (arXiv:2606.06324, 2026) | 2026-07-24 | Preprint | WebFetch (HTML) | ETCLOVG taxonomy; **"Verification and Evaluation: weak readiness checks and validators"** is a first-class harness-flaw layer. A guard that treats `status=success` as evidence when no state effect occurred causes premature finalization. HarnessFix +6.3-18.4%. |
| 7 | "Self-Evolving Agents with Anytime-Valid Certificates" (arXiv:2607.00871, 2026) | 2026-07-24 | Preprint | WebFetch (HTML) | Each self-modification admitted only via an **auditable certificate + persistent ledger**; an oracle is **"kept only if it fails on the unpatched base"** (the anti-vacuous-guard rule in formal dress); gate **favors safety over progress** (abstain rather than false-admit). |

### 2.3 Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not read in full |
|---|---|---|
| https://www.nature.com/articles/s41597-025-06154-7 | Peer-reviewed (2025) | Auth-wall redirect (idp.nature.com) — recency evidence for (non-)actionable-report dataset |
| https://arxiv.org/pdf/2511.12229 (Actionable Warning Is Not Enough, weak supervision) | Preprint (2025) | Snippet; corroborates actionable-vs-valid distinction |
| https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43322.pdf (Tricorder ICSE 2015) | Peer-reviewed | Primary Tricorder paper; ch.20 read instead |
| https://software-lab.org/publications/fse2025_suppressions.pdf | — | (Actually read in full via pdfplumber — row 4) |
| https://habr.com/en/companies/pvs-studio/articles/513952/ (baseline vs diff) | Practitioner | Confirms diff/baseline "only new warnings" model |
| https://help.perforce.com/helix-qac/.../prqa-framework-manualsu15.html (Baseline Diagnostics Suppression) | Vendor doc | "suppress old, show only change-driven diagnostics" |
| https://sdtimes.com/devops/demystifying-differential-and-incremental-analysis... | Industry | Differential analysis = report only new/changed since baseline |
| https://arxiv.org/pdf/2103.08480 (Extreme mutation testing in practice) | Preprint | Industrial extreme-MT; complements Vera-Pérez |
| https://onlinelibrary.wiley.com/doi/full/10.1002/smr.2450 (Betka 2022) | Peer-reviewed | Traditional vs extreme MT tradeoffs |
| https://arxiv.org/html/2607.07405v1 ... + ai-eval.org harness survey, arXiv:2603.02798, 2504.00406, 2509.11787, 2604.27000, pixee, wiz, paloalto, scworld, algomox | mixed | Alert-fatigue + gate corroboration |

**URLs collected total: ~45 unique** (well above the moderate-tier 15+ target).

### 2.4 Recency scan (2024-2026) — PERFORMED
New findings in the window that complement/supersede the canonical Google (2015/2018) work:
- **FSE 2025 (Hu et al.)**: first empirical study of *suppressions* — 50.8% useless, and suppression is how developers encode "known/not-applicable" so genuine warnings survive. Directly supersedes an ad-hoc allowlist: the recalibrated gate should treat `superseded_record` as pyfinagent's *structured suppression* channel.
- **Nature Sci Data 2025**: (non-)actionable report dataset — actionability triage is now a benchmarked ML task, confirming the problem is live in 2025.
- **arXiv:2607.07405 / 2606.06324 / 2607.00871 (all 2026)**: the deterministic-gate / harness-flaw / anytime-certificate line is brand-new and directly on-point for a harness verification gate — none existed at the canonical Google-era. They converge on: gates must be pure/auditable, per-context calibrated, and abstain rather than false-fire.
- No finding *contradicts* the Google "effective false positive / <10% or it gets ignored" canon; the 2025-26 work extends it to suppressions and to agent harnesses.

---

## Section 3 — PLAN recommendations for the contract

**Framing (external → internal):** The preflight is a harness verification gate. Google's
canon (sources 1-2) is unambiguous — a checker that fires >~10% effective-false-positives
is **ignored or disabled**, and the whole tool loses trust (FindBugs was retired for exactly
this). The current preflight reports 151 "broken" of which the genuine residue is **0** — a
~100% effective-FP rate — which is why "nobody acts on it" (step text). The 2026 harness
literature (sources 5-7) says the same in agent terms: a validator that emits an untrustworthy
signal is a first-class *harness flaw* (ETCLOVG "Verification and Evaluation"), and gates must
be pure/auditable and **abstain rather than false-fire**. So the deliverable is a
*trustworthy* checker, not a louder one.

### R1 — REUSE the 75.17 classifier as the path core (the MANDATE)
Import from `scripts.qa.sweep_absent_verification_paths`: **`flat_steps`, `verif_commands`,
`fp_reason`, `classify`, `shape_census`** (and `_extract_candidates`/`_clean` if the preflight
keeps its own row-formatting). This single move fixes D1/D2/D3 at once:
- `classify()` already scans **only `status=="done"`** and **excludes `superseded_record`**
  → work-item (a) status-awareness + the "exclude already-annotated" rule for free.
- `fp_reason` already excludes `handoff/` + `tmp/` transient, abs-host paths
  (`/Library/LaunchAgents/com.py`), URL routes (`/openapi.json`), frontend-relative
  (`lib/icons.ts`), globs, negative assertions → work-item (a) transient-awareness, each
  pinned by an existing unit proof to MIRROR (test lines `:289-332`).
- `verif_commands` handles the **list shape** the preflight's `_extract_command` drops (D2).
- Do NOT re-implement these adjudicators in `preflight_verify_masterplan.py` — a second copy
  would drift from the 75.17 census (the whole point of the importable core, per that file's
  docstring `:16-20`).

### R2 — Switch iteration from recursive `_walk_steps` to canonical `flat_steps` (work-item b)
Replace `_walk_steps` (`:165-177`) with `flat_steps(data)`. This is the **root cause** of the
"?" ids: the recursive walk descends into `phases[].subphases[]` (`38.10-38.13`, `46.0-46.8`)
and `phases[].archived_legacy_steps[]` (`5.1/5.2` duplicates), none of which are canonical
steps. `flat_steps` reads only `ph["steps"]`, so **every emitted `step=X` resolves 1:1 to a
real step → zero "?"**. The canonical denominator becomes **883** (matches
`shape_census` sum 720+126+13+24). Optionally emit a one-line NOTE counting
subphase/legacy entries skipped, so the change is auditable rather than silent.

### R3 — Keep the import leg, but make it status-aware (coverage gap)
`classify()` is **path-only**; the preflight's `_check_imports`/`_extract_imports` has no
classifier equivalent and IS load-bearing (it catches renamed modules). Keep it, but gate it
behind the same `status=="done"` + `superseded_record` filter. Measured: all **5** current
import-broken lines are **pending** phase-5 steps (`5.2/5.8/5.9/5.11/5.12`,
`backend.markets.*` not yet built) → status filter zeroes them. Consider proposing to 75.17's
owner later that `fp_reason` grow an import sibling; for THIS step keep it local + gated.

### R4 — Make the summary internally consistent (work-item c)
Current summary conflates STEP counts, LINE counts, and a wrong denominator. Emit one
auditable line with a reconciled breakdown, e.g.:
```
preflight: 883 canonical steps (dict=720 str=126 list=13 none=24); 731 done/unannotated scanned.
GENUINE: 0 broken across 0 steps (0 missing-path, 0 unimportable).
Excluded: <N> non-done, 14 superseded_record-annotated, <M> transient/non-source refs.
Unverifiable-by-static-checker: 8 done steps with shlex-unparseable commands (23.2.7, 23.3.2, 70.2, 70.3, 70.5, 71.2, 71.4, 71.5).
```
Key rules: (i) report GENUINE lines AND the step count they span so "lines vs steps" can never
diverge un-narrated; (ii) put **unparseable** in its own bucket labelled a *checker limitation*,
never "broken" — those 8 are all `done` and their commands are valid shell the static tokenizer
can't split; (iii) exit code should key off GENUINE only (0 today).

### R5 — Triage the residue: it is EMPTY; annotate NOTHING new (work-item d)
Reproducible finding (`python scripts/qa/sweep_absent_verification_paths.py --json` →
`genuine: {}`; and the R1-R3 simulation → 0 path + 0 import): after recalibration the genuine
residue is **0**. Every genuinely-unrunnable done step is **already annotated**:
- `4.14.4, 4.14.24, 4.17.9` by **75.2.1**;
- `4.17.2/3/4/5/6/7/8/11/12, 4.14.26` (the **go_live_drills cluster** + skill-pair) + `68.5`
  by **75.17** — **owned by 75.17, do NOT re-annotate** (criterion 5 + the coordination rule;
  75.17 test `:158-183` pins exactly 14 holders, one each).
So criterion 5 ("any surviving unrunnable done step gains a `superseded_record`") is satisfied
**vacuously and honestly** — there are none. State this with the sweep output as evidence
(criterion 4: "no count asserted anywhere without reproducible evidence"). If triage *does*
surface an import-class survivor the classifier can't see, annotate it with the §1.4 shape
(BYTE-IDENTICAL command + success_criteria; `criteria_amended:false`).

### R6 — Anti-vacuous-guard test design (criterion 6 + house doctrine)
A "residue == 0 on live" assertion is a **tautology risk** (auto-memory
`feedback_mutation_test_guards_and_fixtures`; the 75.18 anti-vacuous-guard doctrine; the
Vera-Pérez pseudo-tested-method result — source 3 — is the formal proof that a test which
cannot fail proves nothing). MIRROR 75.17's paired design (`:203-214`):
- **Clean-on-live**: recalibrated checker returns 0 genuine against the live masterplan.
- **Genuine-on-a-defect-fixture (POSITIVE, load-bearing)**: a synthetic `status=done` step
  naming a genuinely-absent, non-transient, non-frontend path MUST be reported genuine — proves
  the checker can still detect a real defect, not just always-return-0.
- **Status-fixture**: one `done` + one each of `pending`/`deferred`/`dropped`/`superseded`
  naming the SAME absent artifact; only the `done` (unannotated) one is genuine — regression to
  status-blindness fails (criterion 1).
- **Transient/non-source fixtures**: `handoff/…`, `/openapi.json`, `lib/icons.ts`,
  `/Library/LaunchAgents/com.py`, glob-prefix — each pinned by a fixture, "not by an allowlist
  of observed strings" (criterion 2). Reuse `fp_reason` unit-proofs `:289-332`.
- **Shape fixture BY SHAPE**: assert `isinstance` first (75.2.1 lesson; test `:257-284`),
  including the list shape that the OLD preflight `_extract_command` dropped.
- **Mutation matrix (criterion 6, incl. >=1 FIXTURE mutation)** — record verbatim in
  `live_check_75.19.md`:
  - Production mutations: (m1) remove the `status=="done"` filter → pending 5.4.x reappear →
    a test fails; (m2) remove the `superseded_record` skip → the 14 annotated reappear → fails;
    (m3) revert to recursive `_walk_steps` → "?" ids `46.x` reappear → fails; (m4) drop the
    transient prefix → handoff rows reappear → fails.
  - **Fixture mutation** (the load-bearing one): mutate the POSITIVE defect fixture's path to a
    path that DOES exist → the "genuine detected" assertion must flip to fail; proves the
    fixture can represent both outcomes (the 75.2.1 lesson: mutating production alone missed a
    fixture that couldn't represent the failure).

### R7 — Boundaries / non-goals (from the step BOUNDARY clause)
- **NO immutable verification criteria amended.** Deliverable = a trustworthy CHECKER + (zero)
  annotations. Any `superseded_record` added keeps command + success_criteria BYTE-IDENTICAL
  (asserted via `git show <baseline>`), `criteria_amended:false`.
- **Do not fold the unparseable-8 into "broken"** and do not "fix" their commands (that would
  edit immutable criteria). They are a checker-coverage note.
- **Do not double-annotate the go_live_drills cluster** (75.17 owns it).
- Executor tag is **opus-4.8/xhigh**; `harness_required:true`; verification command is
  `.venv/bin/python -m pytest backend/tests/test_phase_75_19_preflight_calibration.py -q`
  (the test file the contract must create).
- Consider a `--quiet`/exit-code contract for CI wiring, but note 75.17's docstring `:19-20`
  explicitly says nightly-CI wiring is **75.19's job** — so wiring the recalibrated gate into a
  CI/pre-commit cadence is in-scope-adjacent; keep it minimal and status-aware.

### Suggested contract hypothesis
> Replacing `preflight_verify_masterplan.py`'s recursive walk + ad-hoc path heuristic with the
> importable 75.17 classifier (`flat_steps`+`verif_commands`+`fp_reason`+`classify`), adding a
> status-aware import leg, and reconciling the summary, drops the reported defect count from
> 151/222-lines (≈100% effective-FP) to the true genuine residue of **0**, with zero "?" ids
> and an internally-consistent summary — turning an ignored checker into a trustworthy gate,
> and confirming (with reproducible sweep output) that no new `superseded_record` is owed.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Recalibrating preflight_verify_masterplan.py = reuse the 75.17 importable classifier (flat_steps+verif_commands+fp_reason+classify). Three root defects: recursive _walk_steps drags in non-canonical subphases/archived_legacy (the 13 '?' ids 38.10-46.8) and drops list-shaped verification; no status/annotation/transient filter (~100% effective-FP). classify() already scans done-only + skips superseded_record; fp_reason already excludes handoff/transient, frontend-relative, url-routes, abs-host, globs. Keep a status-gated import leg (classifier is path-only; 5 import-broken are all pending phase-5). Fully-recalibrated GENUINE residue = 0 (reproducible via the sweep) -> nothing new to annotate; all 14 genuine defects already dispositioned (3 by 75.2.1, 11 by 75.17 incl. go_live_drills, which 75.19 must NOT re-touch). Summary must reconcile 883 canonical steps / genuine-lines-across-steps / 8 unparseable-done as a checker-limitation bucket. External canon (Google effective-FP <10%-or-ignored; FindBugs retired) + 2026 harness-gate papers + Vera-Perez pseudo-tested-methods mandate an anti-vacuous-guard test: clean-on-live PLUS a positive genuine-defect fixture PLUS >=1 fixture mutation.",
  "brief_path": "handoff/current/research_brief_75.19.md",
  "gate_passed": true
}
```

### Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch/pdfplumber (7)
- [x] 10+ unique URLs total (~45)
- [x] Recency scan (last 2 years) performed + reported (FSE 2025, Nature 2025, 3x 2026 arXiv)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Mandatory internal reads done (preflight target, classifier API, 75.17 test, superseded_record 75.2.1 shape, .gitignore, live baseline RUN, go_live_drills 75.17 status)

