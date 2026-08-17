# Research Brief -- phase-86.110

**Topic:** Test isolation leaks into shared, version-controlled state -- partial
monkeypatching of module-level path constants; structural pytest idioms that make
the leak impossible; enumerating the class transitively; prior art on
"tests must not touch the working tree"; safe restoration of a polluted state file.

**Tier:** simple (caller-set; depth/length knob only -- the >=5 read-in-full floor
and >=10 URL floor are unchanged at every tier per `.claude/rules/research-gate.md`).
**Audit-class:** YES (caller-set) -- loop-until-dry with K_required = 2.

---

## ENVELOPE (born inert -- phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": true,
    "rounds": 5,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.110.md",
  "gate_passed": true
}
```

**Sources read in full (the 7 that count toward the gate):**

1. https://docs.pytest.org/en/stable/how-to/monkeypatch.html
2. https://docs.pytest.org/en/stable/how-to/tmp_path.html
3. https://github.com/infotroph/tree-is-clean
4. https://pytest-pyfakefs.readthedocs.io/en/stable/intro.html
5. https://arxiv.org/html/2504.16777
6. https://wenxiwang.github.io/papers/ICSE2022.pdf
7. https://taoxie.cs.illinois.edu/publications/esecfse19-ifixflakies.pdf

---

## Status log (write-first, incremental)

- [t0] Brief created; envelope written born-inert. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full as binding operating instructions.
- [t1] INTERNAL HALF COMPLETE (enumerated, not hand-listed). See below.
- Next: external rounds 1..N (loop-until-dry, K=2).

---

# PART 1 -- INTERNAL MEASUREMENT (criterion 4: enumerate, do not hand-list)

## 1.1 The enumeration command and its raw output

The class is defined by "patches `_HISTORY_PATH`" x "patches `_HEARTBEAT_PATH`".
Both sides were enumerated repo-wide; the leak set is the set difference.

```
$ grep -rn "_HISTORY_PATH"   --include='*.py' . | grep -v '/\.venv/'
$ grep -rn "_HEARTBEAT_PATH" --include='*.py' . | grep -v '/\.venv/'
```

NOTE (zsh trap, measured): unquoted `--include=*.py` fails with
`(eval):1: no matches found` and produces EMPTY output -- which reads exactly
like "no such sites exist". The `--include` glob MUST be quoted. An unquoted
first attempt in this session returned zero rows for both greps; treating that
as the answer would have concluded there is no leak at all.

**Side A -- every site that rebinds `_HISTORY_PATH` (9 sites, 8 files):**

| File:line | Patches `_HISTORY_PATH` | Also patches `_HEARTBEAT_PATH`? |
|---|---|---|
| `backend/tests/test_phase_38_2_cycle_start_logging.py:31` | yes | **yes** (`:32`) |
| `backend/tests/test_phase_85_4_cycle_loudness.py:76` | yes | **yes** (`:77`) |
| `backend/tests/test_cycle_heartbeat_alarm.py:61` | yes | **NO** |
| `backend/tests/test_phase_85_6_anchor_deadlock.py:256` | yes | **yes** (`:257`) |
| `backend/tests/test_phase_66_1_rail_guard.py:194` | yes | **NO -- LEAK** |
| `backend/tests/test_phase_66_1_rail_guard.py:211` | yes | **NO -- LEAK** |
| `backend/tests/test_phase_85_4_completed_age_alarm.py:34` | yes | **yes** (`:35`) |
| `backend/tests/test_phase_86_38_degradation_visibility.py:157` | yes | **yes** (`:158`) |
| `scripts/smoketest_stages_5_through_13.py:402,404,409` | yes (direct assign + restore) | **NO** |

**Side B -- every `_HEARTBEAT_PATH` reference (9 rows):** the 5 test patches above,
plus production `cycle_health.py:37` (definition), `:558` (the write), `:563`/`:566`
(the read).

## 1.2 Why a hand-list under-covers -- three findings the named pair misses

The caller named two sites (`66_1` near `:194` and `:211`). Enumeration finds three
additional members of the "patches one constant, not both" class:

1. **`test_cycle_heartbeat_alarm.py:61`** patches `_HISTORY_PATH` and never
   `_HEARTBEAT_PATH`. **Not a leak** -- adjudicated by reachability, not by the
   patch table: its tests only call `cycle_health.cycle_heartbeat_alarm()`
   (`:75,:95,:114,:131,:146,:163,:195`), a READER, and they seed history through
   their own local `_write_history(path, ...)` helper (`:36`) into `tmp_path`.
   No writer is reached. This is the case that proves the patch table alone is
   the WRONG denominator -- it over-reports as well as under-reports.
2. **`scripts/smoketest_stages_5_through_13.py:402-409`** swaps `_HISTORY_PATH`
   only (plain assignment + restore in a `finally`), never `_HEARTBEAT_PATH`. It
   is not collected by pytest, so a tests-only sweep misses it entirely; if it
   reaches `record_cycle_end` it writes the REAL heartbeat.
3. The caller's own line-to-name mapping is **inverted**. Measured:
   `:191 def test_rail_guard_cycle_history_row_carries_flags` (patch at `:194`,
   writes `cycle_id="c1"` at `:197`) and
   `:207 def test_cycle_history_row_carries_funnel_counts` (patch at `:211`,
   writes `cycle_id="c2"` at `:213`). The prompt swapped the two names.

**The transitive-reach enumeration (the correct denominator):**

```
$ grep -rln 'record_cycle_end\|_write_heartbeat\|record_cycle_start' \
    --include='*.py' backend/tests scripts
backend/tests/test_phase_36_12_kill_switch_trading_path_block.py
backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py
backend/tests/test_phase_38_2_cycle_start_logging.py
backend/tests/test_phase_66_1_rail_guard.py
backend/tests/test_phase_85_4_completed_age_alarm.py
backend/tests/test_phase_86_38_degradation_visibility.py
scripts/qa/mutation_matrix_86_38.py
```

Cross-checking that set against side A/B: `36_12` and `36_17` appear here but in
NEITHER patch list. They do not leak because they cut the seam one level higher --
`test_phase_36_17...py:271` does
`monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)`, replacing
the **provider** so no real `CycleHealthLog` is ever constructed. That is idiom (b)
from the objective, already shipped in this repo.

## 1.3 The mechanism (production anchors, re-derived)

- `backend/services/cycle_health.py:34-37` -- `_HANDOFF` root, then TWO independent
  module-level constants: `_HISTORY_PATH = _HANDOFF / "cycle_history.jsonl"` (`:36`)
  and `_HEARTBEAT_PATH = _HANDOFF / ".cycle_heartbeat.json"` (`:37`).
- `record_cycle_end` writes history at `:488` **and then unconditionally** calls
  `self._write_heartbeat(cycle_id, "end")` at **`:492`**.
- `_write_heartbeat` (`:555`) does `_HEARTBEAT_PATH.write_text(...)` at **`:558`**,
  wrapped in a bare `except Exception` that only `logger.warning`s (`:559-560`) --
  so the leak is silent by construction.
- `record_cycle_start` has the same shape: `self._write_heartbeat(cycle_id, "start")`
  at `:426`.
- Module singleton `_log = CycleHealthLog()` (`:571`) exposed via `get_log()` (`:574`).

So **one public call writes two paths**, and the constants are read at call time
from module globals -- which is exactly why patching one of them is a partial fix
that still looks green: the assertion in both leaking tests only reads back the
tmp_path history file (`66_1:202`, `:219`), so nothing in the test can observe the
second write.

## 1.4 Current pollution state -- MEASURED (and it has already self-healed)

- **git-tracked: YES.** `git ls-files --error-unmatch handoff/.cycle_heartbeat.json`
  exits 0.
- `git show HEAD:handoff/.cycle_heartbeat.json` ->
  `{"cycle_id": "c7ac27f2", "event": "end", "updated_at": "2026-08-13T19:31:52.305075+00:00"}`
- working tree ->
  `{"cycle_id": "3e5afddb", "event": "end", "updated_at": "2026-08-17T19:47:15.759445+00:00"}`
- **The working tree is NOT currently polluted.** `3e5afddb` appears in
  `handoff/cycle_history.jsonl` (2 rows), i.e. it is a real cycle written by the
  legitimate writer at 2026-08-17T19:47Z. Whatever test value was there has already
  been overwritten by a live cycle.
- **`c2` has no legitimate source, confirmed two ways.** Parsing all 174 rows of
  `handoff/cycle_history.jsonl`: rows with `cycle_id == "c2"` = **0** (a naive
  `grep -c 'c2'` returns 10 -- substring hits inside hashes/timestamps; the parsed
  count is the real one). And scanning every historical blob of the heartbeat file
  across its 20 commits touching it: `"c2"` was **never committed**.
- Discriminating measurement for provenance: `86_38` ALSO writes `cycle_id="c2"`
  (`:181`), but its `health` fixture patches BOTH constants (`:157`, `:158`), so it
  cannot be the source. `66_1:213` is the only unisolated `c2` writer. Note both
  leaking tests write: `:197` writes `c1` first, `:213` writes `c2` second, and in
  pytest's file-definition collection order `:207` runs after `:191`, so **`c2` is
  the last-writer-wins value** -- matching the reported symptom.

## 1.5 Existing in-repo prior art for the structural fix

`backend/tests/conftest.py:6-11,35` already carries an import-time (not autouse-
fixture) guard, installed after fixture rows leaked into the REAL
`pyfinagent_data.llm_call_log` between 2026-05-19 and 2026-07-07. Its stated
rationale -- "Installed at import time rather than as an autouse fixture" (`:35`) --
is the same design question this step is asking, already answered once in this repo
for a different shared resource.

---

# PART 2 -- EXTERNAL RESEARCH

## 2.1 Search queries run (three-variant discipline, per research-gate.md)

| Variant | Query |
|---|---|
| year-less canonical | `pytest monkeypatch module-level path constant autouse fixture prevent tests writing real files` |
| year-less canonical | `pytest tmp_path_factory monkeypatch fixture isolation best practices` |
| year-less canonical | `CI check git status clean after tests fail if working tree modified` |
| year-less canonical | `pyfakefs pytest fake filesystem prevent tests touching real files` |
| current-year frontier | `test pollution shared state detection 2026 flaky tests order dependent` |
| last-2-year window | `ODRepair "order-dependent" flaky tests polluted shared state repair paper pdf Wing Lam` |
| last-2-year window | `"iFixFlakies" OR "polluter" "state-polluting" tests filesystem shared state empirical study` |

## 2.2 Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-17 | official doc | WebFetch | "Prefer patching the reference that your code uses instead of patching the original object"; all `monkeypatch` edits auto-undone; autouse fixture shown as the way to make a whole resource class unreachable; `raising=True` default makes a typo'd attribute name fail loudly. |
| 2 | https://docs.pytest.org/en/stable/how-to/tmp_path.html | 2026-08-17 | official doc | WebFetch | `tmp_path` = per-function unique dir; `tmp_path_factory` = session-scoped for higher-scoped fixtures; retains last 3 invocations; `--basetemp` "will be cleared blindly before each test run". |
| 3 | https://github.com/infotroph/tree-is-clean | 2026-08-17 | tool/doc | WebFetch | Direct prior art for (d). Motivation names our exact case: "detecting any files undesirably written into the working directory, e.g. by tests that ought to be using a proper tempdir." Runs `git status`, fails the build, "reports the full diff". `check_untracked` toggles new-file strictness. |
| 4 | https://pytest-pyfakefs.readthedocs.io/en/stable/intro.html | 2026-08-17 | official doc | WebFetch | Whole-filesystem interception via the `fs` fixture: "No files in the real file system are changed during the tests." LIMIT: "will not work with Python libraries (other than os and io) that use C libraries to access the file system." |
| 5 | https://arxiv.org/html/2504.16777 | 2026-08-17 | peer-reviewed (preprint HTML) | WebFetch (arXiv HTML per gate chain) | Parry, Kapfhammer, Hilton, McMinn 2025. "Filesystem Pollution" is a named cause theme: tests "modifying the filesystem ... and then failing in such a way that they omit to perform proper clean up procedures", which "triggers the failure of a group of subsequent test cases". 5 of 45 clusters; 75% of flaky tests belong to a cluster. |
| 6 | https://wenxiwang.github.io/papers/ICSE2022.pdf | 2026-08-17 | peer-reviewed (ICSE 2022) | curl + local `pypdf` extract (74,851 chars); every quote below regex-verified | **[ADVERSARIAL / scope-limiting]** ODRepair, the state-of-the-art auto-repair for order-dependent tests, explicitly EXCLUDES our case: "While there can be many types of shared state, such as heap-state, file system, databases, etc., heap-state was found to be one of the most common ... As such, we focus on polluted heap-state". Even in scope it repairs only 141 of 327 (43%). |
| 7 | https://taoxie.cs.illinois.edu/publications/esecfse19-ifixflakies.pdf | 2026-08-17 | peer-reviewed (ESEC/FSE 2019) | curl + local `pypdf` extract (71,025 chars); quotes regex-verified | Canonical victim/polluter/cleaner vocabulary: a victim fails because earlier tests "'pollute' the state (e.g., global variable, file system, network) on which the victim depends. We call such state-polluting tests polluters." Notes full isolation (Muslu et al.: "a fresh file system") works but "all forms of isolation add extra overhead". |

NOTE on method: sources 6 and 7 were NOT read via `WebFetch` on the PDF. Per the
measured in-repo finding that WebFetch PDF summaries can emit fabricated text
inside quotation marks (`.claude/agent-memory/researcher/reference_webfetch_pdf_summaries_fabricate_quotes.md`,
measured at steps 83.1.1 and 86.29), both PDFs were downloaded with `curl` and
extracted locally with `pypdf`, and every string quoted above was confirmed by
`re.finditer` against the extraction before being written here.

## 2.3 Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://dl.acm.org/doi/10.1145/3510003.3510173 | peer-reviewed (ICSE'22 ODRepair) | HTTP 403 Forbidden to WebFetch; obtained the same paper as source 6 via the authors' copy |
| https://dl.acm.org/doi/10.1145/3533767.3534404 | peer-reviewed (ISSTA'22, evolution-aware OD detection) | paywalled; superseded for our purpose by sources 6-7 |
| https://arxiv.org/pdf/2501.12680 | preprint (OD flaky tests in JavaScript) | wrong language ecosystem; snippet sufficient |
| https://cs.gmu.edu/~winglam/publications/2025/LevinETAL25Takuan.pdf | peer-reviewed 2025 (Takuan, dynamic invariants for OD debugging) | recency-scan hit; Java/Daikon-specific, no filesystem coverage |
| https://people.cs.gmu.edu/~winglam/publications/2025/RahmanETAL25RankF.pdf | peer-reviewed 2025 (ranking relevant tests for OD) | recency-scan hit; ranking, not isolation |
| https://people.cs.gmu.edu/~winglam/publications/2023/LiETAL23Tuscan.pdf | peer-reviewed 2023 (Tuscan orders) | test-order generation, orthogonal to the fix |
| https://taoxie.cs.illinois.edu/publications/icst19-idflakies.pdf | peer-reviewed (iDFlakies) | detection framework, Java-only |
| https://docs.pytest.org/en/stable/reference/reference.html | official doc | API index; the two how-to pages carry the guidance |
| https://github.com/jwarby/git-is-clean | tool | npm equivalent of source 3 |
| https://www.npmjs.com/package/git-is-clean | tool | duplicate of the above |
| https://j2r2b.github.io/2019/03/26/ensure-no-uncommitted-changes.html | blog | `git status --porcelain` recipe, same mechanism as source 3 |
| https://www.baeldung.com/linux/git-script-check-clean-directory | community | shell recipe only |
| https://github.com/pytest-dev/pyfakefs | code | same content as source 4 |
| https://pypi.org/project/pytest-randomly/ | tool | detection lever cited in 2.4 |
| https://pytest-with-eric.com/pytest-best-practices/pytest-tmp-path/ | blog | lower tier than source 2 |
| https://pytest-with-eric.com/mocking/pytest-monkeypatch/ | blog | lower tier than source 1 |
| https://www.datadoghq.com/knowledge-center/flaky-tests/ | industry | vendor overview |
| https://quashbugs.com/blog/how-to-fix-flaky-tests-complete-diagnosis-guide | industry (2026) | recency hit; no new mechanism |

**URLs collected: 25** (7 read in full + 18 snippet-only).

## 2.4 Recency scan (2024-2026) -- MANDATORY SECTION

Searched the 2024-2026 window explicitly (`test pollution shared state detection
2026 flaky tests order dependent`, plus the two 2025 GMU papers and the 2025 arXiv
preprint). **Result: 1 finding that COMPLEMENTS the canonical sources, 0 that
supersede them.**

- **Complements:** Parry et al. 2025 (source 5) is the first study to quantify
  filesystem pollution as a *co-occurrence* cause -- 5 of 45 clusters, i.e. it is
  real but the *third least common* theme. It also supplies the 75%-of-flaky-tests-
  cluster figure, which is the argument for enumerating a class rather than fixing
  instances.
- **No supersession:** the 2019 iFixFlakies vocabulary (victim/polluter/cleaner)
  and the 2022 ODRepair scoping are both still current; the 2025 papers (Takuan,
  RankFlaky) extend *detection and ranking* for Java heap state and do not touch
  filesystem pollution or Python.
- **Nothing newer changes the pytest idioms.** `monkeypatch` semantics and
  `tmp_path`/`tmp_path_factory` are unchanged in the current stable docs.

## 2.5 Key findings, cited per claim

1. **Patch the seam, not the leaves.** pytest's own guidance is to "Prefer patching
   the reference that your code uses instead of patching the original object", and
   adds that "for code that you control, a safer long-term pattern is to make
   dependencies explicit so they can be passed into the code under test instead of
   patched globally" (pytest docs, source 1). This is the direct answer to (a): N
   constants is N chances to miss one; a single provider is one.
2. **`raising=True` is the free half of the fix.** `monkeypatch.setattr` defaults to
   raising `AttributeError` when the target does not exist (source 1) -- so a
   renamed constant fails loudly, but an *unpatched second* constant is invisible.
   The default protects against typos, never against incompleteness.
3. **Autouse is the documented mechanism for "make a whole resource unreachable".**
   pytest documents an autouse fixture that deletes `Session.request` "so that any
   attempts within tests to create http requests will fail" (source 1). The same
   shape applied to a path provider converts opt-in isolation into opt-out.
4. **Session-scoped isolation needs `tmp_path_factory`,** because function-scoped
   `tmp_path` cannot be consumed by a higher-scoped fixture (source 2).
5. **Working-tree assertion is established CI prior art, and its stated motivation
   is literally this bug class** -- "detecting any files undesirably written into the
   working directory, e.g. by tests that ought to be using a proper tempdir"
   (source 3). It runs `git status`, fails the build, and "reports the full diff".
6. **Whole-filesystem faking exists but is not free.** pyfakefs guarantees "No files
   in the real file system are changed during the tests" (source 4), but "will not
   work with Python libraries (other than os and io) that use C libraries to access
   the file system" -- so it is a heavier, partially-leaky hammer.
7. **[ADVERSARIAL] The automated-repair literature deliberately does not cover this
   case.** ODRepair states it focuses on "polluted heap-state" and lists "file
   system" among the state types it sets aside (source 6). So there is no
   off-the-shelf detector to adopt; the enumeration must be built. Its 141/327 (43%)
   patch rate even in its chosen scope is a caution against expecting automation.
8. **Filesystem pollution is real but comparatively rare -- and that is an argument
   FOR a cheap global guard, not against it.** 5 of 45 clusters (source 5); a rare
   class caught by review is a class that will be missed by review.
9. **The canonical vocabulary for the internal write-up** is victim / polluter /
   cleaner (source 7), and full isolation ("a fresh file system") is acknowledged as
   effective but carrying "extra overhead" (source 7) -- the same trade-off pyfakefs
   embodies.

## 2.6 Consensus vs debate

- **Consensus:** isolate via temp dirs, patch the reference the code actually uses,
  prefer explicit dependency injection over global patching, and detect pollution by
  running tests in a randomized/isolated order.
- **Debate:** *how much* isolation. iFixFlakies (source 7) notes fresh-environment
  isolation works but "all forms of isolation add extra overhead"; pyfakefs (source
  4) takes the maximal position but concedes C-extension leakage. ODRepair (source
  6) declines filesystem state entirely. No source argues for the status quo of
  per-test manual constant patching.
- **Gap in the literature:** nobody addresses *version-controlled* state
  specifically. The git-status assertion (source 3) is a practitioner tool, not an
  academic result; that is the strongest reason to lean on it here.

## 2.7 Pitfalls (from the literature + measured here)

- Patching one of N module-level constants leaves the rest live (measured, 1.3).
- A test asserting only on the isolated artifact cannot observe the leak (measured:
  `66_1:202`, `:219` read back only the tmp history file).
- A silent `except Exception: logger.warning` around the second write makes the leak
  invisible even at runtime (`cycle_health.py:559-560`).
- A patch table is the wrong denominator in BOTH directions -- it under-reports
  (missed `scripts/smoketest_stages_5_through_13.py`) and over-reports
  (`test_cycle_heartbeat_alarm.py` patches one constant and is not a leak).
- `--basetemp` "will be cleared blindly before each test run" (source 2) -- never
  point it at a real directory.
- Unquoted `--include=*.py` under zsh yields empty grep output that reads as "no
  occurrences" (measured this session).

## 2.8 Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Application |
|---|---|---|
| Patch the provider, not the constants (source 1) | `cycle_health.py:574 get_log()`; already done at `test_phase_36_17...py:271` | A `_handoff_dir()` provider (or patching `_HANDOFF`, `:34`) collapses 2 seams into 1. Note `:36`/`:37` bind at import, so patching `_HANDOFF` alone does NOT retarget them -- the provider must be read at call time. |
| Autouse "make it unreachable" (source 1) | `backend/tests/conftest.py:35` already argues import-time vs autouse for the BQ leak | Same shape, second resource. Precedent exists in-repo. |
| `tmp_path_factory` for higher scope (source 2) | new session fixture | Needed only if the guard is session-scoped. |
| git-status assertion (source 3) | `handoff/.cycle_heartbeat.json` is git-TRACKED (measured) | A session-scoped guard that snapshots tracked files and diffs at teardown is the enumeration-free detector for (c)/(d). |
| pyfakefs (source 4) | whole suite | Rejected as primary: heavier, C-extension gaps, and the repo already relies on real tmp_path files. |
| No automated repair covers filesystem state (source 6) | -- | Confirms the guard must be hand-built; do not shop for a plugin. |

## 2.9 Answering (e): restoring the polluted file

**Measured facts that decide it:** the file is git-tracked; HEAD holds `c7ac27f2`
(2026-08-13); the working tree now holds `3e5afddb` (2026-08-17T19:47Z) which IS a
real cycle present twice in `handoff/cycle_history.jsonl`; and `c2` has 0 rows in
the 174-row ledger and was never committed.

Therefore the practical answer is **neither "regenerate" nor "leave and document"**
as posed: **the pollution has already self-healed.** The legitimate writer
(`cycle_health.py:558`, driven by the live autonomous cycle) overwrote the test
value hours ago. The decision rule generalizes:

- The heartbeat is **derived, last-writer-wins, single-line state** whose sole
  legitimate producer runs on every cycle. Hand-restoring it is pointless work that
  the next cycle overwrites, and hand-writing a `cycle_id` would *manufacture* a
  value with no ledger row -- exactly the defect (`c2`) being cleaned up.
- **Rule:** if the file is (i) fully derived, (ii) rewritten wholesale by a live
  writer on a known cadence, and (iii) currently holds a value corroborated by the
  upstream source of truth -- take no content action; just stop the leak. Restore by
  hand only when the state is append-only or accumulating (like
  `cycle_history.jsonl`, where a leaked row would persist forever), or when no
  legitimate writer will run before a reader depends on it.
- The auditable check that the file is clean is not "does it equal HEAD" (it never
  will, legitimately) but **"is its `cycle_id` present in `cycle_history.jsonl`"** --
  which is exactly the probe that catches `c2` and clears `3e5afddb`.

## 2.10 Answering (c): enumerate, don't hand-list

Two complementary enumerations, both cheap and both re-runnable:

1. **Static (the set difference)** -- the two greps in 1.1, then subtract. Catches
   sites that patch one constant. Must be reachability-adjudicated (1.2) or it
   over-reports.
2. **Dynamic (the enumeration-free detector)** -- the git-status assertion (source
   3): snapshot tracked files at session start, diff at teardown, fail naming the
   file. This needs no list at all, so it cannot under-cover, and it generalizes to
   every future writer -- which is the whole point, since a hand-list is a claim
   about a set whose membership rule was never checked.

Detection lever from the literature: run under `pytest-randomly` / `pytest -p
no:randomly` orderings to surface order-dependence (Parry et al. 2025 clustering
rationale, source 5) -- but note the leak here is *order-independent* (it writes the
real file on every run), so the working-tree diff is the higher-yield detector.

## 2.11 Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7)
- [x] 10+ unique URLs total (25)
- [x] Recency scan (last 2 years) performed + reported (2.4)
- [x] Full papers / pages read, not abstracts (PDFs locally extracted + quote-verified)
- [x] file:line anchors for every internal claim (Part 1)

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted (2.6)
- [x] All claims cited per-claim

## 2.12 Audit-class coverage loop (K_required = 2)

| Round | Queries | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | pytest monkeypatch / tmp_path idioms | 2 (sources 1-2) | no |
| 2 | git-clean CI, pyfakefs, 2026 flaky-test frontier | 3 (sources 3-5) | no |
| 3 | ODRepair / iFixFlakies peer-reviewed | 2 (sources 6-7) | no |
| 4 | `pytest plugin detect tests that modify files in repository working directory snapshot diff session` + `read-only bind mount source directory run test suite prevent writes sandbox` | **0** | **DRY** |
| 5 | `conftest autouse fixture guard fail test that writes outside tmp_path repository file` + `dependency injection over module-level global path constants testability Python configuration object` | **0** | **DRY** |

`dry_rounds = 2 >= K_required = 2` -> **`coverage.dry = true`**.

Rounds 4-5 were not empty of information, only of NEW READ-IN-FULL findings, and
the two negative results are themselves load-bearing:

- **No pytest plugin does snapshot-and-diff of tracked files around a session.**
  The nearest neighbours all run the *inverse* query (which tests to run given
  changed files): `pytest-picked`, `pytest-run-changed`, `pytest-cagoule`,
  `pytest-snap`. One search returned explicitly that it "doesn't contain that
  specific implementation" for an autouse guard failing tests that write to repo
  files. This corroborates finding 7 from the opposite direction: the guard has to
  be written here, and it is small.
- **Read-only bind mounts are not available as a fix on this host.** Recursive
  read-only bind mounts are a Linux VFS/kernel feature (Docker requires kernel
  >= 5.12); pyfinagent is local-only on macOS (`.claude/context/owner.md` /
  local-only deployment), so option (d)-by-mount is out and the git-status
  assertion is the portable one.

## 2.13 Internal code inventory

| File | Lines cited | Role | Status |
|---|---|---|---|
| `backend/services/cycle_health.py` | `:34,:36,:37,:426,:488,:492,:555,:558,:559-560,:571,:574` | The writer. Two constants, one public call, two writes, silent except | PRODUCTION -- do not edit in this step |
| `backend/tests/test_phase_66_1_rail_guard.py` | `:191,:194,:197,:202,:207,:211,:213,:219` | **The two leaking tests** (`c1`, `c2`) | DEFECT -- patches `_HISTORY_PATH` only |
| `backend/tests/test_phase_86_38_degradation_visibility.py` | `:155-160` (`health` fixture), `:157,:158,:165,:181` | **Reference-correct pattern** -- patches BOTH | GOOD -- the model to copy |
| `backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py` | `:271` | Patches the PROVIDER `get_log` -- idiom (b) already in-repo | GOOD -- alternative model |
| `backend/tests/test_phase_38_2_cycle_start_logging.py` | `:13,:31,:32` | Patches both; docstring already states the intent | GOOD |
| `backend/tests/test_phase_85_4_cycle_loudness.py` | `:76,:77` | Patches both | GOOD |
| `backend/tests/test_phase_85_4_completed_age_alarm.py` | `:34,:35` | Patches both | GOOD |
| `backend/tests/test_phase_85_6_anchor_deadlock.py` | `:256,:257` | Patches both | GOOD |
| `backend/tests/test_cycle_heartbeat_alarm.py` | `:36,:60,:61,:75` | Patches `_HISTORY_PATH` only, but reaches NO writer | NOT a leak -- false positive of the patch-table method |
| `scripts/smoketest_stages_5_through_13.py` | `:402,:404,:409` | Swaps `_HISTORY_PATH` only; not pytest-collected | AT RISK -- outside a tests-only sweep |
| `backend/tests/conftest.py` | `:6-11,:35` | Existing import-time leak guard (BQ `llm_call_log`) | PRIOR ART -- the integration point |
| `handoff/.cycle_heartbeat.json` | whole file | The polluted artifact; git-TRACKED | Currently NOT polluted (self-healed) |
| `handoff/cycle_history.jsonl` | 174 rows | The real ledger / source of truth | `c2` absent; `3e5afddb` present |
| `scripts/qa/mutation_matrix_86_38.py` | `:81` | References the writer by test name only | No path binding |

## 2.14 Caveats and residual uncertainty

- I did **not execute** the test suite (task boundary: read and measure only). The
  leak is established from source reading plus the ledger/heartbeat measurements,
  not from an observed write during a run. A single `pytest
  backend/tests/test_phase_66_1_rail_guard.py -k funnel` followed by `git diff
  --stat handoff/.cycle_heartbeat.json` would confirm it in one command; that
  belongs to GENERATE, not to the gate.
- `scripts/smoketest_stages_5_through_13.py` is flagged from its constant swap; I
  did not trace whether its exercised path actually reaches `record_cycle_end`.
- Both leaking tests construct their own `CycleHealthLog()` (`66_1:195`, `:212`)
  rather than using `get_log()`, so a provider-only fix at `get_log` would NOT
  cover them -- the fix must target the path binding, or those two call sites must
  change too. This is the single most important design constraint found.
