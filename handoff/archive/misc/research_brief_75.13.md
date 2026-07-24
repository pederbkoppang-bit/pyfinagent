# Research Brief — Step 75.13 (Python dependency integrity: lockfile, undeclared deps, dead + implicit declarations)

Tier: **moderate** (caller-specified). NOT audit-class.
Researcher: Layer-3 Researcher (Workflow launch). Started 2026-07-24.
Status: **COMPLETE** — gate_passed: true (7 sources read in full; recency scan done; 3-variant queries; 11 internal files inspected).

## Step summary (from masterplan 75.13)

P1, executor sonnet-4.6/high. BOUNDARY: lock generated from the LIVE venv snapshot
(`pip freeze` — no resolver run), no `.env` edits, NO package installs beyond declaring
what is already installed. Four findings folded in:
- **deps-01**: commit `backend/requirements.lock` (full `==`-pinned `pip freeze`, >=150 pins,
  regen/sync header) + point `.github/workflows/pip-audit.yml` at the lock.
- **deps-02**: declare `exchange-calendars` pinned in `backend/requirements.txt` (3 import
  sites w/ silent-degrade fallbacks); `gpt-researcher` -> NEW
  `scripts/autoresearch/requirements-autoresearch.txt` (pinned) + `run_nightly.sh` fails
  LOUDLY on ImportError.
- **deps-09**: declare `numpy`, `PyYAML`, `pytest`, `python-dateutil`, `google-cloud-storage`
  in `backend/requirements.txt`.
- **deps-08**: remove dead `fpdf2`, annotate `xlrd` (pandas .xls engine).

## MEASURED venv snapshot (2026-07-24, live `.venv/bin/pip freeze`)

- **Line count: 303** — clears the `>=150 == pins` assert comfortably (2x margin).
- No editable (`-e`), no `git+`, no `@ file://` / direct-URL lines — a clean freeze; no
  quirk that would break the verification asserts or need `--exclude-editable`.
- Named-package exact versions (verbatim `pip freeze` emission):
  | package | `pip freeze` emits | drift vs current floor |
  |---|---|---|
  | pandas | `pandas==3.0.1` | floor `>=2.2.0` — **whole major** behind (2.x → 3.x) |
  | yfinance | `yfinance==1.2.0` | floor `>=0.2.40` — **whole major** (0.2 → 1.2) |
  | numpy | `numpy==2.4.4` | undeclared; installed 2.4.4 |
  | exchange_calendars | `exchange_calendars==4.13.2` | **UNDERSCORE form** (see gotcha) |
  | gpt-researcher | `gpt-researcher==0.14.8` | hyphen form |
  | fpdf2 | `fpdf2==2.8.7` | INSTALLED (so removing the *declaration* does NOT uninstall) |
  | xlrd | `xlrd==2.0.2` | installed |
  | PyYAML | `PyYAML==6.0.3` | **CAPITALS** (matches case-sensitive assert) |
  | pytest | `pytest==9.0.3` (+cov 7.1.0, +timeout 2.4.0) | installed |
  | python-dateutil | `python-dateutil==2.9.0.post0` | hyphen form |
  | google-cloud-storage | `google-cloud-storage==3.10.1` | hyphen form |

### 🔴 GOTCHA #1 (name normalization) — `exchange-calendars` hyphen vs underscore
`pip freeze` emits `exchange_calendars==4.13.2` (**underscore**). The verification does two
different string checks:
- LOCK: `any(x.lower().startswith('exchange') for x in pins)` → underscore form PASSES
  (prefix-only, case-insensitive). So the lock keeps the verbatim `pip freeze` underscore form.
- requirements.txt: `assert 'exchange-calendars' in r` → **HYPHEN, case-sensitive**. The
  executor MUST write `exchange-calendars==4.13.2` (canonical PyPI distribution name, hyphen)
  in `requirements.txt`. Copying the underscore form from the freeze into requirements.txt
  would FAIL the assert. (pip installs both spellings identically — PEP 503 normalization —
  so this is a text-assert quirk, not a functional one.)

### 🔴 GOTCHA #2 (`fpdf2 not in r` is whole-file substring) — no residue allowed
`assert 'fpdf2' not in r` checks the ENTIRE requirements.txt text. A leftover comment like
`# removed fpdf2 (dead)` would FAIL the assert. The executor must delete the line AND leave
no `fpdf2` substring anywhere (comments included). Annotate the removal in the commit
message / experiment_results.md, not in requirements.txt.

### 🔴 GOTCHA #3 (`PyYAML` is case-sensitive) — must be capitals
`assert 'PyYAML' in r` is case-sensitive. `pyyaml` (lowercase) FAILS. `pip freeze` luckily
emits `PyYAML` with capitals, so copying the freeze spelling is safe; writing lowercase is not.



## Research: Python dependency locking, pip-audit, PEP 508, silent-import-fallback risk

### Read in full (7; >=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://pip.pypa.io/en/stable/topics/repeatable-installs/ | 2026-07-24 | official pip doc | WebFetch full | **pip OFFICIALLY endorses freeze-as-lock:** "A requirements file, containing pinned package versions can be generated using pip freeze" — "would pin not only the top-level packages, but also all of their transitive dependencies." Caveats: "it trusts the locations you're fetching the packages from (like PyPI) and the certificate authority chain"; hashes are the stricter add-on (not required for a single-machine deploy-lock). Directly validates deps-01. |
| 2 | https://pip.pypa.io/en/stable/reference/requirements-file-format/ | 2026-07-24 | official pip doc | WebFetch full | `#`-comments ignored; `--requirement`/`--constraint`/`--hash` options; extras `pkg[extra]`; markers `; python_version < "2.7"`; `==` pin example `docopt == 0.6.1`. |
| 3 | https://peps.python.org/pep-0508/ | 2026-07-24 | PEP (peer-tier) | WebFetch full | Dependency-spec grammar `name extras? versionspec? ; marker`; **name matching is case-INSENSITIVE** (PyPI regex runs `re.IGNORECASE`). |
| 4 | https://packaging.python.org/en/latest/specifications/name-normalization/ | 2026-07-24 | official PyPA spec | WebFetch full | **Normalization = `re.sub(r"[-_.]+","-",name).lower()`.** "`exchange_calendars` and `exchange-calendars` normalize to the same form and are interchangeable in requirements files for pip." -> GOTCHA #1 is a TEXT-assert quirk, not a functional one. |
| 5 | https://github.com/pypa/gh-action-pip-audit/blob/main/README.md | 2026-07-24 | official PyPA action doc | WebFetch full | `inputs:` accepts multiple requirements files (`inputs: requirements.txt dev-requirements.txt`); `locked:` is for `pylock.*.toml` (PEP 751), NOT a freeze-style lock; `no-deps: true` / `require-hashes: true` improve accuracy on already-pinned files. |
| 6 | https://semgrep.dev/blog/2026/the-best-free-open-source-supply-chain-tool-the-lockfile/ | 2026-07-24 | industry blog (recency) | WebFetch full | Audited-graph==deployed-graph argument: "With a content-hashed lockfile, you can reason about whether you were impacted by a vulnerability/malicious package as opposed to having to guess based on a version or range specifier, because the build environment is deterministic and reproducible." Auditing an unpinned manifest examines "theoretical possibilities, not actual deployed code." |
| 7 | https://www.index.dev/blog/avoid-silent-failures-python | 2026-07-24 | practitioner blog | WebFetch full | Silent-fallback risk (deps-02): "Always catch specific exceptions instead of using bare except: clauses, which can hide underlying bugs"; "log loudly and fail visibly." Broad `except` masking an ImportError is the anti-pattern the deps-02 loud-fail addresses. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/jazzband/pip-tools/issues/1524 | GitHub issue | pip-compile-vs-freeze rationale; covered by source #1 |
| https://peps.python.org/pep-0685/ | PEP | extra-name normalization; adjacent to #4 |
| https://github.com/pypa/pip-audit | tool README | tool-level behavior; action README (#5) is the CI-relevant doc |
| https://pip.pypa.io/en/stable/topics/secure-installs/ | pip doc | hash-checking depth; out of single-machine scope |
| https://medium.com/.../shift-security-left-...pip-audit-in-ci (Jun 2026) | blog | CI pip-audit recency corroboration |
| https://dev.to/.../uv-audit-vs-pip-audit-and-a-gate-narrower-than-it-looks | blog | uv-audit vs pip-audit; not in scope (we use pip-audit) |
| https://www.index.dev + oneuptime 2026-01 + augmentcode | blogs | silent-failure corroboration for #7 |
| https://news.ycombinator.com/item?id=25255668 | forum | "lockfile == reproducible builds without pip freeze" debate (see Consensus vs debate) |
| https://packaging.python.org/en/latest/specifications/dependency-specifiers/ | PyPA spec | live PEP 508 spec; #3 is the PEP itself |

### Recency scan (2024-2026)
Searched 2026 (frontier), 2025 (window), and year-less canonical variants per topic.
**Findings:** (a) pip docs are at **v26.1.2** (current) and still endorse `pip freeze` for pinning
— no supersession of the freeze-as-lock pattern. (b) `gh-action-pip-audit` is at **@v1.1.0**
and added a `locked:` param for **PEP 751 `pylock.*.toml`** (the new standardized lockfile) —
NOTE: a `pip freeze` `requirements.lock` is NOT a `pylock.toml`, so it belongs in `inputs:`,
not `locked:`. (c) Semgrep's **2026** lockfile post and multiple **Jun-2026** pip-audit-in-CI
walkthroughs reaffirm audited-graph==deployed-graph. (d) **PEP 751** (2025, pylock standard)
is the only genuinely new development; adopting it is OUT OF SCOPE for 75.13 (the step
explicitly specifies a `pip freeze` snapshot, no resolver) — flag as a possible future step.
No 2024-2026 source contradicts the freeze-snapshot-for-a-single-machine-deploy approach.

### Consensus vs debate (external)
- **Consensus:** a committed, fully-pinned lock is the foundational supply-chain control; audit
  the lock (deployed graph), not the floors (hypothetical graph). Pinning + repeatable installs
  + CI scanning is standard hygiene (Semgrep 2026; pip-audit docs; pipenv security docs).
- **Debate:** `pip freeze` vs `pip-compile`/`uv`/Poetry. The pip-tools camp (jazzband #1524, HN
  25255668) argues freeze can't track WHY a transitive is pinned and mixes direct+transitive.
  **For pyfinagent this debate is moot:** the app is a **single-machine, deploy-locked** system
  (local-only deployment; user ADC; Peder's Mac) and the step DELIBERATELY specifies a freeze
  snapshot with NO resolver run — the "hard to maintain / resolution hell" critique applies to
  teams re-resolving across machines, not to a byte-frozen single-host lock that is regenerated
  by re-freezing the audited venv. pip's own docs bless this exact use.

### Pitfalls (from literature)
1. **Auditing the floors, not the deployed graph** — the current `pip-audit.yml` scans
   `requirements.txt` (loose `>=` floors), so it evaluates a hypothetical resolution, not what
   actually runs. Fix = point it at the lock (deps-01). (Semgrep 2026.)
2. **Silent import fallbacks** — broad `except`/`except ImportError: pass` that degrades behavior
   invisibly on rebuild. Present at 4 sites here (3 exchange_calendars + 1 embedding preflight).
   The deps-02 loud-fail + explicit declarations are the countermeasure. (index.dev.)
3. **Name-normalization text traps** — pip treats `exchange_calendars`==`exchange-calendars`, but
   a naive string `assert` does not. See GOTCHA #1/#3.


### Internal code inventory (file:line anchors, all read in full at the relevant spans)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/requirements.txt` | 1-60 | Floors file (no lock) | 60 lines; has `xlrd>=2.0.1` (already commented), `fpdf2>=2.7.0` (line 26, dead), `pytest-cov==7.1.0` (line 58). MISSING: exchange-calendars, numpy, PyYAML, python-dateutil, google-cloud-storage, and explicit pytest |
| `.github/workflows/pip-audit.yml` | 1-60 | CI supply-chain scan | Audits `backend/requirements.txt` + top-level `requirements.txt` (the FLOORS, not a lock). Runs on push/PR paths + weekly cron + workflow_dispatch. `--requirement backend/requirements.txt --strict` at line 48. Must ALSO point at `requirements.lock` |
| `backend/backtest/markets.py` | 11-14, 176-204 | Multi-market calendar | `try: import exchange_calendars as xcals / except ImportError: xcals = None`. `get_trading_calendar()` returns None + logs WARNING when `xcals is None`; `is_trading_day()` **fail-OPEN returns True** when calendar unavailable (:203-204). Silent-degrade: trading-calendar gating vanishes on rebuild-without-xcals |
| `backend/autoresearch/monthly_champion_challenger.py` | 319-344 | Champion/challenger anchor | `is_last_trading_friday()`: `try: import exchange_calendars as xcals ... except Exception as exc:` -> pure-Python last-Friday fallback that "doesn't handle holiday-shifted last-sessions without xcals" (:321-322). Behavior CHANGES silently on rebuild |
| `scripts/go_live_drills/signal_reliability_test.py` | 65-84 | Go-live drill | `get_nyse_trading_days()`: `try: import exchange_calendars ... except ImportError: pass` -> stdlib weekday-minus-hardcoded-2026-holidays fallback (:76-81). Silent-degrade; the hardcoded holiday set rots after 2026 |
| `backend/services/macro_regime.py` | 59, 152-157 | GPR macro feature | `_GPR_CACHE_PATH = _GPR_CACHE_DIR / "data_gpr_export.xls"` (:59, **legacy .xls**); `df = pd.read_excel(_GPR_CACHE_PATH)` (:154) -> pandas needs **xlrd** as the .xls engine. Wrapped in try/except -> returns None on parse fail. Confirms xlrd is a real (implicit) runtime dep |
| `backend/services/compliance_logger.py` | 120-149 | Compliance WORM writer | `_write_gcs()` / `_read_gcs()`: `from google.cloud import storage  # type: ignore` then `storage.Client()` — **NO import guard**. If google-cloud-storage (transitive via google-cloud-aiplatform) is pruned, the WORM audit-log write raises uncaught ModuleNotFoundError. Confirms google-cloud-storage must be declared |
| `backend/tools/earnings_tone.py` | 38-39, 56-57 | Earnings GCS reader | 2 more `from google.cloud import storage` sites (unguarded) — reinforces the google-cloud-storage declaration need |
| `scripts/autoresearch/run_memo.py` | 68-83, 103-128, 131-213 | Nightly memo runner | `from gpt_researcher import GPTResearcher` is a LATE import (:70) inside `run_research()`; `_main_async` wraps it in `except Exception -> ERROR file + return 1` (:114-124) = LOUD on gpt_researcher ImportError. BUT `main()` runs `_embedding_preflight()` FIRST (:201-204) which **returns 0 silently** if the EMBEDDING backend (`langchain_huggingface`) is missing — a phase-51.4 intentional soft-skip |
| `scripts/autoresearch/run_nightly.sh` | 1-75 | Launchd wrapper (75.11) | `if python run_memo.py; then OK; else rc=$?; log FAIL rc; increment consecutive_fails; page after PAGE_AFTER_N (=3) via Slack bot-token; exit rc; fi` (:43-72). The 75.11 (sre-ops-04) paging seam. Non-zero run_memo exit IS already logged + pageable |
| `scripts/autoresearch/` (dir) | — | Manifest target | Contains run_memo.py, run_nightly.sh, topics.txt, __pycache__. **NO requirements-autoresearch.txt yet** — deps-02 creates it |

### 🔴 GOTCHA #4 (deps-02 loud-fail COMPOSES with 51.4 + 75.11 — do not naively "make ImportError loud")
Two distinct import-failure seams already exist and pull in OPPOSITE directions:
1. **gpt_researcher package ImportError** -> ALREADY LOUD. `run_research()`'s late `from gpt_researcher import` raises -> `_main_async` broad `except` writes ERROR file + `return 1` -> `run_nightly.sh` logs `FAIL rc=1` + increments consecutive_fails + **pages after 3** (75.11 seam). Nothing to add for the loud path except a text-assert test.
2. **embedding-backend ImportError** (`langchain_huggingface`) -> INTENTIONALLY SILENT. `_embedding_preflight()` (:201-204) returns 0 with only a stderr note — a **phase-51.4 deliberate away-ops soft-skip** ("skip cleanly ... instead of crashing every night"). run_nightly.sh sees rc=0 -> logs OK -> RESETS consecutive_fails -> never pages. THIS is the "silently drying up the research gate" path.
**Latent masking bug:** `_embedding_preflight()` runs BEFORE the gpt_researcher late import, so if BOTH are missing, gpt_researcher's absence is masked by the embedding soft-skip (exit 0). **Recommended executor fix (composes with both):** add an explicit early `importlib.util.find_spec("gpt_researcher") is None -> print FAIL + return non-zero` guard placed BEFORE `_embedding_preflight()`, so the namesake dep's absence is never masked and always reaches the 75.11 page seam. Do NOT convert the 51.4 embedding preflight to loud (that reverts an intentional design). Add a text-assert test on run_nightly.sh (grep for the FAIL log + page seam) per success-criterion #4.



### Application to pyfinagent (external findings -> file:line anchors)

| Leg | Action | Anchor | External basis |
|---|---|---|---|
| deps-01 lock | `.venv/bin/pip freeze > backend/requirements.lock` (303 pins measured) + header comment (regen: `pip freeze > backend/requirements.lock`; sync: `pip install -r backend/requirements.lock` or `uv pip sync backend/requirements.lock`) | new file | pip repeatable-installs (src #1) blesses freeze-as-lock |
| deps-01 audit | add `--requirement backend/requirements.lock` to the `pip-audit` run step (`.github/workflows/pip-audit.yml:46-50`); can keep floors too or audit lock-only | pip-audit.yml:16-18 (paths), :48 (invocation) | Semgrep 2026 (src #6): audit deployed graph |
| deps-02 exchange | add `exchange-calendars==4.13.2` (HYPHEN) to requirements.txt | markets.py:11-14, monthly_champion_challenger.py:326-327, signal_reliability_test.py:66-74 | name-normalization spec (src #4) |
| deps-02 gpt-researcher | new `scripts/autoresearch/requirements-autoresearch.txt` pinning `gpt-researcher==0.14.8` + declared closure; add early `find_spec("gpt_researcher")` loud guard in run_memo.py BEFORE `_embedding_preflight()`; text-assert test on run_nightly.sh | run_memo.py:70,201-204; run_nightly.sh:43-72 | index.dev (src #7) loud-fail |
| deps-09 riders | add `numpy==2.4.4`, `PyYAML==6.0.3`, `pytest==9.0.3`, `python-dateutil==2.9.0.post0`, `google-cloud-storage==3.10.1` to requirements.txt (all installed; resolution unchanged) | numpy x39, yaml x7, dateutil x1, gcs @ compliance_logger.py:122 + earnings_tone.py:38,56 | PEP 508 (src #3) |
| deps-08 fpdf2 | delete `fpdf2>=2.7.0` (line 26) + `# PDF generation` header (line 25); leave NO `fpdf2` substring | requirements.txt:25-26; zero live imports; `Required-by:` EMPTY (orphan leaf) | — |
| deps-08 xlrd | ENHANCE existing comment (line 20) to name pandas `.xls` engine + `macro_regime.py:154` | macro_regime.py:59 (`.xls`), :154 (`read_excel`) | — |

### Vacuous-guard analysis (does the verification command BITE?)
The command is a pure TEXT `assert` chain. Mutations that make it FAIL (proves it bites):
1. Truncate lock to <150 `==` lines -> `len(pins)>=150` fails. 2. Drop the exchange line from the
lock -> `any(startswith('exchange'))` fails. 3. Omit `exchange-calendars` from requirements.txt ->
big assert fails. 4. Leave ANY `fpdf2` substring (incl. a comment) -> `'fpdf2' not in r` fails.
5. Write `pyyaml` lowercase -> `'PyYAML' in r` fails (case-sensitive). 6. Skip the autoresearch
manifest -> `os.path.exists` fails. 7. Unpinned `gpt-researcher` -> `'gpt-researcher==' in ...`
fails. 8. Don't wire the lock into pip-audit.yml -> `'requirements.lock' in y` fails.

**WEAKNESSES (measure-don't-assert; flag to executor + Q/A so a green isn't over-trusted):**
- `len(pins)>=150` counts ANY line containing `==`, INCLUDING comments (`# foo==bar`). A padded/
  malformed lock could pass. MEASURED: a real freeze has 303 real pins + 0 comment-`==` lines, so
  moot here — but the guard proves "150 lines with `==`", not "150 valid pinned packages."
- The `'<name>' in r` checks prove a SUBSTRING EXISTS, NOT that a valid pinned *requirement line*
  exists — a COMMENT like `# numpy pulled transitively` would satisfy `'numpy' in r`. Executor MUST
  add real install lines (not comment mentions); Q/A should eyeball each as an actual requirement.
- `'requirements.lock' in y` passes on a mere COMMENT in the yml (`# TODO: lock`). Executor must
  add the real `--requirement backend/requirements.lock` invocation; Q/A verifies it's in the run
  step, not a comment.
- The command NEVER checks: pytest declaration (criterion #3 wants it, command omits it; also
  `'pytest'` is already a substring of `pytest-cov`), the run_nightly.sh loud-fail seam (criterion
  #4 wants a text-assert; command omits it), the header/regen comment on the lock, and the
  before==after `pip freeze` boundary. These are CRITERIA-gated (Q/A must check by eye), not
  command-gated. A green command is necessary, NOT sufficient.

### BOUNDARY check (all inert against the live venv)
- **pip-audit.yml edit**: YAML text only; runs GitHub-side; inert locally. OK.
- **fpdf2 removal**: declaration-only. `fpdf2==2.8.7` stays INSTALLED (`Required-by:` empty -> no
  reverse deps break). Editing requirements.txt does NOT uninstall. OK.
- **No pip install/uninstall**: the step only WRITES files. `pip freeze > lock` READS the venv.
  Success-criterion #5 requires `pip freeze` before==after in experiment_results.md -> will be
  identical since nothing is installed. OK.
- **No `.env` edits**: run_nightly.sh SOURCES .env but the deps-02 change is to run_memo.py's
  import guard + a new manifest, not .env. OK.
- **Fresh-install dry-check (criterion #6)**: document (do NOT run) `uv pip sync
  backend/requirements.lock` or `pip install -r backend/requirements.lock`.

### Queue-candidates (OUT OF 75.13 SCOPE -- do NOT expand; file as future step)
import-vs-declared diff surfaced undeclared third-party top-level imports beyond the 75.13 set:
`torch`(x4), `transformers`(x1), `statsmodels`(x1), `fredapi`(x1), `voyageai`(x1), `timesfm`(x1),
`chronos`(x1), `vaderSentiment`(x1). Several are likely behind optional try/except guards
(experimental forecasting). A FUTURE dependency-completeness sweep should classify each
(declare / guard / remove); 75.13 must NOT touch them. (`alpaca`, `starlette` are already covered
transitively/by alpaca-py+fastapi.)

### Research Gate Checklist
Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (>=17 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered every relevant module (lock, requirements, pip-audit.yml, 3
  exchange sites, WORM writer, GPR/xlrd, nightly memo path, run_nightly.sh)
- [x] Contradictions/consensus noted (freeze-vs-pip-compile debate; moot for single-machine)
- [x] Claims cited per-claim

## step_text_corrections
1. Import path: node says `autoresearch/monthly_champion_challenger.py:327`; actual is
   **`backend/autoresearch/monthly_champion_challenger.py:327`** (missing `backend/` prefix).
2. deps-02 loud-fail: the gpt_researcher PACKAGE ImportError is ALREADY loud (run_memo.py broad
   `except`->return 1 -> run_nightly.sh 75.11 logs FAIL + pages after 3). The real SILENT dry-up is
   the phase-51.4 `_embedding_preflight()` exit-0 soft-skip (langchain_huggingface), which is an
   INTENTIONAL away-ops design -- do NOT revert it. Scope the loud-fail to a new
   `find_spec("gpt_researcher")` guard placed BEFORE the preflight (composes with 51.4 + 75.11).
3. GOTCHA #1: pip freeze emits `exchange_calendars` (UNDERSCORE); the verification asserts the
   HYPHEN `exchange-calendars` in requirements.txt. Functionally identical to pip (name-norm), but
   the executor MUST write the hyphen form in requirements.txt or the assert fails.
4. `PyYAML` assert is CASE-SENSITIVE (`pyyaml` fails); `fpdf2 not in r` is WHOLE-FILE (no comment
   residue allowed). GOTCHA #2/#3.
5. xlrd is ALREADY declared + commented (line 20); the ask is to ENHANCE the comment (name the
   pandas .xls engine + macro_regime.py:154), not add a missing line. `'xlrd' in r` already passes.
6. pytest is criterion-required but NOT command-asserted (and `'pytest'` is already a substring of
   `pytest-cov`); add `pytest==9.0.3` explicitly for correctness anyway.

## key_findings
1. Live venv MEASURED: `pip freeze` = **303 lines** (>=150 floor cleared 2x); clean freeze (no
   `-e`/`git+`/`@ file` quirk). Named versions: pandas 3.0.1, yfinance 1.2.0, numpy 2.4.4,
   exchange_calendars 4.13.2, gpt-researcher 0.14.8, fpdf2 2.8.7, xlrd 2.0.2, PyYAML 6.0.3,
   pytest 9.0.3, python-dateutil 2.9.0.post0, google-cloud-storage 3.10.1.
2. All 3 exchange_calendars import sites confirmed WITH silent-degrade fallbacks
   (markets.py fail-open returns True; champion/challenger drops holiday-shift handling;
   drill uses hardcoded-2026 holidays). google-cloud-storage unguarded at compliance_logger.py:122
   (WORM writer) + earnings_tone.py:38,56. xlrd is the pandas `.xls` engine for the `.xls` GPR
   cache (macro_regime.py:59,154). fpdf2 = true orphan (0 live imports; `Required-by:` EMPTY).
3. The deps-02 loud-fail must COMPOSE with 75.11's paging seam + 51.4's intentional embedding
   soft-skip; the gpt_researcher import is already loud, the embedding preflight is intentionally
   silent -- add a namesake `find_spec` guard, don't revert 51.4.
4. pip's own docs endorse freeze-as-lock; Semgrep 2026 gives the audited==deployed rationale;
   name-normalization spec proves the underscore/hyphen equivalence -> the verification's text
   asserts have real weaknesses (substring-not-line, comment-counts-as-pin) the executor must not
   game and Q/A must eyeball.
5. Boundary is fully inert: every change is a file WRITE or a venv READ; fpdf2 removal is
   declaration-only (installed, no reverse deps); no install/uninstall; no `.env` edit.

## plan_recommendations
1. Generate the lock via `.venv/bin/pip freeze > backend/requirements.lock`, then PREPEND a header
   comment block (regen cmd + `uv pip sync`/`pip install -r` sync path + "generated from live venv
   YYYY-MM-DD, N pins"). Header must be `#`-comments so it doesn't perturb the `==`-pin count.
2. In requirements.txt: write `exchange-calendars==4.13.2` (HYPHEN, pinned), `numpy==2.4.4`,
   `PyYAML==6.0.3` (CAPS), `pytest==9.0.3`, `python-dateutil==2.9.0.post0`,
   `google-cloud-storage==3.10.1`; DELETE line 25-26 (`# PDF generation` + `fpdf2>=2.7.0`) leaving
   no `fpdf2` residue; ENHANCE xlrd's line-20 comment to name the pandas .xls engine +
   macro_regime.py:154. Use REAL requirement lines (not comments) so the substring asserts reflect
   actual declarations.
3. pip-audit.yml: add `--requirement backend/requirements.lock` to the run step (line 48-50) AND
   add `backend/requirements.lock` to the `push`/`pull_request` `paths` filters so a lock change
   re-triggers the audit; audit the LOCK (deployed graph). Real invocation, not a comment.
4. Create `scripts/autoresearch/requirements-autoresearch.txt` pinning `gpt-researcher==0.14.8`
   plus its declared closure (at minimum langchain-huggingface + sentence-transformers, the default
   EMBEDDING backend the preflight checks); add an early `importlib.util.find_spec("gpt_researcher")
   is None -> print("[autoresearch] FAIL: gpt_researcher not importable") + return 1` guard in
   run_memo.py BEFORE `_embedding_preflight()`; add a text-assert pytest that greps run_nightly.sh
   for the `FAIL rc` log + the page seam (do NOT duplicate/alter the 75.11 seam).
5. experiment_results.md MUST record `pip freeze` before==after (identical), the verbatim
   verification-command output (exit 0), and the documented (unexecuted) fresh-install dry-check.
6. Q/A must eyeball the measure-don't-assert weaknesses: real requirement lines (not comment
   mentions), real pip-audit invocation (not comment), pytest actually declared, and the
   run_nightly.sh loud-fail seam present -- the command's green is necessary but NOT sufficient.

## Output JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 9,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "75.13 is a low-risk, file-only dependency-integrity step whose boundary is fully inert against the live venv (303-pin freeze measured; >=150 cleared 2x; fpdf2 is an installed orphan so its removal is declaration-only). All import claims verified: 3 exchange_calendars silent-degrade sites, unguarded google-cloud-storage in the WORM writer, xlrd as the pandas .xls engine. Key executor gotchas: pip freeze emits exchange_calendars (underscore) but the assert wants the hyphen form in requirements.txt; PyYAML is case-sensitive; fpdf2-not-in is whole-file. The deps-02 loud-fail must COMPOSE with 75.11 paging + 51.4's intentional embedding soft-skip -- gpt_researcher import is already loud; add a namesake find_spec guard, don't revert 51.4. The verification is a text-assert chain with real measure-don't-assert weaknesses (substring-not-line; comment-counts-as-pin; pytest + loud-seam uncovered) that Q/A must eyeball. pip docs bless freeze-as-lock; Semgrep 2026 gives the audited==deployed rationale.",
  "brief_path": "handoff/current/research_brief_75.13.md",
  "gate_passed": true
}
```
