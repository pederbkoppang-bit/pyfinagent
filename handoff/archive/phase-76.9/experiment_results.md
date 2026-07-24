# Experiment results -- Step 76.9 (P1, operator bug B1: both nightly launchd jobs failing)

Date: 2026-07-24 | Cycle: 154 | Execution: DELEGATED sonnet executor GENERATE
(this report); Main review + mutation matrix + live checks pending after this
report is returned.

## What was built

Two independent P1 launchd-job crash fixes, per the contract's Plan items
1-3.

### 1. `scripts/autoresearch/run_memo.py` -- arxiv 429 tolerance

- **RETRIEVER reorder** (was `:211`, now the same line after the added
  comment block): `"arxiv,semantic_scholar,duckduckgo"` ->
  `"semantic_scholar,arxiv,duckduckgo"`. arxiv is no longer in the
  crash-prone `retrievers[0]` PLANNING slot (gpt-researcher upstream
  issue #1282: the planning call at `skills/researcher.py:62` is
  unguarded, while the sub-query fan-out is tolerant). A comment block
  documents the rationale + the 2026-02 arXiv 429 regression.
- **New pure classifier `_is_network_weather(e) -> bool`**, placed near
  the top of the file (after `slugify`, before `run_research`). Returns
  True iff:
  - `type(e).__name__ == "HTTPError"` and `type(e).__module__` starts
    with `"arxiv"`, OR
  - the exception (or any member of its `__cause__`/`__context__`
    chain) is a `ConnectionError` / `TimeoutError`, or (best-effort,
    guarded by `try/except ImportError`) a `requests.exceptions`
    `ConnectionError`/`Timeout` or an `urllib3.exceptions`
    `ConnectTimeoutError`/`ReadTimeoutError`/`NewConnectionError`/
    `MaxRetryError`, OR
  - `str(e)` (case-insensitive) contains `"429"`, `"503"`, or
    `"rate limit"`.
  No new dependency: `requests`/`urllib3` imports are optional and
  guarded; the arxiv-type check is by name/module string, not an
  `import arxiv` (keeps the classifier importable even without the
  `arxiv` package installed).
- **`_main_async`'s `except Exception as e` block (`:114`)** now
  branches on `_is_network_weather(e)`:
  - **True** -> writes `MEMO_DIR / f"{date}-WARN-topic{idx:02d}.md"`
    (title `# Autoresearch WARN (network) -- <date>`; body: topic,
    `Error: <type>: <msg>`, and the line "External retriever weather;
    run tolerated per phase-76.9, see
    handoff/autoresearch/root_cause.md"), prints
    `[autoresearch] WARN (network) -- wrote <path>`, and `return 0`.
    The filename never contains `-ERROR-`.
  - **False** -> the original ERROR-memo + `return 1` path,
    byte-unchanged (verified by diff -- only the branch above it
    changed; the else-path text is identical to the pre-fix code), so
    the phase-75.11 paging seam in `run_nightly.sh` still fires on real
    faults.
- **`write_memo` header string** updated to keep the retriever list in
  sync with the new order: `"arxiv + semantic_scholar + duckduckgo
  retrievers"` -> `"semantic_scholar + arxiv + duckduckgo retrievers"`.
- All new strings/log lines are ASCII-only (verified by inspection --
  no Unicode arrows/dashes were introduced).

`scripts/autoresearch/run_nightly.sh` was NOT touched (per contract
boundary; confirmed unchanged via `bash -n` below and a `git diff`
scoped to that path showing no hunks).

### 2. New test file: `backend/tests/test_phase_76_9_launchd_fixes.py`

Loads `run_memo.py` via `importlib.util.spec_from_file_location`
(matching the existing pattern in `test_phase_51_4_crons.py`). 9 tests:

- `test_t_429_warn_exit0` -- monkeypatches `run_memo.run_research` to
  raise a REAL `arxiv.HTTPError` built with the installed package's
  actual signature (`HTTPError(url: str, retry: int, status: int)`,
  verified against `.venv/lib/python3.14/site-packages/arxiv/__init__.py:820`
  before use), 429-shaped. `MEMO_DIR` monkeypatched to `tmp_path`.
  Drives the real `_main_async` via `asyncio.run` with a stub
  `argparse.Namespace(topic=..., topic_index=None)`. Asserts: return
  0, exactly one `*-WARN-topic*.md` file, zero `*-ERROR-*` files, WARN
  body names `HTTPError`.
- `test_t_real_fault_exit1` -- same harness, `run_research` raises
  `ValueError("boom")`. Asserts return 1, exactly one `*-ERROR-*`
  memo, zero WARN files, body contains `ValueError` and `boom`.
- `test_t_network_weather_classifier_matches_generic_network_errors`
  (parametrized) + `test_t_network_weather_classifier_rejects_real_faults`
  -- direct unit coverage of `_is_network_weather` for the
  connection/timeout-class and message-token branches, independent of
  the end-to-end harness (added beyond the contract's minimum for
  mutation-resistance on the classifier's non-arxiv branches).
- `test_t_retriever_order` -- asserts the literal
  `'"RETRIEVER": "semantic_scholar,arxiv,duckduckgo"'` string is
  present in `run_memo.py` and the old order string is absent, plus a
  behavioral list-index check (`retriever_list[0] ==
  "semantic_scholar"`, `!= "arxiv"`). **Deviation disclosed**: the
  contract asked for "a behavioral variant ... if extractable" of
  `env_defaults`; `env_defaults` is a local dict built inside `main()`,
  which does argument parsing, `sys.exit` on missing
  `ANTHROPIC_API_KEY`, `os.environ.setdefault` mutation, and (if not
  short-circuited) proceeds toward a real network call -- none of
  which is safely callable as a $0/no-network unit test without
  extensive monkeypatching well beyond the RETRIEVER assertion's
  scope. Per contract's own fallback clause ("if not cleanly
  extractable, keep the source-level assert and note it"), this test
  stays source-level for the *ordering* claim; the *behavioral*
  network-tolerance path (the part that actually matters at runtime)
  IS exercised end-to-end by `test_t_429_warn_exit0` /
  `test_t_real_fault_exit1` above, which do drive the real
  `_main_async`.
- `test_t_ablation_fixture_survives_bad_env` -- builds a fixture repo
  under `tmp_path` with `backend/.env` (a valid `KEY=value` line, a
  comment line, and the orphan `  ON"` line mirroring the real
  L80/L81 shape), a `.venv/bin/activate` stub that prepends a fixture
  `.venv/bin` (containing a `python` shim that execs
  `/usr/bin/python3`) onto `PATH` -- necessary because the real host
  has no bare `python` on `PATH` (only `python3`; confirmed via
  `which python` -> not found), a stub
  `scripts/ablation/run_ablation.py` (prints + `sys.exit(0)`), and
  `handoff/logs/` + `handoff/away_ops/` dirs. FIRST asserts the
  fixture reproduces the raw-source EOF failure (`bash -c '.
  <fixture>/backend/.env'` exits nonzero with "unexpected EOF" in
  stderr -- verified live, see below) so the fixture cannot go
  vacuous (feedback_mutation_test_guards_and_fixtures). THEN runs the
  REAL `scripts/ops/run_ablation.sh` with `SRE_OPS_REPO=<fixture>` and
  asserts exit 0, no "unexpected EOF" in combined stdout+stderr, and
  the fixture's `handoff/logs/ablation.log` contains both `START
  ablation` and `END ablation OK`.

### 3. Docs/state

- **`handoff/autoresearch/root_cause.md`**: the file already existed
  (from phase-39.1, tracked at commit `c65cc9c0`) as the historical
  record of the OLD `anthropic:`-prefix root cause. **Deviation
  disclosed**: the original task spec (this delegation's initial
  prompt) said to create this file NEW; mid-task, Main sent a
  correction identifying that it already exists and instructing an
  APPEND, not a replace/create, preserving the phase-39.1 content
  byte-identical. I complied: the phase-39.1 content (lines 1-115,
  `TL;DR` / `Trace` / `Fix` / `Why this stayed broken` /
  `Verification path` / `Operator action required`) is untouched, and
  I appended a new `## 2026-07-24 update (phase-76.9)` section
  covering (a) the huggingface embedding soft-skip window
  (~2026-06-08 onward, closed by phase-75.13's dep install), (b) the
  arxiv HTTP-429 chain (this step's fix), (c) the ablation `.env` EOF
  crash + the 75.11 sanitize + the 2026-07-24 bootstrap, including the
  verbatim `backend/.env` L80-81 quote block for the operator, and (d)
  a note that this carries forward superseded step 39.1's
  `root_cause_documented` intent. Net effect matches Main's correction
  exactly; see `git diff -- handoff/autoresearch/root_cause.md` for
  the literal appended diff (109 insertions, 0 deletions -- confirms
  nothing in the original file was altered).
- **This file** (`handoff/current/experiment_results.md`) -- step id,
  file list, verbatim command outputs, artifact shapes, and the
  OPERATOR REPORT section below.

## Files changed

- `scripts/autoresearch/run_memo.py` (modified -- RETRIEVER order,
  `_is_network_weather` classifier, WARN-memo branch, write_memo
  header string)
- `backend/tests/test_phase_76_9_launchd_fixes.py` (new)
- `handoff/autoresearch/root_cause.md` (modified -- added verbatim
  `backend/.env` L80-81 quote to the pre-existing phase-76.9 section;
  see deviation note above)
- `handoff/current/experiment_results.md` (this file)

Files explicitly NOT changed (contract boundaries):
`scripts/autoresearch/run_nightly.sh`, `scripts/ops/run_ablation.sh`,
`backend/.env`.

## Verbatim command outputs

### Immutable verification command

```
$ bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"
$ echo "exit=$?"
exit=0
```
(no output on success from either half; exit code captured separately
and confirmed 0)

### `bash -n` on the other changed/adjacent shell script (unchanged,
sanity-checked per contract's "bash -n on the changed shell scripts"
criterion -- run_ablation.sh was not edited but is exercised by the new
test, so it is re-verified here for completeness)

```
$ bash -n scripts/ops/run_ablation.sh
$ echo $?
0
```

### New test file run

```
$ .venv/bin/python -m pytest backend/tests/test_phase_76_9_launchd_fixes.py -q
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
.........                                                                [100%]
9 passed in 0.53s
```

### Regression sweep

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_sre_ops.py backend/tests/test_phase_39_1_autoresearch_env.py backend/tests/test_phase_75_deps.py -q
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
...F....................................                                 [100%]
=================================== FAILURES ===================================
__________________ test_c1_runbook_and_operator_token_drafted __________________
    def test_c1_runbook_and_operator_token_drafted():
>       runbook = _read("handoff/current/ops_rotate_runbook_75.11.md")
FileNotFoundError: [Errno 2] No such file or directory: '.../handoff/current/ops_rotate_runbook_75.11.md'
=========================== short test summary info ============================
FAILED backend/tests/test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
1 failed, 39 passed, 1 warning in 5.17s
```

**This failure is PRE-EXISTING and unrelated to 76.9's changes.** Root
cause: `handoff/current/ops_rotate_runbook_75.11.md` was moved by the
`archive-handoff` hook to
`handoff/archive/misc/ops_rotate_runbook_75.11.md` when a later step
closed (confirmed via `find handoff -iname "*ops_rotate_runbook*"` --
the only copy on disk is the archived one). `test_phase_75_sre_ops.py`
was not touched by this step (out of the contract's boundaries), and
this is not one of the 3 named regression files' relevant assertions
for 76.9's own criteria (it's `test_phase_75_sre_ops.py` criterion 1,
unrelated to the ablation/autoresearch criteria this step targets).
Flagging for Main -- the fix (repointing the test at the archived path,
or restoring the file) is out of this step's scope per the contract's
boundaries (no instruction to touch `test_phase_75_sre_ops.py`), and
the other 8 tests in that file (including the ablation-wrapper
criterion-3 tests, `test_c3_*`, which ARE directly relevant to 76.9's
CAUSE 2) all pass.

## Artifact shapes

- `handoff/autoresearch/<date>-WARN-topic<NN>.md` -- new memo shape,
  written only on network-class exception. Header `# Autoresearch WARN
  (network) -- <date>`, body: `Topic: <topic>`, `Error: <type>: <msg>`,
  and the tolerance-note line. Never matches `-ERROR-` (downstream
  counters unaffected).
- `handoff/autoresearch/<date>-ERROR-topic<NN>.md` -- unchanged shape
  (byte-identical construction to pre-fix code) for non-network faults.
- Fixture `handoff/logs/ablation.log` (test-only, under `tmp_path`) --
  matches the real `run_ablation.sh` log shape: `[<iso8601>] START
  ablation` / `[<iso8601>] END ablation OK`.

## OPERATOR REPORT -- backend/.env L80-81 (verbatim, comment text only, NOT edited by this step)

```
L80: # phase-61.1 (2026-06-12): operator tokens "60.2 FLAG: ON" / "60.3 FLAG: ON" / "57.1 FLAG:
L81:   ON"
```

**Repair recommendation** (one-line operator edit; `backend/.env` is
operator-gated and was NOT read or edited by this step beyond the
verbatim lines Main supplied first-hand): rejoin `  ON"` into L80's
comment as a single logical line, OR prefix L81 with `#` so it becomes
its own (harmless) comment line. Either form removes the unbalanced
quote that a raw `. backend/.env` source dies on; the ablation fix in
this step (proving `run_ablation.sh`'s sanitize survives the malformed
line) makes the crash non-fatal for the ablation cron regardless, but
the underlying `.env` line is still worth cleaning up since any future
script that raw-sources `backend/.env` would hit the same failure.

## Follow-up (cycle 2) -- 2026-07-24, Main -- CONDITIONAL blocker fixed

Cycle-1 Q/A (wf_bd5276e2-354) returned CONDITIONAL with a single blocker: qa.md
section-1a lint gate, F401 `sys` imported but unused at
backend/tests/test_phase_76_9_launchd_fixes.py:24 (new test file only; run_memo.py
clean). Fix applied: the dead `import sys` line deleted (one-line edit, no other change).

Re-verification (verbatim):

```
$ { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | xargs uvx ruff check --select F821,F401,F811
All checks passed!
lint exit=0
$ .venv/bin/python -m pytest backend/tests/test_phase_76_9_launchd_fixes.py -q
9 passed in 0.48s
$ bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"
exit=0
```

The lint was re-run via the Q/A-prescribed robust xargs-from-repo-root form (the
cycle-1 evaluator self-caught a zsh word-split false-pass on its first attempt --
shape #9 -- so the passing run above provably linted the actual changed-file set).
