# live_check 76.9 — verbatim live evidence (Main first-hand, 2026-07-24)

## 1. BEFORE baseline (captured before any 76.9 work, this session)

`launchctl list | grep pyfinagent` (excerpt, verbatim):

```
-	0	com.pyfinagent.ablation
-	1	com.pyfinagent.autoresearch      <-- exit 1
```

Autoresearch failed again THIS morning — `handoff/logs/autoresearch-v4.log` tail (verbatim):

```
  File ".venv/lib/python3.14/site-packages/arxiv/__init__.py", line 732, in __try_parse_feed
    raise HTTPError(url, try_index, resp.status_code)
arxiv.HTTPError: Page request resulted in HTTP 429 (https://export.arxiv.org/api/query?search_query=What+does+the+literature+%282025-2026%29+say+about+news+sentiment+alpha+decay...)
[autoresearch] FAILED -- wrote /Users/ford/.openclaw/workspace/pyfinagent/handoff/autoresearch/2026-07-24-ERROR-topic09.md
[2026-07-24T02:02:19+02:00] END nightly autoresearch FAIL rc=1
```

Ablation crash — `handoff/logs/ablation.launchd-v4.log` (verbatim; NOTE the step text's
`handoff/ablation.launchd.log` path is STALE, logrotate renamed it):

```
backend/.env: line 81: unexpected EOF while looking for matching `"'
backend/.env: line 86: syntax error: unexpected end of file
```

## 2. ABLATION — LIVE launchd re-run exits 0 (criterion 2 live arm)

Plist provenance verified: ProgramArguments -> `scripts/ops/run_ablation.sh` (75.11
sanitize wrapper), installed via OPS-ROTATE-BOOTSTRAP leg 3 (harness_log ops addendum
2026-07-24 ~07:15 UTC, operator-attended; backup `.pre-75.11.bak` kept).

`launchctl kickstart gui/$(id -u)/com.pyfinagent.ablation` issued 13:15 local; result
(`handoff/logs/ablation.log`, verbatim):

```
[2026-07-24T13:15:51+02:00] START ablation
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
all-features-tested
[2026-07-24T13:15:53+02:00] END ablation OK
```

`launchctl list | grep ablation` after (verbatim): `-	0	com.pyfinagent.ablation`

The pre-fix job died on the `.env` source BEFORE writing a single log line (0-byte
StandardOut); this run sourced the malformed file through the sanitize, logged
START/END, exited 0. (`all-features-tested` is correct `--next-untested` behavior:
every feature already carries a TSV row from historical batch runs.)

## 3. Sourcing seam BEFORE/AFTER pair against the REAL backend/.env (verbatim)

```
$ bash -c '. backend/.env'    # the old plist's raw behavior
backend/.env: line 81: unexpected EOF while looking for matching `"'
backend/.env: line 86: syntax error: unexpected end of file
raw-source rc=1

$ # run_ablation.sh's verbatim sanitize block (mktemp + grep KEY= + source)
sanitized-source rc=0
```

## 4. Malformed backend/.env line — reported, NOT edited (criterion 3)

Read READ-ONLY by Main (comment text, no secret values; the researcher subagent is
sandbox-denied this file by design and received structure only):

```
L80: # phase-61.1 (2026-06-12): operator tokens "60.2 FLAG: ON" / "60.3 FLAG: ON" / "57.1 FLAG:
L81:   ON"
```

L81 is the hard-wrapped orphan tail of L80's comment (one unbalanced `"`), a non-KEY=
line — exactly the shape the sanitize drops. OPERATOR REPAIR (one line, operator-gated):
rejoin `  ON"` into L80, or prefix L81 with `#`. Full report in experiment_results.md.

## 5. AUTORESEARCH — mocked-429 arm (criterion 1) + kickstart decision

Immutable command (verbatim): `bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"` → exit 0.

New suite `backend/tests/test_phase_76_9_launchd_fixes.py` → `9 passed` (verbatim outputs
in experiment_results.md), including t_429_warn_exit0 driving the REAL `_main_async` with
a REAL `arxiv.HTTPError(url, 3, 429)` → rc 0 + WARN memo + no ERROR memo.

**Midday kickstart of com.pyfinagent.autoresearch DECLINED, with rationale**: a real
memo run costs ~$1–3 metered LLM spend (run_memo docstring), and LLM API costs require
operator approval per CLAUDE.md; tonight's operator-sanctioned 02:00 cron exercises the
identical path for free. The step criterion's mocked arm is fully satisfied; tonight's
run is the natural real-cron evidence. (Residual risk noted honestly: if the Anthropic
API credits from the phase-72 ACT-NOW token are still unfunded, tonight's run will fail
at the LLM call — that is the pre-existing phase-72 operator token, NOT a 76.9
regression; the arxiv-429 crash class this step fixes is covered by the mocked arm +
the classifier tests either way.)

## 6. Mutation matrix — 5 mutations, 5 killed, 0 survivors (verbatim results)

Pre/post SHA-256 of all three mutated files IDENTICAL
(63e48f0b… run_memo.py / c732be82… run_ablation.sh / bb009e3b… test file);
full suite re-run post-restore: `9 passed`.

| # | Mutation (applied to the REAL file, executed) | Killed by (verbatim result) |
|---|---|---|
| M1 | WARN fall-through disabled (`if _is_network_weather(e):` → `if False:`) | `FAILED ...::test_t_429_warn_exit0` — `1 failed` |
| M2 | RETRIEVER reverted to `arxiv,semantic_scholar,duckduckgo` | `FAILED ...::test_t_retriever_order` — `1 failed` |
| M3 | run_ablation.sh sanitize bypassed (`grep -E '^KEY='…` → `cat` raw) | `FAILED ...::test_t_ablation_fixture_survives_bad_env` — `1 failed` |
| M4 | classifier widened to catch-all (`return True` first line) | `FAILED ...::test_t_real_fault_exit1` + `...classifier_rejects_real_faults` — `2 failed` |
| M5 | **STUB/fixture**: fixture .env quote-BALANCED (`  ON"` → `  ON`) | `FAILED ...::test_t_ablation_fixture_survives_bad_env` — the reproduce-guard refused the vacuous fixture — `1 failed` |

Matrix sequenced AFTER the delegated executor completed (feedback_executor_sees_mutation_transients).

## 7. git diff --stat (step-scoped, quant-agent residue excluded — committed separately)

```
 backend/tests/test_phase_76_9_launchd_fixes.py   | 242 ++++++++++++++++ (new)
 handoff/autoresearch/root_cause.md               | 109 ++++++++ (append-only, 0 deletions)
 handoff/current/contract.md                      | 140 +++++++---
 handoff/current/experiment_results.md            | 317 +++++++++++++++++------
 scripts/autoresearch/run_memo.py                 |  76 +++++-
```

`run_nightly.sh` + `run_ablation.sh`: ZERO hunks (boundaries held). Unrelated tree
changes at commit time: backend quant-optimizer session 60617e0b residue
(optimizer_best.json, quant_results.tsv, exp10 artifact — 20-min cadence intraday runs,
same class as commit 64fc644a) — separate chore commit.
