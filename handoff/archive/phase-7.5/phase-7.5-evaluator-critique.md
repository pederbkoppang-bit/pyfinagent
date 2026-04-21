# Q/A Critique — phase-7 / 7.5 (Reddit WSB scaffold + license doc)

**Verdict:** PASS
**Q/A id:** qa_75_v1
**Date:** 2026-04-20 (cycle 1, first Q/A on 7.5)
**Reviewer:** Q/A agent (merged qa-evaluator + harness-verifier)

---

## 1. Harness-compliance audit (5-item)

| # | Check | Result |
|---|---|---|
| 1 | `phase-7.5-research-brief.md` present; floor (>=5 sources read in full), three-variant query discipline, recency scan, `gate_passed: true` | PASS |
| 2 | Contract mtime (00:15) < experiment-results mtime (00:17) — contract authored before GENERATE | PASS |
| 3 | Experiment results include verbatim verification commands + dual-criterion table + scope caveats | PASS |
| 4 | Log-last honored: last `handoff/harness_log.md` cycle block is phase-5-crypto-removal (23:52 UTC 2026-04-19); NO 7.5 entry yet, as required (log is appended AFTER Q/A PASS, BEFORE masterplan status flip) | PASS |
| 5 | First Q/A on 7.5 — no verdict-shopping risk | PASS |

Protocol audit: 5/5.

## 2. Deterministic checks (A–G)

| Tag | Check | Command | Observed | Verdict |
|---|---|---|---|---|
| A | Syntax | `python -c "import ast; ast.parse(open('backend/alt_data/reddit_wsb.py').read())"` | `SYNTAX OK`, exit 0 | PASS |
| B | License file exists | `test -f docs/compliance/reddit-license.md` | exit 0 | PASS |
| C | Dry-run scaffold | `python -m backend.alt_data.reddit_wsb --dry-run` | `{"ts":"2026-04-19T22:17:47...","dry_run":true,"ingested":0,"scaffold_only":true}` | PASS |
| D | Regression suite | `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | `152 passed, 1 skipped, 1 warning in 12.41s` | PASS |
| E | ASCII decode on both new files | `open(...).read().decode('ascii')` | `ASCII OK backend/alt_data/reddit_wsb.py` + `ASCII OK docs/compliance/reddit-license.md` | PASS |
| F | Scope: only 2 new files + handoff trio | `git status --short` shows `backend/alt_data/reddit_wsb.py` (new), `docs/compliance/reddit-license.md` (new), plus handoff/current/phase-7.5-{contract,experiment-results,research-brief}.md | PASS |
| G | Cashtag regex 2-char floor | `extract_cashtags("Long $AAPL and $NVDA but not $A or $I")` | `['$AAPL', '$NVDA']` exactly | PASS |

Deterministic: 7/7.

Non-blocking observation on D: the root-level `tests/` directory has 6 pre-existing collection errors (`test_deduplication`, `test_end_to_end`, `test_ingestion`, `test_queue_processor`, `test_response_delivery`, `test_tickets_db`) unrelated to phase-7.5. The contract's regression scope is `backend/tests/` which is green.

## 3. LLM judgment

| # | Item | Evidence | Verdict |
|---|---|---|---|
| 1 | No `import praw` / `import os` / `from os ...` at module top of `reddit_wsb.py` (scaffold discipline) | grep `^(import praw|import os|from os)` in reddit_wsb.py → zero matches. Imports are `argparse, hashlib, json, logging, re, sys`, `datetime`, `pathlib.Path`, `typing`. | PASS |
| 2 | User-Agent is Reddit format, NOT SEC EDGAR | `_USER_AGENT = "python:pyfinagent:1.0 (by /u/pederbkoppang)"` — matches Reddit spec `<platform>:<app>:<version> (by /u/<username>)`; distinct from SEC EDGAR `Name email` form used in twitter.py | PASS |
| 3 | License doc has 8 sections matching brief spec | Headings: 1 Scope, 2 Access method, 3 Terms-of-Service record, 4 Rate limits, 5 Retention & PII, 6 Permitted use, 7 Review cadence, 8 References | PASS |
| 4 | §3 ToS record tracks RBP as pending | Section 3 contains a status table with Reddit-app-registration / RBP-submission / RBP-approval / first-live-call all marked `pending` or `not started`, and states "Before any live call, the above gates are completed" | PASS |
| 5 | Rate cap `_RATE_INTERVAL_S = 0.6` matches 100-QPM free tier | Source: `_RATE_INTERVAL_S = 0.6  # 100 QPM free-tier ceiling` (100 requests / 60 s = 1.667 rps → 0.6 s interval). Correct. | PASS |
| 6 | DDL has 14 columns | `_CREATE_TABLE_SQL` columns: post_id, as_of_date, subreddit, author_hash, cashtag, title, text, sentiment_score, sentiment_label, score, upvote_ratio, created_at, source, raw_payload — 14. Partition `as_of_date`, cluster `subreddit, cashtag`. | PASS |
| 7 | PII: `_hash_author('[deleted]')` returns `None` (not a hash of the literal) | `python -c "from backend.alt_data.reddit_wsb import _hash_author; print(_hash_author('[deleted]'))"` → `None`. Correct handling of Reddit's tombstone. | PASS |

LLM judgment: 7/7.

## 4. Contract alignment

| Immutable criterion | Status |
|---|---|
| `ast.parse(backend/alt_data/reddit_wsb.py)` exits 0 | PASS (det. A) |
| `test -f docs/compliance/reddit-license.md` exits 0 | PASS (det. B) |

Both immutable criteria satisfied. No amendment of criteria observed.

## 5. Scope honesty

Experiment results explicitly disclose:
- Scaffold only; PRAW + RBP + live calls deferred to phase-7.12.
- `adv_70_oauth_tos` advisory honored — no OAuth click-through this cycle.
- Pushshift defunct; Arctic Shift documented as successor (from research brief).
- License doc established as template for 7.7 (Revelio) and 7.10 (LinkUp).

Scope statement is honest and bounded.

## 6. Violations

None.

## 7. JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_75_v1",
  "reason": "All 2 immutable criteria met; 7/7 deterministic checks (A-G) green; 7/7 LLM-judgment items green; 5/5 harness-compliance audit green. Scaffold discipline (no PRAW/os imports), Reddit-format User-Agent, 14-col DDL, RBP pending in ToS record, PII tombstone handled, 2-char cashtag floor verified.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "file_existence",
    "dry_run_scaffold",
    "regression_pytest_backend_tests",
    "ascii_decode",
    "scope_git_status",
    "cashtag_regex_behavior",
    "import_discipline_grep",
    "user_agent_format",
    "license_doc_section_count",
    "tos_record_rbp_pending",
    "rate_interval_vs_qpm",
    "ddl_column_count",
    "pii_tombstone_none",
    "harness_compliance_5_item",
    "contract_mtime_before_results"
  ]
}
```
