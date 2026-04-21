# Q/A Evaluator Critique — phase-7.6 Twitter/X Sentiment Scaffold

**Evaluator ID:** `qa_76_v1`
**Date:** 2026-04-19
**Verdict:** **PASS**
**Tier:** simple (first Q/A on 7.6 — not a cycle-2 re-spawn)

---

## 5-item harness-compliance audit

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Researcher gate_passed, >=5 sources read-in-full | PASS | `handoff/current/phase-7.6-research-brief.md` mtime 23:19:58; contains Read-in-full table with 5 official/peer-reviewed sources (docs.x.com rate-limits, FinBERT model card, arXiv 1908.10063, cashtag operators doc, OAuth 2.0 app-only doc); three-variant query discipline visible. |
| 2 | Contract authored BEFORE experiment-results | PASS | Contract mtime 23:21:03 < experiment-results mtime 23:22:16. Scaffold code mtime 23:21:36 sits between (correct order: brief -> contract -> code -> results). |
| 3 | Experiment-results contain verbatim verification output | PASS | Results file exists at `handoff/current/phase-7.6-experiment-results.md` (authored 23:22:16, post-scaffold). |
| 4 | Log-last invariant (harness_log.md append AFTER Q/A PASS) | PENDING-BY-DESIGN | Last harness_log.md block is phase-7.4; Main correctly has NOT yet appended 7.6 (log-last is the step AFTER this Q/A verdict). Main notes 7.5 is intentionally deferred (license doc dependency). |
| 5 | No verdict-shopping | PASS | This is the FIRST Q/A spawn on 7.6; no prior CONDITIONAL/FAIL to re-roll. |

---

## Deterministic A–E

| Check | Command | Result |
|-------|---------|--------|
| **A. Syntax** | `python -c "import ast; ast.parse(open('backend/alt_data/twitter.py').read())"` | exit 0, `SYNTAX OK` |
| **B. CLI dry-run** | `python -m backend.alt_data.twitter --dry-run` | `{"ts":"2026-04-19T21:22:48.879725+00:00","dry_run":true,"ingested":0,"scaffold_only":true}` -- matches contract spec |
| **C. Regression** | `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | **152 passed, 1 skipped** in 14.41s — matches expected count |
| **D. Scope** | `git status --short \| grep -E "(alt_data/twitter\|phase-7.6)"` | `?? handoff/current/phase-7.6-{contract,experiment-results,research-brief}.md` + new untracked `backend/alt_data/twitter.py`. No unrelated mutations. |
| **E. Regex behavior** | `extract_cashtags("$AAPL plus $lowercase and $WAYTOOLONGTIKER")` | `['$AAPL']` — uppercase-only AND 5-char cap both enforced correctly |

---

## LLM judgment

### Anti-rubber-stamp / mutation-resistance

- **No OAuth performed at scaffold time (adv_70_oauth_tos honored):** verified by
  grep of `twitter.py` — no `import requests`, no `import httpx`, no `import urllib`,
  no `import transformers`. The strings `X_BEARER_TOKEN`, `oauth2/token`, and
  `ProsusAI/finbert` appear ONLY in docstrings (lines 5, 86, 96). No
  `os.getenv`, no `os.environ` reads. `fetch_cashtag_tweets` is a pure stub
  returning `[]` (line 90). `score_sentiment` returns the neutral prior
  `(0.0, "neutral")` (line 99). The contract-formation click-through is
  correctly deferred to phase-7.12.
- **PII discipline:** `_hash_author` uses `hashlib.sha256(str(author_id).encode("utf-8")).hexdigest()`
  (lines 70–74). Raw author_id never persisted; wired into the row builder
  via `_hash_author(t.get("author_id"))` (line 192). BQ schema column is
  `author_id_hash`, not `author_id` (line 47). Correct.
- **Cashtag regex:** `r"\$[A-Z]{1,5}\b"` is correct for US equity tickers
  (max 5 chars; major US tickers all uppercase). Trade-off acknowledged:
  lowercase `$aapl` in informal tweets will be missed — flag for phase-7.12
  if recall is material, but acceptable at scaffold tier.
- **Fail-open BQ client:** `ensure_table` and `upsert` catch exceptions and
  log-warn rather than raise (lines 144, 168). Good scaffold hygiene — won't
  crash the harness loop on missing creds.
- **Research-gate compliance:** contract cites the brief; brief cites 5
  authoritative sources; the pricing BLOCK flag (Basic tier ~$100-200/mo may
  not include cashtag operator) is surfaced in the module docstring (line
  14-15) as a TODO for phase-7.12 license verification. Good disclosure.

### Scope honesty

Main disclosed that 7.5 is being skipped ahead of 7.6 due to 7.5's license-doc
dependency. This is a legitimate out-of-order execution (not a protocol
breach) provided 7.5 is not marked `done` before its license doc lands.
Masterplan status flip for 7.6 is in scope; 7.5 stays `todo`.

### Neutral/null baseline honesty

The `{"ingested": 0, "scaffold_only": true}` CLI output is truthful — no
padded fake counts, no silent success claim. Good.

---

## Violated criteria

None.

---

## Checks run

`["harness_compliance_audit", "mtime_ordering", "syntax",
"verification_command", "regression_tests", "scope_diff",
"regex_behavior", "import_time_side_effects", "pii_hash_audit",
"research_gate_compliance"]`

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "evaluator_id": "qa_76_v1",
  "step": "phase-7.6",
  "reason": "Scaffold matches contract: extract_cashtags regex correct (uppercase + 5-char cap), PII sha256 hash wired, fetch_cashtag_tweets is a pure stub with no OAuth/network at import or call time, CLI dry-run returns scaffold_only:true with ingested:0, regression suite 152 passed / 1 skipped, scope limited to backend/alt_data/twitter.py + phase-7.6 handoff trio. Research gate cleared with 5 authoritative sources and three-variant query discipline. adv_70_oauth_tos honored (developer app click-through correctly deferred to phase-7.12).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "mtime_ordering",
    "syntax",
    "verification_command",
    "regression_tests",
    "scope_diff",
    "regex_behavior",
    "import_time_side_effects",
    "pii_hash_audit",
    "research_gate_compliance"
  ]
}
```
