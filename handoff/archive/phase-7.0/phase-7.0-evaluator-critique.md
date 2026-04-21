# Q/A Evaluator Critique -- phase-7 / step 7.0

**Agent:** qa (merged qa-evaluator + harness-verifier)
**Cycle:** 1
**Date:** 2026-04-19
**Verdict ID:** qa_70_v1

---

## 5-item protocol audit

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Researcher spawn proof | PASS | `handoff/current/phase-7.0-research-brief.md` present (29,089 bytes). |
| 2 | Contract PRE-commit | PASS | `phase-7.0-contract.md` mtime 22:15 < `phase-7.0-experiment-results.md` mtime 22:18. |
| 3 | Experiment results complete | PASS | Verbatim 4-assertion output + regression 152/1 + mid-cycle Sec. replacement disclosed. |
| 4 | Log-last discipline | PASS | `handoff/harness_log.md` last block is phase-6.5.9 + phase-6.5 closure; NO 7.0 entry yet. |
| 5 | No verdict-shopping | PASS | First Q/A on 7.0. |

---

## Deterministic checks A-G

### A. Immutable assertions

```
$ test -f docs/compliance/alt-data.md                      -> exit 0
$ grep -q 'Van Buren' docs/compliance/alt-data.md          -> exit 0
$ grep -q 'hiQ' docs/compliance/alt-data.md                -> exit 0
$ grep -q 'X Corp' docs/compliance/alt-data.md             -> exit 0
```

All 4 PASS.

### B. Substance (not just bare grep tokens)

- **Van Buren** (Section 3, lines 110-119): Full caption `593 U.S. ___, 141 S.Ct. 1648 (2021)`, holding on "exceeds authorized access" under 18 U.S.C. Sec.1030(e)(6), 6-3 majority by Justice Barrett, plus concrete relevance sentence ("scraping public URLs does not fall under Sec.1030 at all"). NOT a bare token. PASS.
- **hiQ** (lines 121-132): Full caption `31 F.4th 1180 (9th Cir. 2022)`, on-remand-from-SCOTUS holding, December 2022 settlement ($500K + permanent injunction + data deletion), plus "never scrape LinkedIn" operational directive. NOT a bare token. PASS.
- **X Corp** (lines 134-144): Full caption `No. 3:23-cv-03698-WHA (N.D. Cal. May 9, 2024)`, Judge Alsup named, Copyright-Act-preemption holding, plus "single-district ruling; we treat it as guidance, not binding precedent" honest caveat. NOT a bare token. PASS.

### C. ASCII discipline

```
$ python3 -c "open('docs/compliance/alt-data.md','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK
$ grep -c 'Sec' docs/compliance/alt-data.md        -> 10 occurrences
$ python3 -c "print(chr(0xa7))" -- pattern search  -> zero matches
```

PASS. Initial draft had 16 `Sec.` (U+00A7) positions per the mid-cycle disclosure; re-check confirms zero remain.

### D. Regression

```
$ source .venv/bin/activate && pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 13.46s
```

PASS. Matches experiment_results' claim (7.0 adds no test targets; doc-only).

### E. Scope

```
$ git status --short docs/compliance/
?? docs/compliance/alt-data.md
```

Only new file is the compliance doc. The broader `git status --short` includes many unrelated M/D files (the pre-existing session state: `.claude/agents/harness-verifier.md D`, `.claude/settings.json M`, etc.) but none of those are caused by 7.0; they predate this cycle. Scope limited to doc + handoff as advertised. PASS.

### F. Bonus Phase 8 token (7.8 future-proof)

```
$ grep -q 'Phase 8' docs/compliance/alt-data.md  -> exit 0
```

PASS. Not a 7.0 immutable criterion; recorded for 7.8 pre-provisioning.

### G. Per-Source Policy Table coverage vs phase-7 masterplan steps

Masterplan phase-7 steps read from `.claude/masterplan.json`:
`7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12`.

Section 4 table rows: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8 (deferred), 7.9, 7.10, 7.11, 7.12 = 12 rows (correctly excludes 7.0 itself, which IS the compliance doc). 1:1 coverage of all ingestion/infra steps. PASS.

---

## LLM judgment

**Citation accuracy (spot-checked against general legal knowledge, not re-WebFetched):**
- Van Buren: 141 S.Ct. 1648 (2021), "exceeds authorized access" narrowing of 18 U.S.C. Sec.1030(e)(6), 6-3 majority by Justice Barrett -- matches the actual opinion. Grep-satisfiable, yes, but substantively grounded.
- hiQ: 31 F.4th 1180 (9th Cir. 2022), remand-from-SCOTUS ruling; the Dec-2022 settlement figure ($500K + permanent injunction) matches public reporting. The doc correctly distinguishes "not a CFAA liability, but a breach-of-contract outcome on the click-through ToS LinkedIn required" -- that's the right legal nuance.
- X Corp v. Bright Data: case number 3:23-cv-03698-WHA maps to N.D. Cal. and Judge William Alsup presided; the Copyright Act preemption reasoning (X as non-exclusive licensee) is the actual holding.

**Doc readability / actionability:** the per-source table (Section 4) is specific enough that a developer opening phase-7.5 (Reddit WSB) knows immediately that the access method is "Reddit Data API v1 with OAuth app key," rate limit is "API default," legal basis is "Reddit ToS 2024 permits non-commercial + small-volume research; commercial use requires paid enterprise tier," and the compliance owner is phase-7.5 itself. This is not boilerplate. Scraping Disciplines 5.1-5.6 are enforceable (User-Agent string is verbatim; rate-limit responsibility pushed to phase-7.11 infra layer; robots.txt + no-circumvention + PII hashing declared).

**Risk Register honesty:** rows are R1 low/high, R2 low/high, R3 medium/medium, R4 high/medium, R5 medium/medium. NOT evenly-sprinkled "medium/medium" rubber stamps -- R4 (stale doc) correctly high-likelihood, R1/R2 correctly low-probability-but-high-impact lawsuit scenarios. Plausible estimates. PASS.

**Anti-rubber-stamp -- Sec. replacement verification:** experiment_results discloses mid-cycle `Sec.` -> `Sec.` fix for 16 positions. Q/A independently confirms zero U+00A7 remain and 10 `Sec.` substitutions visible (lines 52, 72, 112, 118, 197, 244, 262, etc.). Clean disclosure matches reality. PASS.

**Scope honesty:** doc-only, regression unchanged at 152/1, no Python code touched. The experiment_results explicitly disclaims "PII hashing discipline is declared, not yet enforced" (caveat 5) -- honest scope framing, not overclaim. PASS.

### FLAG (CONDITIONAL-leaning but non-blocking): Reddit/X OAuth ToS contradiction

Section 2.2 states: "we never click 'I agree,' so there is no ToS contract
formation." Section 3.2 relevance-paragraph repeats: "we do not log into any
source, so there is no ToS contract formation." But Section 4 row 7.5
describes Reddit access as "Reddit Data API v1 with OAuth app key" and row
7.6 describes X/Twitter as "X API v2 with OAuth app key (paid tier for
volume)."

Obtaining an OAuth app key on either platform REQUIRES the developer to (a)
create a developer account (click-through ToS accepted), (b) create an app
registration (second click-through ToS accepted -- often the Data API
Developer Terms), and (c) on paid tiers, accept a commercial-use addendum.
That IS contract formation. The blanket Section 2.2 claim is therefore too
strong as written: it's accurate for pure-public-URL HTTP scraping (the
`X Corp` legal basis) but NOT accurate for the API-gated rows in Section 4
(rows 7.5, 7.6, and arguably 7.3 FINRA developer key, 7.9 pytrends via
Google terms).

This creates an internal contradiction between Section 2.2 (no ToS) and
Section 4 (OAuth app keys accepted for at least 2, possibly 4 rows).

**Severity assessment:** non-blocking for 7.0 close. The immutable
verification criteria for 7.0 do not require ToS-consistency audit; they
require file existence + three case-name tokens. But this will bite
phase-7.5 and 7.6 when those steps try to write "legal basis" verification
against the compliance doc. Recommendation: either (a) revise Section 2.2
to "we never click I agree for pure-HTTP scraping; API-gated sources
accept ToS at developer-account + app-registration, which IS contract
formation -- compliance for those sources is direct ToS compliance, not
absence-of-contract," and (b) add a column to Section 4 distinguishing
"contract formed via OAuth" from "public URL, no contract" rows. Doing
this before 7.5 ships saves a retro-amendment.

Logging as ADVISORY for phase-7.0 close, MUST-FIX before phase-7.5
(Reddit) GENERATE.

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria pass verbatim. Substantive citations (not bare grep tokens) for Van Buren (141 S.Ct. 1648), hiQ (31 F.4th 1180), X Corp (3:23-cv-03698-WHA). ASCII clean; regression 152/1 unchanged; doc-only scope; Per-Source Table 1:1 with phase-7 steps 7.1-7.12; Phase 8 bonus token present. Mid-cycle Sec. replacement independently verified.",
  "violated_criteria": [],
  "violation_details": [],
  "advisories": [
    {
      "id": "adv_70_oauth_tos",
      "severity": "non-blocking for 7.0; must-fix before 7.5 GENERATE",
      "finding": "Sections 2.2 and 3.2 assert 'no ToS contract formation (we never click I agree)' but Section 4 rows 7.5 (Reddit API OAuth), 7.6 (X API OAuth), and arguably 7.3 (FINRA developer key) and 7.9 (pytrends/Google terms) all require click-through ToS at developer-account and app-registration. Internal contradiction that needs a Section 2.2 revision + a Section 4 column distinguishing contract-formed-via-OAuth rows from public-URL-no-contract rows before 7.5 ships."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5_item",
    "immutable_assertions_verbatim",
    "substance_beyond_bare_tokens",
    "ascii_decode",
    "regression_pytest",
    "git_status_scope",
    "phase8_bonus_token",
    "masterplan_cross_link_G",
    "citation_accuracy_spot_check",
    "risk_register_honesty",
    "mid_cycle_fix_verification",
    "oauth_tos_internal_consistency"
  ]
}
```

---

## Final Decision

**PASS (qa_70_v1)** with one non-blocking advisory (Reddit/X OAuth ToS
contradiction between Sections 2.2 and 4 -- must be reconciled before
phase-7.5 GENERATE, but not a 7.0 blocker since 7.0's immutable criteria
are satisfied verbatim).

Orchestrator actions:
1. Append phase-7.0 cycle block to `handoff/harness_log.md` (log-last).
2. Flip `.claude/masterplan.json` step 7.0 status pending -> done.
3. Archive `handoff/current/phase-7.0-*` to `handoff/archive/phase-7.0/`
   (PostToolUse hook).
4. Carry advisory `adv_70_oauth_tos` into the phase-7.5 contract's
   preconditions.
