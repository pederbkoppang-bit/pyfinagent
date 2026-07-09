# Research Brief — phase-67.2

**Consumer-contract-break heuristic (code-review-trading-domain SKILL.md) + fix live NameError in `parse_llm_classification` + behavioral test**

Tier: moderate. Gate: **PASSED** (6 sources read in full; recency scan done; all internal claims file:line-anchored).

---

## Step 67.2 immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `.claude/skills/code-review-trading-domain/SKILL.md` gains a consumer-contract-break heuristic (interface/CLI-flag/output-shape/schema change without every consumer verified; the Q/A greps consumers itself) with a severity rating and a negation list; every pre-existing heuristic name remains present (criteria-erosion guard).
2. `backend/agents/agent_definitions.py::parse_llm_classification` degrades gracefully on malformed input: the exception tuple references only imported names, and the documented repro now returns the default Main-routed `ClassificationResult` instead of raising NameError.
3. A behavioral test exercises the malformed-classification path (not tautological, not over-mocked) and passes in the venv.
4. Fresh Q/A PASS with the 67.1 gates applied to this diff (lint-gate output over the changed files included in the critique).

Verification command (IMMUTABLE): `bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_agent_definitions_classification.py -q -x --timeout=60 && grep -q "consumer-contract-break" .claude/skills/code-review-trading-domain/SKILL.md'`

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/agent_definitions.py` | 23-27 | Module imports = `dataclasses`, `enum`, `typing.Optional`, `resolve_model` only — **no `import json`** | ROOT CAUSE |
| `backend/agents/agent_definitions.py` | 352-402 | `parse_llm_classification()` — `from backend.utils import json_io` (fn-local); parses via `json_io.loads(text)` at :366; `except (json.JSONDecodeError, KeyError, TypeError)` at :396 references UNBOUND `json` | LIVE BUG |
| `backend/utils/json_io.py` | 21, 26-32 | `import json as _json`; `loads()` = thin wrapper re-raising `json.JSONDecodeError`. Docstring (:15-16): *"The helpers intentionally re-raise `json.JSONDecodeError` so existing try/except clauses at call sites keep working unchanged."* | Contract: call sites are EXPECTED to `import json` for their except clause |
| `backend/agents/multi_agent_orchestrator.py` | 49, 975-982 | `_classify_via_llm()` — the SOLE caller. `parse_llm_classification(text)` at :982 is **not wrapped in try/except** | BLAST RADIUS |
| `backend/agents/{planner_agent,risk_debate,orchestrator,debate,evaluator_agent}.py` | 13/15/17/9/30 | House pattern: all five `import json` at module top **and** use `json_io.loads()` **and** `except json.JSONDecodeError` | HOUSE STYLE (fix template) |
| `.claude/skills/code-review-trading-domain/SKILL.md` | 54-70 | Top-15 ranked list (erosion-guard baseline names below) | EDIT TARGET |
| `.claude/skills/code-review-trading-domain/SKILL.md` | 136-155 | Dimension 3 (Code quality) table + negation list = recommended insertion home | EDIT TARGET |
| `.claude/skills/code-review-trading-domain/SKILL.md` | 171 | Existing `rename-as-refactor` [BLOCK] — ADJACENT but distinct (see below) | DISAMBIGUATE |
| `backend/tests/` | 110 files | Convention: `test_phase_X_Y_*.py`, module docstring w/ bug+fix, `from __future__ import annotations`, `sys.path.insert(REPO_ROOT)`, plain `def test_*` + asserts, import fn under test directly. No unittest classes. | TEST TEMPLATE |
| `pytest.ini` | 7-9 | Only marker = `requires_live`; NO `addopts`, NO timeout config | — |
| `handoff/current/research_brief_67_1.md` | 12, 44, 90-106 | 67.1 IS the ruff F821/F401 backend lint gate via `uvx ruff check --select F821,F401`; uses THIS exact bug as its proof case | INTERLOCK |

### Blast-radius trace (criterion 2)

`_classify_via_llm` (multi_agent_orchestrator.py:975) awaits the Communication agent, then calls `parse_llm_classification(text)` at :982 **with no try/except around it**. So when the Communication agent returns non-JSON (rate-limit blurb, prose, truncated stream), `json_io.loads` raises `json.JSONDecodeError`, execution reaches :396, and the `except` clause itself raises `NameError: name 'json' is not defined` — the graceful default at :397-402 is **never reached**. The NameError propagates out of `_classify_via_llm` into the classification path (Slack/iMessage routing). The very fallback built to route malformed Communication output to Main is dead code today. Verified live in the venv (2026-07-09): `parse_llm_classification('not json {')` -> `NameError: name 'json' is not defined`.

### Erosion-guard baseline (criterion 1 — all must remain present)

Top-15 ranked names: `secret-in-diff, kill-switch-reachability, stop-loss-always-set, prompt-injection-path, broad-except-silences-risk-guard, financial-logic-without-behavioral-test, tautological-assertion, perf-metrics-bypass, command-injection, excessive-agency-scope-creep, position-sizing-div-zero, criteria-erosion, sycophantic-all-criteria-pass, supply-chain-dep-pin-removal, unicode-in-logger`. Plus all dimension-table names (Dim1 x13, Dim2 x11, Dim3 x8, Dim4 x6, Dim5 x8). The diff must ADD rows only; renaming/removing any is a criteria-erosion self-violation.

### Two GENERATE prerequisites surfaced (live-env audit)

- **`pytest-timeout` is NOT installed** in `.venv` (`pip show pytest-timeout` -> NOT INSTALLED; `python -m pytest --help` lists no `--timeout`). The IMMUTABLE verification command passes `--timeout=60`, which pytest rejects as an unrecognized argument (exit 4) => **the gate cannot pass until GENERATE `pip install pytest-timeout` into the venv** (and, for reproducibility, records it where pytest itself is pinned). This is exactly the "live-smoke the real env before completing" discipline; do NOT edit the immutable command.
- **67.1 interlock:** 67.1 (in-progress, `depends_on_step`) wires `uvx ruff check --select F821,F401 <changed .py>` as the backend deterministic lint gate and cites agent_definitions.py:396 as its proof case. So the deterministic complement to this heuristic already exists as 67.1. `ruff` is intentionally NOT in the venv (`uvx` ephemeral, mirrors the MCP uvx pattern) — that is why `import ruff` fails; expected, not a defect. Criterion 4's "lint-gate output over the changed files" = run 67.1's ruff gate over `agent_definitions.py` (must go F821-clean post-fix) and over the new test file.

---

## External research

### Read in full (6; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.astral.sh/ruff/rules/undefined-name/ | 2026-07-09 | Official doc (Tier-2) | WebFetch | F821: *"Checks for uses of undefined names… An undefined name is likely to raise `NameError` at runtime."* Pyflakes-derived. This is the exact static check that catches agent_definitions.py:396. |
| 2 | https://arxiv.org/html/2603.23448 | 2026-07-09 | Preprint (Tier-1), 2026 | WebFetch (html) | c-CRAB code-review-agent benchmark. 10 defect categories incl. **"Design – API interfaces and contract issues"** and **"Error Handling – exception behavior."** LLM tools pass only **20.1–32.1%** vs humans 100%; **41.5%** combined. *"Repository-specific context is a likely missing ingredient"* — cross-file/contract defects are where LLM reviewers are WEAKEST. |
| 3 | https://arxiv.org/html/2605.24397 | 2026-07-09 | Systematic Lit Review (Tier-1), 2026 | WebFetch (html) | Two classes: **syntactic** (type sigs, visibility, interface defs -> compile error) vs **behavioral** (same signature, changed runtime semantics/exceptions). In npm, **behavioral = 68.1%** of breaks. Static tools *"reach high accuracy on syntactic breaks but limited coverage on behavioral ones"*; behavioral breaks *"emit no compiler, linker, or runtime exception on their own"* and need tests most projects lack. **>60% of clients use undeclared transitive functionality** -> silent consumer breakage. |
| 4 | https://ar5iv.labs.arxiv.org/html/2209.00393 | 2026-07-09 | Preprint (Tier-1), canonical 2022 | WebFetch (ar5iv) | Sembid semantic-differencing. **33.83% of Patch and 64.42% of Minor** Java version pairs had >=1 breaking API. Semantic breaks outnumber syntactic **2-4x** (1.10%/4.06% SemB vs 0.38%/1.04% SynB). Root causes: **91.67% changed output calculation, 73.33% changed execution logic**; unit tests caught only 16 vs Sembid's 72 (100% vs 16.61% coverage). Interface-stable changes are the dangerous, near-invisible class. |
| 5 | https://arxiv.org/html/2408.14431 | 2026-07-09 | Preprint (Tier-1), 2024 | WebFetch (html) | npm breaking-change comprehension (1,519 breaks / 381 projects). **19% of documented breaking changes cannot be detected by regression testing** — client suites lack coverage for affected APIs. Behavioral patterns: return-value spec (79), option/config handling (231), default behavior (203), **error-handling process (42)**. Mitigation: functionally-similar-code detection + behavioral-break detection beyond type changes. |
| 6 | https://zuplo.com/learning-center/semantic-api-versioning | 2026-07-09 | Practitioner doc (Tier-4) | WebFetch | Breaking-change checklist: *"removing endpoints, changing response formats, renaming required parameters, or altering authentication methods… changing a required field's data type… restructuring the response body."* Directly enumerates the shipped-bug class (renamed field / re-shaped output / re-routed input). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/pdf/2507.17369 (Roseau, Java source-based BC analysis) | Preprint 2025 | `/html/` returned 404; not needed — Sembid + SLR cover the same ground |
| https://tokenwisehq.com/best-llm-for/code-review | Blog 2026 | HTTP 521 (Cloudflare); snippet: GPT-5.1 review must *"spot… the API contract break — without burying the engineer in nonsense"* |
| https://realpython.com/ruff-python/ | Tutorial | HTTP 403 |
| https://arxiv.org/pdf/2301.04563 ("I depended on you and you broke me") | Preprint | Corroborates client-side manifestation; SLR subsumes |
| https://arxiv.org/pdf/2408.14431 (AexPy mention) / AexPy Python BC tool | Tool | Python analogue of Roseau; behavioral cue in Sembid suffices |
| https://apitect.com/blogs/preventing-api-breaking-changes-… | Blog | Deprecation-window practice; Zuplo covers it |
| https://docs.astral.sh/ruff/rules/undefined-export/ (F822) | Official doc | Adjacent lint rule, not load-bearing |
| github.com/astral-sh/ruff issues #8700/#14772/#7914 | Issue tracker | F821 false-positive edge cases (forward refs, deleted-shadow builtins) — informs negation list |

### Recency scan (2024-2026)

Searched three variants per topic (current-year `2026`, recency `2025`, year-less canonical). **Result: 3 new findings in the 2024-2026 window that COMPLEMENT (not supersede) the canonical Sembid (2022):**
1. c-CRAB (2026, #2) establishes that LLM code reviewers are measurably WEAKEST exactly on design/contract/cross-file defects (20-32% pass) — fresh empirical basis that a human-curated heuristic + deterministic lint is the right complement, not "trust the LLM to notice."
2. The 2026 SLR (#3) quantifies behavioral breaks at 68.1% of npm breaks and confirms static tools under-cover them — validates a WARN (judgment-required, grep-assisted) severity rather than a pure deterministic BLOCK.
3. npm-comprehension (2024, #5): 19% of documented breaks are test-invisible — reinforces that the heuristic must force a *consumer grep*, not rely on the test suite alone.
No 2024-2026 source contradicts the canonical Sembid finding; the direction of travel is consistent (behavioral/consumer breaks dominate and evade automation).

---

## Key findings

1. **Consumer/behavioral breaks dominate and evade automation.** Behavioral = 68.1% of npm breaks (SLR #3); they emit no compiler/runtime exception on their own. Semantic breaks outnumber syntactic 2-4x (Sembid #4). => The heuristic must target the SHAPE/ROUTING/OUTPUT-contract layer, and its enforcement is a *consumer grep*, because tests miss 19% of them (#5).
2. **LLM reviewers are weakest precisely here.** c-CRAB (#2): Design/contract + maintainability categories score 7.9-27.0% because they need repository-specific context. This is the empirical justification for a codified pyfinagent-native heuristic (with grep cues) over generic "review carefully" prompting.
3. **F821 is the deterministic floor, not the ceiling.** ruff F821 (#1) statically catches the undefined-`json` NameError — and 67.1 wires exactly that gate. But F821 CANNOT catch renamed dict keys, argv-vs-stdin, SDK-kwarg-vs-CLI-flag, or output re-casing (those are valid names, wrong contract). The SKILL.md heuristic is the judgment complement covering the part F821 provably can't see. Clean division of labor.
4. **House style dictates the fix.** json_io's own docstring says call sites are expected to keep `import json` for their except clause; five sibling modules do exactly that. agent_definitions.py is the lone omission.

---

## RECOMMENDATION A — the fix variant (criterion 2)

**Chosen: variant (a) `import json` at module top, PLUS broaden the tuple to include `AttributeError`.**

Rationale:
- `import json` at the top is the **house-consistent minimal fix** — byte-for-byte the pattern in planner_agent.py:13, risk_debate.py:15, orchestrator.py:17, debate.py:9, evaluator_agent.py:30, and the contract json_io.py:15-16 documents. Because json_io does `import json as _json`, the raised `json.JSONDecodeError` IS the same class as the top-level `json.JSONDecodeError`, so the existing tuple resolves and catches it.
- Reject variant (b) "catch `ValueError`" — it works (JSONDecodeError subclasses ValueError) but silently WIDENS the catch and DIVERGES from house style; not warranted.
- Reject variant (c) "import the exception from json_io" — json_io does not export `JSONDecodeError`; no such symbol.
- **Add `AttributeError`** because a valid-but-wrong-shape payload exposes a second latent gap: `json_io.loads("5")` -> `int 5`, then `data.get("primary", …)` at :368 raises `AttributeError: 'int' object has no attribute 'get'`, which the current 3-tuple does NOT catch and would ALSO propagate out of the un-wrapped caller. The stated criterion is *"degrades gracefully on malformed input"* — a bare JSON scalar/array from the Communication agent is malformed input. Final clause:
  `except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:`
  (This mirrors the house precedent that call sites tune the tuple to their inputs — evaluator_agent.py:441 catches a 3-tuple `(json.JSONDecodeError, ValueError, IndexError)`.)

## RECOMMENDATION B — the SKILL.md heuristic row (criterion 1)

Insert into **Dimension 3 (Code quality)** table (SKILL.md:139-147) and add a negation bullet at :149-153. Also append as **#16** in the Top-15 ranked list (append-only; do not renumber/rename existing 1-15, preserving the erosion guard). House format = `| name | detection cue (with grep) | Severity |`.

Table row:

```
| consumer-contract-break | A diff changes a PUBLIC contract shape -- function/method signature, kwarg name, CLI flag, dict/JSON key, response-field casing, return type, or how a value is passed (argv vs stdin, positional vs `--flag`, SDK kwarg vs CLI flag) -- WITHOUT every consumer verified in the SAME diff. Q/A greps consumers itself: `grep -rn "<old_symbol>" backend/ frontend/ scripts/` for each renamed/removed/re-shaped symbol; any surviving reference to the old shape => escalate to BLOCK. Also flag a module that references a name in an `except (...)` tuple or type annotation that the module never imports (grep the module's `^import`/`^from` vs names used in `except`/annotations -- the agent_definitions.py:396 `json` NameError class). Behavioral-break subset (same signature, changed runtime semantics/exception/output) per the ecosystem literature: changed return-value spec, changed default behavior, changed error-handling path. | WARN (BLOCK if a live unverified consumer is found) |
```

Negation-list bullet:

```
- `consumer-contract-break`: purely ADDITIVE changes (new optional kwarg WITH default, new dict key, new endpoint) that narrow no existing consumer contract are non-breaking -- do NOT flag. A rename where a grep for the old symbol returns zero non-test hits because every consumer was updated in the SAME diff is verified, not a break (the phase-25.6 `paper_snapshots -> paper_portfolio_snapshots` rename is the model). Internal/private (`_`-prefixed) symbols with no cross-module or cross-process consumer are exempt. Changes behind a default-OFF flag that are byte-identical when the flag is absent are exempt. Distinct from `rename-as-refactor` (Dim 4): that fires on rename + semantic change hiding old behavior; consumer-contract-break fires on ANY shape/routing change -- including a pure rename with correct new behavior -- when consumers are not grep-verified. Both may fire; verdict = worst severity.
```

Top-15 ranked append:

```
16. **consumer-contract-break** [WARN->BLOCK] -- interface/CLI-flag/output-shape/dict-key/response-casing/input-routing change (or an `except`/annotation name the module never imports) shipped without grepping every consumer. Operator 2026-05-26 recurring class (argv-vs-stdin, --max-tokens SDK-vs-CLI, Recent Reports alpha/casing). Behavioral breaks = 68.1% of ecosystem breaks and evade tests (arXiv 2605.24397, 2408.14431); F821 covers only the undefined-name subset.
```

**Severity choice = WARN (auto-escalate BLOCK):** grounded in SLR #3 (behavioral breaks need judgment + are test-invisible) and c-CRAB #2 (this class is where blanket automation fails) — a pure deterministic BLOCK would over-fire on benign additive changes; WARN forces the Q/A to run the consumer grep, and a found unverified consumer promotes to BLOCK. Mirrors the existing `llm04-training-code-added` auto-promote precedent (SKILL.md:87).

## RECOMMENDATION C — the behavioral test (criterion 3)

`backend/tests/test_agent_definitions_classification.py`. Match house template: module docstring (bug + fix), `from __future__ import annotations`, `sys.path.insert(REPO_ROOT)`, import `parse_llm_classification`, `AgentType`, `QueryComplexity` directly, plain `def test_*` + asserts, NO mocks (the function is pure string->dataclass parsing, so mocking is neither needed nor desirable — an over-mocked test would trip SKILL.md `over-mocked-test` BLOCK).

Four cases (each non-tautological — asserts real routing/complexity, not `is not None`):
1. `test_malformed_non_json_defaults_to_main` — `parse_llm_classification("not json {")` returns `agent_type == AgentType.MAIN`, `confidence == 0.5`, and `"Parse failed"` in `reasoning`. This is the DOCUMENTED REPRO — implicitly asserts NO NameError (a raise fails the test).
2. `test_valid_json_wrong_shape_defaults_to_main` — `parse_llm_classification("5")` (and `"[1,2]"`) returns `AgentType.MAIN` (proves the AttributeError path degrades gracefully; would still raise under an `import json`-only fix).
3. `test_fenced_json_block_parses` — a real fenced ` ```json\n{"primary":"qa","complexity":"complex"}\n``` ` returns `agent_type == AgentType.QA`, `complexity == QueryComplexity.COMPLEX` (proves the strip logic + happy path survive the fix).
4. `test_bare_valid_json_parses` — `'{"primary":"research","triggers_harness":true}'` returns `AgentType.RESEARCH`, `triggers_harness is True`.

Guards against anti-patterns: no `assert x is not None` (each asserts a specific enum/value); no module-under-test mock; case 3/4 prove the fix did not regress the success path (mutation-resistant — reverting the fix breaks case 1/2; breaking the strip logic breaks case 3).

---

## Application to pyfinagent (external -> internal mapping)

- SLR #3 "behavioral = 68.1%, static tools under-cover" -> heuristic severity = WARN+grep, not deterministic-only; and the fix's `AttributeError` add targets a behavioral (valid-JSON-wrong-shape) break at agent_definitions.py:368.
- c-CRAB #2 "Design/contract is where LLM reviewers fail (7.9-27%)" -> justifies codifying grep cues into SKILL.md rather than trusting Q/A to notice cross-file breaks unaided.
- ruff F821 #1 + 67.1 -> the deterministic complement is already being built; SKILL.md heuristic covers the non-F821 remainder (renamed keys, argv/stdin, casing). Criterion 4's lint-gate output = 67.1's `uvx ruff check --select F821,F401 backend/agents/agent_definitions.py` going clean post-fix.
- Zuplo #6 breaking-change list -> maps 1:1 onto the detection cue (renamed params, re-shaped response, changed error format).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: ruff docs, c-CRAB 2603.23448, SLR 2605.24397, Sembid 2209.00393, npm-BC 2408.14431, Zuplo)
- [x] 10+ unique URLs total (6 read + 8 snippet-only = 14+)
- [x] Recency scan (2024-2026) performed + reported (3 complementary findings)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the bug site, sole caller, json_io contract, 5 house-pattern modules, SKILL.md structure, test conventions, 67.1 interlock, and the pytest-timeout env gap
- [x] Contradictions/consensus noted (no source contradicts Sembid; consensus = behavioral/consumer breaks dominate + evade automation)
- [x] Claims cited per-claim

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
