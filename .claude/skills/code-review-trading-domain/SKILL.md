---
name: code-review-trading-domain
description: Trading-domain code-review heuristics for pyfinagent Q/A evaluator. Provides 5 dimensions (security, trading-domain correctness, code quality, anti-rubber-stamp, LLM-evaluator anti-patterns), Top-15 ranked heuristics with BLOCK/WARN/NOTE severity dispatch, and explicit negation lists. Preloaded into Q/A subagent context at spawn. Not user-invocable — background reference only.
user-invocable: false
---

## Code review heuristics (phase-16.59)

Added 2026-05-16. Source: `handoff/archive/phase-16.59/research_brief_16_59.md`
(complex-tier research; 7 Tier-1/2 sources read in full; gate_passed=true).
Implements the Anthropic Code Review pattern
([code.claude.com/docs/en/code-review](https://code.claude.com/docs/en/code-review))
adapted to the harness-MAS evaluator role. Coverage: OWASP LLM Top-10 2025,
sycophancy mitigations from arXiv 2509.16533 (EMNLP 2025), Cloudflare
"explicit negation list" pattern, and pyfinagent-specific trading-domain
invariants (kill_switch / stop-loss / perf_metrics single-source / MIN_ASSET_VOL).

**Order of operations.** Code-review heuristics run AFTER the deterministic
checks (§1) and existing-results check (§2), BEFORE the LLM judgment (§4).
They do NOT replace the 5-item harness-compliance audit
(researcher / contract-pre-commit / results / log-last / no-verdict-shopping)
— that runs first, unchanged.

### Severity dispatch rule

| Severity | Verdict effect | How to record |
|----------|---------------|---------------|
| **BLOCK** | Auto-FAIL | Add heuristic name to `violated_criteria`; populate `violation_details` triple |
| **WARN**  | Force CONDITIONAL | Add to `violation_details` with `violation_type: Threshold_Not_Met` and severity=`WARN` in details |
| **NOTE**  | PASS-with-flag | Note in `reason` field but do NOT degrade verdict |

A diff may trigger multiple heuristics; verdict = worst severity hit.
On every BLOCK or WARN, quote at least one offending file:line from the
diff so a reviewer can verify the finding (per Cloudflare's "coordinator
reads source code to verify" pattern).

### Simultaneous-presentation rule (cycle-2 spawns)

Per arXiv 2509.16533 (EMNLP 2025), LLM evaluators flip verdicts under
detailed-but-wrong rebuttals when the prior verdict is presented
sequentially. Mitigation: on a cycle-2 spawn after a CONDITIONAL or FAIL,
read in this order in ONE pass before judging:

1. Updated `experiment_results.md` (the new evidence)
2. Updated `evaluator_critique.md` (the new critique appended)
3. The PRIOR verdict from `handoff/harness_log.md` (the rebuttal context)
4. The diff between (1) and the previous cycle's results

If the code did NOT actually change between cycles, a verdict reversal is
sycophancy — return the prior verdict with `violation_type: Circular_Reasoning`.
This codifies the CLAUDE.md "second-opinion-shopping on unchanged evidence
is forbidden" rule.

### Top-15 ranked heuristics (impact × frequency)

1. **secret-in-diff** [BLOCK] — API key/token/credential literal in diff. OWASP LLM02 2025.
2. **kill-switch-reachability** [BLOCK] — execution-path change leaves `kill_switch.is_paused()` unreachable. `kill_switch.py:12-18` (FINRA 15c3-5).
3. **stop-loss-always-set** [BLOCK] — buy path allows `stop_loss_price=None` at entry without fallback. `paper_trader.py:99-114` (phase-25.6).
4. **prompt-injection-path** [BLOCK] — user-supplied string reaches LLM system prompt without sanitization. OWASP LLM01 2025.
5. **broad-except-silences-risk-guard** [BLOCK] — `except Exception: pass` inside risk-guard / kill-switch / stop-loss code path. `paper_trader.py:26,52` (known anti-patterns).
6. **financial-logic-without-behavioral-test** [BLOCK] — Sharpe/drawdown/position-sizing math changed with no test exercising the new path.
7. **tautological-assertion** [BLOCK] — `assert x == x`, `assert result is not None`, or mock-and-assert-called.
8. **perf-metrics-bypass** [WARN] — Sharpe/drawdown/alpha computed outside `services/perf_metrics.py` (single-metric-source rule).
9. **command-injection** [BLOCK] — `subprocess`/`os.system`/`eval`/`exec` with non-literal argument.
10. **excessive-agency-scope-creep** [WARN] — new tool / BQ-write / file-write capability added without least-privilege doc. OWASP LLM06 2025.
11. **position-sizing-div-zero** [WARN] — vol used as divisor without floor guard. `risk_engine.py:33` MIN_ASSET_VOL=1e-6.
12. **criteria-erosion** [WARN] — previously-required evaluation criterion silently dropped across cycles.
13. **sycophantic-all-criteria-pass** [WARN] — evaluator output with every criterion PASS in <3 sentences, no file:line, no quoted evidence.
14. **supply-chain-dep-pin-removal** [WARN] — pinned version removed from `requirements.txt`/`pyproject.toml`/`package.json` without explicit reason. OWASP LLM03 2025.
15. **unicode-in-logger** [NOTE] — logger call with non-ASCII characters (Windows cp1252 crash defense per `security.md`).

### Dimension 1 — Security audit

Per OWASP LLM Top-10 2025 + `security.md` + Semgrep python-security ruleset.

| Heuristic | Detection cue | Severity |
|-----------|---------------|----------|
| secret-in-diff | `grep -iE "(api_key|secret|password|token)\s*=\s*['\"][A-Za-z0-9/+]{16,}"` on diff | BLOCK |
| prompt-injection-path | Trace API param → `messages[0].content` / `system=` without sanitize step | BLOCK |
| command-injection | `subprocess`/`os.system`/`eval`/`exec` with non-literal arg | BLOCK |
| insecure-output-handling | `llm_call(...)` result flowing directly into `query(...)`, `exec(...)`, or file path | BLOCK |
| supply-chain-dep-pin-removal | Removed `==X.Y.Z` pin from dep manifest | WARN |
| yaml-unsafe-load | `yaml.load()` without `Loader=yaml.SafeLoader` | WARN |
| pickle-deserialization | `pickle.load`/`loads` on external/network input | WARN |
| system-prompt-leakage | New endpoint/log/response serializing `agent_config.system_prompt`, full `messages` list incl. system role, or skill `.md` content to external caller. Grep: `json\.dumps.*messages\|logging.*system_prompt\|return.*"system"\s*:` (OWASP LLM07:2025) | WARN |
| rag-memory-poisoning | New `add_memory()` / `add_memories()` call where input originates from an external or unvalidated source (not seed data or authenticated BQ path); or new vector-store import (`chromadb`, `pinecone`, `weaviate`, `pgvector`) without access-control doc. Grep: `add_memori(es\|y)\|import chromadb\|import pinecone\|import weaviate\|import pgvector` (OWASP LLM08:2025 — pyfinagent uses BM25 so Vec2Text inversion N/A; poisoning is the real surface) | WARN |
| unbounded-llm-loop | New `while True` or unbounded `for` loop wrapping an LLM API call; OR removal/reduction of `MAX_TOOL_TURNS`, `MAX_RESEARCH_ITERATIONS`, `MAX_CONSECUTIVE_FAIL`, `MAX_RESEARCH_ITER` bound constants. Grep: `while True` near `messages.create\|generate_content`; diff reducing the named constants (OWASP LLM10:2025 denial-of-wallet) | WARN |
| excessive-agency | New write/delete/execute tool added to agent without least-privilege doc | WARN |
| owasp-headers-bypass | New `APIRouter` registered outside the auth-middleware stack | WARN |

**What NOT to flag (negation list):**
- Secret-looking literals in `tests/fixtures/`, `*_example.py`, or files matching `*.template.*`
- Broad `except` in vendored third-party code under `backend/vendor/` or imported libraries
- `yaml.load` with `Loader=yaml.SafeLoader` already set or in a config-loading helper that wraps SafeLoader internally
- `subprocess.run` with a list argument and shell=False (this is safe; only flag the string + shell=True form)
- `system-prompt-leakage`: `system=agent_config.system_prompt` passed directly to `client.messages.create()` (e.g. `backend/agents/multi_agent_orchestrator.py:985`) is safe — only flag when the full `messages` list or raw `system_prompt` string is serialized to an external response, log line, or endpoint body
- `rag-memory-poisoning`: `FinancialSituationMemory` seed entries at `backend/agents/memory.py:23-54` are safe (static, not external); `load_from_bq_rows()` in authenticated BQ context is acceptable; BM25 corpus is not subject to Vec2Text embedding-inversion attacks
- `unbounded-llm-loop`: existing bounds (`for cycle in range(1, args.cycles + 1)` at `scripts/harness/run_harness.py:1111`; `for iteration in range(1, MAX_RESEARCH_ITERATIONS + 1)` at `backend/agents/multi_agent_orchestrator.py:523`; `for turn in range(max_turns)` at `:1048`) are correct — do NOT flag these; only flag NEW loops that bypass or remove the bound constants

Source: [OWASP LLM Top-10 v2.0 (2025)](https://genai.owasp.org/llm-top-10/) (LLM07 System Prompt Leakage, LLM08 Vector and Embedding Weaknesses, LLM10 Unbounded Consumption added in v2.0; older v1.1 reference [Invicti](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025) preserved for LLM01-LLM06), [security.md](../../rules/security.md).

### Dimension 2 — Trading-domain correctness

pyfinagent-specific invariants. These BLOCK heuristics encode the
non-negotiable risk-guard wiring.

| Heuristic | Detection cue | Severity |
|-----------|---------------|----------|
| kill-switch-reachability | New execution path skips `kill_switch.is_paused()` | BLOCK |
| stop-loss-always-set | Buy path with `stop_loss_price=None` and no fallback | BLOCK |
| perf-metrics-bypass | Sharpe/drawdown/alpha formula inline instead of importing `services/perf_metrics.py` | BLOCK |
| position-sizing-div-zero | Vol used as divisor without `MIN_ASSET_VOL` floor | WARN |
| max-position-check-bypass | `paper_max_positions` guard removed/weakened (`paper_trader.py:131-132`) | BLOCK |
| bq-schema-migration-safety | `NOT NULL` column add without DEFAULT, or column drop on live BQ table | WARN |
| stop-loss-backfill-removal | Removal of `backfill_stop_losses` (`paper_trader.py:466-517`) | BLOCK |
| crypto-asset-class | Re-enables `crypto` asset class (owner directive 2026-04-19 bans it) | BLOCK |
| sod-nav-anchor | Changes to `_sod_nav`/`_peak_nav` in `kill_switch.py:46-52` without audit-log invariant update | WARN |
| paper-trader-broad-except | `except Exception:` swallowing in execution path (existing examples at `paper_trader.py:26,52`) | BLOCK |

**What NOT to flag (negation list):**
- `perf_metrics` import that DOES route through `services/perf_metrics.py` (only flag inline re-implementations)
- Stop-loss code in `tests/` that intentionally exercises the no-stop edge case
- `kill_switch` bypass in test code that explicitly mocks the kill switch
- BQ schema changes in `pyfinagent_staging` dataset (staging is allowed; only flag `pyfinagent_data` / `pyfinagent_pms` writes)
- New code in `backend/vendor/` (third-party; conventions don't apply)

Source: [kill_switch.py](../../../backend/services/kill_switch.py), [risk_engine.py](../../../backend/markets/risk_engine.py), [paper_trader.py](../../../backend/services/paper_trader.py), [.claude/rules/backend-services.md](../../rules/backend-services.md).

### Dimension 3 — Code quality

| Heuristic | Detection cue | Severity |
|-----------|---------------|----------|
| broad-except | `except Exception: pass` or bare `except:` anywhere | WARN |
| no-type-hints | New public `def` without parameter/return annotations | NOTE |
| print-statement | `print()` in non-test, non-script code | WARN |
| global-mutable-state | Module-level mutable dict/list mutated by functions (non-singleton file) | WARN |
| test-coverage-delta | >50 lines new business logic with zero new tests | WARN |
| unicode-in-logger | `logger.{info,warning,error,debug}` with non-ASCII | NOTE |
| magic-number | Numeric literal in financial formula without named constant | NOTE |
| composition-over-inheritance | Inheritance chain >2 levels added | NOTE |

**What NOT to flag (negation list):**
- `print()` in `scripts/`, `tests/`, or `__main__` blocks
- Missing type hints on private (`_`-prefixed) helper functions
- Global state in `*_constants.py`, `settings.py`, or singleton modules tagged as such
- Test files that exceed the >50-lines-with-no-tests rule by definition

Source: Python 3.14 typing conventions; [security.md](../../rules/security.md) ASCII logger rule.

### Dimension 4 — Anti-rubber-stamp on financial logic

**Heuristic class identifier:** `anti-rubber-stamp`

The highest-leverage dimension. Q/A MUST refuse PASS on financial-logic
changes that lack a behavioral test. Per arXiv 2404.18496 + Cloudflare AI
code review production data: telling the LLM "what NOT to do" beats
positive instruction.

| Heuristic | Detection cue | Severity |
|-----------|---------------|----------|
| financial-logic-without-behavioral-test | Diff touches `perf_metrics.py`/`risk_engine.py`/`backtest_engine.py`/`backtest_trader.py` AND no `test_*.py` modified | BLOCK |
| tautological-assertion | `assert.*is not None`, `assert x == x`, `assert mock.*called` | BLOCK |
| over-mocked-test | Test mocks the entire module under test (e.g. `@patch("backend.services.paper_trader.PaperTrader")` in a paper_trader test) | BLOCK |
| rename-as-refactor | Diff is rename + behavior change in same commit; old semantics not preserved | BLOCK |
| pass-on-all-criteria-no-evidence | Evaluator marks every criterion PASS with <3 sentences total, no file:line, no quoted output | BLOCK |
| formula-drift-without-citation | Risk constant (`DEFAULT_TARGET_VOL`, `daily_loss_limit_pct`, `MAX_LEVERAGE`) changed without commit/comment citing source | WARN |

**What NOT to flag (negation list):**
- Pure-refactor diffs that move code without changing logic, where pre/post tests pass without modification
- Added docstrings or type-hint-only changes — these don't need new tests
- Config-only changes (yaml/json/toml) that have no Python logic
- New constants added with citation in the same commit (cite OK = no flag)

Source: [arXiv 2404.18496](https://arxiv.org/html/2404.18496v2), [Cloudflare AI code review](https://blog.cloudflare.com/ai-code-review/), [Anthropic Code Review](https://code.claude.com/docs/en/code-review).

### Dimension 5 — LLM-evaluator anti-patterns

Self-aware heuristics. Q/A grading itself.

| Heuristic | Detection cue | Severity |
|-----------|---------------|----------|
| sycophancy-under-rebuttal | Prior verdict FAIL/CONDITIONAL flipped to PASS without code change between cycles | BLOCK |
| second-opinion-shopping | Fresh Q/A spawned on unchanged `experiment_results.md` (compare mtime across cycles) | BLOCK |
| missing-chain-of-thought | Verdict issued with no file:line / command-output citations | BLOCK |
| 3rd-conditional-not-escalated | Step has 2+ prior CONDITIONALs in `harness_log.md` and current verdict is CONDITIONAL again | BLOCK |
| position-bias | First criterion always PASS regardless of evidence | WARN |
| verbosity-bias | Long output PASS, short output CONDITIONAL — verdict correlates with length not evidence | WARN |
| criteria-erosion | Previously-required criterion missing from current cycle | WARN |
| self-reference-confidence | "Generator confirms X is correct" used as sole basis for PASS | WARN |

**What NOT to flag (negation list):**
- Verdict reversal AFTER the code actually changed (that's the documented cycle-2 flow, not sycophancy)
- Short critiques where every criterion has a file:line citation (concise ≠ sycophantic)
- Repeated PASS across cycles where evidence is identical AND nothing changed (correct steady-state)
- Verdict issued without file:line if the verification command itself produced no output that warrants citation

Source: [arXiv 2509.16533](https://arxiv.org/abs/2509.16533), [SurePrompts LLM-as-Judge guide](https://sureprompts.com/blog/llm-as-judge-prompting-guide), [SycEval arXiv 2502.08177](https://arxiv.org/html/2502.08177v2), CLAUDE.md second-opinion-shopping prohibition.

### Reporting

When code-review heuristics fire, add to the existing JSON output:

```json
{
  "ok": false,
  "verdict": "FAIL",
  "violated_criteria": ["secret-in-diff", "kill-switch-reachability"],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "git diff",
      "state": "backend/api/foo.py:42 contains API_KEY='sk-...'",
      "constraint": "secret-in-diff [BLOCK] — OWASP LLM02 2025",
      "severity": "BLOCK"
    }
  ],
  "checks_run": ["syntax", "verification_command", "code_review_heuristics", "evaluator_critique"]
}
```

Append `"code_review_heuristics"` to `checks_run` whenever any of the 5
dimensions was evaluated (even if no findings).

### Sources (read in full during phase-16.59 research gate)

1. [Anthropic Code Review docs](https://code.claude.com/docs/en/code-review) — multi-agent severity model + REVIEW.md customization
2. [arXiv 2509.16533 EMNLP 2025](https://arxiv.org/abs/2509.16533) — sycophancy under rebuttal; simultaneous-presentation mitigation
3. [arXiv 2404.18496](https://arxiv.org/html/2404.18496v2) — AI-powered code review; multi-agent specialization
4. [SurePrompts LLM-as-Judge guide](https://sureprompts.com/blog/llm-as-judge-prompting-guide) — four structural biases + five mitigations; RCAF
5. [OWASP LLM Top-10 2025](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025) — 4 new entries vs 2023 v1.1
6. [Cloudflare AI code review at scale](https://blog.cloudflare.com/ai-code-review/) — explicit negation lists; coordinator reasonableness filter
7. [OWASP LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — baseline v1.1

Full research brief: `handoff/archive/phase-16.59/research_brief_16_59.md`.
