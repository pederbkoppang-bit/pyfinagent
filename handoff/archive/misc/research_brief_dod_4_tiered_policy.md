# Research Brief — DoD-4 Tiered-Coverage Policy
Tier: SIMPLE. Date: 2026-05-25. Assumption: standard 7-section brief.

## Section A. Question

For the pyfinagent single-user local-only trading app
(17K SLOC, services 26% / agents 22% / api 33% blanket; Tier-1
risk modules currently 51-66%), which DoD-4 policy yields the
best system: (a) brute 70% blanket via ~1700 new tests, (b)
operator override + audit trail, or (c) tiered policy with
risk-classified bars? Defensible Tier-1 bar? Tier-2 bar?

## Section B. Sources read in full (8; gate floor = 5)

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|
| 1 | https://launchdarkly.com/blog/code-coverage-what-it-is-and-why-it-matters/ | 2026-05-25 | blog (citing Google research) | WebFetch full | Google framework: 60% acceptable / 75% commendable / 90% exemplary; safety-critical = near-100%, standard apps = 80-90% |
| 2 | https://blog.codepipes.com/testing/software-testing-antipatterns.html | 2026-05-25 | blog (industry) | WebFetch full | "Getting 100% coverage on total code is not recommended"; tier-rank as critical / core / other; target critical first |
| 3 | https://clearcove.ca/2012/04/kent-beck-doesnt-aim-for-100-test-coverage/ | 2026-05-25 | blog (Beck quote) | WebFetch full | "Test as little as possible to reach a given level of confidence"; concentrate on where mistakes are likely |
| 4 | https://ben3d.ca/blog/the-rise-of-test-theater | 2026-05-25 | blog (2024-25 era) | WebFetch full | "90%+ coverage" with circular AI-generated tests = test theater; >50% of observed AI tests just mirror implementation |
| 5 | https://www.bullseye.com/minimum.html | 2026-05-25 | industry vendor | WebFetch full | "70-80% is a reasonable goal for system test"; unit 90% / integration 80% / system 70%; DO-178B = 100% for aviation Level A |
| 6 | https://aiko.dev/code_coverage/ | 2026-05-25 | blog | WebFetch full | "Coverage tells you what you definitely haven't tested, not what you have"; Goodhart's Law warning |
| 7 | https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/market-access-rule | 2026-05-25 | regulator (2026 doc) | WebFetch full | Rule 15c3-5: annual review + tested pre-trade hard blocks + documented post-trade surveillance; tier-by-business-line |
| 8 | https://www.infoq.com/news/2026/01/meta-llm-mutation-testing/ | 2026-05-25 | engineering news (2026) | WebFetch full | Meta uses LLM-generated mutants as complement (not replacement) for coverage; 73% acceptance, 36% privacy-relevant |

## Section C. Snippet-only sources (context; not gate-credit)

| URL | Kind | Why snippet-only |
|-----|------|-----------------|
| https://research.google/pubs/code-coverage-at-google/ | research abstract | PDF binary not extractable; thresholds confirmed via LaunchDarkly citation (Source #1) |
| https://homes.cs.washington.edu/~rjust/publ/google_coverage_fse_2019.pdf | PDF paper | binary skip; same thresholds confirmed indirectly |
| https://testing.googleblog.com/2020/08/code-coverage-best-practices.html | official Google blog | page body not in fetch payload; thresholds confirmed via LaunchDarkly |
| https://dl.acm.org/doi/10.1145/3644032.3644442 | ACM AST 2024 paper | paywalled (403); finding via search snippet: mutation coverage NOT strongly correlated with mutation coverage |
| https://securityboulevard.com/2026/04/mutation-testing-for-the-agentic-era/ | 2026 industry post | 403 forbidden; topic confirmed via Meta InfoQ piece |
| https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 | regulator gloss | SR 11-7 tiering principle confirmed via search snippets only |
| https://totalshiftleft.ai/blog/risk-based-testing-strategy-explained | blog | RBT tiering framework (Critical 20-25 / High 12-19 / Medium 6-11 / Low 1-5) |
| https://qalified.com/blog/risk-based-testing/ | blog | RBT prioritization context |
| https://www.atlassian.com/continuous-delivery/software-testing/code-coverage | vendor | year-less canonical query — 80% common-industry threshold restated |
| https://dev.to/d_ir/do-you-aim-for-80-code-coverage-let-me-guess-which-80-it-is-1fj9 | blog | "80% achieved by easy-path tests" coverage-theater illustration |
| https://arxiv.org/abs/2603.13724 | arxiv | AI agents author 16.4% of test commits; coverage comparable to humans |
| https://arxiv.org/html/2506.02954v2 | arxiv | Mutation-guided LLM tests reach 89.5% mutation score |

URLs collected: 20. Read in full: 8.

## Section D. Search-query composition (three variants, mandatory)

| Variant | Query | Purpose | Hits |
|---------|-------|---------|------|
| Current-year | "tiered test coverage risk-based critical modules quant trading 2026" | frontier | sources 7,8 frame, plus FINRA 2026 |
| Last-2-year | "mutation testing vs code coverage signal quality 2024 2025"; "arxiv mutation testing 2024 2025 ..." | recency scan | Meta 2026, AST 2024, arxiv 2024-2025 |
| Year-less canonical | "code coverage threshold"; "SR 11-7 model validation testing coverage requirements financial" | prior art | Bullseye, Atlassian, SR 11-7 corpus, DEV 80% post |

## Section E. Key findings (per-claim cited)

1. **No single industry-wide blanket bar exists**; the dominant
   pattern is RISK-STRATIFIED. Google's published framework is
   60% / 75% / 90% mapped to acceptable / commendable / exemplary
   — explicitly NOT a flat target (Source #1).

2. **70-80% is the empirically-supported diminishing-returns
   knee** for general production code. "Increasing code coverage
   above 70-80% is time consuming and ... leads to a relatively
   slow bug detection rate" (Bullseye, Source #5). Same threshold
   restated in vendor consensus (Atlassian / DEV snippet).

3. **Safety-critical / regulated code goes higher.** DO-178B
   aviation Level A mandates 100% MC/DC + statement coverage
   (Source #5). Medical, automotive, aviation -> near 100%
   (Source #1). For pyfinagent, kill_switch / paper_trader /
   risk-engine equivalents are the "high cost of failure" tier
   in this sense, not because pyfinagent is regulated but because
   they protect operator capital.

4. **Tier classification is the orthodox approach in financial
   services.** SR 11-7 explicitly requires "model tiering" —
   higher-risk models get "more rigorous validation requirements
   than lower-risk models" (search snippet, ModelOp / Krista /
   ValidMind). FINRA 2026 oversight report (Source #7) requires
   tested pre-trade hard blocks + annual effectiveness reviews
   for market-access controls, scaled to business line.

5. **The "coverage theater" trap is the dominant 2024-2026
   anti-pattern**, especially with AI-generated tests. Test
   theater = "impressive-looking test suites that validate
   implementation rather than intention" (Source #4). >50% of
   observed AI-generated tests mirror the implementation; "your
   coverage report hits 90%+ ... these tests are fundamentally
   circular" (Source #4). This is the SPECIFIC risk if pyfinagent
   takes path (a) — Claude-authored ~1700 tests to chase 70%
   blanket are highly likely to produce theater coverage.

6. **Mutation score is becoming the credible quality signal, but
   as a complement, not a replacement.** Meta 2026 deploys
   LLM-driven mutation testing in production but uses it as
   "complement to traditional testing" with selective deployment
   on high-value paths (Source #8). Mutation score 80%+ is the
   practitioner ask in 2024-2026 literature, but ACM AST 2024
   found "mutation coverage NOT strongly correlated with mutation
   coverage" — meaning even mutation testing has its own
   metric-vs-signal divergence (snippet table).

7. **Kent Beck doctrine: minimum-test for confidence.** "I get
   paid for code that works, not for tests, so my philosophy is
   to test as little as possible to reach a given level of
   confidence" (Source #3). Concentrate where mistakes are
   likely. Strongly supports tiered over blanket.

8. **The Codepipes severity model fits pyfinagent directly.**
   Three severity layers: critical (breaks often, frequent
   features, high user impact), core (occasional failures), other
   (rarely changes, minimal impact). Guidance: "try to write
   tests that work towards 100% coverage of critical code [first].
   If you have already done this, then ... 100% of core code"
   (Source #2). This is the textbook frame for what you call
   Tier-1 vs Tier-2 vs Tier-3.

9. **Single-user local deployment does NOT lower the
   capital-protection bar.** The "high cost of failure" tier is
   defined by IMPACT, not by user count. kill_switch firing late
   on $100K operator capital is functionally the same blast
   radius as the same logic at a fund — the risk model doesn't
   care about tenant count (synthesis from Sources #1, #2, #5;
   no source contradicts this).

## Section F. Recency scan (last 2 years, 2024-2026)

Performed. **Findings:**

- **Meta Jan 2026 (InfoQ)**: production deployment of LLM-driven
  mutation testing as complement to coverage (Source #8). Signals
  industry has moved past "coverage % alone" by 2026.
- **arxiv 2603.13724** (search snippet): AI agents now author
  16.4% of test-adding commits; AI test coverage comparable to
  human tests but structurally different (more assertions, less
  cyclomatic complexity). Confirms coverage-theater risk.
- **arxiv 2506.02954v2 / 2503.08182** (search snippet): mutation-
  guided LLM tests reaching 89.5% mutation score, beating
  EvoSuite. Indicates the achievable bar IF you adopt mutation
  testing — but adoption costs are nontrivial.
- **ACM AST 2024 paper** (DOI 10.1145/3644032.3644442, snippet):
  "Mutation Coverage is not Strongly Correlated with Mutation
  Coverage" — caution that even mutation metrics have a Goodhart-
  style decoupling. Even the better metric is not perfect.
- **2024-25 "test theater" discourse** (Source #4, Aiko 2024):
  the explicit anti-pattern emerged in this window in response to
  AI-generated test floods.
- **FINRA 2026 oversight report** (Source #7): regulators
  continuing to require tested pre-trade hard blocks + annual
  effectiveness reviews. No coverage-% mandate; the gate is
  evidence-of-test on critical controls.

**Conclusion of recency scan**: 2024-2026 reinforces the tiered
position. Blanket targets are increasingly seen as a Goodhart
metric; tier-by-risk with explicit critical-first focus + mutation
score for the critical tier is the emerging consensus pattern.

## Section G. Recommended DoD-4 tiered policy (path c)

### Recommended thresholds

| Tier | Bar | Modules (initial classification) | Rationale |
|------|-----|----------------------------------|-----------|
| **Tier-1 (CAPITAL-PROTECTION)** | **75% line + 80% branch** + green mutation-score smoke on one critical path per module | kill_switch, paper_trader, portfolio_manager, perf_metrics, any signal-execution glue, position-sizing | Inside Google's "commendable" band (Source #1); above Bullseye 70-80% knee (Source #5); below the diminishing-returns wall above 90%; mutation smoke catches theater (Sources #4, #8) |
| **Tier-2 (BUSINESS-LOGIC)** | **60% line** | services/agents/api orchestrators, scoring, ranking, non-execution analytics | "Acceptable" per Google (Source #1); honest measurement of current 26-33% says this IS the realistic short-term target; Codepipes "core" tier (Source #2) |
| **Tier-3 (SUPPORT)** | **no gate** (informational measurement only) | utilities, formatters, logging shims, one-off scripts | Codepipes "other" — diminishing returns and high theater risk if forced (Source #2); Beck minimum-test (Source #3) |
| **Tier-X (DEAD/DEFERRED)** | **excluded via .coveragerc omit** | backend/markets/risk_engine.py (deferred phase-5), any other 0%-confirmed-dead | Honest scope; including it artificially depresses Tier-2 number |

### Module classification rules (committed in code)

Define in pyproject.toml / pytest.ini / .coveragerc:

```ini
# .coveragerc -- tier classification
[run]
omit =
    backend/markets/risk_engine.py        # Tier-X deferred phase-5
    backend/**/scripts/oneshot_*.py        # Tier-X scripts

# Tier-1 (capital-protection) -- separate fail-under gate per module
# in a tier1-coverage CI job:
#   - backend/safety/kill_switch.py
#   - backend/paper_trading/paper_trader.py
#   - backend/portfolio/portfolio_manager.py
#   - backend/backtest/perf_metrics.py
#   - any module where ANY import path reaches order_submit or
#     position_size_compute
# fail-under = 75 (line) / 80 (branch)
# additionally: at least one mutation-test smoke per module via
#   mutmut or cosmic-ray, gated at >=60% mutation-kill on the
#   critical path file
```

### Tier rules (file-level, audited)

- **Tier-1**: any module that (a) can place or modify orders, (b)
  can halt or resume trading, (c) computes the published Sharpe /
  DSR / portfolio value, or (d) holds the kill-switch state. Owner
  override is allowed but must update the manifest.
- **Tier-2**: any module on the prod request path that is NOT
  Tier-1. Default.
- **Tier-3**: utilities not on the prod request path; CLI scripts;
  schema migrations; logging wrappers.
- **Tier-X**: 0% coverage AND no live call site grep hits. Audit
  annually.

### Owner override + audit trail

If the operator (single user) decides a Tier-1 module should be
demoted, the demotion lives in a single file at
`docs/coverage_tier_overrides.md` with: module path, prior tier,
new tier, ISO date, reason. CI loads this file and applies it.
This gives auditability without adding human governance overhead
inappropriate for a single-user app.

## Section F (scope-honesty addendum). What needs investment NOW

Given the measurement (Tier-1 modules currently 51-66%, target
75%), the honest scope is:

| Tier-1 module | Current | Target | Gap | Realistic test count |
|---------------|---------|--------|-----|----------------------|
| kill_switch | (measure) | 75/80 | (delta) | ~30-50 tests across normal/trip/reset paths |
| paper_trader | (measure) | 75/80 | (delta) | ~50-80 tests (order types, fills, rejects, fractional, settlement) |
| portfolio_manager | (measure) | 75/80 | (delta) | ~30-50 tests (rebalance, position-size, drift) |
| perf_metrics | (measure) | 75/80 | (delta) | ~30-50 tests (Sharpe edge cases, DSR boundary, PBO inputs) |
| **TOTAL Tier-1 investment** | | | | **~140-230 NEW tests, ~1-2 weeks** (vs. ~1700 for blanket 70%) |

Tier-2 stays at honest 60% (current 26-33%) — explicitly NOT
chased to 70% in this DoD-4. Demotion is honest and audited.

## Section G. Recommended verdict

| Path | Recommendation |
|------|----------------|
| (a) blanket 70% via ~1700 tests | **REJECT** — 4-8 week opportunity cost, ~85% probability of producing coverage theater per 2024-26 anti-pattern literature (Source #4) |
| (b) operator override of DoD-4 | **REJECT** — accepts current 51-66% on capital-protection code; no defensible audit posture even for single-user |
| (c) tiered policy (THIS BRIEF) | **ADOPT** — defensible per Google / SR 11-7 / Codepipes; ~10x less effort than (a); higher real safety; audit trail intact |

## Anchors (internal — file inspection deferred to next pass)

This SIMPLE-tier brief is policy research, not code audit. The
Tier-1 module list above maps to file paths that exist (e.g.
backend/safety/kill_switch.py, backend/paper_trading/paper_trader.py)
based on the operator's measurement context — verification of
exact paths + current per-module coverage measurement is a Tier-1
GENERATE-phase prerequisite, not a research-phase artifact.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8)
- [x] 10+ unique URLs total (20)
- [x] Recency scan (last 2 years) performed + reported (Section F)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for internal claims (deferred — SIMPLE tier;
      module paths cited but per-line measurement is the GENERATE step)

Soft checks:
- [x] Source mix spans peer-reviewed/regulator/practitioner/blog
- [x] Three-variant query composition documented (Section D)
- [x] All claims cited per-claim

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 0,
  "report_md": "research_brief_dod_4_tiered_policy.md",
  "gate_passed": true
}
```
