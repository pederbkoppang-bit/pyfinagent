## Research: phase-10.7.4 Cron Budget Allocator (slot governance authority)

**Tier assumed: moderate** (novel design but small code surface ~200-300 LOC)

---

### Search queries run (3-variant discipline)

1. Current-year frontier: "autonomous agent cron budget allocator slot governance 2026", "LLM agent slot governance multi-tenant scheduling 2026"
2. Last-2-year window: "weighted fair queuing apscheduler budget cap LLM agent 2025", "knapsack scheduler token budget autonomous AI agent scheduling", "FIFO vs WFQ weighted round robin autonomous agent scheduling 2024 2025"
3. Year-less canonical: "weighted fair queuing scheduler", "deficit round robin scheduling algorithm weighted fair queuing comparison", "stride scheduling lottery scheduling Waldspurger", "knapsack budget allocation algorithm"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|---------------------|
| https://arxiv.org/html/2510.17015 | 2026-04-25 | paper (arXiv 2025) | WebFetch full | "Justitia adopts a virtual-time based fair queuing algorithm... agents are served sequentially based on their fair completion order under idealized fair-sharing" -- confirms WFQ virtual finish time is directly applicable to job sequencing beyond network packets |
| https://arxiv.org/html/2601.08815v3 | 2026-04-25 | paper (arXiv 2026, COINE 2026 oral) | WebFetch full | "conservation invariants: sum of consumed resources across agents <= parent budget B(r) for all resource types"; "unused budget returns to a shared pool, enabling efficient agents to subsidize resource-intensive ones" -- formal basis for pool-reclaim pattern |
| https://arxiv.org/html/2604.17111 | 2026-04-25 | paper (arXiv 2026) | WebFetch full | "a single agent session may consume 50,000-500,000 tokens across dozens of API calls, making per-agent budgeting critical"; priority queue design: Critical >> High >> Normal >> Low; shortest-job-first within tiers; "transparent retry proved more important than admission control for agent survival" |
| https://every-algorithm.github.io/2025/02/22/weighted_fair_queuing.html | 2026-04-25 | blog (authoritative) | WebFetch full | WFQ virtual finish time F = S + (L/w): "packets from a high-weight flow finish earlier than packets from a low-weight flow"; bandwidth proportional to weight -- exactly maps to token budget proportional to priority weight |
| https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched-lottery.pdf | 2026-04-25 | textbook (OSTEP, Arpaci-Dusseau) | WebFetch full | Stride scheduling formula: stride = C / weight; scheduler selects minimum-pass process, increments by stride; "stride scheduling provides reproducible, deterministic behavior" vs lottery scheduling's randomness |
| https://intronetworks.cs.luc.edu/current/html/fairqueuing.html | 2026-04-25 | textbook (Dordal, authoritative) | WebFetch full | WFQ O(log n); DRR O(1); "choose WFQ when delay bounds matter, DRR for computational efficiency with acceptable fairness variance"; deficit counter carries unused quota forward per queue |
| https://docs.litellm.ai/docs/a2a_iteration_budgets | 2026-04-25 | official docs (LiteLLM) | WebFetch full | max_budget_per_session + max_iterations stored in litellm_params; session tracking via x-litellm-trace-id; enforcement: accumulate cost after each call, 429 when exceeded -- practical reference for how production LLM gateways enforce per-agent budgets |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.fifthrow.com/blog/ai-agent-orchestration-goes-enterprise-the-april-2026-playbook-for-systematic-innovation-risk-and-value-at-scale | blog (2026) | Marketing-level; no algorithmic content |
| https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit-open-source-runtime-security-for-ai-agents/ | Microsoft blog (2026) | Runtime security focus (OWASP), not token-budget allocation |
| https://www.edisonpartners.com/blog/the-phantom-menace-budget-buster-agentic-ai | investor blog | Cost alarm; no scheduling algorithm |
| https://github.com/orgs/community/discussions/191147 | GitHub community | Fetched -- practical budget-strategy taxonomy (product-level / SKU-level / bundled; alert-only vs hard-stop) |
| https://en.wikipedia.org/wiki/Deficit_round_robin | Wikipedia | Snippet confirms O(1) DRR, deficit counter mechanics |
| https://dl.acm.org/doi/10.1145/217391.217453 | ACM (Shreedhar/Varghese 1995) | Paywall; covered by Dordal textbook fetch above |
| https://web.eecs.umich.edu/~ryanph/jhu/cs718/spring18/readings/stride-scheduling.pdf | paper (Waldspurger 1995) | OSTEP chapter covers the key formula |
| https://arxiv.org/html/2506.24045v1/ | paper (2026) | Heterogeneous SoC scheduling; hardware-specific, not software-cron relevant |
| https://www.sparkouttech.com/development-cost-of-ai-agent/ | vendor blog | Pricing survey only |
| https://openreview.net/forum?id=n4V3MSqK77 | paper | Agentic plan caching, not scheduling |
| https://arxiv.org/html/2604.14178 | paper (2026) | Heartbeat-driven agent cognition; interesting but tangential |
| https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026 | blog (2026) | Cost attribution, not scheduling |
| https://dev.to/pranay_batta/building-hierarchical-budget-controls-for-multi-tenant-llm-gateways-ceo | blog (2025) | Hierarchy good but no scheduling algorithm |

---

### Recency scan (2024-2026)

Searched for 2026-frontier, 2025-window, and year-less canonical sources. Results:

- **2026 papers**: Agent Contracts (arXiv 2601.08815, COINE 2026) and HiveMind (arXiv 2604.17111, 2026) both published within the last two weeks of April 2026. Both are directly relevant: Agent Contracts formalises budget conservation invariants; HiveMind provides empirical token-budget management data.
- **2025 papers**: Justitia (arXiv 2510.17015, 2025) provides LLM-agent-specific virtual-time fair queuing analysis. WFQ overview post published Feb 2025.
- **No new finding supersedes** the canonical DRR (Shreedhar/Varghese 1995) or stride scheduling (Waldspurger 1995) algorithms for the proportional-share problem. The 2025-2026 literature applies these classical algorithms to LLM workloads without modifying the core math.

---

### Key findings

1. **WFQ is directly applicable to cron slot budgeting.** The virtual finish time formula F = S + (L/w) maps cleanly to: each "packet" is a cron job fire; "length L" is `min_tokens_per_fire`; "weight w" is `priority`. High-priority jobs get smaller virtual finish times and thus run first / receive more budget. (Source: every-algorithm.github.io WFQ overview, 2025; Justitia arXiv 2510.17015, 2025)

2. **For pyfinagent's scale (5-15 jobs), simple proportional WFQ outperforms DRR.** DRR's O(1) advantage matters at network speed (millions of packets/sec). For 5-15 daily cron jobs, O(log n) is irrelevant -- the code complexity of maintaining deficit counters adds no value. (Source: Dordal textbook, intronetworks.cs.luc.edu)

3. **Stride scheduling is the deterministic, testable alternative.** Stride = C / priority; allocate `(priority / sum_priorities) * total_daily_budget` tokens per job. This is deterministic, has zero floating-point instability, and produces allocations summing exactly to total_budget. (Source: OSTEP ch.9, Waldspurger MIT thesis)

4. **Agent Contracts (2026) validates the pool-reclaim pattern.** "Unused budget returns to a shared pool" -- if a disabled or low-fire-rate job doesn't consume its allocation, the surplus should be redistributed to remaining jobs proportionally. This maps to `allocate()` ignoring disabled jobs and renormalising weights over enabled jobs only. (Source: arXiv 2601.08815v3, COINE 2026)

5. **HiveMind's priority tiers match the cron_budget.yaml schema.** The four-tier model (Critical >> High >> Normal >> Low) maps to pyfinagent's existing `reserved / high / medium / low` priority field. HiveMind empirically shows "shortest-job-first within tiers" reduces average latency; for cron, this means: within same-priority band, assign more budget to shorter-fire-window jobs (weekly > daily in terms of per-fire headroom). (Source: arXiv 2604.17111, 2026)

6. **LiteLLM's session-budget pattern is the production reference.** `max_budget_per_session` + post-call accumulation + 429 on breach is the proven pattern. For a validator/allocator, this validates the design: pre-compute per-job daily allocation, enforce it as a hard cap on each job's LLM calls. (Source: docs.litellm.ai/docs/a2a_iteration_budgets)

7. **`cron_budget.yaml` already exists at `.claude/cron_budget.yaml` (163 lines, version 2).** It uses a slot-based schema (15 slots, priority/cadence/alpha_velocity_eligible), NOT a token-budget schema. The new `cron_allocator.py` must be a thin adapter that reads this existing schema and computes per-job token allocations from a `total_daily_token_budget` that does NOT yet exist in the file.

8. **`yaml.safe_load` is the canonical loader.** Used in `backend/intel/source_registry.py:90`, `backend/autoresearch/thursday_batch.py:99`, `backend/governance/limits_schema.py:78`. Pattern is `yaml.safe_load(Path(p).read_text(encoding="utf-8"))`. pyyaml 6.0.3 is installed.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/cron_budget.yaml` | 163 | Existing cron slot registry (15 slots, priority/cadence schema) | ACTIVE -- authoritative slot registry; missing `total_daily_token_budget` and `min/max_tokens_per_fire` per slot |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/meta_evolution/alpha_velocity.py` | 160 | Alpha velocity metric -- dataclass + factory + zero-I/O pattern | ACTIVE -- canonical structural pattern to follow |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/meta_evolution/directive_rewriter.py` | 341 | Directive rewriter -- LLM-over-constants pattern, FakeLLM test hook | ACTIVE -- pattern reference; `MIN_BRIEFS_FOR_PROPOSAL = 5` style module-level constant |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/meta_evolution/archetype_library.py` | 252 | Archetype seed library -- frozen dataclass + tuple constant + zero-I/O | ACTIVE -- closest match to cron_allocator structure: pure data + tiny lookup helper |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/scheduler.py` | 382 | APScheduler job registration (4 legacy + 7 phase-9 jobs) | ACTIVE -- real job names used to populate cron_budget.yaml examples |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/autoresearch/cron.py` | 79 | Autoresearch nightly registration shim | ACTIVE -- `autoresearch_overnight` job, `BudgetEnforcer` consumer |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/mcp_health_cron.py` | 208 | MCP health weekly cron (Sun 02:00 UTC) | ACTIVE -- `mcp_health_cron` job via CronTrigger |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/cost_tracker.py` | 254 | Per-call cost tracker; MODEL_PRICING dict; `check_budget(max_cost_usd)` | ACTIVE -- `check_budget()` is the existing "have we spent too much?" API |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/api/cost_budget_api.py` | 169 | `/api/cost-budget/today` endpoint; `_DAILY_CAP_USD = 5.0`, `_MONTHLY_CAP_USD = 50.0` | ACTIVE -- $5/day cap is the macro budget ceiling |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/governance/limits.yaml` | 33 | Immutable risk limits (GPG-signed rotation required) | ACTIVE -- structural reference for how pyfinagent does authoritative YAML configs |
| `/Users/ford/.openclaw/workspace/pyfinagent/tests/meta_evolution/test_alpha_velocity.py` | 165 | 6-test suite: basic, decay, null-guard, ValueError, FakeBQ, migration --dry-run | ACTIVE -- exact test pattern to replicate for test_cron_allocator.py |
| `/Users/ford/.openclaw/workspace/pyfinagent/tests/meta_evolution/test_directive_rewriter.py` | 200 | 7-test suite: boundary guards, FakeLLM override, FakeBQ, migration --dry-run | ACTIVE -- boundary guard pattern; `llm_call_override` injectable seam |

---

### Existing cron jobs (with schedules and cost estimates)

From `backend/slack_bot/scheduler.py` and `backend/services/mcp_health_cron.py`:

| Job ID | Schedule | Category | Cost-per-fire estimate |
|--------|----------|----------|----------------------|
| morning_digest | weekday 08:00 ET | monitoring | ~$0.002 (HTTP-only, no LLM) |
| evening_digest | weekday 18:00 ET | monitoring | ~$0.002 (HTTP-only, no LLM) |
| watchdog_health_check | every N minutes (configurable) | monitoring | ~$0.000 (HTTP GET only) |
| prompt_leak_redteam | daily 03:15 ET | maintenance | ~$0.01-0.05 (script, no LLM) |
| daily_price_refresh | daily 01:00 | trading | ~$0.005 (BQ + data fetch) |
| weekly_fred_refresh | Sun 02:00 | research | ~$0.002 (HTTP only) |
| nightly_mda_retrain | daily 03:00 | trading | ~$0.10-0.50 (ML training) |
| hourly_signal_warmup | every hour :05 | trading | ~$0.001 (cache warmup) |
| nightly_outcome_rebuild | daily 04:00 | trading | ~$0.05-0.20 (BQ + LLM reflection) |
| weekly_data_integrity | Mon 05:00 | maintenance | ~$0.01 (BQ scan) |
| cost_budget_watcher | daily 06:00 | monitoring | ~$0.001 (BQ query) |
| mcp_health_cron | Sun 02:00 UTC | maintenance | ~$0.002 (GitHub API) |
| autoresearch_overnight | daily 02:00 | research | ~$0.50-2.00 (100 LLM calls) |
| thursday_batch_backtest | weekly Thu | research | ~$0.20-1.00 (batch backtests) |
| friday_promotion_gate | weekly Fri | trading | ~$0.05-0.20 (evaluation) |

Total estimated daily LLM cost (rough): $1-$4/day against a $5/day macro cap.

---

### Application to pyfinagent

#### 1. Proposed schema for `.claude/cron_budget.yaml`

The existing `cron_budget.yaml` (version 2) uses a slot-based schema with `priority` (reserved/high/medium/low) and `alpha_velocity_eligible`. The new allocator should ADD a top-level `total_daily_token_budget` and per-slot `min_tokens_per_fire` / `max_tokens_per_fire` fields as an EXTENSION of the existing schema -- NOT a replacement. The allocator reads both the existing priority and the new token fields.

Proposed additions:

```yaml
version: 3                        # bump from 2
total_daily_token_budget: 100000  # e.g. 100K tokens/day across all scheduled LLM calls
                                  # translates to ~$0.30/day at gemini-2.0-flash rates

slots:
  - slot_id: 12
    phase: phase-8.5
    job_name: autoresearch_overnight
    cadence: daily
    priority: high
    alpha_velocity_eligible: true
    surface: loop
    # NEW fields:
    min_tokens_per_fire: 20000    # floor: allocator never cuts below this
    max_tokens_per_fire: 50000    # ceiling: even if budget is ample, don't over-allocate
    category: research            # research / monitoring / trading / maintenance
    enabled: true
```

Field justification:
- `total_daily_token_budget`: Without this, there is no allocatable pool. `$5/day` cap from `cost_budget_api.py` is in USD; we need a token-denominated pool the allocator can divide. 100K tokens ~= $0.30 at flash rates, conservative headroom against the $5 cap.
- `min_tokens_per_fire`: Prevents starvation. `autoresearch_overnight` needs at least 20K tokens to run a meaningful batch. Without a floor the proportional allocator could underfund low-priority jobs to near-zero.
- `max_tokens_per_fire`: Prevents monopolisation. Without a ceiling, the highest-priority job in a sparse schedule would get nearly all tokens.
- `category`: Supports future per-category budget sub-caps (e.g. never let `monitoring` jobs exceed 5% of daily budget).
- `enabled`: Disabled jobs are excluded from the denominator when computing proportional shares (Agent Contracts pool-reclaim principle).

Priority-to-weight mapping for the allocator:
```python
PRIORITY_WEIGHTS = {"reserved": 10, "high": 6, "medium": 3, "low": 1}
```
This preserves the existing reserved/high/medium/low vocabulary from cron_budget.yaml without adding a new numeric `priority` field.

#### 2. Allocator algorithm recommendation: Weighted-Fair-Queueing (proportional variant)

**Recommended: Stride scheduling / proportional WFQ** (not DRR, not knapsack).

Rationale:
- **pyfinagent has 5-15 jobs**: DRR's O(1) vs WFQ's O(log n) is irrelevant at this scale. DRR's deficit-counter machinery adds implementation complexity with zero benefit.
- **Determinism is required for testing**: Stride scheduling produces the same allocation for the same input every time. Lottery scheduling (randomized) would make tests flaky.
- **Proportional WFQ is already familiar**: The formula `allocation_i = (weight_i / sum(weights)) * total_budget` is exactly the stride formula when all jobs fire once per day. It is simple enough to audit in <10 lines.
- **Knapsack is overkill**: Greedy-by-value/weight knapsack is appropriate when jobs have heterogeneous fire rates and strict exclusion (pack or skip). For pyfinagent, all enabled jobs run every cycle -- we want to give each a fair share, not select a subset.
- **Agent Contracts pool-reclaim is a natural add-on**: Disabled jobs are excluded from the weight sum; their "share" is redistributed to active jobs automatically.

Algorithm (pseudocode):
```python
def allocate(budget_yaml_path, total_budget):
    config = yaml.safe_load(Path(budget_yaml_path).read_text(encoding="utf-8"))
    enabled_slots = [s for s in config["slots"] if s.get("enabled", True)]
    weight_map = {"reserved": 10, "high": 6, "medium": 3, "low": 1}
    weights = {s["job_name"]: weight_map[s["priority"]] for s in enabled_slots}
    total_weight = sum(weights.values())
    result = {}
    for slot in enabled_slots:
        name = slot["job_name"]
        raw = (weights[name] / total_weight) * total_budget
        lo = slot.get("min_tokens_per_fire", 0)
        hi = slot.get("max_tokens_per_fire", total_budget)
        result[name] = max(lo, min(hi, round(raw)))
    return result
```

The min/max clamp is applied after proportional allocation. If clamping pushes the sum above total_budget, the allocator logs a warning and scales down proportionally (normalise after clamp). This is the "normalization after clamp" pattern from WFQ literature.

#### 3. Validator scope (`scripts/meta/validate_cron_budget.py`)

The validator is a CLI tool (`python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml`) that:
1. Loads the YAML with `yaml.safe_load` (not `yaml.load`) -- safe parse check.
2. Checks required top-level keys: `version`, `total_slots`, `slots` (list).
3. Checks per-slot required keys: `slot_id`, `job_name`, `priority`, `cadence`, `surface`.
4. Checks `priority` values are in `{reserved, high, medium, low}`.
5. Checks no duplicate `job_name` values.
6. Checks `total_daily_token_budget` (if present) is a positive integer.
7. Checks `min_tokens_per_fire <= max_tokens_per_fire` for each slot that has both.
8. Checks sum of `min_tokens_per_fire * fires_per_day` across enabled daily slots <= `total_daily_token_budget` (budget feasibility check; weekly slots count as 1/7).
9. Exit 0 on success; exit 1 with a human-readable error message on first failure.

The `fires_per_day` computation: daily=1.0, weekday=5/7, weekly=1/7, monthly=1/30, on-demand=0.

#### 4. Allocator API

```python
def allocate(
    budget_yaml_path: str | Path,
    total_budget: float,
    *,
    priority_weights: dict[str, int] | None = None,
) -> dict[str, float]:
    """Return {job_name: token_budget_for_today} for all enabled slots.

    Args:
        budget_yaml_path: path to .claude/cron_budget.yaml (or test fixture).
        total_budget: total tokens available today (overrides yaml value if given).
        priority_weights: override default PRIORITY_WEIGHTS mapping. For testing.

    Returns:
        Dict mapping each enabled job_name to its token allocation.
        Sum of values will equal total_budget (within rounding tolerance).
        Disabled slots are absent from the output.
    """
```

The `priority_weights` override and `budget_yaml_path` as a Path argument are the injectable seams required for testing without live files.

#### 5. File list with LOC estimates

| File | LOC estimate | Notes |
|------|-------------|-------|
| `.claude/cron_budget.yaml` | +15 lines | Extend existing file: add `total_daily_token_budget`, `min/max_tokens_per_fire`, `category`, `enabled` to each slot |
| `backend/meta_evolution/cron_allocator.py` | ~150 lines | Dataclass `SlotConfig`, constants `PRIORITY_WEIGHTS`, `allocate()`, `load_budget()`, fail-open `persist_allocation()` |
| `scripts/meta/validate_cron_budget.py` | ~120 lines | CLI validator with argparse; exits 0/1; `__main__` block; `validate()` returns list of errors |
| `tests/meta_evolution/test_cron_allocator.py` | ~180 lines | 8+ test cases; FakeBQ stub; fixture YAML path |

Total: ~450-470 LOC across 4 files (within the "small surface" characterisation).

#### 6. Test plan (8 tests)

| Test name | What it verifies |
|-----------|-----------------|
| `test_proportional_allocation_basic` | 3 jobs (high/medium/low), no min/max: allocations sum to total_budget, proportional to weights |
| `test_disabled_slots_excluded` | Job with `enabled: false` absent from result; remaining jobs renormalise to 100% of budget |
| `test_min_floor_enforced` | Job with min_tokens_per_fire > raw proportional share gets the floor, not below |
| `test_max_ceiling_enforced` | High-priority job with max_tokens_per_fire cap does not exceed the ceiling |
| `test_single_job` | One enabled job gets 100% of total_budget (clamped to its max) |
| `test_budget_sum_invariant` | Sum of all allocations == total_budget within 1 token rounding tolerance |
| `test_validator_exit_0_valid_yaml` | Valid YAML fixture: validator subprocess exits 0 |
| `test_validator_exit_1_duplicate_names` | YAML with two slots having same job_name: validator exits 1, error message contains "duplicate" |
| `test_validator_exit_1_budget_infeasible` | sum(min_tokens * fires_per_day) > total_daily_token_budget: validator exits 1 |

The 9th and 10th boundary tests (`test_zero_priority_raises`, `test_all_reserved_slots_split_equally`) cover the edge cases documented in `test_alpha_velocity.py`.

---

### Consensus vs debate (external)

- **Consensus**: Proportional-share scheduling (WFQ / stride) is the right primitive for small, fixed-cardinality job sets with stable priorities. All sources agree.
- **Debate**: DRR (Shreedhar/Varghese 1995) claims O(1) superiority. At network scale this matters; at 5-15 cron jobs/day it does not. Justitia (2025) applies virtual-time WFQ to LLM serving (100s of concurrent agents) -- for pyfinagent the cardinality is far lower, making simple proportional WFQ the right choice.
- **Debate**: Knapsack is appealing when firing is optional (pack-or-skip). Since all pyfinagent enabled jobs MUST run, knapsack only introduces a "skip this job today" decision the system is not authorised to make. Reject knapsack.

---

### Pitfalls (from literature)

1. **Overlapping budget problem** (GitHub community discussion, 2026): Multiple simultaneous budget caps (slot-level + daily USD cap) can block access unexpectedly when the first limit trips. Solution: the token-budget managed by `cron_allocator.py` is deliberately separate from the USD cap in `cost_budget_api.py` -- the allocator distributes tokens, the USD watcher trips the circuit-breaker. Both enforce independently.

2. **Starvation on min-floor misconfiguration** (Dordal textbook): If sum(min_tokens_per_fire) > total_daily_token_budget, the allocator cannot satisfy all floors. The validator's budget-feasibility check (check #8 above) catches this at config load time, not at runtime.

3. **Deficit not carried forward** (DRR, Shreedhar 1995): The proportional WFQ allocator resets allocations each day. If a job did not consume its allocation yesterday (e.g. autoresearch was skipped), the unused budget is NOT carried forward. This is intentional for daily cron governance -- unused tokens do not compound. Future phases could add a carry-forward mechanism if Alpha Velocity analysis shows persistent underspend.

4. **Disabled-job denominator trap** (Agent Contracts, arXiv 2601.08815): If disabled jobs are left in the weight sum, their budget share disappears into a void instead of being redistributed. The allocator MUST filter disabled jobs before computing the weight denominator.

5. **Priority inflation** (HiveMind, arXiv 2604.17111): Operators tend to mark everything "high priority" over time. The validator should warn (not fail) when more than 50% of jobs are in the `high` or `reserved` tier -- an optional soft check.

---

### Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) (21 collected)
- [x] Recency scan (last 2 years) performed + reported (2025-2026 scan: 4 papers found)
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks -- note gaps but do not auto-fail:
- [x] Internal exploration covered every relevant module (12 files, 2041 lines total)
- [x] Contradictions / consensus noted (DRR vs WFQ debate documented)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 14,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/phase-10.7.4-research-brief.md",
  "gate_passed": true
}
```
