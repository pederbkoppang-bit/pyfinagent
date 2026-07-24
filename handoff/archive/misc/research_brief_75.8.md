# Research Brief — Step 75.8 (tier=complex, NOT audit-class)

**Status:** IN PROGRESS (write-first, incremental)
**Researcher spawn:** Layer-3 harness MAS
**Step:** 75.8 — gauntlet stub-refusal + promotion_gate dry-run write guard + governance/limits divergence WARNING

## Step scope (3 sub-fixes)
- (a) gap6-01: `scripts/risk/gauntlet.py:134` runs `_run_regime_stub` UNCONDITIONALLY; non-dry-run must raise NotImplementedError; stub must refuse to write `dry_run:false` reports; promotion_gate must reject stub-fingerprint reports (per-regime `bt_drawdown==drawdown`).
- (b) gap6-10: `promotion_gate.py:150` `--dry-run` still writes allocation_pct/stage init AND the gauntlet stamp into real optimizer_best.json — guard both writes behind `if not args.dry_run`.
- (c) gap3-02 OBSERVABILITY-ONLY: six governance/limits.yaml RiskLimits have zero enforcement consumers vs live kill-switch (`paper_daily_loss_limit_pct=4.0` vs governed 2%). Add `backend/governance/divergence.py` wired as startup WARNING only; write divergence doc + draft token GOV-LIMITS-DECIDE.

---

## Internal code inventory (file:line anchors) — ALL CLAIMS VERIFIED

### (a) gap6-01 — gauntlet stub fabrication
| File:line | Fact | Verified |
|---|---|---|
| `scripts/risk/gauntlet.py:132-134` | `run(strategy, dry_run, seed)` → `per_regime = [_run_regime_stub(r, rng) for r in REGIMES]` — stub called **UNCONDITIONALLY** | YES |
| `scripts/risk/gauntlet.py:150` | `dry_run` only annotates: `"dry_run": bool(dry_run)` in the report dict | YES |
| `scripts/risk/gauntlet.py:80` | stub returns are pure seeded noise: `rng.normal(loc=0.0003, scale=0.018, size=n)` | YES |
| `scripts/risk/gauntlet.py:90-91` | per-regime `bt_drawdown == drawdown` (both = `dd`), `forced_exits:0` → ratio 1.0 | YES |
| `scripts/risk/gauntlet.py:121-122` | MC block also `bt_drawdown == p99_drawdown` (both = `dd_p99`), `breaches:0` | YES |
| `scripts/risk/gauntlet.py:159-163` | report written to `handoff/gauntlet/<strategy>/report.json` with the `dry_run:false` label if run without `--dry-run` | YES |
| `backend/backtest/gauntlet/evaluator.py:29,41,68` | 4 hard gates: drawdown-ratio ≤2.0, forced_exits==0, mc p99-ratio ≤2.0, breaches==0. Stub passes ALL four **by construction** | YES |
| `scripts/risk/promotion_gate.py:107,116` | `evaluate_gauntlet(report)` blocks only on `overall_pass=False`; never checks `dry_run` or the stub fingerprint | YES |
| `scripts/risk/promotion_gate.py:165-174` | on pass, SHA-256 stamps the report hash into `optimizer_best.json` | YES |
| REGIMES catalog `backend/backtest/gauntlet/regimes.py` | **7 regimes, exactly 1 intraday_only** (`flash_crash_2010`, line 110) → **6 non-skipped** in dry-run | YES |
| callers | **NO python code imports `gauntlet.run`**; only CLI + docstring refs. `phase4_9_redteam.py` calls `autonomous_harness.promote_strategy`, not gauntlet.run → adding NotImplementedError on non-dry-run breaks no importer | YES |

**STEP-TEXT PRIORITY DRIFT (flag):** masterplan bundles gap6-01 under a **P0** step, but `audit_phase75/confirmed_findings.json` rated gap6-01 **P1** — rationale: "`promote_strategy` has no production caller yet (only `phase4_9_redteam.py`) so no live capital currently flows through the synthetic authorization — latent-serious gate-integrity defect." So this is a **latent** defense-in-depth fix, not an actively-bleeding one. Fix is correct regardless; framing matters for Q/A.

### (b) gap6-10 — promotion_gate --dry-run still writes
| File:line | Fact | Verified |
|---|---|---|
| `scripts/risk/promotion_gate.py:86` | `--dry-run` parsed | YES |
| `scripts/risk/promotion_gate.py:150-158` | write-path #1: `if "allocation_pct" not in existing:` → `update_optimizer_best(...)` writes real file — **no `if not args.dry_run` guard** | YES |
| `scripts/risk/promotion_gate.py:165-174` | write-path #2: gauntlet-stamp `OPTIMIZER_BEST.write_text(...)` — **no `if not args.dry_run` guard** | YES |
| `scripts/risk/promotion_gate.py:182` | `dry_run` only echoed into output report | YES |
| `backend/services/promotion_gate.py:125-145` | `update_optimizer_best()` writer: `path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")` — the exact byte format the byte-identical test must expect | YES |

**STEP-TEXT PRIORITY DRIFT (flag):** audit rated gap6-10 **P3 polish**, and notes it is **as-designed** ("docstring lines 3-6 document that --dry-run writes the canary default; the step-4.8.5 acceptance command in harness_log cycle 82 depended on that write") and **init-only** ("existing stage preserved :159-161; stamp requires --require-gauntlet PASS → dry-run cannot advance a stage"). Implication for the fix: **the module docstring lines 3-7 MUST be updated** (they currently promise --dry-run writes allocation_pct; after the guard that is false — a lying docstring otherwise). Real (non-dry-run) init path still works; `optimizer_best.json` already has `allocation_pct:0.05` so the init path is already dead against the live file (good — no prod impact).

### (c) gap3-02 — governance/limits.yaml zero enforcement consumers
| File:line | Fact | Verified |
|---|---|---|
| `backend/governance/limits.yaml:30` | governed `max_daily_loss_pct: 0.02` (**FRACTION** = 2%) | YES |
| `backend/governance/limits.yaml:31` | governed `max_trailing_dd_pct: 0.10` (FRACTION = 10%) | YES |
| `backend/config/settings.py:529` | live `paper_daily_loss_limit_pct = 4.0` (**PERCENT** = 4%) → **DIVERGENT** (2% governed vs 4% actual) | YES |
| `backend/config/settings.py:530` | live `paper_trailing_dd_limit_pct = 10.0` (PERCENT = 10%) → **MATCHES** (10% both) | YES |
| `backend/services/paper_trader.py:1094-1095,1107-1108` | LIVE kill-switch reads `settings.paper_daily_loss_limit_pct` (the real enforcement site; audit cites `check_and_enforce_kill_switch`) | YES |
| `backend/api/paper_trading.py:500-501,558-559` | API breach evals also read the settings value, not limits.yaml | YES |
| `backend/api/settings_api.py:165` | value is **hot-mutable** to [0.5, 25.0] via `PUT /api/settings` — the exact runtime mutation the immutable-limits watcher exists to prevent, yet the watcher guards a file with zero value-consumers | YES |
| `backend/main.py:277-286` | lifespan calls `load_once()` but **discards the return** — only logs the digest; no boot-assert / no `min()` vs settings. This is where the divergence WARNING wires in | YES |
| repo-wide grep | the six field NAMES appear ONLY in `backend/governance/*` + `scripts/{audit,governance}` lint tooling — **zero runtime value-consumers confirmed** | YES |
| `scripts/governance/lint_limits_usage.py` | EXISTING **source AST scanner** (flags literals/env-backdoors/legacy settings attrs) — NOT a runtime value comparator. rule (c) is WARN-only and calls `settings.paper_daily_loss_limit_pct` a "legacy" attr that "should be migrated to the immutable snapshot (phase-4.9 follow-up)" → migration never happened. `divergence.py` is genuinely non-duplicative | YES |

**Precision note:** "zero enforcement consumers" is accurate but subtle — limits.yaml IS loaded/validated/digest-watched at boot; what has zero consumers is the six VALUES (no code reads `limits.max_daily_loss_pct` to enforce anything). `divergence.py` must call `limits_schema.load()` to READ governed values, normalize units (fraction ×100 → percent), and compare vs `settings`.

### Test conventions (from existing phase-75 tests)
- Naming: `backend/tests/test_phase_75_promotion_gate.py` (matches `test_phase_75_*` family). Does NOT exist yet.
- Header idiom: `REPO_ROOT = Path(__file__).resolve().parents[2]` + `sys.path.insert(0, ...)`; "All offline: no BQ, no network."
- Mocking discipline (test_phase_75_mcp_truth.py): prefer `create_autospec(..., instance=True)` over bare `Mock()`; assert BEHAVIOR/OUTCOME, not just envelope shape. Source-string scans allowed only as SUPPLEMENT, never sole evidence (mutation doctrine).
- tmp_path + monkeypatch: promotion_gate module constants `OPTIMIZER_BEST/OUT/GAUNTLET_ROOT` are import-time from `REPO` → test must `monkeypatch.setattr(promotion_gate, "OPTIMIZER_BEST", tmp_file)` etc. + `monkeypatch.setattr(sys, "argv", [...])` then call `main()`.
- `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1` exists (limits_loader.py:112) to stop the watcher SIGKILL during tests that touch limits.yaml — the divergence test should NOT need to mutate limits.yaml (it only READS), but note the env escape exists.

### Anti-vacuous-guard analysis (the 6 success criteria)
1. **NotImplementedError (crit 1):** behavioral — `run(dry_run=False)` must RAISE (real test, not a source scan). Add BOTH guards so each is independently mutation-testable: (i) `if not dry_run: raise NotImplementedError` at top of `run()`; (ii) defense-in-depth `assert report["dry_run"] is True` before the write. Prove "no stub path emits dry_run:false" via: `run(dry_run=True)` → `report["dry_run"] is True`, AND `run(dry_run=False)` raises. Together these are exhaustive over the two branches.
2. **Stub-fingerprint rejection (crit 2):** needs TWO fixtures — (a) all non-skipped regimes `bt_drawdown==drawdown` → REJECTED; (b) realistic divergent report that STILL passes the other 3 gates (ratio≤2, forced_exits 0, breaches 0) → passes. Without the symmetric (b) case the test is vacuous. EDGE: `all()` of an empty regime list is True — guard must be `non_skipped and all(bt==dd ...)` so a 0-regime report is not falsely fingerprinted.
3. **Byte-identical --dry-run (crit 3):** VACUOUS TRAP — `optimizer_best.json` already has `allocation_pct` AND a gauntlet hash, so BOTH writers are already no-ops against the live file; a test using the live file passes even with NO guard. Fixture MUST (i) LACK `allocation_pct` (fires writer #1) and (ii) have a DIFFERENT/absent gauntlet hash under `--require-gauntlet` with a passing report (fires writer #2). REQUIRED control assertion: same fixture under NON-dry-run MUST mutate the file (proves the writers are live + fixture exercises them). Byte compare against the exact `json.dumps(..., indent=2)+"\n"` format.
4. **Divergence pairs (crit 4):** must flag daily-loss 4.0-vs-2.0 divergent AND report trailing-dd 10.0-vs-10.0 as NON-divergent (symmetric/discriminating — else a checker that flags everything passes vacuously). Unit-normalize (limits fraction ×100). Prove pure: `pytest.raises(nothing)` + file/settings unmutated.
5. **git diff surface (crit 5):** zero edits to evaluator thresholds / kill-switch enforcement / DSR-PBO constants / limits.yaml VALUES. `governance_limits_divergence_75.md` must exist with drafted `GOV-LIMITS-DECIDE` token.
6. **ast.parse (crit 6):** trivially satisfied but ensure all 3 touched files parse.

---

## External research

### Search-query discipline (3 variants per topic)
- **Current-year frontier (2026):** "fail-safe defaults refuse to fabricate placeholder stub production code NotImplementedError 2026"; "configuration drift detection runtime divergence observability warn-only 2026"; "backtest overfitting fabricated evidence promotion gate provenance PBO deflated Sharpe 2026".
- **Last-2-year window (2024-2026):** the config-drift 2026 guides (IBM/Octopus/Nudge/StateTech/XOps), Warne 2026-01, ScienceDirect CPCV (2024), turbinefi 2026, arXiv 2603.20319 / 2605.24564.
- **Year-less canonical:** "parse don't validate make illegal states unrepresentable"; "dry-run flag still writes side effects bug CLI design idempotence"; "poka-yoke software mistake-proofing fail closed secure by design principle". Surfaced the founding essays (King 2019, Minsky 2010 slogan, Shingo/Toyota poka-yoke, Bailey-Borwein-LdP PBO).

### Read in full (>=5 required; counts toward the gate) — 7 sources
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-07-23 | authoritative blog (founding essay) | WebFetch HTML | "Refuse, don't return dubious values" — a function should succeed with a guaranteed result or fail outright; make illegal states unrepresentable. Grounds fix (a): the non-dry-run path must FAIL, not fabricate. |
| 2 | https://henrikwarne.com/2026/01/31/in-praise-of-dry-run/ | 2026-07-23 | blog (RECENT 2026-01-31) | WebFetch HTML | dry-run must have NO side effects; "since I know --dry-run will not change anything, it is safe to run without thinking." Conditionals must gate every side effect; core logic stays flag-unaware. Grounds fix (b). |
| 3 | https://docs.pytest.org/en/stable/how-to/tmp_path.html | 2026-07-23 | official doc | WebFetch HTML | tmp_path returns an isolated `pathlib.Path`; canonical byte-compare = store initial bytes, run op, assert `read_text()`/`read_bytes()` unchanged; `list(tmp_path.iterdir())` to prove no write. Grounds the crit-3 test. |
| 4 | https://www.softwaretestinghelp.com/poka-yoke/ | 2026-07-23 | practitioner | WebFetch HTML | Shift from "judgment" (asking people to be careful) to "constraint" (make the mistake impossible by design); prevention > detection. Frames all three fixes as constraint-not-judgment. |
| 5 | https://octopus.com/devops/configuration-management/configuration-drift/ | 2026-07-23 | vendor/practitioner doc | WebFetch HTML | drift = gradual deviation of active config from baseline; detect by continuous comparison vs version-controlled baseline; detection is an "early warning system" before remediation. IaC dry-run modes (`terraform plan`, `ansible --check`) are the canonical no-write preview. Grounds gap3-02 warn-only + fix (b). |
| 6 | https://github.com/angular/angular-cli/issues/6810 | 2026-07-23 | real bug report | WebFetch HTML | **Exact gap6-10 pattern**: `--dry-run` printed "no changes will be written" while STILL modifying `app.module.ts`. Labeled P1 regression. "reassuring message ... while simultaneously performing the exact side effects the flag was supposed to prevent." |
| 7 | https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-07-23 | **peer-reviewed** (Bailey, Borwein, López de Prado, Zhu) | pdfplumber (binary PDF → text, per research-gate PDF chain) | PBO domain anchor: an overfit backtest fits "past noise rather than future signal ... performance out-of-sample is, of course, utterly disappointing"; "backtest overfitting is a deterministic fact." CSCV swaps all IS/OOS partitions. This is WHY a gauntlet stamping seeded noise as PASS is catastrophic — it manufactures the exact false positive the gate exists to catch. |

### Identified but snippet-only (context; does NOT count toward gate) — ~18
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://deviq.com/principles/make-illegal-states-unrepresentable/ | principle doc | reinforces #1; redundant |
| https://www.ibm.com/think/topics/configuration-drift | vendor doc | HTTP 403 (blocked); substituted Octopus |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | peer-reviewed | paywalled abstract; CPCV > traditional OOS methods |
| https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026 | blog (2026) | recency corroboration only |
| https://arxiv.org/pdf/2603.20319 | preprint (2026) | "Implementation Risk in Portfolio Backtesting" — tangential |
| https://arxiv.org/pdf/2605.24564 | preprint (2026) | look-ahead-bias LLM backtesting — tangential |
| https://news.ycombinator.com/item?id=27263136 | forum | "if dry-run/else soup ... bugs caused by your dry-run support" quote |
| https://github.com/argoproj/argo-workflows/issues/12944 | bug report | 2nd dry-run-not-respected instance |
| https://en.wikipedia.org/wiki/Poka-yoke | encyclopedia | Shingo/Toyota origin |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | peer-reviewed | DSR (already a project gate) |
| https://github.com/esvhd/pypbo | code | PBO reference impl |
| https://quality.arc42.org/approaches/fail-safe-defaults | doc | fail-safe defaults |
| https://en.wikipedia.org/wiki/Method_stub | encyclopedia | stub definition |
| https://komodor.com/learn/kubernetes-configuration-drift-causes-detection-and-prevention/ | vendor | k8s drift |
| https://moss.sh/devops-monitoring/configuration-drift-detection/ | blog | drift detection |
| https://devsecopsschool.com/blog/fail-safe-defaults/ | blog (2026) | fail-safe defaults guide |
| https://github.com/cli/cli/issues/1145 | feature req | dry-run flag design |
| https://docs.pytest.org/en/stable/how-to/monkeypatch.html | official doc | monkeypatch sys.argv / setattr pattern |

### Recency scan (2024-2026) — PERFORMED
- **New/complementary:** (1) Warne "In Praise of --dry-run" (2026-01-31) reaffirms the no-side-effect contract and the "flag pollutes the code" factoring cost — directly current for fix (b). (2) 2026 config-drift literature (Octopus/IBM/Nudge/XOps) converges on "detect + warn before remediate," and names **IaC dry-run modes (`terraform plan`, `ansible --check`) as the canonical no-write preview** — a direct template for both fix (b) and gap3-02's warn-only rollout. (3) 2024-2026 backtest-overfitting work (ScienceDirect CPCV, arXiv 2603/2605) refines PBO/DSR but does **not** supersede the canonical Bailey-Borwein-LdP result.
- **Verdict:** no 2024-2026 finding overturns the canonical principles (parse-don't-validate, poka-yoke constraint-over-judgment, PBO). They complement and give a modern (IaC dry-run) implementation template.

### Key findings (per-claim cited)
1. **Refuse, don't fabricate** — a stub for an unbuilt engine must raise, not return a plausible value; the non-dry-run gauntlet path returning seeded noise is the anti-pattern. (King 2019, lexi-lambda.github.io)
2. **Constraint over judgment (poka-yoke)** — design so the mistake is impossible, not so a human remembers to pass `--dry-run`. Two independent guards (top-of-`run()` raise + pre-write `assert dry_run`) make the illegal state structurally unreachable. (softwaretestinghelp.com; Shingo/Toyota)
3. **dry-run = zero side effects, gate every writer** — the failure mode is a reassuring message printed while the write still happens; every side-effect path must check the flag. (Warne 2026; angular-cli #6810; HN 27263136)
4. **IaC no-write preview is the template** — `terraform plan` / `ansible --check` show the would-be mutation and write nothing; fix (b) should print the would-be mutation, matching this idiom. (Octopus 2026)
5. **Config-drift → detect + warn first** — divergence checkers compare active-vs-baseline and emit warnings before enforcement; exactly gap3-02's warn-only rollout with an operator gate for the flip. (Octopus 2026)
6. **Why fabricated gate evidence is catastrophic** — the whole PBO/DSR apparatus exists because overfit/unverified backtest evidence looks great IS and fails OOS; a gate authorized by random noise inverts its own purpose. (Bailey-Borwein-LdP-Zhu, PBO paper)

### Application to pyfinagent (external → internal file:line)
- **(a) NotImplementedError + refuse-to-write** [King, poka-yoke]: add `if not dry_run: raise NotImplementedError(...)` at the top of `run()` (`gauntlet.py:132`) + defense-in-depth `assert report["dry_run"] is True` before `report.json` write (`gauntlet.py:161`). Two independently-mutatable guards. No importer breaks (only CLI callers).
- **(a) stub-fingerprint rejection at the consumer** [PBO / provenance]: in `promotion_gate.py` after `_load_gauntlet_report` (line 103), block when `non_skipped and all(r["bt_drawdown"] == r["drawdown"] for non-skipped r)`. **Guard the empty list** (`all([])==True`). Scope note: `autonomous_harness.promote_strategy` (258-289) is the OTHER consumer and lacks this — flag for the planner.
- **(b) dry-run write guard** [Warne, angular #6810, Octopus]: wrap BOTH `promotion_gate.py:150-158` (alloc init) and `:165-174` (gauntlet stamp) in `if not args.dry_run:` … `else: print(would-be mutation)`. **Update the module docstring lines 3-7** (they currently promise --dry-run writes allocation_pct — that becomes false).
- **(c) divergence.py** [Octopus config-drift, warn-only]: new `backend/governance/divergence.py` pure fn that calls `limits_schema.load()`, normalizes units (**fraction ×100 → percent**), compares vs `settings.paper_daily_loss_limit_pct/paper_trailing_dd_limit_pct`, returns `[(name, settings_pct, governed_pct, divergent_bool)]`. Wire as WARNING-only in `main.py` lifespan (fail-open try/except, mirroring the existing governance block at `main.py:277-286`). Write `governance_limits_divergence_75.md` + draft `GOV-LIMITS-DECIDE`.

### Consensus vs debate
Strong consensus: refuse-over-fabricate, dry-run = no writes, constraint-over-judgment, warn-before-enforce. Only nuance in the literature: Warne notes dry-run "pollutes the code" (the if/else cost) — the King/HN answer is to **factor side effects** so the flag gates thin write-wrappers, not business logic; here the writers are already isolated (two `write_text` calls), so the guard is clean.

### Pitfalls (from literature + internal)
- **Reassuring-message-but-still-writes** (angular #6810) — the promotion report echoing `dry_run:true` while writing is the same trap; the test must assert the FILE, not the printed message.
- **"if dry-run/else soup" bugs** (HN) — keep guards at the two write sites only; don't thread the flag through logic.
- **Unit mismatch** — comparing fraction 0.02 to percent 4.0 naively mis-flags trailing-dd (0.10 vs 10.0). Normalize first; only daily-loss actually diverges (2% vs 4%); trailing-dd MATCHES (10% both).
- **Vacuous tests** — see the anti-vacuous analysis above; the crit-3 byte-identical and crit-2 fingerprint tests are both vacuous without discriminating fixtures + control assertions.

### Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7: King, Warne, pytest, poka-yoke, Octopus, angular #6810, PBO-via-pdfplumber)
- [x] 10+ unique URLs total (~35 collected)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages/paper read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (gauntlet, promotion_gate x2, evaluator, governance x3, settings, main lifespan, paper_trader kill-switch, autonomous_harness, regimes, lint, json_io, existing tests, optimizer_best.json, audit artifacts)
- [x] Contradictions/consensus noted
- [x] All claims cited per-claim

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All three sub-fixes verified against source with file:line. (a) gauntlet.py:134 runs _run_regime_stub unconditionally; dry_run only annotates (line 150); stub bt_drawdown==drawdown passes all 4 evaluator gates by construction; NO python importer of gauntlet.run so NotImplementedError breaks nothing. (b) promotion_gate.py:150-158 + 165-174 both write the REAL optimizer_best.json with no args.dry_run guard. (c) six limits.yaml values have zero runtime value-consumers (only loaded/digest-watched at main.py:277-286); live kill-switch reads settings.paper_daily_loss_limit_pct=4.0 vs governed 0.02(=2%). Step-text corrections: audit rates gap6-01 P1 / gap6-10 P3 / gap3-02 P1 (masterplan's P0 is the bundle priority; none touches live capital yet); fix (b) must update promotion_gate docstring lines 3-7; fingerprint guard belongs on autonomous_harness.promote_strategy too; divergence.py MUST normalize fraction-vs-percent (only daily-loss diverges, trailing-dd matches). Vacuous-test traps flagged for crit-2 (needs realistic-passes fixture + empty-list guard) and crit-3 (fixture must lack allocation_pct + divergent hash + a non-dry-run control that proves writers fire). 7 sources read in full incl. peer-reviewed PBO paper.",
  "brief_path": "handoff/current/research_brief_75.8.md",
  "gate_passed": true
}
```
