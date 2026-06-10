# Research Brief — phase-53.5 (E2E smoke capstone)

**Tier:** moderate. **Date:** 2026-06-10. **Gate: PASSED** (`gate_passed: true`).

> NOTE (audit trail): the `researcher` subagent (`a80b95565062938a4`) ran the gate and
> persisted this brief; a `run_harness.py` invocation (triggered by a done-phase
> verification command during the first portable `aggregate.sh` run) then CLOBBERED this
> rolling file with optimizer content. Reconstructed faithfully here from the researcher's
> returned summary + gate envelope. The #2-full-skip fix (this cycle) prevents the recursion.

## Gate envelope (as returned by the researcher)

```json
{"tier":"moderate","external_sources_read_in_full":7,"snippet_only_sources":12,
 "urls_collected":19,"recency_scan_performed":true,"internal_files_inspected":13,
 "gate_passed":true}
```

## Decisive findings

### 1. The clobber (md5-proven this session)
`python scripts/harness/run_harness.py --dry-run --cycles 1`:
| File | Behavior |
|------|----------|
| `contract.md` | CLOBBERED (always; `write_contract` runs before the dry-run branch) |
| `research_brief.md` | CLOBBERED (on the current SATURATED plateau, ≥10 saturated params) |
| `experiment_results.md` | UNTOUCHED (the dry-run `continue` skips the generator) |
| `harness_log.md` | APPENDED (the cycle entry) |
Exits **0**, credential-free (lazy `BigQueryClient`, never queries in dry-run). Sequencing
for Main: write all 5 handoff files → backup {research_brief, contract, experiment_results}
→ run the dry-run → restore the clobbered files (keep the appended harness_log cycle).

### 2. aggregate.sh is NOT portable-green as written (4 defects, all fixed this cycle)
- **#1** mis-treats `phase-5 status=deferred` as a non-done blocker (deferred is intentional).
- **#2** (re-run every done-phase verification command) crashes on a malformed/list step
  (`AttributeError: 'list' object has no attribute 'get'`), and even guarded it is a FULL
  live/historical-drift audit: ~30 of 488 commands fail on a clean rerun (live MCP servers,
  booted backend on :8765, since-moved/removed modules `paper_metrics_v2`/`reconciliation`,
  transient handoff artifacts). It is NOT a portable smoke → SKIP it in portable.
- **#5** `npm run build` races the live dev server's `.next` (passes standalone + on a clean
  CI runner) → quiesce the dev server for the local run.
- **#7** the `CRITICAL` grep false-matches the word "critical" in benign prose ("0 critical",
  "2 critical findings", node_modules "1 critical" vuln) → match real incident markers.
Check #5 (phase-4.6 sub-smoke) already SKIPs correctly (file absent → `[ -x ]` false).

### 3. The credential-free pytest subset
No pytest markers/config exist. Credential-free subset = `--ignore` the 6 live/state/
always-fail files: `test_phase_23_2_{10,12,14,16}*.py` + `test_agent_map_live_model.py` +
`test_rainbow_canary.py`. Durable fix (a `requires_live` marker) = follow-up.

### 4. e2e-smoke.yml outline (mirrors env-syntax-lint.yml house shape)
`on:` `workflow_dispatch` + `schedule: cron '17 6 * * *'` + `pull_request: branches:[main]`;
`permissions: contents: read` (least-privilege; no secrets; never `pull_request_target` —
2026 GitHub Actions security roadmap). `runs-on: ubuntu-latest`, `timeout-minutes: 20`,
`actions/checkout@v4`, `actions/setup-python@v5` (3.14, cache pip), `actions/setup-node@v4`
(cache npm). Steps cheapest-first (Fowler test-pyramid): ast syntax → credential-free
pytest → npm ci + tsc --noEmit + build → run_harness --dry-run → intel_e2e --fixtures →
phase6_e2e --dry-run. The CI lane does NOT invoke aggregate.sh (its #2 is a live audit).

## Sources (read in full)

GitHub Docs workflow-syntax; GitHub 2026 Actions security roadmap; GitHub dependency-caching;
actions/setup-python; AltexSoft smoke-testing 101; Harness CI/CD smoke testing; Martin Fowler
"The Practical Test Pyramid". Recency scan (2024-2026): the 2026 security roadmap
(least-privilege, no `pull_request_target` for untrusted) is the current guidance; no source
contradicts the credential-free-subset + cheapest-first design.

## Recommended plan
Add e2e-smoke.yml (outline §4); make aggregate.sh portable-green (fix the 4 defects;
SMOKE_PORTABLE skips the #2 live-audit); run the local smoke + the harness dry-run with the
backup/restore sequence (§1).
