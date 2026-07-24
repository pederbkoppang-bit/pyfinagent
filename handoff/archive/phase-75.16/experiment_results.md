# Experiment results -- Step 75.16 (Cloud Functions + Docker deploy-surface retirement/hardening)

Date: 2026-07-24. **Execution model: Sonnet executor GENERATE (7th
delegated); Main review + independent re-measurement. Executor draft at
`experiment_results_75.16_draft.md`.**

## BOUNDARY DEVIATION (executor's, disclosed up front by the executor itself)

Two `pip index versions` PyPI lookups (functions-framework, pyarrow) were
made early in the executor session to source pin versions -- a NETWORK
call violating the letter of the $0/no-network boundary. Information-only
(nothing installed, no state changed, $0 spent); every other pin came
from requirements.lock/pip show; zero gcloud/docker calls. The executor
disclosed it unprompted at the top of its draft. Main's judgment: breach
of letter, not intent (the boundary's purpose is no deploys/no metered
spend); prominently surfaced for the Q/A to weigh.

## What shipped (legs a-h; 22 files, +224/-607, net -383 lines)

- **(a)** scripts/deploy/deploy_agents.sh DELETED (would today upload the
  repo root INCLUDING backend/.env as four PUBLIC --allow-unauthenticated
  functions; all four cd targets nonexistent, no set -e).
- **(b)** functions/ingestion/cloudbuild.yaml DELETED with
  functions/ingestion/RETIRED.md documenting the evidence (orphaned E-L
  refactor, zero callers/schedulers, entry-point mismatch, live path =
  the in-backend service).
- **(c)** Ingestion status decisions factored into the NEW pure
  functions/ingestion/response.py (500 fetch-exception / 500 load-failure
  / 200 empty-success / 200 success -- Cloud Scheduler acks any 2xx and
  ignores bodies, so the old 200-with-Failure-body read as success);
  data_fetchers.py re-raises genuine errors.
- **(d) THE LIVE PATH**: functions/quant/main.py -- timeout=(5,30) on
  both SEC requests.get; the full traceback now goes ONLY to
  logging.critical; the stream yields a SINGLE-line `ERROR: ...` with the
  FINAL_JSON:/ERROR: tokens untouched (the orchestrator's line-prefix
  parser contract, verified in the diff by Main).
- **(e)** Earnings function: model id env-var'd (default aligned to the
  live earnings_tone service; 2.5-family retirement noted); NLP failure
  now a distinguishable status (never {'error':...}-as-data); 4-key JSON
  validation; wildcard CORS -> the backend localhost/Tailscale allowlist
  idiom.
- **(f)** All 3 functions requirements.txt fully ==-pinned; all 3 added
  to pip-audit.yml (paths + per-file --requirement steps).
- **(g)** backend/Dockerfile -> python:3.14-slim + real requirements
  install; frontend/Dockerfile -> npm ci from the committed lockfile.
- **(h)** 5 migrations + extend_historical_data.py re-anchored to
  Path(__file__).resolve().parents[2]; 4 unreferenced scripts/debug/*.py
  DELETED (grep-zero-references evidence in the draft).
- NEW backend/tests/test_phase_75_deploy_surface.py (44 tests, import-safe
  -- no google-cloud imports); the 75.15 collection pin legitimately moved
  1474/1490 -> 1518/1534 (+44 unmarked tests; deselected count -- the
  canary -- unchanged at 16).

## Verification (Main-independent)

- Immutable command: **exit 0** (re-run after every Main edit); the
  executor's M8 proof shows it FAILED on every leg against the pre-fix
  tree (git-show reconstructions -- no stash) and exits 0 post-fix.
- New tests: **44 passed**; both guard files together: **60 passed**.
- **CI-equivalent tail (Main re-run)**: `1510 passed, 0 failed, 2 skipped,
  16 deselected` (= 75.15's 1466 + the 44 new).
- Raw suite (executor): 8 failed -- a STRICT SUBSET of the 9-red baseline;
  the absent one (23_2_15) is the documented PATH-shell-dependent test
  (green in the executor's shell, red in Main's -- exactly the 75.15
  finding; zero new failures).
- **THE HEADLINE DELTA PROOFS, Main-reproduced**: M1 (traceback restored
  into the yield under a RENAMED variable -- the immutable command's
  escape hatch): the new AST guard KILLS it (1 failed) while the immutable
  command stays exit 0 on the same mutant -- measured both ways. M2
  (`==`-in-comment pin dodge) equivalently proven by the executor.
  Matrix: 6/6 killed + the M7 stub-fixture discipline (vacuous-fixture
  variant fails) + M8 pre/post proof.
- Ruff: clean over the git-derived scope after Main removed 3 MORE
  pre-existing F401s in touched files (earnings `Part`, ingestion
  `datetime`, migrate_bq_schema `sys` -- each proven pre-existing via
  git-show-HEAD lint; the 75.5 precedent, now applied in 5 consecutive
  steps).
- py_compile clean on all touched scripts; yaml.safe_load clean on
  pip-audit.yml; the ci_gates suite green at the moved pin.

## Queued this step

- **75.16.1** (at contract time): earnings function missing deps
  (vertexai, google-cloud-storage -- non-deployable as committed) +
  the untimed :120 requests.get; retire-vs-repair decision at research.

## Not verified live

No deploy was executed (boundary): the quant hardening reaches the LIVE
function only when the operator next redeploys it (until then the
deployed copy still streams tracebacks -- the URL is live TODAY, so the
redeploy is operator-actionable: OPS-QUANT-REDEPLOY suggested at next
convenience, $0-adjacent, one gcloud command documented in RETIRED.md's
sibling note). Docker images unbuilt. pip-audit additions first exercise
on the next push.
