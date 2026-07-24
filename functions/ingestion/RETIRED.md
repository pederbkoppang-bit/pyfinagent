# functions/ingestion/cloudbuild.yaml -- RETIRED (phase-75.16, leg b)

`cloudbuild.yaml` was deleted from this directory on 2026-07-24. It is not
needed to run or redeploy anything live; git history (this file's own commit,
and the file's full history before it) preserves the content if it is ever
needed again.

## Why deleted, not fixed

Evidence gathered during the phase-75.16 research gate
(`handoff/current/research_brief_75.16.md`):

- **Entry-point mismatch.** The deleted config deployed
  `--entry-point=ingestion_agent_http`, but the only HTTP entry point defined
  in this directory's `main.py` is `ingest_market_data_el`. Per Google's Cloud
  Run functions deploy reference, `--entry-point` "must be a function name ...
  that exists in your source code" -- a deploy from this config would have
  failed outright.
- **Zero callers or schedulers.** Nothing in the repo (Cloud Scheduler config,
  cron, orchestrator code) invokes this build or its target function name.
  The live ingestion agent the backend actually calls
  (`backend/agents/orchestrator.py::run_ingestion_agent` ->
  `INGESTION_AGENT_URL`) is a **separate, already-deployed** Cloud Function
  whose source predates this directory's April 2026 refactor -- it is
  untouched by this deletion.
- **Single-commit, never-iterated.** The entire `functions/ingestion/`
  directory (including this file) was introduced in one commit
  (`fe5acdea`, "Major project restructure: clean root directory", 2026-04-13)
  and never touched again -- a half-built refactor, not a maintained
  pipeline (see `get_historical_universe()` in `main.py`, still a "CRITICAL
  TODO" placeholder static list).
- **Placeholder config values.** The deleted file's `--set-env-vars` carried
  literal placeholders (`BUCKET_NAME=your-gcs-bucket-name`,
  `USER_AGENT_EMAIL=your-email@example.com`) -- it was never filled in for a
  real deploy.
- **`--allow-unauthenticated` on an unmaintained function** is exactly the
  pattern flagged elsewhere in phase-75.16 (leg a,
  `scripts/deploy/deploy_agents.sh`) -- deleting the config removes the
  attack surface rather than hardening a config nobody uses.

`main.py`, `utils/data_fetchers.py`, `config.py`, `status.py`, and
`requirements.txt` in this directory are still hardened per the phase-75.16
immutable verification criteria (HTTP 500 on genuine failure, re-raised fetch
exceptions, pinned deps) even though the function is not deployed -- criterion
compliance does not depend on live-traffic status.
