# Sprint Contract -- phase-4.6 step 4.6.1

Started: 2026-04-17 (Cycle 30)
Step: 4.6.1 - Backend boot + /api/health returns 200
Status: in-progress

## Research Gate (Passed)

Two agents spawned in parallel:

1. **researcher subagent** (evidence-based): 18 unique URLs covering
   uvicorn subprocess testing (Safir/LSST ships this exact pattern),
   FastAPI testing tradeoffs (TestClient in-process vs out-of-process),
   Kubernetes readiness probes (fixed 0.5s poll is idiomatic), uvicorn
   SIGTERM semantics (uvicorn 0.42.0 has child-process reap fixed),
   find_spec semantics (filesystem-only, does NOT execute module),
   macOS arm64 subprocess safety (posix_spawn not fork; 127.0.0.1 not
   localhost).

2. **Explore subagent** (codebase audit): `backend/agents/mcp_servers/`
   is authoritative; `backend/mcp/` are legacy stubs. `/api/health` at
   `backend/main.py:251` uses `importlib.util.find_spec` for the three
   MCP servers. Auth middleware bypasses `/api/health` via `_PUBLIC_PATHS`
   at `backend/main.py:199`. Settings require `gcp_project_id` and
   `rag_data_store_id` (Pydantic Field(...) required) at Lifespan init.

## Hypothesis

My draft `scripts/smoketest/steps/boot_backend.py` will boot
`backend.main:app` on port 8765, /api/health will return 200 with
`mcp_servers.{data,backtest,signals}.status == "ok"`, and latency will
be well under 5s. Five research-flagged bugs must be fixed first:

1. PIPE-buffer deadlock (stdout=PIPE with no drainer); fix: redirect
   to DEVNULL (log level is already warning).
2. No early-exit when uvicorn crashes; fix: check `proc.poll()` each
   iteration.
3. Overly-broad `except Exception` in finally; fix: narrow to
   `subprocess.TimeoutExpired`.
4. `time.time()` for latency; fix: `time.monotonic()`.
5. (Optional, in main.py) `find_spec` misses import-time broken deps;
   defer to a later step -- the smoketest correctly tests the health
   endpoint as-is.

## Success Criteria (immutable)

- curl /api/health returns status=ok
- response includes mcp_servers with data, backtest, signals all ok
- latency under 5s

## Verification Command (immutable)

python scripts/smoketest/steps/boot_backend.py --port 8765 --timeout 30

## Plan

1. Apply fixes 1-4 to boot_backend.py.
2. Run verification command.
3. If PASS: spawn qa-evaluator + harness-verifier IN PARALLEL for
   EVALUATE phase (per CLAUDE.md harness protocol).
4. If both evaluators PASS: LOG to harness_log.md and mark 4.6.1 done.
5. If FAIL: diagnose from proc stderr, increment retry_count, fix, re-run.

## References (cited in research)

- https://safir.lsst.io/user-guide/uvicorn.html (Safir spawn_uvicorn)
- https://fastapi.tiangolo.com/tutorial/testing/ (TestClient vs subprocess)
- https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/ (fixed-interval readiness)
- https://github.com/Kludex/uvicorn/discussions/2257 (SIGTERM single-worker)
- https://github.com/Kludex/uvicorn/issues/2289 (child-process reap, fixed)
- https://sre.google/sre-book/service-level-objectives/ (health SLO tiers)
- https://docs.python.org/3/library/importlib.html#importlib.util.find_spec (find_spec semantics)
- https://docs.python.org/3/library/time.html#time.monotonic (monotonic clock)
