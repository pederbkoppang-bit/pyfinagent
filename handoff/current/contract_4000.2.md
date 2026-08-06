# Contract -- 4000.2: build the E2E smoke script, and prove its checks can fail

Step id: 4000.2 (phase-4000, P1, depends_on 4000.1=done). Written 2026-08-06
AFTER the research gate (wf_706ab423-79f, gate_passed=true, tier=simple,
7 sources read in full, 19 URLs) and BEFORE any artifact. Suffixed filenames
per the phase CONCURRENCY RAIL.

## Research-gate summary -- findings that BIND the design

- R1 (spawn-premise corrected): CLAUDE_CODE_BINARY is NOT a usable test stub
  hook -- claude_code_client.py:70 tries shutil.which() BEFORE the module-level
  _DEFAULT_SEARCH_PATHS literal (:49-54, frozen at import). An env-var stub
  would silently invoke the REAL CLI from a unit test. Tests stub by PATH-
  prepending a fake executable named `claude` (documented pytest monkeypatch
  pattern); the script additionally accepts an explicit --claude-binary arg,
  and resolution-without-arg goes through the IMPORTED _resolve_claude_binary.
- R2: pytest-subprocess is in-process-only and WRONG here; tests drive the
  script as a real child via subprocess.run and stub the backend with a REAL
  socket (pytest-httpserver, threaded=True, check_assertions()).
- R3 (counter architecture): the rail subprocesses are spawned by the BACKEND;
  no in-process wrapper can count them, llm_call_log is BUFFERED (flush at 100
  rows/60s, piggybacked, no flush endpoint -- api_call_log.py:38-39,:299-304),
  and there is NO cancel endpoint for a running analysis. The <=30 cap is a
  SAFETY CEILING on the observable surface: (i) <=2 tickers by arg validation;
  (ii) a pre-start gate refusing a new analysis when the expected per-analysis
  count exceeds remaining budget; (iii) a during-window watcher polling the log
  source that aborts the OBSERVER, restores the flag, and exits a distinct code
  -- disclosing that in-flight backend calls continue; (iv) the authoritative
  count is post-window after a >=65s flush wait. A cap trip is a LOUD LEGITIMATE
  OUTCOME, not a test bug.
- R4 (bound): live pins measured 2026-08-06: gemini_model=claude-sonnet-4-6,
  deep_think_model=claude-opus-4-8 -- the orchestrator builds clients from
  SETTINGS (orchestrator.py:652-659), so a full single-ticker analysis is
  ~25-33 rail calls and may exceed 30 ALONE. This also explains the baseline's
  sonnet-dominant rail traffic. The Layer-2 model_tiers roles are NOT the
  analysis-path bound (agent_definitions.py only).
- R5 (E5 has no out-of-process surface): rail_guard_status() is process-local;
  its only reader is the autonomous cycle path the smoke deliberately avoids.
  consecutive_failures CANNOT be read by the smoke -- asserting it would be a
  fabricated-SAFE check. E5 is implemented as the measurable surrogate
  predicate: zero in-window ok=false rail rows AND zero 'rail_guard_skipped:'
  markers in analysis output AND no breaker_open P1 page observed; the direct-
  read gap is disclosed, and "expose rail_guard_status on an observability
  endpoint" is queued as its own step (4000.8) at this step's flip.
- R6 (zero needs a positive control): a '0 metered rows' read is
  indistinguishable from 'not flushed yet'; every zero-based check pairs with
  a POSITIVE control (rail rows N>0) before it may pass.
- R7 (restore discipline): PUT /api/settings/ writes backend/.env ON DISK and
  cache-invalidates (settings_api.py:418-452), so a crashed --live run leaves
  the flag mutated ACROSS restarts. Restore lives in an ExitStack finally as a
  PUT (never in-memory), re-raising per contextlib doctrine; --keep-on is
  stack.pop_all() taken ONLY on explicit flag + all-checks-pass.
- R8 (polling): AIP-151 task contract + AWS Full Jitter for the interval
  (sleep = random(0, min(cap, base*2^attempt))), hard deadline per analysis.
- R9 (E3 split, refining frozen baseline (f) within its rule): E3a envelope
  truth on the smoke's OWN probe call (raw modelUsage sum == total_cost_usd,
  tolerance 1e-6; probe runs in an EMPTY temp cwd to avoid the repo-context
  overhead measured in 4000.1); E3b backend rows: llm_call_log.model equals
  the configured tier for the calling path (from GET /api/settings). The
  backend's own envelopes are not externally accessible; disclosed.

- R10 (ADDENDUM, cycle 3, from Q/A cycle-2 finding 1 -- disclosed E4 scope):
  the frozen baseline's E4 has FOUR legs. The smoke implements legs 1-3
  (terminal-completed; report present/schema-parseable; no synthetic 0.0/HOLD
  shape, recommendation read from either the string or the
  RecommendationDetail.action form). Leg 4 -- 'the persisted analysis row
  exists' -- is NOT PROVABLE from the sync poll (GET /api/analysis/{id} reads
  the in-memory task dict, analysis.py:392), so it is disclosed in the emitted
  E4 verdict, queued as its own masterplan step (4000.9, at this step's flip,
  alongside 4000.8), and owed as 4000.3 live_check evidence against the real
  persisted surface. This mirrors the accepted E5-gap handling (R5).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json 4000.2)

1. "scripts/qa/smoke_cc_rail_e2e.py exists with --dry and --live modes; --dry performs no settings mutation of any kind, proven by a test asserting zero PUT requests against the stubbed settings endpoint during a dry run."
2. "The <=30-rail-call and <=2-ticker budgets are enforced in code: a fixture that would drive 31 calls aborts with a distinct non-zero exit and a message naming the cap, asserted by test."
3. "The preflight resolves the claude binary through backend/agents/claude_code_client.py's own resolution logic (imported, not reimplemented), asserted by test through the stub search path."
4. "Each check E1-E6 emits a machine-readable per-check verdict; the overall exit code is non-zero iff any check fails, proven in both directions by fixtures (an all-pass canned window exits 0; each single-check-fail fixture exits non-zero)."
5. "E3 iterates ALL keys of the envelope's modelUsage map: a fixture whose envelope carries two models -- the configured tier first and a foreign model second -- must FAIL E3, proving the check cannot be satisfied by reading only the first key."
6. "Flag restoration lives in a finally block: a fixture that crashes the analysis mid-window still issues the restore PUT, asserted by test against the stub."
7. "Tests drive the script via subprocess.run([...]) asserting on .returncode -- never on an imported function's return value -- so neutering the script's exit plumbing turns the suite red."
8. "MUTATION COVERAGE, named in a test-file comment: (m1) breaking the E1 row-selection rule so it matches zero rows must turn the E1 fixture red; (m2) neutering the E2 delta check to always-zero must turn the E2 fixture red; the executor runs the matrix once, records verbatim output in the handoff, and restores."
9. "A --dry run against the REAL running backend in its CURRENT flag state completes, reports the honest rail state, makes zero flag changes; its verbatim output is recorded in experiment_results.md."
10. "git diff scope: scripts/qa/, backend/tests/, handoff/ only."

## Design (binding for GENERATE)

Script scripts/qa/smoke_cc_rail_e2e.py, stdlib + google-cloud-bigquery only:
- Modes: --dry (no mutation; health + flag GET + binary resolution + optional
  probe [--no-probe] + 7d rail-health report + recent per-analysis call
  estimate) and --live (--ticker xN<=2, --expected-calls-per-analysis required,
  --keep-on optional, full E1-E6 with window bracketing).
- Data sources pluggable for cross-process testability: --backend-url (default
  http://localhost:8000) and --llm-log-source (default 'bigquery'; an http URL
  serves canned rows in tests). BQ path uses ADC exactly like the 4000.1 recon.
- Exit codes: 0 all-pass; 1 check-fail; 2 preflight-fail; 3 budget-cap trip
  (message names the cap); 4 usage error. JSON verdicts on stdout, one line
  per check: {"check":"E1","ok":true,...} + a final summary object.
- E-checks per baseline (f) with R5/R6/R9 refinements above.
- E6 is REPORT-shaped (numbers + stated basis), never a pass/fail leg in
  4000.2; its threshold consumption belongs to 4000.3's decision rule.

Tests backend/tests/test_phase_4000_2_cc_rail_smoke.py: pytest-httpserver
threaded stub backend (settings GET/PUT, health, analysis POST/poll, llm-log
endpoint); PATH-prepended stub `claude`; every scenario via subprocess.run +
returncode; scenarios: dry-zero-PUT, cap-31-rows exit 3, 3-tickers exit 4,
two-model-envelope E3 fail, crash-mid-window restore-PUT-still-issued,
all-pass exit 0, per-check single-fail nonzero (parametrized), binary
resolution through imported logic, negative control (stub claude records
invocation; real CLI never reachable on the stripped PATH).

## Non-scope

No live --live run (4000.3, operator-gated). No changes to claude_code_client
or llm_client routing. No BQ writes. pytest-httpserver: verify availability in
the venv FIRST; if absent, fall back to a stdlib ThreadingHTTPServer stub in
the test file (76.9.2 prior art) rather than adding a dependency without
operator sign-off.

## References

- handoff/current/research_brief_4000.2.md (envelope; pytest-httpserver docs,
  CPython contextlib docs, AWS backoff-jitter study, AIP-151, pytest
  monkeypatch docs, claude CLI reference).
- handoff/archive/phase-4000.1/cc_rail_baseline_4000_1.md (frozen E1-E7 +
  measured baselines).
- backend/tests/test_phase_76_9_2_max_bridge.py (in-repo prior art: real
  bridge process against stub upstream).
