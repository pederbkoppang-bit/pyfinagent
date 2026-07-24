# Experiment results — Step 76.9.2 (durable Anthropic Max-rail routing for nightly autoresearch)

Date: 2026-07-24 | Cycle: 156 | Execution: MAIN-on-Fable GENERATE (opus-tagged; goal phase-78 FIX ORDER item 1)

## What was built

### 1. `scripts/ops/anthropic_max_bridge.py` (NEW, stdlib-only) — the durable transport

Repo-versioned hardening of the live-proven scratchpad bridge:
`client (SDK/langchain, plain HTTP :18797) → claude-code-proxy (https :18796) → claude -p (Max plan)`.
SSE→non-streaming-JSON aggregation (LOAD-BEARING: anthropic 0.96.0's non-streaming
`create()` on an SSE response silently returns raw text instead of a Message —
`_response.py:266-278`, strict validation off by default — so cert-only fixes cannot
work); SSE verbatim passthrough for `"stream": true` clients; proxy `error` events
surfaced as PROXY-ERROR text (never silent empties); /health relayed through the
chain; 600s upstream timeout (> proxy's 180s per-call); upstream verify=False
(loopback-only; nothing else anchors that cert; the gateway itself runs
NODE_TLS_REJECT_UNAUTHORIZED=0; no live external proxy client exists — re-verified).
Env seams: `ANTHROPIC_BRIDGE_UPSTREAM` (test stub), `ANTHROPIC_BRIDGE_PORT`.

### 2. `scripts/ops/templates/com.pyfinagent.anthropic-bridge.plist` (NEW, template)

NOT bootstrapped (62.0 rail blocks launchctl bootstrap from sessions — by design).
**OPERATOR TOKEN `OPS-BRIDGE-BOOTSTRAP`**: cp template → ~/Library/LaunchAgents +
`launchctl bootstrap gui/$(id -u) …`. The same operator action should rebind the
proxy plist so its new CLAUDE_PATH takes effect (see §4).

### 3. `scripts/autoresearch/run_nightly.sh` (EDIT) — flag-gated routing, loud-fail

- `AUTORESEARCH_USE_MAX_RAIL` (default OFF/absent = single false if-guard,
  byte-identical behavior — fixture-proven). Flag lives in backend/.env
  (operator-gated; reaches the script via the existing sanitized `set -a` sourcing).
  **OPERATOR TOKEN**: add `AUTORESEARCH_USE_MAX_RAIL=1` to backend/.env to go live;
  remove it (or =0) to revert — one env change (criterion 2).
- ON: preflight `curl -sf -m 10 …/health`; failure → fail-state increment + page
  through the factored 75.11 seam (`_record_fail_and_page`, body moved verbatim) +
  `exit 78` — **NEVER a silent fallback to the metered API** (criterion 5).
  Success → export `ANTHROPIC_API_URL` + `ANTHROPIC_BASE_URL` (langchain_anthropic
  1.4.8 reads API_URL first, `_client_utils` reads BASE_URL — both needed) +
  `ANTHROPIC_API_KEY=max-rail-dummy-key` (overrides the sourced real key; any
  leakage to api.anthropic.com would 401 = provable $0 metered).
- `AUTORESEARCH_REPO` override added as a TEST SEAM (disclosed; production launchd
  never sets it — hardcoded default byte-identical).

### 4. Operator-infra edits (~/.openclaw + proxy plist; goal-authorized; backups kept)

- `claude-code-proxy.js` MODEL_MAP += `claude-opus-4-8/4-7→opus`,
  `claude-sonnet-5→sonnet`, `claude-fable-5→fable` (CLI aliases probe-verified on
  2.1.218; `--model sonnet` runs claude-sonnet-5 per modelUsage); `resolveModel`
  unknown→sonnet SILENT-DOWNGRADE TRAP replaced with verbatim claude-* passthrough
  (bad ids now fail loudly at the CLI). `node --check` clean; proxy kickstarted.
- Proxy plist += `CLAUDE_PATH=/Users/ford/.local/bin/claude` (the installer-maintained
  symlink, auto-rewritten on version updates — the /opt/homebrew symlink from this
  morning stays as interim cover but is no longer load-bearing once the operator
  rebinds the plist). Criterion 4: both the plist fix AND the documented symlink.
- Backups: `claude-code-proxy.js.bak-76.9.2-20260724`, plist `.bak-76.9.2`.
  Reference copy checked into `scripts/ops/reference/claude-code-proxy.js` + README
  (deployed copy stays authoritative; git gets reviewability).
- Criterion 3 (clients unbroken): re-verified NO live external client of the proxy
  (openclaw.json has no baseUrl override; only stale .bak.1 ever pointed at :18796;
  combined-certs.pem orphaned) — and the changes are additive (new MODEL_MAP keys;
  passthrough only for ids that previously mis-ran as sonnet).

### 5. Tests — `backend/tests/test_phase_76_9_2_max_bridge.py` (NEW, 11 tests)

sse_aggregate unit (text/usage/stop accumulation; PROXY-ERROR surfacing; garbage
tolerance); E2E against the REAL bridge process + stub upstream via
ANTHROPIC_BRIDGE_UPSTREAM (health relay, aggregation, stream passthrough, loud 502
on dead upstream); run_nightly fixture runs of the REAL script (flag OFF inert —
routing env provably absent; flag ON + dead bridge → rc=78 + fail-state + run_memo
NEVER executed; flag ON + healthy stub → both URLs + dummy key observed by the stub,
REAL sourced key overridden); default-OFF documented assert.

## Verification (verbatim)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_76_9_2_max_bridge.py -q
11 passed in 3.73s

$ bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"
IMMUTABLE exit=0

$ {changed-py robust form} | xargs uvx ruff check --select F821,F401,F811
All checks passed!
lint exit=0

$ node --check ~/.openclaw/claude-code-proxy.js   # OK
$ plutil -lint scripts/ops/templates/com.pyfinagent.anthropic-bridge.plist   # OK
```

run_memo.py: ZERO hunks (immutable command guards it; boundary held).

## Live evidence + mutation matrix

See `handoff/current/live_check_76.9.2.md`: the criterion-1 real run (rc=0 memo
through the DURABLE bridge), criterion-5 live dead-rail run (rc=78 verbatim), the
Opus MODEL_MAP probe, the M1-M6 matrix, and the first-attempt run that was
externally stopped mid-report (disclosed; no state damage; rerun completed).

## Operator tokens from this step

1. `AUTORESEARCH_USE_MAX_RAIL=1` in backend/.env → nightly goes $0 Max-rail
   (revert = remove the line). Recommended after reviewing live_check.
2. `OPS-BRIDGE-BOOTSTRAP`: bootstrap the bridge plist + rebind the proxy plist
   (commands in the template header). Until then the bridge must be started
   manually and the flag left OFF for unattended nights.
