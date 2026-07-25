# Contract — Step 76.9.2 (durable Max-rail routing for nightly autoresearch)

Date: 2026-07-24 | Cycle: 156 | Executor: MAIN-on-Fable (opus-tagged step; goal phase-78 FIX ORDER item 1) | Gates: Fable via Workflow (goal constraint; stall → opus)

## Research-gate summary (gate PASSED — Workflow wf_12c8ead9-d26, tier=moderate, Fable)

`handoff/current/research_brief_76.9.2.md` — 6 read in full, 12 snippet-only, 18 URLs,
recency scan done, 17 internal files, gate_passed=true. Deciding findings:

1. **Aggregation is REQUIRED regardless of transport fix**: anthropic 0.96.0
   non-streaming `create()` on an SSE response tries `response.json()`, swallows the
   failure, and returns raw `response.text` instead of a Message — silent type
   corruption (`_response.py:266-278`, `_strict_response_validation` default False;
   corroborated externally by sub2api #867). This alone eliminates cert-regen-only and
   HTTP-listener-only options.
2. Cert-regen bar is high anyway: Python 3.13+ default-enables VERIFY_X509_STRICT;
   OpenSSL strict wants keyUsage+SKI+CA:TRUE+non-empty SAN; the measured cert has ZERO
   extensions. And (a)/(b) edit un-versioned ~/.openclaw infra.
3. **RE-VERIFIED: no live external proxy client** (openclaw.json no baseUrl; only the
   stale .bak.1 ever pointed at :18796; combined-certs.pem orphaned; the gateway plist
   sets NODE_TLS_REJECT_UNAUTHORIZED=0) → proxy-side edits LOW-RISK (criterion 3).
4. `langchain_anthropic` 1.4.8 reads ANTHROPIC_API_URL first (`chat_models.py:947-951`),
   `_client_utils.py` reads ANTHROPIC_BASE_URL → export BOTH; `run_memo.py:288`
   requires non-empty ANTHROPIC_API_KEY → dummy key (metered billing impossible).
5. Live CLI probe (2.1.218): `--model sonnet` ran as **claude-sonnet-5** (modelUsage
   ground truth); aliases fable/opus/sonnet documented → MODEL_MAP additions
   probe-verified. Current unknown→sonnet fallback is a silent-downgrade trap → pass
   claude-* ids through verbatim.
6. `~/.local/bin/claude` is the installer-maintained path (rewritten by this morning's
   auto-update); the /opt/homebrew symlink is manual → plist CLAUDE_PATH must point at
   `~/.local/bin/claude`.
7. run_nightly.sh insertion point: between :31 (venv activate) and :43 (run_memo),
   AFTER the sanitized `set -a` sourcing :19-27 (flag auto-lands from .env; dummy key
   overrides the sourced real key); loud-fail routes through the existing
   fail-state+paging block (:54-71).

## Hypothesis

A repo-versioned SSE-aggregating bridge (hardened from the live-proven scratchpad
pattern) + a default-OFF flag in run_nightly.sh gives autoresearch a $0 Max-rail
transport with one-flag revert, loud preflight failure, and zero possibility of
silent metered fallback.

## Immutable success criteria (verbatim from .claude/masterplan.json 76.9.2)

1. "A real run_memo run completes rc=0 through the DURABLE routing (not the session scratchpad bridge) with a dummy metered key proving $0 leakage, evidence verbatim"
2. "The routing is flag-gated so the operator can revert to the direct metered API with one env change; default documented"
3. "Other OpenClaw clients of the proxy are enumerated and shown unbroken by any cert/listener change (or the change is additive-only)"
4. "The claude-binary PATH fix is made durable (plist CLAUDE_PATH or documented symlink) and the proxy MODEL_MAP covers the three autoresearch roles or the fallback is documented"
5. "MUTATION: break the routing (wrong port) -> the run fails loudly, never silently falls through to the metered API"

Immutable verification command (verbatim):
`bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"`

## Plan

1. **`scripts/ops/anthropic_max_bridge.py`** (NEW, stdlib-only): hardened from the
   scratchpad bridge — 127.0.0.1:18797; upstream default https://localhost:18796
   overridable via `ANTHROPIC_BRIDGE_UPSTREAM` (test seam); /health relayed through the
   chain; SSE→non-streaming-JSON aggregation incl. `error`-event surfacing; SSE
   verbatim passthrough when the client sends `"stream": true`; 600s upstream timeout
   (> proxy's 180s per-call); upstream verify=False (loopback-only; nothing anchors the
   cert; the gateway itself runs NODE_TLS_REJECT_UNAUTHORIZED=0); ASCII logs.
2. **`scripts/ops/templates/com.pyfinagent.anthropic-bridge.plist`** (NEW, template —
   NOT bootstrapped; 62.0 rail): KeepAlive service running the bridge under .venv
   python. Operator token **OPS-BRIDGE-BOOTSTRAP** documented in experiment_results.
   The live test runs the repo script directly in background (durable ARTIFACT, not
   the scratchpad copy — satisfies criterion 1's "not the session scratchpad bridge").
3. **`scripts/autoresearch/run_nightly.sh`** (EDIT — in scope this step): flag
   `AUTORESEARCH_USE_MAX_RAIL` (default OFF/absent → byte-identical single if-guard).
   ON: preflight `curl -sf -m 10 http://127.0.0.1:18797/health`; on failure invoke the
   fail-state+paging path and exit non-zero (LOUD FAIL — criterion 5); on success
   export ANTHROPIC_API_URL + ANTHROPIC_BASE_URL = http://127.0.0.1:18797 and
   ANTHROPIC_API_KEY=max-rail-dummy-key. Flag lives in backend/.env (operator-gated
   flip; reaches the script through the existing sanitized sourcing). Add a
   `NIGHTLY_REPO`-style override only if the fixture tests need it (disclose).
4. **Operator-infra edits (~/.openclaw + proxy plist; goal-authorized, backups kept)**:
   proxy.js MODEL_MAP += `claude-opus-4-8→opus`, `claude-sonnet-5→sonnet`,
   `claude-fable-5→fable`; replace unknown→sonnet with verbatim claude-* passthrough
   (silent-downgrade trap); proxy plist EnvironmentVariables += CLAUDE_PATH=
   /Users/ford/.local/bin/claude (installer-maintained; today's /opt/homebrew symlink
   becomes non-load-bearing, left in place + documented). `launchctl kickstart -k` the
   proxy to reload. Verbatim diffs recorded in live_check; a reference copy of the
   edited proxy.js checked into `scripts/ops/reference/` for versioning (deploy
   authority stays with the operator's ~/.openclaw copy).
5. **Tests** `backend/tests/test_phase_76_9_2_max_bridge.py` (NEW): unit-test
   `sse_aggregate` on synthetic SSE streams (text deltas, usage, stop_reason,
   error-event surfacing); end-to-end REAL bridge against a stub plain-HTTP upstream
   via ANTHROPIC_BRIDGE_UPSTREAM (aggregation + stream-passthrough + /health);
   run_nightly.sh fixture runs (76.9 fixture pattern): flag OFF → no base-url exports
   reach the stub run_memo; flag ON + bridge down → non-zero exit + fail-state (LOUD);
   flag ON + stub healthy → both URL vars + dummy key visible to the stub. Fixture
   must be reproduce-first where a failure shape is claimed.
6. **Mutation matrix (Main, after GENERATE)**: M1 wrong port in the exported base-url
   → live run fails loudly, zero api.anthropic.com traffic (criterion 5, run against
   the real bridge); M2 flag-guard inverted → flag-OFF fixture red; M3 preflight
   removed → bridge-down fixture red; M4 aggregation neutered (return raw text) →
   sse_aggregate unit red; M5 FIXTURE mutation: stub upstream serves valid JSON
   instead of SSE → the aggregation e2e assert flips (fixture load-bearing); M6 STUB:
   fixture run_memo stub never checks env → env-assert tests red.
7. **Live checks (Main)**: real run_memo rc=0 + non-ERROR memo through the REPO bridge
   (dummy-key $0 proof); MODEL_MAP probe (one bridge call on claude-opus-4-8 asking
   model self-ID — expect opus, not silent sonnet); proxy kickstart evidence; M1 live.
8. Q/A via qa-verdict Workflow (model fable per goal; stall → opus rerun) → Cycle 156
   log → flip (auto-push now unblocked).

## Boundaries

- backend/.env NOT edited (flag documented; operator flips — token in results).
- No launchd bootstrap of the new plist (62.0 rail; OPS-BRIDGE-BOOTSTRAP token).
- No new Python deps (bridge is stdlib-only).
- scripts/autoresearch/run_memo.py UNCHANGED (immutable command guards it).
- ~/.openclaw edits: proxy.js + proxy plist ONLY, each with a timestamped .bak;
  additive/verified-low-risk per criterion 3 evidence.

## References

- Brief: handoff/current/research_brief_76.9.2.md (SDK/langchain/CLI/cert citations)
- Fact-check wf_3b9205bc-666 (client enumeration); live test Cycle 154 addendum
- Patterns: 75.11 run_ablation.sh wrapper + plist template; 76.9 fixture tests
