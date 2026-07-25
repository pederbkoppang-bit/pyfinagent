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

### 5. Tests — `backend/tests/test_phase_76_9_2_max_bridge.py` (NEW, **12 tests**)

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
12 passed in 4.14s

$ bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"
IMMUTABLE exit=0

$ git diff --name-only HEAD -- '*.py' | tr '\n' '\0' | xargs -0 -r uvx ruff check --select F821,F401,F811
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

---

## Cycle-2 update (2026-07-25): mutation matrix COMPLETE; criterion 1 still OPEN

**M2 and M3 executed** (the two that were pending because they mutate the script bash
was executing; the script was idle, so they ran cleanly):

| # | Mutation | Result |
|---|----------|--------|
| M2 | flag-guard inverted (`= "1"` → `!= "1"`) | `3 failed, 8 passed` → `test_nightly_flag_off_is_inert` **RED** (plus the two flag-ON fixtures, expected — inverting the guard breaks every branch) |
| M3 | preflight removed (`curl -sf …/health` → `true`) | `1 failed, 10 passed` → `test_nightly_flag_on_bridge_down_fails_loud` **RED**, and only that one |

```
=== BASELINE ===     11 passed in 3.62s
=== POST-REVERT ===  11 passed in 3.35s
SHA pre  = bae41ae8c6b2bacf2137fd89e403183196e350743c376f66b2612154f86cfb96
SHA post = bae41ae8c6b2bacf2137fd89e403183196e350743c376f66b2612154f86cfb96
IDENTICAL = True
```

The full **M1–M6 matrix is now complete** and every mutation killed its intended guard.

**Criterion 1 remains NOT MET, and is not claimed.** Attempt 5 (the first with all three
retrievers live, after 76.9.3 restored DuckDuckGo) wedged after a clean research phase
and was killed at the 30-minute cap. Full evidence in `live_check_76.9.2.md` §8. The
useful new fact: the bridge served **27/27 HTTP 200** with the last LLM call completing
at 00:22:52, and the client then sat at 0% CPU with no in-flight request — so the
blocker is a distinct hang, not a routing failure. It is queued as **76.9.5** (P1),
which requires the wedge to be *captured* rather than inferred and explicitly keeps the
bridge's aggregation under suspicion (a 200 does not prove a well-formed body).

Bridge lineage note: the process serving all of this is the **repo** script
(`scripts/ops/anthropic_max_bridge.py`, PID 85602) — it survived the previous session,
so the "durable routing, not the scratchpad bridge" half of criterion 1 is satisfied;
what is missing is a completed `rc=0` run through it.

**Status recommendation for this step: NOT done.** Criteria 2, 3, 4 and 5 are met with
verbatim evidence; criterion 1 is blocked on 76.9.5. Marking it done would be a false
claim of a passing end-to-end run.

---

## Cycle-3 update (2026-07-25): criterion 1 is now **MET** — this supersedes the cycle-2 recommendation above

The cycle-2 section above ends "Status recommendation for this step: **NOT done**",
which was correct when written and is **no longer true**. Recorded as a supersession
rather than an edit, so the sequence stays auditable.

**What changed.** The Q/A's cycle-1 FAIL identified the blocker with a live raw-socket
probe: the SSE **passthrough** branch emitted no HTTP/1.1 body delimiter, so keep-alive
clients (httpx / the anthropic SDK, which gpt_researcher reaches via `stream=True` at 6
sites) hung forever. **My hardening had introduced it.** Fixed in
`scripts/ops/anthropic_max_bridge.py` (`Connection: close` + `close_connection = True`),
guarded by a new raw-socket keep-alive test, and proven by mutation **M7**: reverting
the fix turns the NEW guard RED while the OLD urllib guard stays GREEN — demonstrating
that guard's vacuity rather than asserting it. Full detail in `live_check_76.9.2.md` §9.

**Criterion 1 evidence** (`live_check_76.9.2.md` §10): attempt 6 ran 01:16:47 → 01:22:47
and logged `END nightly autoresearch OK`, which `run_nightly.sh:94-96` emits **only** in
the `if python …/run_memo.py; then` success branch; that same branch reset the fail-state
to `consecutive_fails: 0` (it had been 2). It wrote a **16,634-char non-ERROR** memo with
synthesized prose and a real bibliography, served by the repo-versioned bridge (pid
50256), with `ANTHROPIC_API_KEY=max-rail-dummy-key` and **zero** `api.anthropic.com` /
`401` / `authentication_error` occurrences in the run output.

**Also fixed this cycle** (from the cycle-2 verdict): a stale `11 passed` in a block
labelled verbatim (now 12, regenerated); a `(NEW, 11 tests)` count (now 12); a
`{changed-py robust form}` placeholder sitting inside a verbatim block (now the real
command, executed to confirm it runs); and a POST-count claim that attributed a
whole-log grep to one process lifetime (re-measured: 6 since this bridge started).
Historical mutation-matrix captures were deliberately left at their original values —
those runs really did execute against 11 tests, and rewriting them would falsify the
record.

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Real run rc=0 through the DURABLE routing + $0-leakage proof | **MET** (live_check §10) |
| 2 | Flag-gated, one-env-change revert, default documented | MET |
| 3 | Other proxy clients enumerated/unbroken or additive-only | MET |
| 4 | Durable CLAUDE_PATH + MODEL_MAP coverage | MET |
| 5 | MUTATION: broken routing fails loudly, never silent metered fallback | MET (§1, live rc=78) |

**Status recommendation: all five criteria met.** Pending an independent verdict — I am
not closing this on my own reading of a run I wanted to succeed.
