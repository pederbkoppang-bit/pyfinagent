# Q/A Critique — goal-browser-mcp (Playwright MCP: browser-driving for Claude Code)

**Evaluator:** Q/A (Layer-3 harness MAS, merged qa-evaluator + harness-verifier)
**Date:** 2026-06-01
**Verdict:** PASS (restart_pending: true — documented post-restart live-dispatch check)
**Mode:** in-place working-tree read (changes UNCOMMITTED; no worktree, no checkout)

---

## 1. Harness-compliance audit (5 items)

1. **Researcher spawned before contract** — PASS. `research_brief.md` exists
   (`# research_brief -- goal-browser-mcp`), `gate_passed: true`, **7 sources read
   in full** (floor 5; table lists 7 WebFetch-full + 15 snippet-only = 22 URLs),
   recency scan present (5 findings incl. live RCE #1495 / Google-login-block
   advisories). Contract "Research-gate summary (PASSED)" cites researcher
   `a983c70b128d3bdd5` + brief path + decisive findings with file:line anchors
   (middleware.ts:22, MarketFilter.tsx:62/71/77, cockpit-helpers.tsx:198/226).
2. **Contract written before GENERATE** — PASS. `contract.md` has step id
   (`goal-browser-mcp`), all 4 immutable criteria + the OUT-OF-SCOPE clause copied
   verbatim, a dependency-ordered plan (A-D), the restart caveat, and a references
   block.
3. **experiment_results present** — PASS. Lists what was built (A+B+C), the
   NEW/MODIFIED file list, the verbatim launch-bug + smoke + click-through +
   dev-restore output, and a per-criterion (1-4) status section.
4. **Log-last discipline** — PASS. `grep -in goal-browser-mcp handoff/harness_log.md`
   returns NOTHING — no cycle header yet. This is a `/goal` step, not a masterplan
   phase. Correct: log + status flip come AFTER this PASS.
5. **No verdict-shopping** — PASS. No prior `goal-browser-mcp` verdict in
   `harness_log.md` (grep for `goal-browser` + result/PASS/FAIL/CONDITIONAL = none).
   FIRST verdict on this evidence. sycophancy-under-rebuttal / second-opinion-shopping
   / 3rd-conditional heuristics all N/A (no prior cycle).

---

## 2. Deterministic checks (re-run independently — did NOT trust generator output)

### 2.1 `.mcp.json` parses + playwright server shape (binding gate)
```
$ python3 -c "json.load(open('.mcp.json'))['mcpServers']['playwright']"
type=stdio  command=npx  alwaysLoad=false
args = [-y, @playwright/mcp@0.0.75, --executable-path <bundled Chromium>,
        --user-data-dir <repo>/.playwright-mcp-profile, --allowed-hosts localhost,
        --viewport-size 1440,900]
```
- **command:"npx"** ✓
- **pinned `@playwright/mcp@0.0.75`** ✓ (NOT `@latest` — verified `@latest` absent).
- **`--allowed-hosts localhost`** ✓
- **`alwaysLoad:false`** ✓
- **NO dangerous flags** ✓ — programmatic scan for `--caps`/`vision`/`pdf`/`devtools`/
  `--allow-unrestricted-file-access`/`--allowed-origins`/`browser_run_code` → `NONE`.

### 2.2 `--executable-path` actually exists on disk (the criterion's hard requirement)
```
$ test -x "/Users/ford/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
EXEC_X=YES   (-rwxr-xr-x  52064  17 apr.  ...Google Chrome for Testing)
```
The path in `.mcp.json` resolves to a real executable. NOT a phantom path.

### 2.3 Smoke test RE-RUN independently → EXIT 0, genuinely drove a browser
```
$ python3 scripts/mcp_servers/smoke_test_playwright_mcp.py ; echo EXIT=$?
using bundled Chromium: /Users/ford/Library/Caches/ms-playwright/chromium-1208/.../Google Chrome for Testing
spawning: npx -y @playwright/mcp@0.0.75 --headless --isolated --executable-path <bundled>
OK initialize -- server=Playwright
OK tools/list -- 23 tools; sample: ['browser_click','browser_close','browser_console_messages',
                                     'browser_drag','browser_drop','browser_evaluate','browser_file_upload','browser_fill_form']
OK required browser tools present: ['browser_navigate','browser_click','browser_snapshot']
OK browser_navigate -- http://localhost:3000/login
OK browser_snapshot -- real login DOM read (matched token: 'sign in')
SMOKE PASS: ...drove a real navigation + DOM read on the live dev server.
EXIT=0
```
Reproduced cleanly. It really drove Chromium: navigated the live dev server's
`/login` and matched the REAL login token **`'sign in'`** in the a11y snapshot —
NOT an error string. 23 tools enumerated; all 3 required present. This is the
in-session proof the server attaches and its tools are callable at the protocol
level. `SMOKE_SYNTAX=OK` (ast.parse).

### 2.4 Anti-false-positive (the prior "google"-in-error-text false-pass)
Read the script (`smoke_test_playwright_mcp.py:50-55, 177, 191-201`). The earlier
loose test false-passed by matching `"google"` in the "Google Chrome.app not found"
launch error. The CURRENT script CANNOT false-pass:
- `LOGIN_TOKENS = ("sign in","passkey","ai financial analyst")` — **deliberately
  omits "google"** (inline comment says exactly why, `:51-52`).
- `ERROR_MARKERS = ("### error","is not found at","run \"npx playwright install","executable doesn't exist")`
  — a launch/nav error is REJECTED (`:177` checks `isError` OR any marker on nav;
  `:191` rejects markers on snapshot) BEFORE the token match.
- Token match requires one of the SPECIFIC login tokens (`:194-201`), else FAIL.
So a broken config (wrong/missing executable, unpinned bad version, missing tool)
surfaces as a non-zero exit, not a green check. **Mutation-resistant** (see §4).

### 2.5 `.gitignore` ignores the profile dir
```
$ git check-ignore -v .playwright-mcp-profile/
.gitignore:65:.playwright-mcp-profile/   .playwright-mcp-profile/
```
Present and effective. Diff adds it with a comment explaining it may hold a
logged-in session.

### 2.6 Runbook SAFETY section — required topics all present
`docs/runbooks/browser-mcp.md:113-136` (grep-confirmed each):
no **trades**/money/**credentials**/account-or-security-**settings**/**consent**
dialogs (`:118-121`); page content is **untrusted** data / OWASP LLM01 indirect
prompt injection (`:122-125`); **localhost**-only `--allowed-hosts` with the
"not a hard boundary" caveat (`:126-127`); keep capabilities **minimal**, no
code-exec, cites **RCE #1495** (`:128-131`); no real creds in the automated
browser (`:132-133`); **pin the version**, never `@latest` (`:134`); alwaysLoad
rationale (`:135-136`). Comprehensive and specific.

### 2.7 Scope — no trading logic touched
```
$ git status --short | grep -vE 'handoff/'
 M .gitignore
 M .mcp.json
?? .playwright-mcp/          <-- see NOTE in §4
?? docs/runbooks/browser-mcp.md
?? scripts/mcp_servers/smoke_test_playwright_mcp.py
```
Scan for `backend/(backtest|screener|autonomous|risk|services/paper_trader|
services/kill_switch|markets)` etc. in the diff → **NONE**. Only the 4 in-scope
files (+ a runtime output dir). OUT-OF-SCOPE honored.

### 2.8 secret-in-diff
`grep -iE '(api_key|secret|password|token)\s*=\s*['"'"'"][A-Za-z0-9/+]{16,}'` over
the tracked diffs AND both new files → none (rc=1).

### 2.9 dev-server gate restored (verifying the click-through was reversed)
```
$ curl -s -L --max-redirs 0 http://localhost:3000/paper-trading -o /dev/null -w '%{http_code}'
/paper-trading HTTP 302
```
**302 = gated/restored.** Confirms the generator unset `LIGHTHOUSE_SKIP_AUTH` and
kickstarted the frontend back to the auth-gated state after the click-through, as
experiment_results claims (`:88-92`). I did NOT toggle skip-auth or restart the
frontend (per prompt instruction) — I only re-verified the restored state.

---

## 3. The 4 immutable criteria — independent verification

1. **Attaches in a fresh session + browser-driving tools callable** — PASS
   (protocol-PASS + documented restart check). The smoke test's real MCP
   `initialize` → `notifications/initialized` → `tools/list` returns server=Playwright
   with 23 tools incl. all of `{browser_navigate, browser_click, browser_snapshot}`,
   and a live `browser_navigate`+`browser_snapshot` round-trip on the dev server.
   This is the strongest IN-SESSION evidence the server attaches and tools are
   callable. The `mcp__playwright__*` tool DISPATCH inside THIS Claude Code session
   needs a restart (MCP config is session-snapshotted — same rule as agent `.md`
   edits). That gap is HONESTLY disclosed in both contract (`:92-99`) and
   experiment_results (`:108-113`) with a concrete post-restart check (`/mcp` lists
   the server, or re-run the smoke test). This mirrors the established
   qa-roster-live post-restart pattern — see §5 for why this is acceptable as a
   PASS rather than a CONDITIONAL.
2. **Version-controlled, reproducible install/config** — PASS. `.mcp.json` pinned
   server (§2.1) + `.gitignore` entry (§2.5) + reproducible smoke test (§2.3,
   re-ran to exit 0) + `docs/runbooks/browser-mcp.md`. The criterion says
   "`.mcp.json` pin + smoke test, OR a docs/runbooks/ entry" — all three exist.
3. **SMOKE TEST passes, captured verbatim, incl. EU pill → "vs DAX"** — PASS.
   experiment_results `:69-87` captures the click-through verbatim:
   `INITIAL bench= vs SPY | All checked= True` → click EU (`EU ref: e174`) →
   `AFTER EU bench= vs DAX | EU checked= True` → click US → `AFTER US bench= vs SPY
   | US checked= True`. This is REAL `browser_click` interaction with a per-snapshot
   ref (`e174`), asserting BOTH the benchmark-label flip (SPY→DAX→SPY) AND the
   radio checked-state — not a screenshot, not faked (see §5 for the genuineness
   analysis). The contract's "if all-US, click All↔US instead" fallback was not
   needed because skip-auth exposed the full filter; the stronger EU→DAX assertion
   was achieved. I did NOT re-run this leg (per prompt — it requires the reversible
   skip-auth toggle the generator already ran + restored, §2.9); I verified it was
   done and reversed.
4. **Safety guardrails in the runbook** — PASS. §2.6 — all required guardrails
   present and specific, grounded in the research (RCE #1495, OWASP LLM01,
   "not a security boundary").

---

## 4. Code-review heuristics (5 dimensions) — findings

- **Dim 1 Security** (the load-bearing dimension here — adding a browser-control MCP
  is a materially larger attack surface): secret-in-diff = none (§2.8).
  `excessive-agency` / OWASP LLM06: a new action-capable tool IS added, but with a
  documented least-privilege posture — `--allowed-hosts localhost`, `alwaysLoad:false`,
  NO `--caps vision/pdf/devtools`, NO code-exec capability, NO
  `--allow-unrestricted-file-access`, version pinned, profile git-ignored, and a
  written guardrail doc. The RCE #1495 vector (`browser_run_code`/code-exec) is NOT
  in the default capability set and is NOT enabled. `browser_evaluate` is in the
  default 23-tool set (runs JS in the PAGE sandbox, distinct from #1495's
  system-command escape) — its risk is page-level, mitigated by localhost-only +
  untrusted-content guardrail. supply-chain-dep-pin: version IS pinned (the opposite
  of a pin-removal) → no flag. Net: least-privilege is documented → **WARN-class
  excessive-agency is satisfied/mitigated, downgraded to NOTE.**
- **Dim 2 Trading-domain correctness**: no kill-switch / stop-loss / perf-metrics /
  paper_trader / risk_engine / backtest path touched (§2.7). No
  llm-output-to-execution path. Clean — N/A.
- **Dim 3 Code quality**: smoke test is well-structured — typed helpers, explicit
  90s timeout, `finally` block that terminates/kills the child and tails stderr,
  dynamic Chromium resolution (not hard-coded build number), dev-server reachability
  guard that SKIPS (not fails) phase 2 if localhost is down. `print()` statements are
  in a `scripts/` smoke test (negation-list exempt). No broad-except in a risk path
  (the `except Exception` at `:88-89/108-109/210-219` are in JSON-parse-retry /
  reachability-probe / cleanup — appropriate, not swallowing a risk guard). Good.
- **Dim 4 Anti-rubber-stamp**: no `perf_metrics.py`/`risk_engine.py`/`backtest_*`
  change → financial-logic-without-behavioral-test BLOCK N/A. The artifact IS itself
  a behavioral test, and I re-ran it (§2.3). No tautological/over-mocked assertions —
  the smoke test asserts against REAL server responses (23 tools, real login token),
  not mocks. Not a rubber-stamp.
- **Dim 5 LLM-evaluator anti-patterns**: this critique cites file:line + verbatim
  command output throughout. First verdict on this evidence → no
  sycophancy-under-rebuttal, no second-opinion-shopping, no 3rd-conditional due.

**NOTE (housekeeping, does NOT degrade verdict):** an untracked `.playwright-mcp/`
output dir exists at repo root (default Playwright MCP `--output-dir`; holds
`console-*.log` + `page-*.yml` accessibility snapshots from the click-through runs).
It is NOT covered by the `.playwright-mcp-profile/` gitignore entry, so a careless
`git add -A` could commit DOM snapshots of the cockpit. This is NOT one of the 4
criteria (criterion #2 asks for `.playwright-mcp-profile/`, which IS ignored, §2.5)
and not in the declared file scope, so it does not block. Recommend the operator add
`.playwright-mcp/` to `.gitignore` too, or clear the dir before the next `git add`.

Worst severity across all dimensions: **NOTE** (no BLOCK, no WARN). Verdict not degraded.

---

## 5. LLM judgment — restart caveat, click-through genuineness, scope honesty

- **Criterion #1 restart caveat — HONEST and acceptable as PASS (not CONDITIONAL).**
  The criterion text is "attaches in a fresh session and its browser-driving tools
  are callable (show tool list or ping)." The smoke test literally does an MCP
  handshake + `tools/list` (shows the tool list) AND a live ping (navigate+snapshot)
  against a freshly-spawned server process — that IS "attaches + tools callable +
  tool list shown," satisfied NOW at the protocol layer. The ONLY thing deferred is
  the `mcp__playwright__*` dispatch INSIDE the current Claude Code session, which is
  a hard platform constraint (config snapshotted at session start), not a defect or
  an unfinished task. It is disclosed identically in contract + results with a
  concrete, cheap post-restart check (`/mcp`, or re-run the smoke test). This is the
  same shape the project already accepts for agent-roster changes (qa-roster-live).
  Forcing CONDITIONAL here would punish an honest, unavoidable platform fact that the
  generator both disclosed AND gave the strongest available in-session proof for.
  Verdict: protocol-PASS with `restart_pending: true` flagged in the envelope so the
  next session runs the live-dispatch check.
- **Criterion #3 EU→"vs DAX" click-through — genuine, not fakeable in this setup.**
  The captured output shows a per-snapshot ref (`EU ref: e174`) being passed to
  `browser_click`, and the assertion reads the benchmark label BEFORE (`vs SPY`) and
  AFTER (`vs DAX`), plus the `EU checked` aria-state flip, then a SECOND click (US)
  flipping it back (`vs SPY`, `US checked`). A fabricated result would have to
  invent the ref-string mechanics, the bidirectional state transitions, AND match the
  exact label strings from `cockpit-helpers.tsx:198` / `format.ts` MARKET_BENCHMARK_LABEL
  — and the experiment_results candidly documents a real friction (`browser_click`
  takes `target`, not `ref`/`element`, "discovered live from the tool's inputSchema
  after an initial param error", `:85-87`), which is the kind of detail that only
  surfaces from actually driving the tool. Corroborating: my independent §2.3 smoke
  run proves the SAME server can drive a real navigation+snapshot on this dev server,
  and §2.9 confirms the skip-auth gate was toggled and restored. I judge it genuine.
- **Mutation resistance** (would the smoke test catch a broken config?): YES.
  - Wrong/missing `--executable-path` → launch error → `ERROR_MARKERS` ("is not
    found at" / "executable doesn't exist") reject on nav/snapshot → non-zero exit.
  - Missing required tool (`browser_navigate`/`click`/`snapshot`) → explicit FAIL
    `:160-163`.
  - Unpinned/broken version that fails to spawn → `initialize` never returns →
    TimeoutError / non-`result` → FAIL `:144-147`.
  - The "google"-in-error false-pass is structurally closed (§2.4). Strong.
- **Scope honesty**: experiment_results discloses the launch-bug it hit AND the loose
  smoke test's false-pass, then how it fixed BOTH — i.e. it volunteers the exact
  anti-pattern the harness exists to catch, rather than hiding it. The restart gap is
  disclosed, not buried. The click-through used the documented reversible skip-auth
  path and restored the gate (§2.9). "DONE" claims map to verifiable artifacts.
  Nothing overclaimed.

---

## Verdict

**PASS** (`restart_pending: true`). All 4 immutable criteria satisfied:
(#1) protocol-level attach + 23-tool list + live navigate/snapshot proven in-session,
with the `mcp__playwright__*` live-dispatch honestly deferred to a post-restart `/mcp`
check (qa-roster-live pattern — judged an acceptable PASS, not a CONDITIONAL);
(#2) `.mcp.json` pinned `@0.0.75` (executable-path verified `test -x`, no dangerous
caps) + `.gitignore` + reproducible smoke test (I re-ran → exit 0) + runbook;
(#3) smoke test re-runs to exit 0 and CANNOT false-pass (specific login tokens +
error-marker rejection), and the EU→`vs DAX`→US→`vs SPY` click-through is genuine
real-interaction evidence with the dev gate restored afterward;
(#4) runbook SAFETY section complete and specific (no trades/money/creds/settings/
consent; untrusted page content; localhost-only; RCE #1495 / minimal caps / no
code-exec; pin version).
No trading logic touched; no secret-in-diff; worst code-review severity = NOTE
(one housekeeping flag: gitignore the `.playwright-mcp/` output dir too).

checks_run: harness_compliance_audit, mcp_json_parse_and_shape, executable_path_test_x,
no_dangerous_caps_scan, smoke_test_rerun, anti_false_positive_review, gitignore_check,
runbook_safety_coverage, scope_no_trading_logic, secret_scan, dev_gate_restored_check,
smoke_syntax, code_review_heuristics, harness_log_prior_verdict_scan
