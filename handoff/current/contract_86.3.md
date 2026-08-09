# Contract — phase-86.3

**Step:** `86.3` — *P1 TEST SUITE PAUSES THE LIVE TRADING BOOK: any full
`backend/tests` run POSTs a real pause/resume cycle to `localhost:8000`, and
worktree isolation does not contain it.*
**Cycle:** 186 · **Date:** 2026-08-09 · **Absorbs:** step `36.21` (superseded)

**STATUS: PLAN COMPLETE.** Research gate closed `gate_passed: true` before any
GENERATE work began.

---

## 1. Research-gate summary

**Brief:** `handoff/current/research_brief_86.3.md` · **Verdict:** `gate_passed: true`

```json
{"tier":"moderate","external_sources_read_in_full":8,"snippet_only_sources":22,
 "urls_collected":30,"recency_scan_performed":true,"internal_files_inspected":15,
 "coverage":{"audit_class":false},"gate_passed":true}
```

**Findings that changed the design (all four re-verified by Main at source
before being acted on — the gate is evidence, not authority):**

1. **The seam already exists.** `backend/tests/conftest.py:47-61` patches
   `urllib.request.urlopen` at conftest **import** time, denying exactly one host
   (`slack.com`) by explicit design. The offending module resolves `urlopen` by
   module-attribute lookup at call time, so that patch is already in its path.
   **86.3 is a predicate widening, not a new mechanism.**
2. **No published tool has verb granularity.** pytest-socket, pytest_network,
   pytest-recording, pytest-test-categories ADR-001, httpretty, responses, respx
   all gate on *host* or on *library*, never on *method*. The policy required
   here is "GET `:8000` yes, POST `:8000` no", so it must be hand-rolled.
   Separately, `responses`/`respx`/urllib3 recipes are the **wrong layer** —
   stdlib `urllib.request` does not pass through urllib3, so the most-cited
   conftest recipe would miss this defect entirely.
3. **Two collection-time traps.** *(Re-verified: `sed -n '46,56p;110,114p'`.)*
   `_backend_is_up()` is called **inside the `skipif` decorator**, so it runs at
   **module import** — no fixture of any scope, session included, is early
   enough to intercept it. And `:53` catches only
   `(urllib.error.URLError, OSError, TimeoutError)`, so a `RuntimeError` raised
   on the **GET probe** would escape, error the whole module's collection, and
   take `test_phase_23_2_4_audit_log_clean_transitions` down with it —
   **breaking inherited criterion 7.** ⇒ **the guard must let GETs through and
   block only mutating verbs.**
4. **My §4.4 census was wrong in both directions.** *(Both re-verified.)*
   - **False positives:** `test_phase_36_7...` (`:10`, a `curl` inside a
     docstring) and `test_phase_75_17...` (`:303`, a `curl` string assigned to
     `cmd` and passed as an argument, never executed).
   - **Missing:** `backend/tests/test_phase_76_9_2_max_bridge.py` runs its own
     `ThreadingHTTPServer` and POSTs to it on an **ephemeral port**
     (`s.bind(("127.0.0.1", 0))` at `:130`). Benign — but **a host-level
     `127.0.0.1` block would break it.** ⇒ **the policy must key on host AND
     port, never host alone.**

**Correction to the brief, made by Main:** the brief proposes adding a
`urllib3.connectionpool` patch to close "`requests` + `httpx` for free". That is
half right — `httpx` uses **httpcore**, not urllib3, so a urllib3 patch covers
`requests` and anything else on urllib3, but **not** httpx. Recorded so the
coverage claim in §5 is accurate.

**Launch-path disclosure, recorded before the fact:** this gate ran via the
**Agent-tool fallback**, not the first-class Workflow rail. The operator
instruction of 2026-07-27 requires both Layer-3 agents on the Workflow
structured-output rail, but `.claude/workflows/` holds only
`harness-self-audit.js`, `probe-qa-tool-surface.js` and `qa-verdict.js` —
**there is no `research-gate.js`**. That gap is masterplan step `36.27`
(`pending`; priority field `P1`, title mis-tagged `[P2 --]`). The fallback is
documented in CLAUDE.md, so this is a permitted path and not a breach — but the
envelope above is **prose I parsed by eye, not schema-enforced**, so an
over-claimed envelope would not have been caught mechanically. Mitigated here by
re-verifying all four load-bearing findings at source. **36.27 is queued as the
very next step**, so the gates for `86.1`, `86.2` and `36.17` run enforced.

**Disclosure, recorded before the fact:** this gate was launched via the
**Agent-tool fallback**, not the first-class Workflow rail. Per the operator
instruction of 2026-07-27 both Layer-3 agents should launch via the Workflow
structured-output rail, but `.claude/workflows/` contains only
`harness-self-audit.js`, `probe-qa-tool-surface.js` and `qa-verdict.js` —
**there is no `research-gate.js`**. That gap is masterplan step `36.27`
(`pending`, priority field `P1`, title mis-tagged `[P2 --]`). The fallback is
documented in CLAUDE.md, so this is a permitted path, not a breach — but the
envelope below is **prose parsed by eye, not schema-enforced**, which means a
malformed or over-claimed envelope would not be caught mechanically.
**36.27 is queued as the next step after this one**, so the four gates that
follow run on the enforced rail.

---

## 2. Hypothesis

A test suite must not be able to mutate live safety state, and the protection
must hold for a **new** test module that opts into nothing. There are **two
distinct channels** into `handoff/kill_switch_audit.jsonl`, and neither one's
fix covers the other:

- **HTTP** — a test POSTs to the running backend, and *the server* appends the
  row. No client-side file redirect can prevent this.
- **Filesystem** — a test calls a mutating `kill_switch` method in-process and
  `_append_audit` writes directly. No HTTP guard can prevent this.

Therefore the fix is **two guards at one scope**, both installed at conftest
**import time** (before collection imports any module), not as opt-in fixtures.

---

## 3. Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. running the whole backend/tests suite with a live backend listening appends ZERO rows to handoff/kill_switch_audit.jsonl -- measured as line count AND sha256 before/after, not asserted
2. the test still provides its original value (it exists to prove pause/resume does not deadlock) -- it must run against an isolated target (a test app / ephemeral port / injected client), not be deleted or blanket-skipped to obtain the zero
3. a mutation proves the guard bites: point the test back at the live URL and show the row count rising again
4. the fix is stated in terms of the CHANNEL, not the file: enumerate every test that reaches a live host and say which are now contained and which are not
5. no other test changes status vs a measured baseline; fresh Q/A PASS

**Inherited from `36.21`** (absorbed; recorded in the masterplan entry):

6. the protection is **SESSION-SCOPED** so a NEW test module cannot regress this by simply not opting in, and a test proves the protection itself fires
7. `test_phase_23_2_4_audit_log_clean_transitions` still passes with its trigger allowlist **BYTE-UNCHANGED**

**Verification command (immutable):**
```
bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py -q --timeout=120'
```

**live_check:** `live_check_86.3.md` with the line count and sha256 of
`handoff/kill_switch_audit.jsonl` before and after a full `backend/tests` run
with the backend up, plus the enumeration of live-host-reaching tests.

---

## 4. Measured baseline (this session, before any change)

### 4.1 Live journal

```
handoff/kill_switch_audit.jsonl
  lines : 62
  sha256: 90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653
```

### 4.2 Live kill-switch state (GET only — no mutating call was made)

```
paused: false   sod_date: "2026-08-08"   sod_nav: 23830.46   peak_nav: 24666.57
breach.armed: false   breach.daily_baseline_stale: true
breach.daily_baseline_missing: false   breach.trailing_baseline_missing: false
breach.trailing_dd_pct: 3.3755 (limit 10.0)
```

`armed: false` is case C (UTC date past `sod_date`); the trailing leg still
fires. Not touched, not "fixed".

### 4.3 The writer, identified

`backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py`

- `BACKEND_URL = "http://localhost:8000"` (line 42)
- both live tests gated by `@pytest.mark.skipif(not _backend_is_up())`, which
  probes `GET /api/health` — **measured 200 right now** (backend pid 36970 under
  launchd)
- `test_..._live_pause_resume_pause_cycle_under_5s` POSTs `pause → resume →
  pause`, then a **4th** restore `resume` when the book was unpaused pre-cycle
  → **4 rows per run**, matching the measured 22:29:41-43Z / 22:36:59-22:37:01Z
  clusters exactly.

### 4.4 Channel census — derived, with the rule stated

**Rule:** a file is MUTATING if it targets a live host **and** contains an
explicit `method="POST|PUT|DELETE|PATCH"`, **or** `Request(..., data=...)`
(urllib defaults to POST when `data` is present), **or** a
`.post(/.put(/.delete(/.patch(` call.

> The `data=` clause was **added after a first pass omitted it**. The first rule
> keyed only on an explicit `method=` and would have missed any urllib POST
> written the idiomatic way. Recorded because the corrected rule is the one the
> result rests on.

Population: every file under `backend/tests/` matching `localhost:8000|127.0.0.1:8000`.

**Corrected by the research gate and re-verified by Main** (see §1 finding 4).
The rule above keys on *text*; the corrected rule keys on an **executed call
site to a real network client whose host:port resolves at runtime** — docstrings,
strings passed as arguments, and AST-analysis subjects do not count.

| File | Class |
|---|---|
| `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | **MUTATING** — `method="POST"` at `:72`, the only one |
| `test_phase_23_2_9_ticker_meta_latency.py` | read-only GET on `:8000` (also `requires_live`-marked) |
| `test_phase_23_2_13_governance_watcher.py` | read-only GET on `:8000` |
| `test_phase_23_2_7_red_line_nav_match.py` | read-only GET on `:8000` |
| `test_phase_76_9_2_max_bridge.py` | **POSTs — but to its OWN `ThreadingHTTPServer` on an ephemeral port** (`s.bind(("127.0.0.1", 0))` at `:130`). Legitimate and must keep working. |
| ~~`test_phase_36_7_kill_switch_rotation_rearm.py`~~ | **not live-host** — `curl` in a docstring at `:10` |
| ~~`test_phase_75_17_verification_paths.py`~~ | **not live-host** — `:303` `curl` string is an argument to `fp_reason()`, never executed |
| ~~`test_phase_75_deploy_surface.py`~~ | **not live-host** — AST-inspects another file; `localhost:3000` is a CORS regex assertion |
| ~~`test_phase_80_2_error_response_contract.py`~~ | **not live-host** — in-process `TestClient` (and the model to copy) |

**Exactly one** mutating live-*backend* test under `backend/tests/`. Same
corrected rule over `tests/`: **zero** mutating (the `verify_phase_*.py` scripts
issue GETs, and pytest does not collect them — they do not match `test_*`).

**`76_9_2` is why the policy keys on host AND port.** A host-level `127.0.0.1`
block — which is what `pytest-socket --allow-hosts` would force — breaks a
legitimate test. Recorded because it is the single easiest way to get this
fix wrong.

### 4.5 The filesystem channel — and why a census cannot secure it

A file-level scan ("does this file mention `_AUDIT_PATH` anywhere?") reports
**no** unredirected direct writers. **That result is wrong, and I can prove it
with a case already known to be a defect.**

`backend/tests/test_book_safety_69.py` imports `kill_switch`, calls a mutating
method, and mentions `_AUDIT_PATH` — so the file-level scan marks it **safe**.
But the mutating call is:

```python
def test_peak_reset_dark_by_default(monkeypatch):
    st = ks.get_state()                              # the REAL singleton
    ...
    out = st.reset_peak(12345.0, trigger="flatten")  # no redirect in THIS function
```

The file's three `_AUDIT_PATH` redirects are in *other* functions. This is
step `86.1`'s landmine, and my census called it clean.

**Conclusion that shapes the design:** a per-file census is not a sound basis
for this guarantee, and a per-call-site census is not maintainable — a new test
regresses it by simply being written. The protection must be **structural**:
the live path is unwritable during a test session, so no audit is required at
all. This is exactly criterion 6 (inherited from `36.21`).

### 4.6 Existing seam — already in the repo

`backend/tests/conftest.py` (the **only** conftest in either Python test tree)
already installs, **at import time**:

- `PYFINAGENT_TEST_NO_BQ=1` (phase-61.2), and
- a `urllib.request.urlopen` wrapper (phase-82.58) that refuses POSTs to
  `slack.com` — with the docstring *"Scoped to slack.com ONLY — this is not a
  general network jail."*

Import-time install is deliberate there ("it must be active before collection
imports any module") and is precisely the property criterion 6 needs.

**Interception verified independently**, without touching the backend: a
standalone snippet replicating the offending test's exact call pattern
(function-local `import urllib.request`, then `urllib.request.urlopen(...)`)
was intercepted by a module-level patch — the guard raised **before any network
call was made**.

```
INTERCEPTED: GUARD FIRED on 'http://localhost:8000/api/paper-trading/pause'
```

**Scope boundary, stated plainly:** `backend/tests/conftest.py` protects
`backend/tests/` and below. The second tree `tests/` has **no** conftest, so a
guard placed there does **not** reach `tests/services/`, `tests/api/`,
`tests/slack_bot/` or the `verify_phase_*.py` scripts. 86.3's criterion 1 is
scoped to `backend/tests`, so this is in-scope-complete — but it is **not**
repo-wide, and the artifacts must say so rather than implying a general jail.

### 4.7 Suite baseline (recorded, not re-run)

From `handoff/current/live_check_85.4.md` §5, live tree, 2026-08-08:

```
26 failed, 3017 passed, 12 skipped, 5 xfailed, 1 xpassed
```

All 26 node ids are recorded there. Two are expected to move **for reasons
already known**, and this must be accounted for under criterion 5 rather than
reported as a regression:

- `test_book_safety_69.py::test_valid_nav_still_breaches` — fixed by `85.5.1`.
- `test_phase_23_2_4_...::test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s`
  — **is itself one of the 26**. It fails today because `POST /resume` returns
  409 on the stale anchor (`daily_baseline_stale: true`, measured in §4.2).
  This step changes its target, so its status is expected to change; that is
  **this step's own test**, not "another test".

A worktree run gives a different shape (19-20 failed) because it lacks
gitignored files. **Compare live-to-live only.**

---

## 5. Plan

1. **Guard A (HTTP) — a NEW repo-root `conftest.py`, installed at import time.**
   Policy: refuse a request when its **verb is mutating**
   (`POST/PUT/DELETE/PATCH`) **AND** its target host:port is the live backend
   (`localhost` / `127.0.0.1` / `[::1]` **on port 8000**). `GET` always passes —
   required by §1 finding 3, or the module fails to collect and criterion 7
   breaks. Port-keyed, not host-keyed — required by §1 finding 4, or
   `76_9_2` breaks.
   - **Placement:** repo root, because `pytest.ini` lives there, so rootdir is
     the repo root and a root conftest is loaded for **both** test trees —
     including `tests/`, which has **no conftest at all** today. Import-time
     install is strictly stronger than a session-scoped autouse fixture: it is
     live before *collection*, which is the only thing early enough to matter
     (§1 finding 3). This is what satisfies inherited criterion 6.
   - **The existing `backend/tests/conftest.py` slack guard is left byte-
     untouched.** Root conftest loads first, so the slack wrapper simply chains
     on top of the new one. Moving it would risk losing slack protection for any
     invocation that resolves a different rootdir, for no benefit.
   - **Verb detection must handle the bare-URL form:**
     `urlopen(url_string, data=b"...")` is a **POST** even though a `str` has no
     `get_method()`. The guard reads `req.get_method()` when available and
     otherwise infers POST from a non-`None` `data` argument. *(This is the same
     class of miss as the `data=` gap I already made once in §4.4.)*
   - **Second client family:** also patch
     `urllib3.connectionpool.HTTPConnectionPool.urlopen` with the same policy,
     which closes `requests` and anything else riding urllib3. **It does NOT
     cover `httpx`** (httpx uses `httpcore`), and it does not cover raw
     `socket`. Stated as a known, bounded gap — not papered over.
2. **Guard B (filesystem) — explicitly NOT in this step.** The in-process door
   (`_append_audit` writing directly when `_AUDIT_PATH` still points at the live
   file) is a **different channel** and no HTTP guard can see it. It is
   deliberately left open here:
   - the one **known** in-process writer is `86.1`'s `reset_peak` landmine,
     which is inert today (`kill_switch_peak_reset_enabled = False`, measured)
     and has its own step, queued next-but-one;
   - a global `_AUDIT_PATH` redirect would **break criterion 7** —
     `test_phase_23_2_4_audit_log_clean_transitions` reads the **live** journal
     and requires ≥3 parseable rows;
   - and §4.5 shows I cannot *prove* the absence of other in-process writers,
     because my own census produced a false negative on the one case I already
     knew about.
   ⇒ A file-door **preventer** (refuse `_append_audit` when `_AUDIT_PATH` is
   still the live path — which leaves both tmp-redirected writers and live
   *readers* working) is the right design, but it is a second channel with its
   own blast radius and belongs in its own research-gated step. **Queued, not
   silently skipped.**
3. **Repair the test's value, do not delete it.** Re-point
   `test_..._live_pause_resume_pause_cycle_under_5s` at an isolated in-process
   target. Prior art already exists in-repo for these exact endpoints:
   `tests/api/test_pause_resume_timeout.py` drives `resume_trading(req)`
   directly under an autouse tmp `_AUDIT_PATH`. The deadlock property (three
   transitions, each under 5s, no re-entrant lock) is preserved.
4. **Prove both guards bite** (criterion 3 + 6), then
5. **Full-suite run, live tree, backend up** — sha256 + line count before and
   after (criterion 1), and a set-diff of failures against §4.7 (criterion 5).

**Ordering is a safety requirement, not a preference:** the guards go in
*before* any full-suite run. A "baseline" suite run taken first would commit
the very defect under repair and would itself violate criterion 1.

### 5.1 Criterion 3 — how it is proven without harming the live book

Criterion 3 says *"point the test back at the live URL and show the row count
rising again."* Read literally against the operator's book, satisfying it would
require deliberately pausing live trading — committing the defect to prove the
fix, and breaking criterion 1 in the act. It will be split:

- **Guard-ON arm — against the real live URL.** Safe *because* the guard
  intercepts at `urllib.request.urlopen`, i.e. **before a connection is
  opened**. Asserts the refusal and that the live journal's sha256 is unchanged.
- **Guard-OFF arm — against an isolated stub** on an ephemeral port that
  appends to a `tmp_path` journal. Shows the row count rising when the guard is
  removed.

The guard is client-side and cannot see what is listening, so the stub exercises
the identical code path. **This is a deliberate deviation from the literal
wording of criterion 3, disclosed here rather than silently taken**, and the
criterion text is **not** edited. If Q/A judges the substitution insufficient,
that is a legitimate CONDITIONAL and I will not argue it away.

---

## 6. Explicit non-goals

- **Not** gitignoring `handoff/kill_switch_audit.jsonl` (banned by 36.21).
- **Not** relaxing the trigger allowlist in
  `test_phase_23_2_4_audit_log_clean_transitions` (criterion 7 — byte-unchanged).
- **Not** deleting or blanket-skipping the deadlock test (criterion 2).
  Marking it `requires_live` would be a blanket skip and is rejected.
- **Not** fixing `36.28` (the read-half coupling) or `86.1` (the `reset_peak`
  landmine). Guard B may incidentally *contain* 86.1's blast radius; 86.1 still
  needs its own step because containment is not the same as the test no longer
  touching the real singleton.
- **No** threshold, limit, or gate is weakened. No `backend/.env` write. No flag
  promotion. `historical_macro` untouched.
- **No** real-money action. Paper trading only.

## 7. References

- `handoff/current/research_brief_86.3.md` *(pending)*
- `handoff/current/killswitch_cluster_reconciliation_2026-08-09.md` — §2 merge
  of `36.21` into this step
- `handoff/current/live_check_85.4.md` §5 — the 26-node-id baseline
- `handoff/current/experiment_results_85.5.1.md`, `live_check_85.5.1.md` — the
  original measurement of the 8 rows
- `backend/tests/conftest.py` — the import-time seam
- `tests/api/test_pause_resume_timeout.py` — in-process prior art
- masterplan `86.3`, `36.21` (superseded), `36.28`, `86.1`, `36.27`
