# Experiment results — phase-86.3

**Step:** `86.3` — the test suite POSTs a real pause/resume cycle to the live
trading book. **Absorbs** `36.21`. **Cycle 186**, 2026-08-09.

---

## 1. What was built

| File | Change |
|---|---|
| `conftest.py` | **NEW — repo root.** Import-time egress policy: refuse a MUTATING verb (`POST/PUT/DELETE/PATCH`) aimed at the live backend origin (loopback host **AND** port 8000). GETs always pass. Also patches `urllib3.connectionpool.HTTPConnectionPool.urlopen` with the same policy. |
| `backend/tests/test_phase_86_3_live_egress_guard.py` | **NEW —** 12 tests proving the guard fires, does not over-reach, and that removing it lets the rows rise. |
| `backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` | **MODIFIED —** the live pause/resume/pause cycle now runs **in-process** against the real app with the kill switch detached. Adds a mutation lock and a function-scoped autouse live-journal byte-compare. |
| `backend/tests/conftest.py` | **UNTOUCHED** (deliberately — see §3.2). |

**Not changed:** no threshold, no limit, no gate, no `backend/.env`, no flag
promotion, `historical_macro` untouched. No masterplan `verification` block
edited. No real-money action.

---

## 2. The mechanism, restated in one line

`test_phase_23_2_4_pause_resume_no_deadlock_live.py` POSTed
`pause → resume → pause` (+ a 4th restore `resume` when the book was unpaused)
to `http://localhost:8000` whenever `GET /api/health` answered 200 — **4 rows
per run** into the operator's live journal, and the live armed book paused for
the duration.

`36.21` had recorded the same 4-row signature on 2026-07-26 but attributed it to
"an ORDERING or STATE-POLLUTION effect … `kill_switch._state` or `_AUDIT_PATH`
attached to the real path". **That hypothesis is refuted.** Its file census
grepped `resume_trading|\.resume\(\)|\.pause\(\)`; the offending file calls none
of those — it calls `_post_state_transition("/api/paper-trading/pause", ...)`
over HTTP:

```
$ grep -nE 'resume_trading|\.resume\(\)|\.pause\(\)' \
      backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py
(no match)
```

It was invisible to the census meant to find it. That is why running files one
at a time never reproduced the write.

---

## 3. Design, and the two constraints that forced it

### 3.1 GETs must pass — or criterion 7 breaks

`_backend_is_up()` is called **inside the `@pytest.mark.skipif(...)`
decorator**, so it evaluates at **module import**. No fixture of any scope —
session included — is early enough to intercept it. And it catches only
`(urllib.error.URLError, OSError, TimeoutError)`:

```python
    except (urllib.error.URLError, OSError, TimeoutError):
        return False
```

A `RuntimeError` raised on the GET probe would escape, error the whole module's
collection, and take `test_phase_23_2_4_audit_log_clean_transitions` down with
it — the very test criterion 7 requires to keep passing with its trigger
allowlist byte-unchanged. **Hence: block verbs, never the host.**

### 3.2 The policy keys on host AND port — or a legitimate test breaks

`backend/tests/test_phase_76_9_2_max_bridge.py` runs its own
`ThreadingHTTPServer` and POSTs to it on an **ephemeral** port
(`s.bind(("127.0.0.1", 0))` at `:130`). A host-level `127.0.0.1` block — which
is what `pytest-socket --allow-hosts` forces — breaks it. This is the single
easiest way to get the fix wrong, and it is why `pytest-socket` was rejected as
the primary mechanism (the research gate's Option B: no published tool has
**verb** granularity; they all gate on host or on library).

### 3.3 Placement: repo root, import time

`pytest.ini` is at the repo root, so rootdir is the repo root and a root
`conftest.py` is loaded for **both** test trees — including `tests/`, which has
**no conftest at all**. Import-time install (not an autouse fixture) is what
covers *collection*, per §3.1. Together these are what satisfy the criterion
inherited from `36.21`: **a NEW test module is protected by existing, not by
remembering to opt in.**

`backend/tests/conftest.py`'s phase-82.58 slack guard was left **byte-untouched**.
Root conftest loads first, so the slack wrapper simply chains on top. A test
asserts the slack guard is still in the chain, so this step cannot silently
displace it.

---

## 4. Verification — verbatim

### 4.1 The immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py -q --timeout=120'
5 passed, 1 warning in 2.05s
```

Live journal across that run: `90e0303130fc…` → `90e0303130fc…` (unchanged).

The five collected node ids:

```
$ python -m pytest backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py --collect-only -q
test_phase_23_2_4_existing_pytest_regression_files_exist
test_phase_23_2_4_existing_regression_files_reference_phase_23_1_22
test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s
test_phase_86_3_mutation_the_audit_redirect_is_load_bearing
test_phase_23_2_4_audit_log_clean_transitions
```

> **CORRECTED after EVALUATE pass 1.** This block previously read
> `4 passed, 1 warning in 3.75s`, captured **before**
> `test_phase_86_3_mutation_the_audit_redirect_is_load_bearing` — the
> criterion-3 substitute proof — existed. It was therefore a "verbatim" block
> that no longer reproduced, and it did not evidence the load-bearing test
> passing. Caught by the Q/A, which measured `5 passed` and noted the block was
> already contradicted by §4.2 in this same document (17 − 12 = 5). **Regenerated
> from the shipped tree rather than edited**, per the rule that a verbatim
> capture is re-run, never hand-adjusted.

### 4.2 Guard + rewrite together

```
$ python -m pytest backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py \
                   backend/tests/test_phase_86_3_live_egress_guard.py -q --timeout=180
.................                                                        [100%]
17 passed, 1 warning in 4.23s
```
Live journal `90e0303130fc…` before and after — unchanged.

### 4.3 ruff

```
$ ruff check conftest.py backend/tests/test_phase_86_3_live_egress_guard.py \
             backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py
All checks passed!
```

Six findings were raised and **all six fixed, none suppressed**: `I001`
(import order) and `RUF100` (stale `noqa`) autofixed; `S110`/`BLE001` ×2 fixed
by narrowing `except Exception` to `(AttributeError, TypeError, ValueError)`
and `(ImportError, AttributeError)` respectively, each with a debug log instead
of a bare `pass`. *(85.4 shipped with ruff red on a file it created; running it
here is the direct lesson from that.)*

### 4.4 The urllib3 leg actually installed

```
$ python -c "import conftest, urllib3.connectionpool as cp; print(cp.HTTPConnectionPool.urlopen.__name__)"
_guarded_pool_urlopen
```

Checked explicitly because the `try/except` around it could have swallowed a
failure and left the claim false.

---

## 5. Two guard tests failed first — both for real reasons

Recorded because in both cases the guard was **working** and my *assertion* was
wrong. A green run would have hidden the topology.

1. **`urlopen.__name__ == "_guarded_urlopen"` → FAILED**, found
   `'_no_slack_egress'`. The guards **chain**: root conftest installs mine
   first, then `backend/tests/conftest.py` wraps it. The outermost name is not
   mine.
2. **A closure-only chain walk → FAILED**, found only `{'_no_slack_egress'}`.
   `_no_slack_egress` holds its delegate in a **module-level global**
   (`_REAL_URLOPEN = urllib.request.urlopen`), not a closure cell, so its
   `__closure__` is `None` and a closure-only walk sees nothing through it.

Final form walks closure cells **and** `_REAL_*` module globals. It is
demonstrably **not vacuous** — it failed twice against a live tree before it
passed.

---

## 5b. My rewrite crashed the entire suite, and passing in isolation hid it

The first full-suite run after the rewrite **died at 13%**:

```
backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py ..E
object type name: ValueError
object repr     : ValueError('I/O operation on closed file.')
lost sys.stderr
```

**The crash was in my own file**, on the third test — the rewritten cycle. It
had passed every time I ran it alone (`4 passed`), alongside four adjacent
kill-switch files (`87 passed`), and with the guard file (`17 passed`).

**Cause:** I built the client as

```python
with authed_test_client(app, raise_server_exceptions=False) as client:
```

Entering `TestClient` as a **context manager runs the app LIFESPAN** — schedulers,
logging handlers, background threads — in the middle of a pytest session, and
their teardown collided with pytest's capture teardown. The in-repo model,
`test_phase_80_2_error_response_contract.py:38`, constructs the client
**without** the context manager for exactly this reason; I had copied the
pattern but not that detail.

**Fix:** drop the `with`. Requests still route through the full ASGI stack; only
startup/shutdown events are skipped, and this test asserts nothing about them.

**Why this is recorded rather than quietly fixed:** it is a clean instance of
"green in isolation, fatal in the suite". Every single-file run I did was
green, and each one would have been offered as evidence. Only the criterion-1
whole-suite run could see it — which is precisely why criterion 1 is written
the way it is, and why I did not treat the file-scoped greens as sufficient.

---

## 6. Criterion 3 — how it was proven, and what was NOT done

Criterion 3: *"point the test back at the live URL and show the row count rising
again."* Read literally against the operator's book, that means deliberately
pausing live trading to prove a fix — committing the defect under repair, and
breaking criterion 1 in the act. **Split, and disclosed rather than taken
silently:**

- **Guard-ON arm — against the REAL live URL.** Safe precisely because the
  guard raises **before a connection is opened**. Covers all four mutating
  verbs plus the bare-URL `urlopen(url, data=...)` form.
- **Guard-OFF arm — against an isolated stub** on an ephemeral port writing to a
  tmp journal. The unguarded `urlopen` is restored and the same 4-POST cycle
  replayed: **0 → 4 rows.** The guard is client-side and cannot see what is
  listening, so the stub exercises the identical code path.

**Likewise for the in-process rewrite**, the real mutation (delete the
`_AUDIT_PATH` redirect, let the cycle write to the live journal) was **not
run**. It is replaced by proving the two facts whose conjunction is the claim:
(a) the module default `_AUDIT_PATH` **is** the live file — asserted; and (b)
appends follow `_AUDIT_PATH` wherever it points — demonstrated against a
**byte-identical copy** of the live journal in tmp, with the original asserted
unchanged afterwards.

**If Q/A judges either substitution insufficient, that is a legitimate
CONDITIONAL and I will not argue it away.** The criterion text was not edited.

---

## 7. Channel enumeration (criterion 4) — and a census that was wrong

**Rule:** an executed call site to a real network client whose host:port
resolves at runtime. Docstrings, strings passed as arguments, and
AST-analysis subjects do **not** count.

| File | Class | Contained now? |
|---|---|---|
| `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | **MUTATING** → live backend | **YES** — rewritten in-process; guard also refuses it |
| `test_phase_23_2_9_ticker_meta_latency.py` | read-only GET | n/a — GETs deliberately allowed |
| `test_phase_23_2_13_governance_watcher.py` | read-only GET | n/a |
| `test_phase_23_2_7_red_line_nav_match.py` | read-only GET | n/a |
| `test_phase_76_9_2_max_bridge.py` | POSTs to its OWN ephemeral-port server | **unaffected by design** — port-keyed policy |

**My first census was wrong in both directions**, and the research gate caught
it. Recorded because the corrected rule is what the result rests on:

- **False positives:** `test_phase_36_7...` (`curl` inside a docstring, `:10`)
  and `test_phase_75_17...` (`:303` `curl` string passed as an argument to
  `fp_reason()`, never executed). Also `test_phase_75_deploy_surface.py`
  (AST-inspects another file) and `test_phase_80_2...` (in-process TestClient).
- **Missing:** `test_phase_76_9_2_max_bridge.py` — my grep only looked for
  `:8000`, so an ephemeral-port POST was invisible. **That omission would have
  produced a host-level block and broken a legitimate test.**
- My first *rule* also omitted `Request(..., data=...)`, under which urllib
  defaults to POST. Corrected before use.

### Channels NOT contained — stated plainly

- **`httpx`** — rides `httpcore`, not urllib3. Not covered.
- **Raw `socket`** — not covered.
- **SUBPROCESS / child processes — ADDED after EVALUATE pass 1; the Q/A caught
  this and it belonged in the first enumeration.** A guard installed at conftest
  import time exists only in the **pytest process**. A test that shells out runs
  unguarded. Measured: `test_phase_4000_2_cc_rail_smoke.py:202` runs
  `subprocess.run([sys.executable, str(SCRIPT), *argv])`, and
  `scripts/qa/smoke_cc_rail_e2e.py:469` defaults `--backend-url` to
  `http://localhost:8000` — and that script **mutates**
  (`http_json("PUT", f"{base}/api/settings/", …)` at `:289-290`,
  `POST /api/analysis/` at `:307`). No live egress today: `base` is an ephemeral
  stub (`ThreadingHTTPServer(("127.0.0.1", 0))` at `:176`) and **all 12 call
  sites pass an explicit `--backend-url`** — `grep -n "run_smoke(" | grep -v
  live_args | grep -v backend-url` returns nothing. Latent, not active.
  `test_phase_82_11_autoresearch_failure_paging.py:610` is a second, benign
  instance (dead port). **Folded into 86.6.**
  *This is the same mistake in kind as the one already recorded in this step: a
  worktree relocates file paths but not a socket; a conftest guard covers the
  parent process but not a child. I made the process-boundary version of it
  after writing the transport-boundary version down.*
- **The filesystem channel** — a test calling a mutating `kill_switch` method
  in-process while `_AUDIT_PATH` still points at the live file writes directly,
  and **no network guard can see it.** Not closed by this step. See §8.

---

## 8. The filesystem channel — why it is out of scope, and what proves I cannot hand-wave it

A file-level census ("does this file mention `_AUDIT_PATH` anywhere?") reports
**zero** unredirected in-process writers. **That answer is wrong, and the
counter-example is a defect already on the masterplan:**

`backend/tests/test_book_safety_69.py` imports `kill_switch`, calls a mutating
method, and mentions `_AUDIT_PATH` — so the census marks it **safe**. But:

```python
def test_peak_reset_dark_by_default(monkeypatch):
    st = ks.get_state()                              # the REAL singleton
    out = st.reset_peak(12345.0, trigger="flatten")  # no redirect in THIS function
```

The file's three redirects are in *other* functions. That is step **`86.1`**'s
landmine, and **my census called it clean.** So I cannot claim the absence of
other in-process writers, and I do not.

A **preventer** is the right design — refuse `_append_audit` when `_AUDIT_PATH`
is still the live path, which leaves tmp-redirected writers *and* live readers
working. A blanket `_AUDIT_PATH` redirect is **not** viable: it would break
criterion 7, since `test_phase_23_2_4_audit_log_clean_transitions` reads the
**live** journal and requires ≥3 parseable rows.

That is a second channel with its own blast radius. **Queued as its own
research-gated step, not silently skipped.** Today's exposure is bounded: the
one known in-process writer (`86.1`) is inert while
`kill_switch_peak_reset_enabled` is `False` (measured), and `86.1` is next but
one in the queue.

---

## 10. Criteria 1, 5 and 7 — the whole-suite measurement

Full detail and verbatim captures: `handoff/current/live_check_86.3.md`.

**Criterion 1 — GREEN.** Full `backend/tests`, backend confirmed up
(`api/health=200`):

```
BEFORE  62 lines  90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653
        12 failed, 3072 passed, 12 skipped, 5 xfailed, 1 xpassed in 332.58s
AFTER   62 lines  90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653
```

**Delta = 0 rows.** The same suite appended **8 rows** on 2026-08-08.

**Criterion 7 — GREEN, proven by hash** rather than by eye:

```
$ git show c4ff90fa~1:...test_phase_23_2_4_....py | awk '/def test_..._audit_log_clean_transitions/,0' | shasum -a 256
80fcd6a7ae6340ab0b48d9e28d3625a0459c5df3f6c3c67b720a4352cc1da721
$ awk '/def test_..._audit_log_clean_transitions/,0' backend/tests/test_phase_23_2_4_....py | shasum -a 256
80fcd6a7ae6340ab0b48d9e28d3625a0459c5df3f6c3c67b720a4352cc1da721
```

Byte-identical. It still reads the **live** journal and passes in the run above.

**Criterion 5 — NOT a clean pass, and I am not reporting it as one.**

- **New failures: ZERO.** The current 12 are a strict subset of the baseline 26.
- **But 14 tests went failing → passing, and I claim exactly ONE of them.**
  - **11** — `test_64_3_currency_path` ×3, `test_64_4_multi_market_e2e`,
    `test_dod4_tier1_coverage_investment`, `test_phase_70_3_atomic_swap`,
    `test_phase_70_4_gate_observability` ×2, `test_price_tolerance_gate` ×3 —
    are **step 36.28's live-pause coupling**. The 2026-08-08 baseline was taken
    while the book was PAUSED (ask #21); it is unpaused today. **Measured, not
    argued:** forcing the singleton back to `paused=True` (throwaway plugin,
    `_AUDIT_PATH` redirected to tmp so it wrote nothing) makes all 11 fail
    again, node-for-node — `11 failed, 95 passed in 2.98s`.
  - **1** — `test_book_safety_69::test_valid_nav_still_breaches` — fixed by
    **85.5.1**, documented in 86.5's audit basis.
  - **1** — `test_phase_23_2_15_known_pass_scripts_still_pass` — passes in
    **both** conditions today, so its 08-08 failure had another live-system
    cause. **Not root-caused, and not claimed.**
  - **1** — this step's own rewritten cycle test.

**The baseline is confounded.** It was captured under a different live
kill-switch state — which is precisely the coupling 36.28 exists to remove. A
future comparison should either re-baseline under a known pause state or land
36.28 first.

---

## 9. Honest limits
- The guard protects the **live backend origin only**. It is not a network
  jail, by design (§3.2), and `tests/`-tree scripts named `verify_phase_*.py`
  are not collected by pytest at all, so they are outside every claim here.
- The rewritten cycle no longer exercises the **real HTTP transport** to a
  separate process. It exercises the full ASGI stack in-process. The subject —
  a re-entrant lock deadlock inside the app process — is preserved, and the
  audit-delta assertion became *stronger* (exact, against an isolated journal,
  instead of a best-effort tail of a shared file). But cross-process transport
  behaviour is genuinely no longer covered by this test. Stated as a real
  reduction, not argued away.
