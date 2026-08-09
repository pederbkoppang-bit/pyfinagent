# Research Brief — step 86.1

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Date:** 2026-08-09. **Read-only session** — no production code touched, no
pytest run, no POST to localhost:8000, `reset_peak` never invoked.

## Objective (from the caller)

`backend/tests/test_book_safety_69.py::test_peak_reset_dark_by_default` calls
`st.reset_peak(12345.0, trigger='flatten')` on `ks.get_state()` — the REAL
kill-switch module singleton — with NO `_AUDIT_PATH` redirect. It passes today
ONLY because `settings.kill_switch_peak_reset_enabled` is False, so `reset_peak`
returns early BEFORE taking the lock or appending a row. The day the KS-PEAK-RESET
operator token is applied (step 79.6), this test writes a real `peak_reset` row
into the live `handoff/kill_switch_audit.jsonl` and drops the LIVE trailing
high-water mark from ~24666 to 12345. `peak_reset` is in `_BASELINE_EVENTS`, so it
is replayed on every boot and is authoritative: a lower peak means the
trailing-drawdown leg fires LATER — a real LOOSENING of a safety limit, introduced
by running the test suite.

Questions: (1) two-arm testing of flag-gated code without the ON arm touching
production state; (2) the class "safe only by accident of configuration" and how
to detect it systematically; (3) pytest idioms for detaching a module-level
singleton, and why patching an accessor is not enough; (4) why an authoritative
DOWNWARD move of a safety high-water mark needs stronger provenance than a ratchet.

## Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| year-less canonical | `testing feature flag both arms enabled disabled test pollution production state` |
| year-less canonical | `PolDet detecting state-polluting tests test dependency Gyori ISSTA` |
| year-less canonical (direct-fetch prior art) | `unittest.mock` "Where to patch"; pytest `monkeypatch` how-to; Fowler *Feature Toggles*; Fowler *High-Water Mark*; Kafka KIP-101 |
| last-2-year (recency) | `flaky test state pollution detection 2025 feature flag dark launch test isolation production data` |
| last-2-year (recency) | `arXiv 2025 order-dependent flaky tests shared state pollution empirical study` |
| current-year (recency) | `arXiv feature flags technical debt empirical study 2024 2025 stale flag latent code path` |

---

## Read in full (8; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://martinfowler.com/articles/feature-toggles.html | 2026-08-09 | authoritative blog (canonical, year-less) | WebFetch, full | "It's most important to test the toggle configuration which you expect to become live in production, which means the current production toggle configuration plus any toggles which you intend to release flipped On." — "It's also wise to test the fall-back configuration where those toggles you intend to release are also flipped Off." — "With multiple toggles in play we have a combinatoric explosion of possible toggle states." — Ops toggles include long-lived "Kill Switches"; toggles are "inventory which comes with a carrying cost". Recommends "exposing an endpoint which allows for dynamic in-memory re-configuration of a feature flag" for tests. |
| 2 | https://mir.cs.illinois.edu/marinov/publications/GyoriETAL15PollutionDetection.pdf | 2026-08-09 | peer-reviewed (ISSTA 2015, PolDet) | curl → **pdfplumber** (63,721 chars; PDF chain per `.claude/rules/research-gate.md` Step 3) | "PolDet finds tests that modify some location on the heap shared across tests **or on the file system**". Latent-pollution money quote: "even if the current test suite does not have any test t′ that can be affected by the polluting test t, it is still valuable to know that t is a polluting test, so it could be fixed **even before t′ is added** and the test order is changed." Also: "Ideally a polluting test should be caught **right when the developer is about to add it** to the test suite because that is when the developer is in the best position to fix" it. Eval: 26 projects / 6105 tests → 324 polluting, 194 relevant. Prior study: "78% of the polluting tests pollute the shared state right when they are added". |
| 3 | https://docs.python.org/3/library/unittest.mock.html | 2026-08-09 | official docs | WebFetch, full ("Where to patch") | "There can be many names pointing to any individual object, so for patching to work you must ensure that you patch the name used by the system under test." — "The basic principle is that you patch where an object is *looked up*, which is not necessarily the same place as where it is defined." — with `import a` + `a.SomeClass`, "we have to patch `a.SomeClass` instead". |
| 4 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-09 | official docs | WebFetch, full | "All modifications will be undone after the requesting test function or fixture has finished." — "Prefer patching the reference that your code uses instead of patching the original object in the standard library." — `setattr(..., raising=True)` "will raise if the target of the set/deletion operation does not exist" — `monkeypatch.context()` to "apply patches only in a specific scope" — global autouse fixtures can block network/filesystem access suite-wide. |
| 5 | https://martinfowler.com/articles/patterns-of-distributed-systems/high-watermark.html | 2026-08-09 | authoritative blog (canonical, year-less) | WebFetch, full (page is an excerpt; backward-movement material is in the book chapter, NOT on the page — stated rather than inferred) | The high-water mark is "an index into the log file that records the last log entry known to have successfully replicated to a Majority Quorum of followers", and "All servers in the cluster should only transmit to clients the data that reflects updates below the high-water mark." I.e. the mark IS the safety boundary readers rely on. |
| 6 | https://cwiki.apache.org/confluence/display/KAFKA/KIP-101+-+Alter+Replication+Protocol+to+use+Leader+Epoch+rather+than+High+Watermark+for+Truncation | 2026-08-09 | official design doc (Apache Kafka) | WebFetch, full | The Q4 anchor. Truncating on the high watermark alone is unsafe because it lags: "should that follower become leader before it has caught up, some messages may be lost due to the truncation" — "Message m2 has been lost permanently." Fix = Leader Epoch, a token that NAMES the leadership period: "the follower gets the appropriate LeaderEpoch from the leader's vector of past LeaderEpochs and uses this to truncate only messages that do not exist in the leader's log." A downward move must be justified against a *named* prior state, not merely asserted. |
| 7 | https://arxiv.org/html/2509.00466v1 | 2026-08-09 | preprint (JS-TOD, 2025) | WebFetch, full (arXiv **html** chain, not `/pdf/`) | "Our evaluation using JS-TOD reveals two main causes of test order dependency flakiness: **shared files** and shared mocking state between tests." Detection = AST extraction + reorder at 3 levels + 10 reorders x 10 reruns. NOTE (honest): the HTML does NOT carry the 13-vs-42 per-cause counts the search snippet claimed, and does NOT discuss latent/config-triggered pollution — I am not citing those. |
| 8 | https://arxiv.org/html/2510.26171v1 | 2026-08-09 | preprint (Oct 2025) | WebFetch, full (arXiv **html** chain) | Polluter/victim: "A polluter modifies or pollutes some shared state between the polluter and the victim"; victim "fails for not having the expected state". Why brute-force order search fails: "a test suite is needed to run on all permutations of the test cases, which is often not practical"; when cleaners outnumber polluters "the chance of finding a failing order can only be 1.2%". Static signal: "in nearly every instance, OD test pairs ... shared one or more static attributes from the test class." 27 modules / 189 OD tests; 96.61% coverage, 65.92% avg test reduction, 72.19% fewer re-runs. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://dl.acm.org/doi/10.1145/2771783.2771793 | peer-reviewed (PolDet, ACM) | paywalled duplicate of #2 |
| https://experts.illinois.edu/en/publications/reliable-testing-detecting-state-polluting-tests-to-prevent-test- | index page | metadata only |
| https://taoxie.cs.illinois.edu/publications/icst19-idflakies.pdf | peer-reviewed (iDFlakies) | complementary (classification, not pollution detection) |
| https://people.cs.gmu.edu/~winglam/publications/2023/LiETAL23Tuscan.pdf | peer-reviewed (Tuscan orders, 2023) | superseded for our purpose by #8, which builds on it |
| https://arxiv.org/abs/2504.16777 | preprint (systemic flakiness, 2025) | 75%-of-flaky-in-clusters finding noted; not needed for this defect |
| https://arxiv.org/html/2511.14002 | preprint (FlakyGuard, 2025) | LLM auto-repair; out of scope (we want prevention) |
| https://arxiv.org/pdf/2101.09077 | preprint (flaky tests in Python) | older; Python-specific census, no new mechanism |
| https://arxiv.org/pdf/2501.06972 | industry paper (Google, AI code migrations) | contains the stale-experiment-flag framing ("its value effectively becomes a constant ... the flag and the associated code remains") — snippet only |
| https://arxiv.org/abs/2601.11693 | preprint (technical lag as latent debt) | adjacent framing only |
| https://launchdarkly.com/ (flag-testing guides via search) | vendor docs | vendor tier; Fowler #1 is the stronger canonical source |
| https://medium.com/draftkings-engineering/mastering-feature-flags-testing-feature-flags-2bfdff31905f | industry blog | secondary to #1 |
| https://reflectoring.io/testing-feature-flags/ | community blog | lowest tier |
| https://testrigor.com/blog/feature-flags-how-to-test/ | vendor blog | lowest tier |
| https://docs.gitlab.com/administration/feature_flags/ | official docs (GitLab) | ops guide, not a testing pattern |
| https://www.ministryoftesting.com/articles/testing-with-feature-flags-what-we-expected-and-what-actually-happened | community | anecdote: a background process stayed active with the flag OFF and only surfaced when flipped — same class as ours, but community tier |
| https://docs.cypress.io/cloud/features/flaky-test-management | vendor docs | JS/UI focus |
| https://www.harness.io/blog/flaky-tests-the-quiet-killer-of-productivity-in-your-ci-pipeline | industry blog | generic |
| https://scrolltest.com/flaky-tests-detection-quarantine-prevention-guide-2026/ | community | generic 2026 guide |
| https://reproto.com/how-to-fix-flaky-tests-in-2025-a-complete-guide-to-detection-prevention-and-management/ | community | generic |
| https://pie.inc/blog/flaky-tests-explained/ | community | generic |
| https://getautonoma.com/blog/flaky-tests | community | generic |
| https://www.netdata.cloud/academy/flaky-tests/ | community | generic |
| https://www.getunleash.io/blog/using-feature-flags-to-manage-technical-debt | vendor blog | flag-debt framing, secondary to #1 |
| https://flagshark.com/blog/feature-flag-technical-debt-guide/ | vendor blog | source of the "75% of toggles persist up to 49 weeks" claim — UNVERIFIED, not cited as fact |
| https://arxiv.org/pdf/1905.00357 · https://arxiv.org/pdf/1808.08174 · https://arxiv.org/pdf/2103.02669 · https://arxiv.org/pdf/2207.01047 · https://www.arxiv.org/abs/2501.12680 · https://arxiv.org/abs/2603.28592 · https://arxiv.org/abs/2311.12019 | preprints | adjacent flakiness/debt literature, no new mechanism for this defect |
| https://oneuptime.com/blog/post/2026-01-30-flag-unit-testing/view · https://www.thegreenreport.blog/... · https://yrkan.com/blog/feature-flags-testing-strategy/ · https://data.epo.org/gpi/EP3612941B1 · researchgate ML-flaky-tests | community / patent | lowest tier |

**Unique URLs collected: 44** (8 read in full + 36 snippet-only rows above,
counting the merged rows individually).

## Recency scan (2024-2026) — PERFORMED

Three scoped passes (see query table). **Result: 3 new findings that complement,
none that supersede, the canonical sources.**

1. **arXiv:2510.26171 (Oct 2025)** — the polluter/victim model is unchanged since
   PolDet 2015, but the paper quantifies why *finding a victim by re-running orders*
   is a bad detection strategy: "the chance of finding a failing order can only be
   1.2%" when cleaners outnumber polluters. Directly applicable: our polluter has
   **no victim at all today** (the flag is OFF), so no order-permutation strategy
   could ever surface it. Detection must be state-based (PolDet-style), not
   outcome-based.
2. **arXiv:2509.00466 (Sept 2025, ICST'25)** — "shared **files**" is still one of
   the two dominant OD causes in a modern (JS/Jest) ecosystem, a decade after
   PolDet. The filesystem channel is not a legacy Java concern.
3. **arXiv:2501.06972 (Google, 2025, snippet-only)** — the stale-flag framing:
   an experiment's "value effectively becomes a constant, but the flag and the
   associated code remains". Our case is the mirror image and is *worse*: the flag
   is a constant `False` **today** and a test has silently taken a dependency on
   that constant, so the code is not dead — it is armed.

No 2024-2026 work supersedes Fowler's two-configuration rule, PolDet's
before/after state-diff, `unittest.mock`'s where-to-patch rule, or KIP-101's
epoch-before-truncate rule.

---

## Internal code inventory (all line numbers RE-DERIVED 2026-08-09; step text's "~694" was stale)

| File:line | Symbol | Finding |
|---|---|---|
| `backend/services/kill_switch.py:48` | `_AUDIT_PATH` **module global** | `Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"` — the LIVE journal |
| `backend/services/kill_switch.py:49` | mkdir at import | `_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)` |
| `backend/services/kill_switch.py:74-75` | `_AUDIT_ARCHIVE_SUBDIR` / `_AUDIT_ARCHIVE_GLOB` | `"audit"` / `"kill_switch_audit*.jsonl"` |
| `backend/services/kill_switch.py:89-91` | `_audit_archive_dir()` | **DERIVED** from `_AUDIT_PATH` (`_AUDIT_PATH.parent / "audit"`) — redirecting `_AUDIT_PATH` alone moves the archive dir too |
| `backend/services/kill_switch.py:94-111` | `_audit_source_paths()` | reads `_AUDIT_PATH` **at call time**, archives first, live file LAST |
| `backend/services/kill_switch.py:~200-231` | `KillSwitchState.__init__` | ends with `self._load_from_audit()` → **construction replays the journal**, so the redirect must precede construction |
| `backend/services/kill_switch.py:397-430` | `_apply_authoritative_peak` | the replay's guarded assignment for `peak_reset`; rejects only non-positive / non-finite. Docstring: "Such an assignment can LOWER the peak" |
| `backend/services/kill_switch.py:432-443` | `_append_audit` (`@staticmethod`) | opens the **module global** `_AUDIT_PATH` at `:440`; **swallows** write errors via `logger.warning` at `:442-443` |
| `backend/services/kill_switch.py:450-473` | `_snapshot_locked` | returns **NINE** keys: `paused, pause_reason, sod_nav, sod_date, peak_nav, paused_at, auto_resume_alerted_at, baseline_provenance, sod_provisional` |
| `backend/services/kill_switch.py:581-634` | `update_peak` | RATCHET only (`elif nav > self._peak_nav`); writes plain `peak_update` rows |
| `backend/services/kill_switch.py:596-599` | the authority rule | "the replay grants authority only to a row that NAMES what it superseded (`prior_peak` coercing to a positive finite NAV)" |
| `backend/services/kill_switch.py:670-701` | **`reset_peak`** | `def` at `:670`; DARK early return `return None` at `:693-694`; `with self._lock:` at `:695`; assignment `self._peak_nav = float(new_peak)` at `:697`; `_append_audit("peak_reset", old_peak=..., new_peak=..., trigger=..., operator=...)` at `:698-700` |
| `backend/services/kill_switch.py:704` | `_state = KillSwitchState()` | module singleton, constructed at import |
| `backend/services/kill_switch.py:709` | `_BASELINE_EVENTS` | `frozenset({"sod_snapshot", "peak_update", "peak_reset"})` |
| `backend/services/kill_switch.py:742-743` | `get_state()` | `return _state` |
| `backend/services/kill_switch.py:793, :995, :1033, :1047, :1053-1054` | direct `_state` access | module-level functions bypass `get_state()` entirely (`_state.snapshot()`, `_state.resume(...)`, `_state._append_audit(...)`, `_state._lock`) |
| `backend/config/settings.py:39` | `kill_switch_peak_reset_enabled` | `bool = Field(False, ...)` — "Default OFF = DARK (reset_peak is a no-op…)" |
| `backend/tests/test_book_safety_69.py:186-192` | `test_peak_reset_dark_by_default` | **THE LANDMINE** |
| `backend/tests/test_book_safety_69.py:99` | stale docstring | "and this one already omitted **three**" → must be **seven** (9 real keys − the 2 the mock supplied at `:83`) |
| `backend/tests/test_book_safety_69.py:107-111` | the good idiom, same file | redirect `_AUDIT_PATH` **and** `_audit_archive_dir`, then assert `_audit_source_paths()` holds nothing outside `tmp_path` |
| `backend/tests/test_book_safety_69.py:195-207` | `test_peak_reset_active_when_token_enabled` | the flag-**ON** arm IS isolated (`_AUDIT_PATH` → tmp at `:197`; fresh `KillSwitchState()` at `:198`; flag forced ON at `:202`) |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py:118-140` | `ks_tmp_audit` | **PREVENTER (file channel only)** |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py:142-190` | `isolated_state` | **PREVENTER (file + in-memory)** |
| `backend/tests/test_phase_36_12_kill_switch_trading_path_block.py:41-57` | `_live_audit_file_is_write_protected` (autouse) | **DETECTOR ONLY** |
| `conftest.py` (repo root, phase-86.3, shipped today) | HTTP egress guard | **PREVENTER (HTTP channel only)**; explicitly excludes the filesystem channel |
| `backend/tests/conftest.py` (61 lines) | phase-82.58 slack.com guard | **contains NO audit-file guard** — measured by grep |

### Measured live state (read-only, 2026-08-09)

- `handoff/kill_switch_audit.jsonl` (LIVE): **62 rows**, census
  `{pause: 44, resume: 10, sod_snapshot: 8}` — **zero peak rows**.
- `handoff/audit/kill_switch_audit{,-v2,-v3,-v4}.jsonl` (replayed archives):
  **889 rows**, **20 `peak_update`, 0 `peak_reset` ever**.
- **max peak nav across the whole replay = 24666.57**; last peak row
  `2026-06-22T18:10:36Z nav=24124.77`.

This confirms the caller's premise numerically **and sharpens it**: because the
live file holds no peak rows, the 24666.57 mark is reconstructed *purely from the
archives* on every boot. A `peak_reset` row appended today carries the newest `ts`,
so after `_load_from_audit`'s `ts` merge-sort it is the LAST peak-affecting row and
`_apply_authoritative_peak` assigns `12345.0`. `update_peak`'s ratchet cannot heal
it (`kill_switch.py:410-414`: "a subsequent ratchet then re-anchors from whatever
row comes next, which is lower than the true mark and silently under-measures every
future drawdown"). The 24666.57 mark would be **permanently destroyed**. At a 10%
trailing limit the trip point moves from NAV 22199.9 to NAV 11110.5 — on a book
whose last recorded peak-adjacent NAV was ~24124, the trailing leg becomes
effectively unreachable.

### What each isolation fixture PREVENTS vs merely DETECTS (caller asked precisely)

- **`ks_tmp_audit`** (36.7:118-140) — **PREVENTS** the file channel. Redirects
  `_AUDIT_PATH` to tmp *before* the body runs and asserts
  `ks._audit_archive_dir() == archive`, proving the derived path moved with it.
  It deliberately does **not** swap `_state` ("The module-global `_state` is left
  untouched"), so a test that mutates the singleton's in-memory baseline is still
  unprotected. File-only.
- **`isolated_state`** (36.7:142-190) — **PREVENTS both channels.** Redirects
  `_AUDIT_PATH` **and** installs a detached `object.__new__(ks.KillSwitchState)`
  over the module global `_state`, plus resets `_disarmed_logged`. Its docstring
  names the exact reason and it matches what I measured: "`evaluate_breach` /
  `check_auto_resume` read `_state` directly, so mutating the real one would
  corrupt the live book's in-memory baseline for the rest of the session"
  (confirmed at `kill_switch.py:793/:995/:1033/:1047/:1053`). `object.__new__`
  also *skips* `__init__`, so no boot replay of real archives happens.
- **`_live_audit_file_is_write_protected`** (36.12:41-57, autouse) — **DETECTS
  ONLY.** It reads the live file's bytes before, yields, and asserts equality
  after. By the time it fires the bytes are already on disk; it converts a silent
  corruption into a loud failure but stops nothing. It is also **module-scoped**
  (defined inside that one test module), so `test_book_safety_69.py` inherits
  nothing from it, and it cannot observe an in-memory `_state` mutation at all.
  Its own docstring is honest: "Belt to the … braces".
- **repo-root `conftest.py`** (phase-86.3, today) — **PREVENTS the HTTP channel
  only.** It replaces `urllib.request.urlopen` (and `urllib3`'s pool `urlopen`) at
  import time and raises *before a connection is opened* for MUTATING verbs aimed
  at loopback:8000. Its own "KNOWN, BOUNDED GAPS" section states: "The FILESYSTEM
  channel is NOT covered: a test that calls a mutating `kill_switch` method
  in-process while `_AUDIT_PATH` still points at the live file writes directly, and
  no network guard can see that… The one KNOWN in-process writer is step 86.1's
  `reset_peak` landmine."

---

## Key findings

**F1 — the isolation asymmetry is inverted, and that is the whole defect.** The
flag-**ON** arm (`:195-207`) is fully isolated; the flag-**OFF** arm (`:186-192`)
is not. The author isolated the arm they *knew* would write and left unisolated
the arm they *reasoned* could not — and that reasoning reads a value the operator
owns. Fowler's rule is that both configurations are first-class test subjects:
"It's also wise to test the fall-back configuration where those toggles you intend
to release are also flipped Off" (feature-toggles.html). The correct invariant is
that **isolation is established by the fixture, never derived from the value under
test** — a test asserting "this is a no-op" needs *more* isolation than one
asserting "this writes", because the no-op assertion is exactly the claim that can
become false.

**F2 — a second, independent landmine in the same 7 lines: the test's own greenness
is coupled to operator config.** With `KILL_SWITCH_PEAK_RESET_ENABLED=true` in
`backend/.env`, `assert out is None` at `:191` FAILS. So flipping 79.6 both
corrupts live state *and* turns this test red. Pinning the flag explicitly with the
same handle the ON arm already uses — `monkeypatch.setattr(get_settings(),
"kill_switch_peak_reset_enabled", False)` (cf. `:202`, `:219`) — fixes that half and
is what Fowler means by "dynamic in-memory re-configuration of a feature flag".
This is the phase-36.28 class (test greenness coupled to live state) and is in
scope here because it lives in the same seven lines.

**F3 — patching the accessor is vacuous twice over** (`:187-188`). `st =
ks.get_state()` at `:187` binds **the real singleton** *before* the patch, so
`monkeypatch.setattr(ks, "get_state", lambda: st)` at `:188` rebinds the accessor to
return *the same object* — a no-op by identity. And even with a fresh instance it
would not help, because `evaluate_breach` / `check_auto_resume` / the module helpers
read the global `_state` **directly** (`:793`, `:995`, `:1033`, `:1047`,
`:1053-1054`), never through `get_state()`. This is textbook: "you patch where an
object is *looked up*, which is not necessarily the same place as where it is
defined" (`unittest.mock` docs) and "Prefer patching the reference that your code
uses" (pytest docs). The repo's own correct idiom is `monkeypatch.setattr(ks,
"_state", fresh)` (`isolated_state`, 36.7:180).

**F4 — three handles must move, and one need not.** (a) `_AUDIT_PATH` — because
`_append_audit` is a `@staticmethod` reading the *module global* at call time
(`:440`), so even a detached instance writes to the live file. (b) `_state` —
because `reset_peak` also assigns `self._peak_nav` at `:697`, corrupting the live
in-memory baseline for the remainder of the pytest session even if the file write
is redirected. **A redirect-only fix is a half-fix and is the single most likely
wrong answer.** (c) `_disarmed_logged` — only if the test asserts on the disarm log
(this one does not). NOT needed: `_audit_archive_dir`, which is derived (`:89-91`)
and moves with `_AUDIT_PATH`; the file's existing belt-and-braces patch at `:108`
/`:159` is harmless and matches the step's own success criterion, so keep it.

**F5 — ordering is load-bearing.** `KillSwitchState.__init__` ends with
`self._load_from_audit()`, so `ks.KillSwitchState()` replays the journal at
construction. The redirect must precede construction — the comment at
`test_book_safety_69.py:105-106` ("Redirect BEFORE anything can append") and
`ks_tmp_audit`'s docstring both say so. `object.__new__` (36.7:167) is the
alternative that skips the replay entirely.

**F6 — no order-permutation strategy could ever have found this.** There is no
victim: nothing else reads the live peak inside the suite. PolDet's argument is the
one that applies — "even if the current test suite does not have any test t′ that
can be affected by the polluting test t, it is still valuable to know that t is a
polluting test, so it could be fixed even before t′ is added" — and arXiv:2510.26171
quantifies the futility of the outcome-based alternative ("the chance of finding a
failing order can only be 1.2%"). Detection must be **state-based**: diff the shared
resource before/after, exactly what 36.12's byte-compare does, and what nothing
covering `test_book_safety_69.py` currently does.

**F7 — the census cannot be trusted, and that is measured, not argued.** Step 86.6's
audit_basis records that a file-level scan ("does this file mention `_AUDIT_PATH`?")
reported ZERO unredirected in-process writers and was WRONG, because
`test_book_safety_69.py` mentions it three times *in other functions*. A per-call-site
census is also unmaintainable, since a new test regresses it by existing. So the
systematic answer is structural, not enumerative: a session-scoped detector (repo-wide
sha256 before/after) plus the 86.6 preventer.

**F8 — why a DOWNWARD move needs stronger provenance than a ratchet (Q4).** The
error costs are asymmetric. A ratchet (`update_peak`) is *safe under noise in both
directions*: a spurious LOW value is discarded by the `max()` comparison, and a
spurious HIGH value is conservative — it tightens the trip point and fires the
switch EARLIER. An authoritative downward assignment is the mirror: it is
**unrecoverable** (nothing in the module can restore a destroyed peak;
`_apply_authoritative_peak`'s docstring: "the prior peak … is retained; assigning
this would have lowered or nulled the trailing high-water mark") and it **loosens**
the trip point. KIP-101 is the canonical statement of the same asymmetry: truncating
on the high watermark alone loses data permanently ("Message m2 has been lost
permanently"), and the fix is not a better threshold but a **provenance token** —
the Leader Epoch — so a replica "truncate[s] only messages that do not exist in the
leader's log". pyfinagent already encodes this rule for the *format* — "the replay
grants authority only to a row that NAMES what it superseded" (`:596-599`) — and
`reset_peak` does stamp `old_peak` (`:698`), so its rows ARE authoritative. **The
provenance gap is in the WRITER, not the format**: the row carries `trigger` and
`operator` but nothing that distinguishes a production caller from a pytest process,
so a test's `trigger='flatten'` row is indistinguishable from a real flatten. That
observation is the design seed for 86.6, not for 86.1.

## Consensus vs debate (external)

**Consensus:** test both toggle configurations (Fowler); patch the name the SUT
looks up (Python + pytest docs, identical rule); shared-state pollution is
detected by before/after state diffing, and files are a first-class channel
(PolDet 2015, still true in 2025 per arXiv:2509.00466); a downward move of a
safety boundary requires provenance beyond the boundary value itself (KIP-101).

**Debate / tension:** *detector vs preventer*. PolDet is a detector run
"occasionally for the entire suite, or … only for the newly added tests", and its
precision is 194/324 ≈ 60% — a repo-wide detector will produce noise. 86.6's
audit_basis argues for a preventer instead ("a PREVENTER, not a detector — 36.12's
… and 86.3's own module-scoped byte-compare both fire AFTER the bytes are on
disk"). The resolution for pyfinagent is that these are not alternatives: the live
kill-switch journal is a *single, named* resource, so a byte/sha compare on it has
~100% precision and zero noise, and can coexist with a preventer. Second tension:
Fowler warns about "a combinatoric explosion of possible toggle states", which
argues against blanket both-arm testing; that does not apply here, because the arm
count is 2 and both arms already exist in the file.

## Pitfalls (from the literature and from this repo)

1. **Half-fix: redirecting the file but not the singleton.** With the flag ON,
   `reset_peak` still assigns `_peak_nav = 12345.0` on the real `_state` at `:697`.
   (F4)
2. **Weakening an assertion to make the test "safe".** The step's criterion 5
   forbids it; `assert out is None` and `peak_nav == before` must both survive.
3. **Proving the bug by actually writing to the live journal.** The required
   mutation (criterion 2) must be staged with `_AUDIT_PATH` already redirected —
   show the pre-fix *shape* writing a `peak_reset` row into the tmp file that stands
   in for the live one, then show the fixed shape does not. Never force the flag ON
   against the real singleton.
4. **Reaching for `chmod` to make the live file unwritable.** `_append_audit`
   swallows write errors (`:442-443`), so that produces a `logger.warning` and a
   silent no-op, not a failure. This is precisely why 36.12 chose a byte-compare.
5. **Constructing `KillSwitchState()` before redirecting** — `__init__` replays
   the real archives (F5).
6. **Trusting a file-level `_AUDIT_PATH` grep as a census** — measured wrong (F7).
7. **`git add -A`.** Other Claude sessions are active in this repo; commit with an
   explicit pathspec (standing memory: *Uncommitted is NOT protected* /
   *Two Claude sessions — only ONE flips steps*).

## Application to pyfinagent (external → file:line)

- Fowler "test the fall-back configuration" → the DARK arm at
  `test_book_safety_69.py:186-192` must be a *pinned* configuration
  (`monkeypatch.setattr(get_settings(), "kill_switch_peak_reset_enabled", False)`,
  mirroring `:202`), not an inherited one. (F1, F2)
- `unittest.mock` "patch where it is looked up" + pytest "patch the reference your
  code uses" → replace `monkeypatch.setattr(ks, "get_state", …)` at `:188` with
  `monkeypatch.setattr(ks, "_state", fresh)`, the idiom already proven at
  `test_phase_36_7_…:180`. (F3)
- PolDet's before/after shared-state diff → the file's own `:109-111` precondition
  (`_audit_source_paths()` contains no live file) plus a module-level autouse
  byte/sha compare of `handoff/kill_switch_audit.jsonl` satisfies the step's
  criterion 3 ("asserted by the test run itself rather than checked by hand"),
  reusing 36.12:41-57 verbatim rather than inventing a new shape. (F6)
- KIP-101 leader-epoch → **do not** change `reset_peak`'s row format in 86.1; the
  naming rule is already satisfied (`old_peak` at `:698`). The writer-provenance
  idea belongs to 86.6. (F8)

## Where 86.1 ends and 86.6 begins (caller asked explicitly)

**86.1 (this step) — one known site, test-side only.**
Scope = `backend/tests/test_book_safety_69.py`. Isolate
`test_peak_reset_dark_by_default` (redirect `_AUDIT_PATH` [+ `_audit_archive_dir`
per the criterion] and detach `_state`), pin the flag explicitly, prove it with a
forced-ON mutation staged against tmp, assert the live journal is byte-identical
across the whole file's run, and correct "omitted three" → **seven** at `:99`.
**No production code change.** Its verification command runs only this one file, so
the blast radius is one test module.

**86.6 (queued) — the structural preventer for the filesystem channel, repo-wide.**
Scope = production code + repo-root conftest. Candidate direction (to be validated,
not assumed): refuse inside `_append_audit` when `_AUDIT_PATH` still resolves to the
live production file during a test session — tmp-redirected writers unaffected, live
READS untouched. Hard constraints inherited: must NOT blanket-redirect (because
`test_phase_23_2_4_audit_log_clean_transitions` READS the live journal and needs >=3
parseable rows, and its trigger allowlist is a byte-unchanged regression lock); must
MEASURE what a refusal does to each production caller given `_append_audit` swallows
exceptions (`:442-443`); must not gitignore the journal. **Part B** adds the
subprocess channel (a conftest guard exists only in the pytest process;
`test_phase_4000_2_cc_rail_smoke.py:202` shells out to
`scripts/qa/smoke_cc_rail_e2e.py`, which defaults `--backend-url` to
`http://localhost:8000` at `:469` and MUTATES).

**The boundary rule:** 86.1 may add a *file-local* autouse byte-compare (a
DETECTOR for one module, reusing 36.12's shape). It must NOT touch `_append_audit`
or install a repo-wide guard — doing so pre-empts 86.6's research gate and puts
production code inside a step whose verification command only runs one test file.

## Step 86.1 — verbatim from `.claude/masterplan.json`

`verification.command`:

```
bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'
```

`verification.success_criteria` (verbatim, in order):

1. "the test no longer touches the real singleton: it redirects _AUDIT_PATH (and _audit_archive_dir) to tmp BEFORE constructing state, and a precondition asserts _audit_source_paths() contains no live file"
2. "a MUTATION proves it: force kill_switch_peak_reset_enabled True in the test's own settings and show that the pre-fix form WOULD have written a peak_reset row to the live journal while the fixed form writes only to tmp -- demonstrated, not argued"
3. "the live handoff/kill_switch_audit.jsonl is byte-identical before and after the whole file runs, asserted by the test run itself rather than checked by hand"
4. "the stale 'omitted three' docstring figure is corrected to seven"
5. "no assertion is weakened and no other test changes status; fresh Q/A PASS"

`verification.live_check` (verbatim):

> "live_check_86.1.md with the before/after line count and sha256 of handoff/kill_switch_audit.jsonl across a full run of test_book_safety_69.py, plus the verbatim mutation transcript showing the pre-fix form writing a peak_reset row under a forced-ON flag"

**Note on criterion 1 vs the measured code:** the criterion says "redirects
_AUDIT_PATH (and _audit_archive_dir)". `_audit_archive_dir` is *derived* from
`_AUDIT_PATH` (`kill_switch.py:89-91`), so patching it is redundant — but it is the
file's own existing idiom at `:108`/`:159` and the criterion is immutable, so do
both. Criterion 1 as written does **not** mention detaching `_state`; F4 shows a
redirect-only fix still corrupts the in-memory baseline via `:697`. Satisfy the
criterion **and** detach the singleton.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (8: 6 via WebFetch, 1 via
      the WebFetch→pdfplumber PDF chain, both arXiv sources via `/html/` not `/pdf/`)
- [x] 10+ unique URLs total (44)
- [x] Recency scan (2024-2026) performed + reported (3 complementary findings)
- [x] Full pages/papers read, not abstracts (PolDet: 63,721 chars extracted)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the caller's scope
- [x] Contradictions / consensus noted (detector-vs-preventer tension)
- [x] Claims cited per-claim
- [!] GAP, stated: `arxiv.org/html/2509.00466v1` does not contain the 13-vs-42
      per-cause counts the search snippet asserted; I did not cite them.
      `martinfowler.com/…/high-watermark.html` is an excerpt and does not cover
      backward movement — KIP-101 carries that claim instead.

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 36,
  "urls_collected": 44,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.1.md",
  "gate_passed": true
}
```
