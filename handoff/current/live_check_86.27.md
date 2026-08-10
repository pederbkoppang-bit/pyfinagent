# live_check -- phase-86.27

**Tree under test:** `9bda4e6d` (pinned; `HEAD` is not a stable anchor here --
the auto-changelog hook commits on top of every fix within minutes).
**Pre-fix tree:** `cad386472dc161d121c069cce7a3032598d1f75b`.
**Measured:** 2026-08-10, 08:33-09:30 CEST, against the RUNNING backend
(pid 43839, started 2026-08-09 22:11:52, `/api/health` 200).

---

## A. NO MUTATING REQUEST WAS SENT TO THE LIVE BACKEND. Proof first.

This step measures a hole in a guard that protects a live trading book, so the
first thing to establish is that measuring it did not exploit it.

* **Reachability** was probed with a read-only `GET /api/health`. GETs are
  allowed by policy and change nothing.
* **Refusal** was probed with `_REAL_URLOPEN` replaced by a **non-networking
  sentinel**. A mutating PUT that the old guard would have allowed lands in the
  sentinel and is counted; not one byte reaches the backend.

```
mutating requests that actually left this process: 0 (all 12 were absorbed by the sentinel)
```

`handoff/kill_switch_audit.jsonl` before and after **every** run in this step,
including the mutation matrix and the full-suite run:

```
ea78508bee73887c82df2346da408c72...   64 lines   BEFORE
ea78508bee73887c82df2346da408c72...   64 lines   AFTER
```

The end-to-end guard test never addresses port 8000 at all: it stands up its own
server bound to `0.0.0.0` on an **ephemeral** port and asserts
`port != LIVE_BACKEND_PORT` before doing anything else.

---

## B. Criterion 1 + 2 -- the before/after refusal table

Re-runnable: `python scripts/qa/reproduce_86_27_spellings.py` (exit 0).
The "BEFORE" column runs the pre-fix predicate reconstructed **verbatim** from
`conftest.py@cad38647`.

```
pre-fix predicate reconstructed from conftest.py@cad386472dc1
live backend port: 8000

spelling                         reachable    BEFORE        AFTER         note
----------------------------------------------------------------------------------------------------------------------
127.0.0.1                        HTTP 200     REFUSED       REFUSED       control -- the canonical spelling
localhost                        HTTP 200     REFUSED       REFUSED       control
LOCALHOST                        HTTP 200     REFUSED       REFUSED       control -- case-folded by urlsplit
0.0.0.0                          HTTP 200     REFUSED       REFUSED       control -- added by phase-86.6
127.1                            HTTP 200     NOT-REFUSED   REFUSED       the step's eight: short form
0                                HTTP 200     NOT-REFUSED   REFUSED       the step's eight: bare zero
2130706433                       HTTP 200     NOT-REFUSED   REFUSED       the step's eight: 32-bit integer
localhost.                       HTTP 200     NOT-REFUSED   REFUSED       the step's eight: trailing dot
127.000.000.001                  HTTP 200     NOT-REFUSED   REFUSED       the step's eight: zero-padded
[::ffff:127.0.0.1]               HTTP 200     NOT-REFUSED   REFUSED       the step's eight: IPv4-mapped
192.168.86.85                    HTTP 200     NOT-REFUSED   REFUSED       the step's eight: THIS MACHINE'S LAN ADDRESS
ford-sin-mini.lan                HTTP 200     NOT-REFUSED   REFUSED       the step's eight: THIS MACHINE'S HOSTNAME
0x7f.0x0.0x0.0x1                 HTTP 200     NOT-REFUSED   REFUSED       invented 2026-08-10: hex-dotted
017700000001                     HTTP 200     NOT-REFUSED   REFUSED       invented 2026-08-10: 32-bit octal
[::ffff:7f00:1]                  HTTP 200     NOT-REFUSED   REFUSED       invented 2026-08-10: hex IPv4-mapped
%31%32%37%2e%30%2e%30%2e%31      HTTP 200     NOT-REFUSED   REFUSED       found by research: PERCENT-ENCODED
----------------------------------------------------------------------------------------------------------------------
mutating requests that actually left this process: 0 (all 12 were absorbed by the sentinel)

Every spelling that REACHES this machine on the live port is refused, and no refusal was lost.
```

**All 8 spellings the step named were NOT-REFUSED before. All 12 previously
un-refused spellings are refused now. All 4 controls still refused -- no
existing refusal was lost.**

### The newly-invented spelling that criterion 2 turns on

Criterion 2 forbids an allowlist extension by judging the fix on a spelling
present in **no list in the repo**. Three of the sixteen rows above were
invented on 2026-08-10 (`0x7f.0x0.0x0.0x1`, `017700000001`, `[::ffff:7f00:1]`)
and a fourth came from the research round (`%31%32%37%2e%30%2e%30%2e%31`).

But a spelling written into an artifact stops being novel the moment it is
written. So the test does not rely on these: it **derives spellings from this
machine's interface table at runtime** and **proves each absent from the tracked
tree** with `git grep --fixed-strings -c` before using it
(`test_a_spelling_absent_from_the_entire_REPO_is_still_refused`).

**Pasting these strings into a list to make a future failure go away does not
work**: a pasted spelling stops being absent, drops out of the probe set, and
the test's `>= 3` floor turns red. That property was verified accidentally and
usefully -- the first draft included loopback's integer form `2130706433`, which
IS in the repo six times, and the assertion fired:

```
AssertionError: '2130706433' already appears in the repo, so it does not test the
class -- it tests an entry someone added:
  .claude/masterplan.json:1
  handoff/current/evaluator_critique_86.6.md:3
  ...
```

---

## C. Criterion 3 -- ephemeral stubs still work

```
40 passed in 27.92s   <- pre-fix baseline (cad38647)
40 passed in 27.25s   <- post-fix (9bda4e6d)
```
(`test_phase_86_6_subprocess_channel.py` + `test_phase_4000_2_cc_rail_smoke.py`;
22 of those are the 4000.2 module, by `--collect-only -q`.)

**Does any stub ever bind 8000?** Measured, and asserted as a test rather than
claimed (`test_no_stub_in_this_repo_can_ever_bind_the_live_port`):

```
sysctl net.inet.ip.portrange.first: 49152
sysctl net.inet.ip.portrange.last : 65535
```
8000 is **outside** the ephemeral range, so `bind(("127.0.0.1", 0))` can never
be handed it. The only `bind((` in the test tree is
`test_phase_76_9_2_max_bridge.py:130` -> `("127.0.0.1", 0)`, and 4000.2's stub is
`ThreadingHTTPServer(("127.0.0.1", 0))` at `:176`. Observed ephemeral ports
during an instrumented run: 60484, 60486, 60492, 60499, 60506, 60513, 60519 --
all inside 49152-65535.

---

## D. Criterion 4 -- TOTAL on junk, and it was NOT before

24 inputs x 3 predicates, **0 raised**:

```
input                                  is_live_backend    address_is_live_backend  conftest._is_live_backend
None / 12345 / 3.14 / b'...' / bytearray / object() / '' / '   ' / '://' /
'http://[::1' / 'http://127.0.0.1:notaport/' / 'not a url' / '%%%' / tuples /
dict / list / () / 0 / -1 / True             ... all False, none raised
inputs tested: 24   RAISED: 0  []
```

**This criterion was FAILING before the step**, in conftest's copy:

```
conftest._is_live_backend(12345)    -> RAISED AttributeError: 'int' object has no attribute 'decode'
conftest._is_live_backend(object()) -> RAISED AttributeError: 'object' object has no attribute 'decode'
```
`urlsplit` raises `AttributeError` on a non-string and the old `except ValueError`
did not catch it. Deleting the copy in favour of the total shared predicate is
what fixes it -- not an added `try/except`.

One row deserves calling out rather than hiding: a 300-character hostname on
port 8000 returns **True** (refuse). It is unresolvable, so it fails safe. That
is over-refusal of a URL that could not have reached anything, which is the
correct direction.

---

## E. Criterion 5 -- the cost of the resolution this step introduces

This step DOES introduce a name-resolution call, so the criterion applies.
Measured over a **full `backend/tests/` run on a frozen tree** (`9bda4e6d`,
`git status` clean for `conftest.py`/`scripts/qa`/`backend/tests` before and
after), with read-only instrumentation chained on top of the guard:

```
16 failed, 3351 passed, 12 skipped, 5 xfailed, 1 xpassed in 360.18s (0:06:00)

===== phase-86.27 criterion-5 GUARD COST (full suite) =====
targets_this_machine calls : 59
  total                    : 61.38 ms
  WORST SINGLE CALL        : 32.293 ms (host='example.com')
_canonical_addresses calls : 30 (cache misses only)
  WORST SINGLE CALL        : 31.523 ms (host='example.com')
REAL (network) getaddrinfo : 198
socket.connect events      : 219
  to port 8000             : 52
  hook cost per event      : 3.684 us
  hook cost total          : 0.81 ms
==========================================================
```

**Read it as follows, including the number that does NOT mean what it looks
like.**

* **The guard's total cost over a six-minute suite is 61.38 ms of resolution
  plus 0.81 ms of hook.** Worst single call: **32.3 ms**, and it is
  `example.com` -- a genuinely remote name in a test fixture, resolved once and
  then memoised.
* **`REAL (network) getaddrinfo: 198` is NOT attributable to this guard.** The
  instrumentation patches `socket.getaddrinfo` itself, which is process-global,
  so 198 counts every resolution anywhere in the suite. The guard-attributable
  upper bound is the **30** `_canonical_addresses` cache misses, and only the
  subset of those where `AI_NUMERICHOST` failed reached the network at all.
  Stating this because the raw line invites exactly the wrong conclusion.
* **Only 59 calls in a 3,379-test suite**, because the PORT is tested first:
  every ephemeral stub, every frontend URL and every unparseable string returns
  before any resolution is attempted.
* **An unresolvable host does not hang the run.** `getaddrinfo` has no timeout
  parameter and `setdefaulttimeout` does not apply to name resolution -- a
  `.local` name was measured at 5.0 s, which is the real hazard. It is bounded
  here by the port gate and by memoisation, and asserted by
  `test_an_unresolvable_host_neither_hangs_nor_raises`, which requires the
  second call to be measurably cheaper than the first and the first to be under
  10 s.
* **Not measured:** a pre-fix full-suite wall clock on this machine today, so
  no A/B duration claim is made. The absolute numbers above are the measurement
  the criterion asks for.

### The 16 failures: a measured DELTA, not a count

The 86.6 records put the pre-existing baseline at 14 failures out of 3,291. This
run shows 16 out of 3,379 -- but a count difference proves nothing when both the
test population and the environment moved. So the SET was compared.

**A `git worktree` at the pre-fix commit `cad38647` was created and the 16
failing nodeids re-run there:**

```
13 failed, 3 skipped, 1 warning in 32.27s
```

**13 of the 16 fail identically at the pre-fix tree** -- including both
`test_phase_75_17_verification_paths` tests, which matters because this step
added masterplan step 86.29 and a masterplan-diff test was the one plausible way
that edit could have broken something. It did not.

The other **3 SKIPPED in the worktree rather than passing** (a worktree has no
runtime `backend.log`), so that run could not classify them. They were therefore
diagnosed directly, in the main tree:

```
backend.log must contain >=1 'governance: immutable limits loaded'; got 0
no 'Skipping BUY' line in backend.log OR its newest archive
backend.log must contain at least 1 'Prewarming ticker-meta cache' line; got 0
3 failed in 0.11s
```

All three read a **file** and assert on strings the running backend should have
logged; the backend last restarted 2026-08-09 22:11:52 and the log has since
rotated. They complete in 0.11 s and open no socket at all, so an HTTP guard
cannot be their cause.

**Failures attributable to phase-86.27: 0.** (They are their own finding --
tests whose colour depends on live log content, the sibling of the wall-clock
class phase-86.24 covers. Noted, not fixed here.)

---

## F. Criterion 6 -- what replaces the drift alarm

`test_the_two_live_origin_predicates_AGREE` compared conftest's copy against
`live_backend_origin`'s. **It could never have caught this**: both returned
`False` on all eleven bypass spellings -- they AGREED, and were BOTH WRONG. An
equality oracle over two implementations of one specification detects
*divergence* only; a fault they SHARE is invisible to it by construction.

Two things now cover it:

1. **There is one predicate.** conftest imports it. The old test is now
   trivially true and its docstring says so in terms, so a green result there
   is not mistaken for coverage; it is kept only as a tripwire against a second
   copy reappearing (`test_conftest_no_longer_carries_a_second_copy_of_the_predicate`
   asserts the deletion, including that `_LOOPBACK_HOSTS = frozenset` has not
   re-grown in conftest).
2. **The oracle is REACHABILITY, measured at test time.** A wildcard-bound
   server on an ephemeral port; every spelling that actually reaches it must be
   classified as this machine. Because the reachable set is measured rather than
   enumerated, a spelling nobody has invented is covered the moment it works.
   A positive control (`8.8.8.8`, `1.1.1.1`, `203.0.113.7`,
   `2001:4860:4860::8888` must classify remote) stops a `return True` predicate
   from passing it, and a `>= 8` floor stops a broken probe from passing it
   vacuously.

---

## G. Criterion 7 -- mutation matrix

`python scripts/qa/mutation_matrix_86_27.py` (re-runnable; every mutation is
applied to a COPY in a temp dir, and the tracked file is asserted unchanged at
the end -- it is never opened for writing).

```
id   verdict   probe                                      mutation
M1   KILLED    control=True   mutant=False   revert the address predicate to the four-string allowlist
M2   KILLED    control=True   mutant=False   drop the wildcard (is_unspecified) leg -- 0.0.0.0 reaches the book
M3   KILLED    control=True   mutant=False   drop the IPv4-mapped IPv6 unwrapping (probed on a mapped LAN address)
M4   KILLED    control=True   mutant=False   fail OPEN instead of fail SAFE on an unresolvable host
M5   KILLED    control=False  mutant=True    drop the port check -- every 4000.2 ephemeral stub becomes 'live'
M6   KILLED    control=False  mutant=True    remove canonicalisation entirely
M7   KILLED    control=False  mutant=True    blanket TRUE -- refuses everything, incl. genuinely remote hosts

tracked source UNCHANGED (sha-equal to start): True
All 7 mutants killed.
```

**M1 is the criterion's own mutation: reverting the fix.** It is killed by
`address_is_live_backend(('127.1', 8000))` going True -> False.

**TWO OF THESE SURVIVED ON THE FIRST RUN, and both were my PROBES being wrong
rather than the guards being weak.** Recorded because a matrix that is green on
the first attempt is the one worth distrusting.

- **M3** survived when probed with `::ffff:127.0.0.1`, because CPython already
  reports `IPv6Address('::ffff:127.0.0.1').is_loopback == True`. For loopback
  the branch really is redundant; it carries weight only for a mapped
  **non-loopback** address. The probe is now a mapped LAN address derived at
  runtime, and the gap is separately pinned by
  `test_a_mapped_NON_loopback_address_is_recognised`. Same shape as 86.6's
  M4/M5 pair -- a leg is proven load-bearing only by a case the other legs
  cannot reach.
- **M6** survived when probed with `2130706433`, where the control answer and
  the fail-safe answer coincide at True, so the cell could not discriminate. It
  is probed with a remote numeric address now.

Neither was fixed by weakening a guard or relaxing an assertion.

**The end-to-end guard carries its own mutation inside the test**, rather than
in this matrix: `test_e2e_every_reachable_spelling_is_REFUSED_on_a_mutating_verb`
runs the child process TWICE, once with `install_socket_guard()` and once
without, and asserts that the disarmed control **sent every mutating PUT** before
asserting the armed run refused them. Without that control, a refusal could be
an unreachable stub rather than a working guard.
