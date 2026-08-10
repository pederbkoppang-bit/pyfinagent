# phase-86.27 -- GENERATE

**Step:** 86.27 (P1) -- the live-backend HTTP guard is a host-string allowlist.
**Contract:** `handoff/current/contract_86.27.md`
**Research:** `handoff/current/research_brief_86.27.md` (gate PASSED, recomputed
by `research-gate.js`; 19 sources read in full, 43 URLs, audit-class
`coverage.dry` after 10 rounds)
**Code SHA:** `9bda4e6d` (`907890a4` is the auto-changelog commit on top; code
byte-identical, `git diff 9bda4e6d HEAD -- conftest.py scripts/qa backend/tests`
is empty)

---

## 1. What was wrong, and why the obvious fix is the wrong fix

`uvicorn --host 0.0.0.0` binds the IPv4 wildcard -- `lsof` reads
`TCP *:8000 (LISTEN)` -- so **every** address of this machine reaches the
running book. The guard decided "is this the live backend" by testing
`urlsplit(url).hostname` against four literal strings.

Measured: **12 spellings reachable (`GET /api/health` -> 200) and NOT refused on
a mutating PUT.** The step named eight; three more were invented in the first
five minutes of looking; the research round added a twelfth of a different kind.

**The count went 8 -> 11 -> 12 across three independent looks. That is the
finding.** The residual is not a list of eight, it is an open-ended population,
and every previous attempt at this defect failed by treating the sample as the
population:

- 86.6 cycle 1: HTTP row said COVERED; `0.0.0.0` reached the live book.
- 86.6 cycle 2: added `0.0.0.0`, declared it correct; eight more reached it.
- 86.6 cycle 3: corrected the row, left the class open. **86.27 exists because
  the third fix was still an instance fix.**

### The case that ends the argument for string matching

`%31%32%37%2e%30%2e%30%2e%31`. `urlsplit()` hands the guard that literal string
while **urllib percent-decodes the host itself and connects to
`127.0.0.1:8000`**. So a *perfect* canonicaliser applied to
`urlsplit().hostname` still fails, because `getaddrinfo` cannot resolve it.
Repairing that at the string layer means re-implementing urllib's decode chain
-- a second parser obliged to agree with the first forever, which is the
anti-pattern itself (Stenberg "don't mix URL parsers"; Snyk "single parsing
point"; CWE-180; CWE-184).

**Conclusion: parse-and-match on a host string is not sound here at ANY list
length, because the guard's parser is not the requester's parser.**

## 2. What was built

**One authority.** `scripts/qa/live_backend_origin.py` is now the only
definition. `conftest.py` imports it instead of carrying a copy. The predicate
is `(the address is one of THIS machine's) AND (the port is the backend's)`:

- `_is_this_machine(addr)` -- `ipaddress` classification (`is_loopback` or
  `is_unspecified`, `.ipv4_mapped` unwrapped) unioned with this machine's
  interface addresses from `psutil.net_if_addrs()`;
- `targets_this_machine(host)` -- canonicalises via
  `getaddrinfo(..., AI_NUMERICHOST)`, which does **no network I/O** and is the
  same libc parser the socket will use, so the whole numeric class (short form,
  decimal, hex, octal, zero-padded, IPv4-mapped) is handled including spellings
  nobody has invented. Genuine names fall through to one real resolution,
  memoised. **Fails safe**: a host it cannot canonicalise is treated as live.

**The authority is at the socket, not the string.** A PEP-578 `socket.connect`
hook sees the address the socket is about to use. It has no HTTP verb -- that
event fires before any verb is on the wire -- so the `urlopen`/urllib3 wrappers
carry the verb across on a `threading.local()`. This is a deliberate
**improvement on the research brief's own sketch**, which left the string
predicate as the policy layer; carrying the verb makes the *address* layer the
policy layer, which is the only layer where the address is canonical.

Consequences, all measured rather than argued: **no DNS inside the guard** (the
resolution already happened and it is the caller's own), **zero TOCTOU** (there
is no second resolution to race -- the flaw CVE-2025-31490 punishes), and class
coverage by construction.

**The child process gets the same guard.** `smoke_cc_rail_e2e.py` installs it in
`main()` unless `--allow-live-backend`, and carries the verb at its single
`http_json` seam. A child loads no conftest, so nothing else covered it.

### Files

| file | change |
|---|---|
| `scripts/qa/live_backend_origin.py` | rewritten -- the single authority; adds `targets_this_machine`, `_is_this_machine`, `address_is_live_backend`, `install_socket_guard`, `mutating_scope`, `MUTATING_VERBS` |
| `conftest.py` | imports the authority (copy deleted); wrappers carry the verb; socket guard installed; loud fail-safe fallback if the import ever fails |
| `scripts/qa/smoke_cc_rail_e2e.py` | installs the socket guard in the child; verb carried at `http_json` |
| `backend/tests/test_phase_86_27_live_origin_class.py` | NEW -- 50 tests incl. the reachability ground truth and the end-to-end guard proof |
| `backend/tests/test_phase_86_6_subprocess_channel.py` | docstring only -- states that the old drift alarm is now trivially true and is not coverage |
| `scripts/qa/mutation_matrix_86_27.py` | NEW -- 7 cells, hermetic (temp copies only) |
| `scripts/qa/reproduce_86_27_spellings.py` | NEW -- the re-runnable before/after table |

## 3. Verification

**Immutable command**, pre- and post-fix on this machine:

```
bash -c 'source .venv/bin/activate && python -m pytest \
  backend/tests/test_phase_86_6_subprocess_channel.py \
  backend/tests/test_phase_4000_2_cc_rail_smoke.py -q'

40 passed in 27.92s    <- pre-fix, cad38647
40 passed in 27.25s    <- post-fix, 9bda4e6d
```

Full per-criterion evidence is in `handoff/current/live_check_86.27.md`.

## 4. Two things the tests caught in MY OWN work

Recorded because they are the step's own thesis applied to itself.

**(a) The novel-spelling test refused to certify on a spelling that was not
novel.** The first draft derived candidates from every interface address --
including loopback, whose integer form `2130706433` appears six times in this
repo. The assertion fired with the list of files. Loopback is excluded now, and
the test derives from **non-loopback** interface addresses, which nobody could
have enumerated in advance because they are properties of this host.

**(b) Two mutants survived the first matrix run, and both were MY probes'
fault, not the guard's.**

- **M3** (drop the `.ipv4_mapped` unwrapping) survived because I probed it with
  `::ffff:127.0.0.1` -- and CPython already reports
  `IPv6Address('::ffff:127.0.0.1').is_loopback == True`. For loopback the branch
  genuinely is redundant. It is load-bearing only for a mapped **non-loopback**
  address, which is now the probe, and the gap is pinned as its own test.
  Same shape as 86.6's M4/M5 pair: a leg is proven load-bearing only by a case
  the other legs cannot reach.
- **M6** (remove canonicalisation) survived because I probed with
  `2130706433`, where the control answer and the fail-safe answer coincide at
  True. A mutant is only killed by a case where the guard's answer and its
  failure mode DIFFER; the probe is a remote numeric address now.

Both were fixed by correcting the PROBE, never by weakening a guard or relaxing
an assertion.

## 5. Not claimed

- **The socket layer does not close every channel, and the gaps are named.**
  `httpx`/`httpcore` and raw `socket` use outside the wrappers set no verb flag,
  so they are not refused -- already declared gaps at `conftest.py:52-58` and
  unchanged by this step. **New and worth naming: urllib3 POOLS connections**, so
  a mutating request reusing a socket opened earlier by a GET emits no fresh
  `socket.connect` and the authoritative layer never sees it; the urllib3
  wrapper's own string check is what covers that case, and it is best-effort.
  stdlib urllib does not pool.
- **A network dependency was introduced into one pre-existing assertion.**
  `test_is_live_backend_table`'s `https://example.com:8000 -> False` row now
  requires DNS: pre-fix the answer came from a string comparison, post-fix it
  comes from resolving `example.com` and finding a public address. Offline, that
  host fails to resolve, fails safe to True, and the row would go red. The row
  is left as-is rather than weakened, the behaviour is pinned by
  `test_an_unresolvable_host_neither_hangs_nor_raises`, and this is disclosed
  rather than discovered later.
- **`psutil` is not declared in any requirements file** -- it is installed
  transitively (via `gpt-researcher` / `unstructured`). Its absence is handled:
  the enumeration returns None, and the predicate degrades to "only a globally
  routable address is provably remote", which over-refuses. That degraded path
  has a unit test; it has **not** been exercised by uninstalling psutil.
- **No live cycle has exercised this guard**, and none will: the guard exists in
  the test suite and in a QA script, not in the trading path. Nothing in this
  step changes production behaviour.
- **The eight/eleven/twelve counts are what THIS machine answered to on ONE
  day.** They are not the population, and the design is built so that the
  population does not need to be known.

## 6. Measured results

| criterion | evidence | result |
|---|---|---|
| 1 reproduce first | `scripts/qa/reproduce_86_27_spellings.py`, exit 0 | 8/8 named spellings NOT-REFUSED before; 12 previously-un-refused now REFUSED; 4 controls unchanged; **0 mutating requests sent** |
| 2 novel spelling | spellings derived from the interface table at runtime, each proven absent from the tracked tree by `git grep` | refused; a pasted spelling drops out of the probe set and trips the `>= 3` floor |
| 3 ephemeral stubs | immutable command | **40 passed** pre-fix and post-fix; 8000 is outside this machine's ephemeral range 49152-65535 (asserted as a test) |
| 4 total on junk | 24 inputs x 3 predicates | **0 raised** -- and the pre-fix conftest copy RAISED `AttributeError` on 2 of them, so this criterion was failing before |
| 5 resolution cost | frozen-tree full suite, 360.18s | 59 calls, **61.38 ms total**, worst single **32.3 ms**; hook **3.684 us/event**, 0.81 ms total |
| 6 drift alarm | one predicate + reachability ground truth | old alarm documented as trivially true; ground truth measures the reachable set instead of enumerating it |
| 7 mutation | `scripts/qa/mutation_matrix_86_27.py` | **7/7 killed** (2 survived on the first run -- both my probes' fault, both diagnosed) |

**Full-suite delta attributable to this step: 0 failures.** 16 failed / 3351
passed; 13 of the 16 reproduced at the pre-fix commit `cad38647` in a git
worktree, and the other 3 fail on missing `backend.log` strings in 0.11 s
without opening a socket.

`handoff/kill_switch_audit.jsonl` = `ea78508bee73887c82df2346da408c72...`,
64 lines, byte-identical before and after every run in this step.
