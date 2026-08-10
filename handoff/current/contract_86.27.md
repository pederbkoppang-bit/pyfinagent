# Contract -- phase-86.27

**Step:** 86.27 (P1) -- the live-backend HTTP guard is a host-string allowlist,
and the machine answers to many more names than the list contains.
**Date:** 2026-08-10
**Research:** `handoff/current/research_brief_86.27.md` (1187 lines, 81,197 chars)

---

## 1. Research gate -- PASSED

| | |
|---|---|
| launch | Workflow rail, `.claude/workflows/research-gate.js` |
| tier | `complex` · **audit-class: YES** (loop-until-dry, K=2, satisfied at round 10) |
| sources read in full | **19** (floor 5) |
| URLs collected | **43** (floor 10) · snippet-only 24 |
| recency scan | performed -- 7-row table, 2024-08 .. 2026-07 |
| internal files inspected | 13 |
| `gate_passed` | **true**, RECOMPUTED by the script, not taken from the agent |
| artifact cross-check | all 19 claimed URLs found in the brief; 43 <= 75 distinct URLs present |

**Two runs, and the first one's failure is itself evidence worth recording.**
Run `wf_e03b2ad2-e7e` (08:28-08:50 CEST) did the whole audit-class job and wrote
the complete brief, then **died on the return leg** -- `subagent completed
without calling StructuredOutput` after 224,317 tokens and 69 tool uses.
Write-first discipline meant the artifact survived; only the envelope was lost.
Per `.claude/rules/research-gate.md`, an errored return is a FAILED gate and
never `gate_passed`, so the self-report in the brief was **not** accepted. Run
`wf_fb0e675e-2e9` re-ran the gate LEAN against the finished brief; its verifier
corrected one figure **downward** (`internal_files_inspected` 14 -> 13) and
flagged a prose/table slip ("6 findings" vs a 7-row table), neither
gate-blocking. The numbers above are the enforced ones.

## 2. The defect, reproduced before any fix

`conftest.py:125 _is_live_backend` decides "is this the live backend" by testing
`urlsplit(url).hostname` against `_LOOPBACK_HOSTS` -- **4 literal strings**
(`localhost`, `127.0.0.1`, `::1`, `0.0.0.0`; `LOCALHOST` matches only because
`urlsplit` case-folds). `scripts/qa/live_backend_origin.py:45` carries a second
copy of the same predicate.

The backend runs `uvicorn --host 0.0.0.0`. Measured read-only:
`lsof -nP -iTCP:8000 -sTCP:LISTEN` -> `Python 43839 ford 10u IPv4 TCP *:8000
(LISTEN)`. **The IPv4 wildcard means every address of this machine reaches the
running book, while the guard recognises four strings someone thought of.**

Reproduced 2026-08-10 08:33 CEST with `_REAL_URLOPEN` replaced by a
non-networking sentinel (so no mutating byte was ever sent) and reachability
probed read-only via `GET /api/health`. **11 spellings reachable (HTTP 200) and
NOT refused on a mutating PUT** -- the step's eight, plus three invented on the
spot: `0x7f.0x0.0x0.0x1` (hex-dotted), `017700000001` (32-bit octal),
`[::ffff:7f00:1]` (hex IPv4-mapped). The research round added a twelfth of a
structurally different kind: `%31%32%37%2e%30%2e%30%2e%31`.

**The count went 8 -> 11 -> 12 across three independent looks. That is the whole
finding: the residual is an open-ended population, not a list.**

## 3. Hypothesis

**Parse-and-match on the host string cannot be made sound here, at any list
length, because the guard's parser is not the requester's parser.** Measured on
Python 3.14.4: `urlsplit()` hands the guard
`%31%32%37%2e%30%2e%30%2e%31` while urllib percent-decodes and connects to
`127.0.0.1:8000`. A *perfect* canonicaliser applied to `urlsplit().hostname`
still fails that case, because fixing it at the string layer means
re-implementing urllib's decode chain -- a second parser that must agree with
the first forever. (Stenberg, "don't mix URL parsers"; Snyk, "single parsing
point"; CWE-180 canonicalise-then-validate; CWE-184 incomplete denylist.)

**Therefore the decision must move to where the address is already canonical.**
The PEP-578 `socket.connect` audit event carries the **resolved** `(address,
port)` -- measured: every one of the 12 spellings arrives there as
`('127.0.0.1', 8000)` / `('192.168.86.85', 8000)`. It fires **before** the
connect and aborts it when the hook raises. That gives:

- **no DNS inside the guard** -- the resolution already happened, and it is the
  caller's own;
- **zero TOCTOU** -- there is no second resolution to race (which is what
  CVE-2025-31490 punishes naive resolve-and-compare for);
- **class coverage by construction** -- a spelling nobody has invented yet still
  arrives as an address.

**And the invariant is `(address is one of MY addresses) AND (port == 8000)`,
not the port alone.** Port-only keying is simpler and total, and it is
**rejected**: it flips the frozen row `https://example.com:8000 -> allow`
(`test_phase_86_6_subprocess_channel.py:130`) to refuse. Measured, no stub in
this repo ever binds 8000 (ephemeral range is 49152-65535, `sysctl
net.inet.ip.portrange`), so port-only would have been *safe* -- but it would
have changed an already-graded assertion for no gain over the address test.

## 4. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. REPRODUCE FIRST: show, for each of the eight measured spellings, that a mutating PUT was NOT refused before the fix -- using a non-networking sentinel so no request reaches the live backend, and read-only GET for reachability.
2. The fix is judged on a spelling NOT enumerated anywhere in the repo at fix time: invent one (or derive one from the machine's interfaces at runtime) and show it is refused. An allowlist extension that only covers the eight named spellings FAILS this criterion.
3. The 4000.2 stub servers still work: ephemeral-port targets remain allowed, proven by the existing 22-test module passing, and by a measurement of whether any stub ever binds port 8000.
4. The guard remains TOTAL on junk input -- unparseable URLs, None, bytes, and non-string objects must not raise out of the guard path. Assert this directly.
5. If a name-resolution call is introduced, its worst-case latency inside the guard is MEASURED over a full suite run and reported, and an unresolvable host does not hang or crash the run.
6. The drift alarm's limitation is addressed or explicitly documented: an agreement check cannot detect a shared error, so state what now covers that.
7. Mutation-test every new guard, including reverting the fix; a guard whose mutant survives does not count.

**Verification command (immutable):**
```
bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_86_6_subprocess_channel.py backend/tests/test_phase_4000_2_cc_rail_smoke.py -q'
```
Pre-fix baseline on this tree, run 2026-08-10 08:31: **40 passed in 27.92s**
(22 of them the 4000.2 module, measured by `--collect-only -q`).

**live_check:** `live_check_86.27.md` with the verbatim before/after refusal
table over all eight measured spellings PLUS at least one newly-invented
spelling; proof no mutating request was sent to the live backend during
measurement; the ephemeral-stub regression result; and the junk-input totality
assertions.

## 5. Plan

**S1 -- one shared module, deleting the second parser.**
`scripts/qa/live_backend_origin.py` becomes the single authority and gains:
`is_this_machine(addr)` (`ipaddress` classification -- `is_loopback` OR
`is_unspecified`, `.ipv4_mapped` unwrapped -- unioned with this machine's
interface addresses); `targets_this_machine(host)` (canonicalise via
`getaddrinfo(..., AI_NUMERICHOST)` -- zero network, measured 0.016-0.123 ms --
falling back to a real resolve only for genuine names, memoised, **fail-safe:
unresolvable => treated as live**); `is_live_backend(url)` rebuilt on those; and
`install_socket_guard()`. `conftest.py` **imports** it rather than keeping a
copy.

**S2 -- the socket seam, with the verb carried across.** The socket event knows
the address but not the HTTP verb; the `urlopen`/urllib3 wrappers know the verb
but only a string. So the wrappers set a `threading.local()` mutating flag
around the delegated call and the `socket.connect` hook enforces
`mutating AND is_this_machine(addr) AND port == 8000`. **This is a deliberate
improvement on the research brief's own sketch** (its §5 leaves the string
predicate as the policy layer and the socket hook as a backstop); carrying the
verb makes the *address* layer the policy layer, which is the only layer where
the address is canonical. Refusal derives from `BaseException`, reusing the
`LiveStateWriteRefused` precedent at `conftest.py:238` -- an `Exception` is
swallowed by `kill_switch._append_audit`.

**S3 -- the child process installs the same guard.**
`scripts/qa/smoke_cc_rail_e2e.py` already imports the predicate at `:468`; it
gains `install_socket_guard()` in `main()` unless `--allow-live-backend`.

**S4 -- criterion 6: replace the agreement check with a GROUND TRUTH.** Stand up
a server bound to the IPv4 **wildcard** on an **ephemeral** port -- the same
binding shape uvicorn uses -- MEASURE which spellings actually reach it, and
assert every measured-reachable spelling is classified as this machine. The
oracle is reachability, not a list, so a newly-invented spelling is covered
automatically and **the live backend is never addressed at all**. The old
`test_the_two_live_origin_predicates_AGREE` becomes trivially true (one
implementation) and will be documented as such rather than left looking like
coverage.

**S5 -- criterion 2's test must be self-policing.** Generate the novel spelling
**at test time** and assert `grep -c` over the whole repo returns 0 for it, so
the test cannot silently degrade into "someone added another string".

**S6 -- criterion 4:** assert totality directly on `None`, `bytes`, `int`,
`object()`, `""`, `"://"`, `"http://[::1"`, `"http://127.0.0.1:notaport/"`.
NOTE, measured pre-fix: today's `conftest._is_live_backend` **RAISES
`AttributeError` on `12345` and on `object()`**; `live_backend_origin`'s copy
does not. So criterion 4 is currently FAILED by one of the two predicates, and
the merge fixes it.

**S7 -- criteria 5 and 7:** report the measured guard cost (already measured:
**1.927 us per connect event, 0.10 ms across a 99-second run**;
`handoff/current/captures_86.27/guard_path_volume.txt`) and re-measure over the
full suite with the guard installed. Mutation-test every new guard including
reverting the fix.

## 6. Explicitly NOT in scope

- `backend/main.py::_TAILSCALE_ORIGIN_RE` shares the *shape* but fails **closed**
  (a missed spelling denies a legitimate origin). Usability risk, not this
  defect. **Do not "fix" it here.**
- `backend/api/auth.py:150-153` is already the sound pattern -- it keys on
  `request.client.host`, the actual socket peer. Its only flaw is a dead
  `"localhost"` element at `:152` (a peer address is never that string). Noted,
  not changed.
- `httpx`/`httpcore`, raw `socket` use outside the wrappers, and **pooled
  connection reuse** (a PUT on a connection opened earlier by a GET emits no new
  `socket.connect`) remain open and will be named as open. They are already
  declared gaps at `conftest.py:52-58`.
- The non-`open` filesystem residual carried over from 86.6
  (`os.rename`/`os.remove`/`os.truncate`, `PYFINAGENT_LIVE_STATE_GUARD=off`) is
  **not** folded in here; it is a numbered ask in the day report.

## 7. References

- `handoff/current/research_brief_86.27.md` -- 19 sources, incl. OWASP SSRF
  Prevention Cheat Sheet, PEP 578, PortSwigger SSRF, Claroty/Snyk/Stenberg on
  parser confusion, CWE-180, CWE-184, CPython socket + ipaddress + urllib.parse,
  Oligo "0.0.0.0-day", pytest-socket, ssrf_filter, sockfilter.
- `handoff/current/captures_86.27/book_baseline_am.txt` -- the running book at
  08:44:59 CEST (unpaused, unbreached, next cron 14:00 ET).
- `handoff/current/captures_86.27/guard_path_volume.txt` -- criterion-5 volume
  and cost measurement.
- `handoff/current/experiment_results_86.6.md` §4 -- the criterion-9 HTTP row
  this step exists to close; 86.6 recorded it PARTIAL and did **not** claim the
  class.
