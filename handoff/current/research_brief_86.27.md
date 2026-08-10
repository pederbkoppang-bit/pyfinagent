# Research Brief — phase-86.27

**Topic:** How should a test-suite egress guard decide "this HTTP request targets the
operator's LIVE local backend" when the server binds the IPv4 wildcard (`uvicorn --host
0.0.0.0`), so the set of host strings that reach it is UNBOUNDED?

**Tier:** complex · **Audit-class:** YES (loop-until-dry, K=2 — satisfied at round 10)
**Researcher:** Layer-3 Workflow rail · **Date:** 2026-08-10
**Status:** COMPLETE — `gate_passed: true` (19 sources read in full, 43 URLs, 10 rounds,
2 consecutive dry rounds)

**HOW TO READ THIS FILE.** It was written incrementally (write-first discipline), so it is
ordered as it was produced: sections `[A]`-`[K]` are the live append log of measurements
and sources; the `CONSOLIDATED FINDINGS` block near the end is the answer. If you only
read one thing, read **"Key findings"** and **"Application to pyfinagent"**.

### The one-paragraph answer
Parse-and-match on the host string is **never** sound here — not because the list is too
short, but because **the guard's parser is not the requester's parser** (measured:
`urlsplit()` reports `%31%32%37%2e%30%2e%30%2e%31` while urllib connects to
`127.0.0.1:8000`). The three candidate directions are not alternatives: the sound design
collapses all three into **one predicate evaluated on the destination socket** —
`port == 8000 AND resolved_address ∈ this machine's addresses`, defaulting to refuse
unless provably remote. Measured, this costs **no DNS at all** (the `socket.connect` audit
event already carries the canonical address), has a **zero TOCTOU window**, and preserves
every row of the frozen behaviour table including `https://example.com:8000 → allow`.

---

## 0. The question restated precisely

The current guard (`conftest.py:81 _LOOPBACK_HOSTS`) is a **five-string allowlist**. The
step will be judged on whether a **newly invented spelling, present in no list in the
repo**, is refused. Therefore:

- An allowlist EXTENSION cannot pass — the adversary (the Q/A mutation) picks the string
  AFTER the list is frozen.
- The design question is which of three sound alternatives to adopt:
  - (a) **resolve-and-compare** against the machine's own addresses;
  - (b) **key the refusal on the INVARIANT** (the destination socket / the port) rather
    than the unbounded variable (the host string);
  - (c) **invert the default to fail-safe** (refuse unless provably NOT this machine).

Sections below fill in: external security-engineering doctrine, measured internal facts,
and the mapping.

---

## 1. Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| _(populated as each source is read)_ |

## 2. Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|

## 3. Recency scan (2024-2026)

_(pending)_

## 4. Key findings

_(pending)_

## 5. Internal code inventory (measured)

_(pending)_

## 6. Consensus vs debate

_(pending)_

## 7. Pitfalls

_(pending)_

## 8. Application to pyfinagent

_(pending)_

## 9. Research Gate Checklist

_(pending)_

## 10. JSON envelope

_(pending)_

---
# LIVE APPEND LOG (incremental — sections above get consolidated at the end)

## [A] MEASURED INTERNAL FACTS (round 1)

### A1. Does ANY stub server in this repo bind port 8000? **NO. Measured, zero.**

Command + output (2026-08-10):
```
$ grep -rn '\.bind((' --include='*.py' backend/tests tests scripts
backend/tests/test_phase_76_9_2_max_bridge.py:130:        s.bind(("127.0.0.1", 0))
$ grep -rn 'bind((' --include='*.py' . | grep 8000     # exit 1, no output
```
Every HTTP server constructed anywhere in the repo:

| File:line | Bind address | Port | Value |
|---|---|---|---|
| `backend/tests/test_phase_86_3_live_egress_guard.py:271` | `127.0.0.1` | `0` | ephemeral |
| `backend/tests/test_phase_76_9_2_max_bridge.py:137` | `127.0.0.1` | `up_port` | `_free_port()` → ephemeral |
| `backend/tests/test_phase_76_9_2_max_bridge.py:347` | `127.0.0.1` | `port` | `_free_port()` → ephemeral |
| `backend/tests/test_phase_4000_2_cc_rail_smoke.py:176` | `127.0.0.1` | `0` | ephemeral |
| `scripts/ops/anthropic_max_bridge.py:179` | `127.0.0.1` | `PORT` | `ANTHROPIC_BRIDGE_PORT`, **default 18797** (`:57`) |

**Nothing binds 8000. The single production non-ephemeral local listener is 18797.**
This is the fact that decides whether port-only keying is safe, and it says: it is.

### A2. Ephemeral port range in force on this macOS box — 8000 CANNOT collide

```
$ sysctl net.inet.ip.portrange.first net.inet.ip.portrange.last
net.inet.ip.portrange.first: 49152
net.inet.ip.portrange.last: 65535
```
8000 < 49152, so `bind(("127.0.0.1", 0))` can **never** return 8000 on this machine.
The defensive `assert port != 8000` at `test_phase_86_3_live_egress_guard.py:274` is
therefore *always* true here — harmless, but it is not the thing protecting the stubs;
the kernel's portrange is.

### A3. The two predicates, and why their agreement test is structurally blind

* `conftest.py:81` `_LOOPBACK_HOSTS = frozenset({"localhost","127.0.0.1","::1","0.0.0.0"})`
* `conftest.py:125` `_is_live_backend()` → `parts.hostname in _LOOPBACK_HOSTS and parts.port == 8000`
* `scripts/qa/live_backend_origin.py:39` `LOOPBACK_HOSTS` — **the same four strings**
* `scripts/qa/live_backend_origin.py:45` `is_live_backend()` — same predicate, plus `.lower()`

`test_phase_86_6_subprocess_channel.py::test_the_two_live_origin_predicates_AGREE`
compares `f(u) == g(u)` over a URL table. An equality check over two functions detects
**divergence**, never **incorrectness**: for every one of the eight bypass spellings both
return `False`, i.e. they agree *and are both wrong*. This is the classic shared-mode
failure of N-version programming — two implementations written from the same flawed
specification fail identically, so voting/agreement cannot detect the fault. The only
thing that can is an ORACLE that is independent of the spelling (reachability, or the
resolved address), not a second copy of the same list.

### A4. The CLASS question — other host-string allowlists in this repo

| Site | Shape | Keys on | Same defect? |
|---|---|---|---|
| `conftest.py:81` `_LOOPBACK_HOSTS` | 4-string set | **URL string** (attacker/author-chosen) | **YES — fail-OPEN** |
| `scripts/qa/live_backend_origin.py:39` `LOOPBACK_HOSTS` | 4-string set | **URL string** | **YES — fail-OPEN** (same list) |
| `backend/main.py:531` `_TAILSCALE_ORIGIN_RE` | regex `^http://(localhost\|100\.64-127\.\d+\.\d+):\d+$` | `Origin` header string | Same SHAPE, **opposite direction**: a missed spelling *denies* a legitimate origin (fail-CLOSED, usability bug). Note `127.0.0.1:3000` is **not** matched — only `localhost`. |
| `backend/api/auth.py:150-153` `DEV_LOCALHOST_BYPASS` | `request.client.host in ("127.0.0.1","::1","localhost")` | **the SOCKET peer address**, not a URL | **NO — this one is the SOUND pattern.** The value comes from the accepted connection, so it is an address, never a spelling. `127.1` cannot appear here; the kernel has already canonicalised it. |

**`auth.py:152`'s `"localhost"` entry is dead code** — `request.client.host` is a peer
address and can never be the string `localhost`. Its presence is the fingerprint of the
host-string mental model leaking into a place where only addresses exist. That is
evidence *for* the recommendation below, not against it: the sound site keys on the
socket, and the only wrong element in it is the one string that came from the URL world.

## [B] SOURCES READ IN FULL — running list

1. **OWASP SSRF Prevention Cheat Sheet** (official docs tier) —
   `https://raw.githubusercontent.com/OWASP/CheatSheetSeries/refs/heads/master/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md`
   — accessed 2026-08-10, WebFetch, full markdown source.
   * **"Deny-lists are bypass-prone. Prefer allow-lists."**
   * Prescribed order is **format-validate → DNS-resolve → compare the resolved IP**, i.e.
     validation happens on the *address*, never on the *string*.
   * **"Do not accept complete URLs from the user because URLs are difficult to validate
     and the parser can be abused."**
   * Explicitly names the bypass encodings: **Hex, Octal, Dword, URL and Mixed encoding**;
     records that .NET `IPAddress.TryParse` is bypassable by Hex/Octal/Dword/Mixed.
   * Names the TOCTOU: *"a DNS resolution will be made when the business code will be
     executed"* — a second, later resolution than the validator's.
   * Deny-list table includes `127.0.0.0/8`, `0.0.0.0/8`, `::1/128` — i.e. **CIDR ranges,
     not name strings**.
   * Does **not** endorse socket-level keying; its frame is pre-connection URL/IP validation.
   * **Direct bearing on 86.27:** the doctrine says an allowlist is only viable in "Case 1"
     where the target set is *identified and closed*. 86.27's target set (the ways this
     machine can be named) is the *unbounded* side, so a name-allowlist is Case-2 shaped
     and the cheat sheet's own answer is to resolve first and compare addresses.

2. **CPython `socket` module reference** (official docs tier) —
   `https://docs.python.org/3/library/socket.html` — accessed 2026-08-10, WebFetch.
   * **CONFIRMED, and this is load-bearing for the latency trade-off:** the documented
     signature is
     `socket.getaddrinfo(host, port, family=AF_UNSPEC, type=0, proto=0, flags=0)` —
     **there is no `timeout` parameter.** A `getaddrinfo` call cannot be bounded from the
     stdlib API.
   * `setdefaulttimeout()` is documented as setting the default timeout "for new socket
     objects" — it governs socket operations, not name resolution. It does **not** bound
     `getaddrinfo`.
   * `gethostbyname()` likewise has no timeout and is IPv4-only/deprecated in favour of
     `getaddrinfo`.
   * **`AI_NUMERICHOST` "will disable domain name resolution and will raise an error if
     host is a domain name."** ← This is the escape hatch: it turns `getaddrinfo` into a
     pure, non-blocking, **non-networking** parser for literal addresses.
   * `socket.if_nameindex()` returns `(index, name)` tuples — interface *names*, not
     addresses, so it is **not** sufficient on its own for "is this one of my addresses".
   * `getfqdn()` is itself a resolver (it calls `gethostbyaddr`) — so it inherits the same
     unbounded-blocking problem.

## [C] MEASURED PYTHON BEHAVIOUR (round 1) — the three parsers disagree

All measured on this box, Python **3.14.4**, 2026-08-10.

### C1. `urlsplit()` does NOT normalise. Confirmed empirically.

`urlsplit(f"http://{h}:8000/x").hostname` returns the spelling **verbatim** for every
non-canonical form. The ONLY transformations it performs are (a) **lowercasing** and
(b) **IPv6 bracket-stripping**:

```
'127.1'            -> hostname='127.1'             port=8000
'0'                -> hostname='0'                 port=8000
'2130706433'       -> hostname='2130706433'        port=8000
'127.000.000.001'  -> hostname='127.000.000.001'   port=8000
'0x7f000001'       -> hostname='0x7f000001'        port=8000
'017700000001'     -> hostname='017700000001'      port=8000
'localhost.'       -> hostname='localhost.'        port=8000
'LOCALHOST'        -> hostname='localhost'         port=8000   <-- lowercased
'[::1]'            -> hostname='::1'               port=8000   <-- brackets stripped
'[::ffff:127.0.0.1]'-> hostname='::ffff:127.0.0.1' port=8000
```
So `conftest.py:129`'s `parts.hostname in _LOOPBACK_HOSTS` is a **raw string comparison
against an un-normalised token**. `urlsplit` gives the guard no help whatsoever.
(Note: `scripts/qa/live_backend_origin.py:56` adds `.lower()`, which is redundant —
`urlsplit` already lowercased — so the two predicates' *only* textual difference is a
no-op. They are the same function.)

**Also measured — a latent crash surface:** `urlsplit("http://::1:8000/x").port` raises
`ValueError: Port could not be cast to integer value as ':1:8000'`. An UNBRACKETED IPv6
literal makes `.port` raise. `conftest.py:130` catches `ValueError` and fails OPEN;
`live_backend_origin.py:50` catches it too. So this is handled — but it is handled by
*allowing* the request, which is the wrong direction for a safety guard.

### C2. `ipaddress.ip_address()` REJECTS every bypass spelling — and that is the trap

| spelling | `ipaddress.ip_address()` |
|---|---|
| `127.0.0.1` | `IPv4Address('127.0.0.1')` is_loopback=**True** |
| `::1` | `IPv6Address('::1')` is_loopback=**True** |
| `0.0.0.0` | `IPv4Address('0.0.0.0')` is_unspecified=**True** (is_loopback=False!) |
| `::ffff:127.0.0.1` | `IPv6Address(...)` is_loopback=**True**, `.ipv4_mapped == IPv4Address('127.0.0.1')` |
| `127.1` | **ValueError** |
| `0` | **ValueError** |
| `2130706433` | **ValueError** |
| `127.000.000.001` | **ValueError** |
| `0x7f000001` | **ValueError** |
| `017700000001` | **ValueError** |
| `localhost` / `localhost.` / `ford-sin-mini.lan` | **ValueError** |

Confirmed against the CPython docs (accessed 2026-08-10,
`https://docs.python.org/3/library/ipaddress.html`): *"Leading zeroes are not tolerated
to prevent confusion with octal notation"*, and **changed in 3.9.5**: *"Leading zeros are
no longer tolerated and are treated as an error. IPv4 address strings are now parsed as
strict as glibc `inet_pton()`."* (This is the CVE-2021-29921 hardening.)

**THE TRAP, stated precisely.** `ipaddress` is strict, so the naive hardening
```python
try:    return ipaddress.ip_address(host).is_loopback     # "now it's principled!"
except ValueError: return False                            # <-- the bypass lives HERE
```
**ALLOWS every one of `127.1`, `0`, `2130706433`, `127.000.000.001`, `0x7f000001`,
`017700000001`.** `ipaddress` is not a bypass surface because it accepts too much; it is
a bypass surface because its `ValueError` branch is exactly the set you needed to catch,
and the obvious except-clause maps that set to "allow". Any 86.27 design that reaches for
`ipaddress` as the canonicaliser MUST be audited on this branch.
`is_unspecified` is a second trap: `0.0.0.0` — the spelling 86.6 just fixed — has
`is_loopback == False`. A loopback-only test re-opens the hole 86.6 closed.

### C3. `socket.getaddrinfo()` — the parser the SOCKET actually uses — accepts everything

```
'127.1'            -> ['127.0.0.1']       (0.01 ms)
'0'                -> ['0.0.0.0']         (0.00 ms)
'2130706433'       -> ['127.0.0.1']       (0.00 ms)
'127.000.000.001'  -> ['127.0.0.1']       (0.00 ms)
'0x7f000001'       -> ['127.0.0.1']       (0.00 ms)   <-- HEX
'017700000001'     -> ['127.0.0.1']       (0.00 ms)   <-- OCTAL
'localhost'        -> ['127.0.0.1','::1'] (1.74 ms)
'localhost.'       -> ['127.0.0.1','::1'] (0.69 ms)
'LOCALHOST'        -> ['127.0.0.1','::1'] (0.22 ms)
'ford-sin-mini.lan'-> ['192.168.86.85']   (9.49 ms)
'127.0.0.1.'       -> gaierror            (25.57 ms)  <-- trailing-dot on an IP does NOT resolve
```

**TWO SPELLINGS FOUND THAT ARE IN NO LIST IN THE REPO AND NOT IN THE STEP'S LIST OF
EIGHT: `0x7f000001` (hex) and `017700000001` (octal).** Both reach `127.0.0.1` through
libc's legacy `inet_aton` multi-format parsing. This is the empirical proof that the
population is open: one round of measurement grew the known set from 8 to 10. It is
exactly the OWASP-named "Hex, Octal, Dword and Mixed encoding" family.

### C4. `AI_NUMERICHOST` — a TOTAL, ZERO-NETWORK canonicaliser. This is the key result.

`socket.getaddrinfo(host, port, flags=socket.AI_NUMERICHOST)` disables name resolution
(CPython socket docs: *"AI_NUMERICHOST will disable domain name resolution and will raise
an error if host is a domain name"*). Measured:

```
AI_NUMERICHOST '127.1'             -> ['127.0.0.1']         (0.123 ms)
AI_NUMERICHOST '2130706433'        -> ['127.0.0.1']         (0.032 ms)
AI_NUMERICHOST '0x7f000001'        -> ['127.0.0.1']         (0.018 ms)
AI_NUMERICHOST '017700000001'      -> ['127.0.0.1']         (0.016 ms)
AI_NUMERICHOST '0'                 -> ['0.0.0.0']           (0.092 ms)
AI_NUMERICHOST '::ffff:127.0.0.1'  -> ['::ffff:127.0.0.1']  (0.023 ms)
AI_NUMERICHOST 'localhost'         -> gaierror              (0.017 ms)
```
It canonicalises **the entire numeric class — including newly invented spellings — in
0.016-0.123 ms with NO network I/O and NO DNS**, because it is the *same libc parser the
socket will use*. Names raise `gaierror` immediately (0.017 ms), so the branch is cheap
and total. This dissolves the "blocking DNS in a hot guard" objection for everything
except literal hostnames.

### C5. Quantified hang risk for the UNRESTRICTED (resolving) getaddrinfo

```
definitely-does-not-exist-9f8a7b6c.invalid       gaierror     9.0 ms
definitely-does-not-exist-9f8a7b6c.example.com   gaierror    31.7 ms
nonexistent-host-xyz.local                       gaierror  5003.3 ms   <-- mDNS, 5 SECONDS
```
**Worst case measured: 5.0 s for a single unresolvable `.local` name**, and per C6 there
is no way to bound it. On a suite with hundreds of urlopen calls this is the outage the
step description warned about — but only on the NAME branch, which C4 shows is avoidable
for every numeric spelling.

### C6. There is NO timeout on `getaddrinfo`. Confirmed against the docs.

`socket.getaddrinfo(host, port, family=AF_UNSPEC, type=0, proto=0, flags=0)` — no
`timeout` parameter exists (CPython socket reference, accessed 2026-08-10).
`socket.setdefaulttimeout()` is documented as applying to "new socket objects", i.e.
socket operations, **not** name resolution. The only bounding mechanisms are external:
run it in a thread/process and abandon it (the thread still leaks until libc returns), or
avoid resolution entirely via `AI_NUMERICHOST`.

### C7. The machine's own identity — measured

```
lsof -nP -iTCP:8000 -sTCP:LISTEN  ->  Python 43839 ford 10u IPv4 TCP *:8000 (LISTEN)
lsof -nP -iTCP:18797 -sTCP:LISTEN ->  Python   650 ford  3u IPv4 TCP 127.0.0.1:18797 (LISTEN)
socket.gethostname()  -> 'ford-sin-mini.lan'
socket.getfqdn()      -> 'ford-sin-mini.lan'
UDP-connect trick     -> '192.168.86.85'          (default-route source addr only)
psutil.net_if_addrs() -> AVAILABLE, 30 entries (incl. MACs + IPv6 temporaries)
netifaces             -> NOT INSTALLED (not a project dep)
socket.if_nameindex() -> [(1,'lo0'),(2,'gif0'),(3,'stf0'),(4,'anpi0'),...]  names, NOT addresses
```
* `TCP *:8000` is the **direct confirmation of the wildcard bind** — the premise of the
  whole step, verified read-only rather than assumed.
* `18797` is bound **loopback-only**, so the bridge is not reachable off-box; and it is
  not 8000, so it is irrelevant to port-keying.
* For "is this one of MY addresses" without DNS: **`psutil.net_if_addrs()` is the only
  complete enumerator actually installed** (`netifaces` is absent; `if_nameindex()`
  returns names, not addresses; the UDP trick returns exactly ONE address and would MISS
  every other interface — it is not a sound predicate for this question).

## [D] THE DECISIVE MEASUREMENT — the socket layer already has the canonical address

Measured 2026-08-10, Python 3.14.4, via `sys.addaudithook` (PEP 578).

### D1. The `socket.connect` audit event carries the **RESOLVED** address, not the spelling

```
[127.1       ] urllib.Request  url='http://127.1:9/nope' method='GET'
[127.1       ] socket.getaddrinfo   -> port 9
[127.1       ] socket.connect       -> ('127.0.0.1', 9)      <-- CANONICAL
[2130706433  ] socket.connect       -> ('127.0.0.1', 9)      <-- CANONICAL
[0x7f000001  ] socket.connect       -> ('127.0.0.1', 9)      <-- CANONICAL
[localhost   ] socket.connect       -> ('::1', 9, 0, 0)
[localhost   ] socket.connect       -> ('127.0.0.1', 9)
```
`socket.create_connection` calls `getaddrinfo` **first** and then `sock.connect(sa)` with
the resolved `sockaddr`. So a guard at the socket layer sees the canonical address for
**every** spelling — including ones nobody has invented yet — and pays **zero** DNS cost,
because the resolution is the caller's own, not the guard's. The `localhost` case emits
TWO connect events (`::1` then `127.0.0.1`) as the address list is walked; the guard sees
each candidate.

### D2. The hook fires BEFORE the connect and CAN prevent it

```
=== Q: does the hook fire BEFORE the connect? ===
  REFUSED before connect: blocked ('127.0.0.1', 9)
=== R: and via urlopen with a bypass spelling? ===
  urlopen REFUSED at the socket layer: blocked ('127.0.0.1', 9)
```
A `BaseException` raised from the hook aborts the operation before any packet is sent.
This matches PEP 578: *"If any hook returns with an exception set, later hooks are
ignored"* and the operation is aborted.

### D3. THE HONEST CAVEAT — the socket layer is necessary but NOT sufficient alone

```
=== P: raw socket.connect(("127.1", 9)) ===
  hook saw socket.connect args[1]=('127.1', 9)     <-- RAW STRING, not resolved
  result: ConnectionRefusedError   (i.e. it DID reach 127.0.0.1:9)
```
If a caller hands a **hostname directly to `connect()`** and lets the OS resolve it
internally, the hook sees the raw spelling. So the socket-layer guard must still
canonicalise its `address[0]` defensively — which is **free** via `AI_NUMERICHOST` (C4,
0.016-0.123 ms, no network). **Neither layer alone is complete; socket-layer +
`AI_NUMERICHOST` canonicalisation is.** Stating this rather than selling the socket hook
as a total answer.

### D4. This kills the stated latency objection

The step description warns that resolve-and-compare "puts a BLOCKING DNS lookup inside a
guard that runs on every urlopen". Measured, that is true only for the *URL-string*
formulation. At the socket layer the DNS has **already happened** — the guard reads a
tuple. The 5.0 s mDNS worst case (C5) is never paid **by the guard**; it is paid by the
request the test was making anyway. Cost of the guard proper: a tuple unpack, an
`AI_NUMERICHOST` parse (~0.02 ms) and a set membership test.

### D5. It also kills the TOCTOU

OWASP names the rebinding window explicitly: *"a DNS resolution will be made when the
business code will be executed"* — i.e. the validator resolves once and the client
resolves again later, and the two answers can differ. At the socket layer **there is no
second resolution**: the address in the hook IS the address about to be connected. The
check-to-use gap is zero. This is the structural reason the socket seam beats
resolve-and-compare-on-the-string, independent of latency.

## [E] THE BINDING INTERNAL CONSTRAINT SET (measured from the existing tests)

`backend/tests/test_phase_86_6_subprocess_channel.py:120-136` freezes this table, and any
86.27 design must keep every row:

| URL | required | why |
|---|---|---|
| `http://localhost:8000` | **True (refuse)** | live backend |
| `http://127.0.0.1:8000` | **True** | live backend |
| `http://0.0.0.0:8000` | **True** | live backend (86.6's fix) |
| `http://localhost:8000/api/settings/` | **True** | path is irrelevant |
| `http://127.0.0.1:59999` | **False (allow)** | ephemeral stub — 59999 IS in 49152-65535 |
| `http://127.0.0.1:3000` | **False** | the frontend |
| `http://localhost` | **False** | port 80 |
| **`https://example.com:8000`** | **False** | **right port, WRONG HOST** |
| `""` / `"not a url"` | **False** | must not raise |

**`https://example.com:8000` is the row that REFUTES option (b) in its pure form.**
Port-only keying ("refuse anything on 8000") turns that row True and breaks an existing,
already-graded assertion. So "key on the invariant" cannot mean *port alone*; it must mean
*the destination socket* — the (resolved address, port) pair. `example.com` resolves to a
public address, is provably not this machine, and stays allowed; every spelling of this
machine on 8000 is refused. That is the version of (b) that survives the test table, and
it is simultaneously (a) and (c): the comparison is against this machine's own addresses,
and the default on port 8000 is refuse-unless-provably-remote.

Also binding: `test_phase_4000_2_cc_rail_smoke.py:176` and
`test_phase_76_9_2_max_bridge.py:130,137,347` all bind `127.0.0.1` on **ephemeral** ports;
`scripts/ops/anthropic_max_bridge.py:179` binds `127.0.0.1:18797`. None is 8000, so a
(local-address AND port==8000) predicate leaves all of them untouched.

### E1. Why `test_the_two_live_origin_predicates_AGREE` cannot ever catch this

`test_phase_86_6_subprocess_channel.py:139-175` asserts
`mod._is_live_backend(u) == is_live_backend(u)` over a 9-URL table. Both functions are
byte-equivalent modulo a redundant `.lower()` (C1). For each of `127.1`, `0`,
`2130706433`, `127.000.000.001`, `0x7f000001`, `017700000001`, `localhost.`,
`[::ffff:127.0.0.1]`, `192.168.86.85`, `ford-sin-mini.lan` **both return `False`** —
they AGREE, and are BOTH WRONG. An equality oracle over two implementations of the same
specification detects only *divergence*; a **common-mode fault** (shared wrong spec) is
invisible to it by construction. The test's own docstring even warns that a drift table
omitting the drifting case "is not a drift alarm; it is decoration" — the deeper point is
that no table of URLs can fix it, because the fault is in the *specification both share*.
The only oracle that catches it is one that does NOT consult a list of spellings:
reachability, or the resolved address.

## [F] REPRODUCTION — the current guard vs the socket, on 16 URLs

Measured 2026-08-10, Python 3.14.4. **No request was ever sent**: an audit hook raised a
`BaseException` at `socket.connect`, so the "ACTUAL socket addr" column is the address
Python was about to connect to, captured pre-flight. `guard?` applies the CURRENT
predicate (`urlsplit(u).hostname in {"localhost","127.0.0.1","::1","0.0.0.0"} and
port == 8000`).

```
URL                                            urlsplit.hostname          guard?  ACTUAL socket addr
--------------------------------------------------------------------------------------------------
http://%67oogle.com:8000/x                     %67oogle.com               allow   ('2a00:1450:400f:802::200e', 8000, 0, 0)
http://%31%32%37%2e%30%2e%30%2e%31:8000/x      %31%32%37%2e%30%2e%30%2e%31 allow  ('127.0.0.1', 8000)   <<< BYPASS
http://127.1:8000/x                            127.1                      allow   ('127.0.0.1', 8000)   <<< BYPASS
http://0x7f000001:8000/x                       0x7f000001                 allow   ('127.0.0.1', 8000)   <<< BYPASS
http://017700000001:8000/x                     017700000001               allow   ('127.0.0.1', 8000)   <<< BYPASS
http://2130706433:8000/x                       2130706433                 allow   ('127.0.0.1', 8000)   <<< BYPASS
http://0:8000/x                                0                          allow   ('0.0.0.0', 8000)     <<< BYPASS
http://localhost.:8000/x                       localhost.                 allow   ('::1', 8000, 0, 0)   <<< BYPASS
http://127.000.000.001:8000/x                  127.000.000.001            allow   ('127.0.0.1', 8000)   <<< BYPASS
http://[::ffff:127.0.0.1]:8000/x               ::ffff:127.0.0.1           allow   ('::ffff:127.0.0.1', 8000, 0, 0)  <<< BYPASS
http://192.168.86.85:8000/x                    192.168.86.85              allow   ('192.168.86.85', 8000)  <<< BYPASS
http://ford-sin-mini.lan:8000/x                ford-sin-mini.lan          allow   ('192.168.86.85', 8000)  <<< BYPASS
http://user:pw@127.0.0.1:8000/x                127.0.0.1                  REFUSE  (blocked)
http://example.com@127.0.0.1:8000/x            127.0.0.1                  REFUSE  (blocked)
http://127.0.0.1:8000@example.com/x            example.com                allow   InvalidURL
https://example.com:8000/x                     example.com                allow   ('2606:4700:10::6814:179a', 8000, 0, 0)  [correctly allowed]
```

### F1. THREE spellings found here that are in NO list in this repo and NOT in the step's list of eight

| # | spelling | class | why it matters |
|---|---|---|---|
| 9 | `0x7f000001` | **hex** `inet_aton` | OWASP's "Hex"; PortSwigger's family |
| 10 | `017700000001` | **octal** `inet_aton` | **PortSwigger names this exact string** |
| 11 | `%31%32%37%2e%30%2e%30%2e%31` | **percent-encoded** | see F2 — structurally different and decisive |

The known population went 8 → 11 in a single measurement round. That is the empirical
answer to "can we just enumerate them": no.

### F2. Spelling #11 is the one that ENDS the argument for string-based guarding

`urlsplit()` hands the guard `%31%32%37%2e%30%2e%30%2e%31`. **urllib percent-decodes the
host itself before resolving**, so the socket goes to `127.0.0.1:8000`. Therefore:

* A *perfect* canonicaliser applied to `urlsplit().hostname` STILL fails, because
  `getaddrinfo("%31%32%37%2e%30%2e%30%2e%31")` is a `gaierror` — the guard sees a string
  the resolver cannot resolve, while the requester resolves a *different* string.
* To fix it at the string layer you would have to **re-implement urllib's exact decode
  chain** — i.e. maintain a second parser that must agree with the first, forever. That
  is precisely the anti-pattern Stenberg names: *"if you parse a URL with parser A and
  make conclusions about the URL based on that, and then pass the exact same URL to
  parser B and it draws different conclusions ... it opens up ... for downright security
  vulnerabilities"* (daniel.haxx.se, accessed 2026-08-10), and Snyk's remedy: *"Use as
  few different parsers as possible"* / *"single parsing point"* (snyk.io, accessed
  2026-08-10).
* `http://%67oogle.com:8000` is the benign control for the same mechanism: `urlsplit`
  reports host `%67oogle.com`, the socket goes to a Google IPv6 address. The parser and
  the requester genuinely disagree on Python 3.14.4, today.

**Conclusion, stated as the answer to the objective's question: parse-and-match on the
host string is NEVER sound here.** Not because the list is too short, but because the
guard's parser is not the requester's parser. Any host-string predicate is a second
parser by construction.

### F3. What the current guard DOES get right (do not weaken it)

`http://user:pw@127.0.0.1:8000/x` and `http://example.com@127.0.0.1:8000/x` are both
correctly REFUSED — `urlsplit` strips userinfo properly, so PortSwigger's `@`-embedding
trick does not bypass this particular predicate. Any replacement must keep that.

## [G] SOURCES READ IN FULL — rounds 2-5 (continuing the list from [B])

3. **CPython `ipaddress` reference** — `https://docs.python.org/3/library/ipaddress.html`
   (2026-08-10, WebFetch). *"Leading zeroes are not tolerated to prevent confusion with
   octal notation."* **Changed in 3.9.5:** *"Leading zeros are no longer tolerated and are
   treated as an error. IPv4 address strings are now parsed as strict as glibc
   `inet_pton()`."* `IPv6Address.ipv4_mapped`: *"For addresses that appear to be IPv4
   mapped addresses in the range `::FFFF:0:0/96` ... this property reports the embedded
   IPv4 address. For any other address, this property will be `None`."* Network membership
   via `IPv4Address(...) in IPv4Network('127.0.0.0/8')`.

4. **PEP 578 — Python Runtime Audit Hooks** — `https://peps.python.org/pep-0578/`
   (2026-08-10, WebFetch). *"This proposal does not attempt to restrict functionality, but
   simply exposes the fact that the functionality is being used ... The availability of
   audit hooks alone does not change the attack surface."* *"Hooks cannot be removed or
   replaced."* Overhead *"between 1.05x faster to 1.05x slower"*, i.e. negligible.
   *"If any hook returns with an exception set, later hooks are ignored"* → raising aborts
   the operation. Network events: `socket.address` `(socket, address)` and
   `urllib.Request` `(url, data, headers, method)`; `socket.connect` is listed among the
   socket events folded under `socket.address`.

5. **PortSwigger Web Security Academy — SSRF** — `https://portswigger.net/web-security/ssrf`
   (2026-08-10, WebFetch). Blacklist bypasses, verbatim: *"Use an alternative IP
   representation of `127.0.0.1`, such as `2130706433`, `017700000001`, or `127.1`."*
   Also: *"Register your own domain name that resolves to `127.0.0.1`"*, *"Obfuscate
   blocked strings using URL encoding or case variation"*, and redirect abuse. Whitelist
   bypasses: `@`-embedded credentials, `#` fragments, DNS-hierarchy subdomains, single and
   double URL-encoding *"to exploit inconsistent parsing between validation and request
   execution layers"*. **This independently corroborates C3/F1** — `017700000001` and
   `2130706433` are named by PortSwigger and measured reaching `127.0.0.1` here.

6. **Claroty Team82 — "Exploiting URL Parsing Confusion"** —
   `https://claroty.com/team82/research/exploiting-url-parsing-confusion` (2026-08-10,
   WebFetch). Five confusion classes (scheme, slash, backslash, URL-encoded data, scheme
   mixup); 8 CVEs incl. four Python ones (CVE-2021-23385/32618/23401/23393). The Log4j
   `allowedLdapHost` bypass is the archetype: `${jndi:ldap://127.0.0.1#.evilhost.com:1389/a}`
   — *the validating parser reads the host as `127.0.0.1` and passes the allowlist, while
   the fetching parser connects elsewhere*. Same two-parser shape as this step, inverted.

7. **pytest-socket (miketheman)** — `https://github.com/miketheman/pytest-socket`
   (2026-08-10, WebFetch). *"Entries may be hostnames, IP addresses, or CIDR network
   ranges such as `192.168.0.0/24`."* **Host-level ONLY — there is no port dimension.**
   That is decisive for this repo: `--allow-hosts=127.0.0.1` would allow the live backend
   AND the stubs alike, and blocking `127.0.0.1` would kill all 22 collected 4000.2 tests
   plus 12 max-bridge tests. **pytest-socket cannot express this policy**, which retro-
   justifies `conftest.py`'s hand-rolled guard (its docstring constraint 2 is correct).

8. **Daniel Stenberg (curl maintainer) — "Don't mix URL parsers"** —
   `https://daniel.haxx.se/blog/2022/01/10/dont-mix-url-parsers/` (2026-08-10, WebFetch).
   *"if you parse a URL with parser A and make conclusions about the URL based on that,
   and then pass the exact same URL to parser B and it draws different conclusions and
   properties from that, it opens up not only for strange behaviors but in some cases for
   downright security vulnerabilities."* Remedy: *"Use a single parser to extract the URL
   components you need in one place and then work on the individual components from that
   point on."* **This is the exact indictment of `_is_live_backend`**: `urlsplit` is
   parser A, urllib's own host handling is parser B, and F2 measures them disagreeing.

9. **Snyk — "URL confusion vulnerabilities in the wild"** —
   `https://snyk.io/blog/url-confusion-vulnerabilities/` (2026-08-10, WebFetch). Four
   confusion classes. Concretely: *"For `http://google.com` URL-encoded as
   `http://%67oogle.com`: urllib and requests unexpectedly dispatched requests"*, and
   urllib3 vs others disagreeing on a scheme-less host. Recommendations: *"Use as few
   different parsers as possible"*; *"Single parsing point for decentralized systems"* —
   parse once at entry, pass components onward.

10. **CPython `urllib.parse` reference** —
    `https://docs.python.org/3/library/urllib.parse.html` (2026-08-10, WebFetch).
    **The official confirmations this brief needed:**
    * *"**Percent-encoded sequences are not decoded.**"* ← the mechanism behind bypass #11.
    * *"**Warning:** `urlsplit()` does not perform validation."*
    * *"The `urlsplit()` and `urlparse()` APIs do not perform validation of inputs ...
      We recommend that users of these APIs where the values may be used anywhere with
      **security implications code defensively**. Do some verification within your code
      before trusting a returned component part ... **Is there anything strange about that
      `hostname`?**"* — CPython itself tells you not to trust `.hostname` for a security
      decision.
    * `hostname` is documented as *"Host name (lower case)"* → confirms C1's lowercasing
      and confirms `live_backend_origin.py:56`'s `.lower()` is a no-op.
    * *"Reading the `port` attribute will raise a `ValueError` if an invalid port is
      specified"* — the C1 unbracketed-IPv6 crash is documented behaviour.

11. **Yunus Aydın — "SSRF Vulnerability: Bypassing Protection with DNS Rebinding"**
    (CVE-2025-69660, published **2026-03-14**) —
    `https://aydinnyunus.github.io/2026/03/14/ssrf-dns-rebinding-vulnerability/`
    (2026-08-10, WebFetch). TOCTOU chain: validate → TTL expires → attacker repoints to a
    private IP → fetch hits the private IP. Mitigation is **DNS pinning**: *resolve once
    during validation and use the resolved IP for all subsequent requests*, keeping the
    original hostname only in the `Host` header. **Note for 86.27:** the socket-layer seam
    (D) achieves pinning's guarantee *for free and more strongly* — there is no second
    resolution to pin, because the check happens on the resolved sockaddr itself.

12. **arkadiyt/ssrf_filter (Ruby)** — `https://github.com/arkadiyt/ssrf_filter`
    (2026-08-10, WebFetch). Prior art for "do this properly": positions itself against
    naive implementations that *"[have] TOCTTOU bugs and other issues"*, and claims to
    handle *"URIs/IPv4/IPv6, redirects, DNS, etc, correctly"*. README does not enumerate
    its ranges (they live in source), so treated as corroborating the *shape* of the
    doctrine, not as a range list. **This is the project against which the 2026 NAT64
    bypass (HackerOne #3634400, `64:ff9b:1::/48`) was reported** — see the adversarial
    note in [H].

13. **CWE-184: Incomplete List of Disallowed Inputs** —
    `https://cwe.mitre.org/data/definitions/184.html` (2026-08-10, WebFetch). *"The
    product implements a protection mechanism that relies on a list of inputs (or
    properties of inputs) that are not allowed by policy ... but the list is
    incomplete."* Mitigation, verbatim: *"**Do not rely exclusively on detecting
    disallowed inputs. There are too many variants to encode a character, especially when
    different environments are used, so there is a high likelihood of missing some
    variants.** Only use detection of disallowed inputs as a mechanism for detecting
    suspicious activity."* ChildOf CWE-693 (Protection Mechanism Failure) and **CWE-1023
    (Incomplete Comparison with Missing Factors)**. **This is the CWE identity of the
    86.27 defect** — see [H2] for the polarity subtlety.

14. **CWE-180: Incorrect Behavior Order: Validate Before Canonicalize** —
    `https://cwe.mitre.org/data/definitions/180.html` (2026-08-10, WebFetch). *"The
    product validates input before it is canonicalized, which prevents the product from
    detecting data that becomes invalid after the canonicalization step."* Mitigation:
    *"**Inputs should be decoded and canonicalized to the application's current internal
    representation before being validated.**"* Demonstrative example is the `/safe_dir/../`
    path check — validate *after* `getCanonicalPath()`. **The 86.27 guard is a textbook
    CWE-180:** it validates the spelling, and canonicalisation (`getaddrinfo`) happens
    afterwards, inside the requester.

15. **CPython gh-146245 — "socketmodule.c: Reference and buffer leaks via audit hook
    failures"** — `https://github.com/python/cpython/issues/146245` (2026-08-10,
    WebFetch). **[ADVERSARIAL to the socket-hook recommendation.]** Raising from an audit
    hook leaked `idna`/`pstr` references at `getaddrinfo` (socketmodule.c:6983) — measured
    in the report at **~657 objects per 1000 calls** — and a `Py_buffer` at `sock_sendto`
    (:4810). Status **closed, fixed**, backported to **3.13 and 3.14** (PRs #146247/48/74/75).
    Two things blunt it for this design: (a) the affected events are `getaddrinfo` and
    `sendto`, **not `socket.connect`**, which is the seam recommended here; (b) measured
    on this box (below) the leak is not reproducible at the reported magnitude.

16. **thesis/sockfilter** — `https://github.com/thesis/sockfilter` (2026-08-10, WebFetch).
    Prior art for the exact pattern: *"Block socket creation based on hostname/port to
    suppress unwanted network activity"*; *"The intended use is for making sure your unit
    tests don't make network connections."* Its predicate is
    `socket_address_allowed(address) -> address.host in [...] and address.port == 80`.
    **Note the flaw it shares with 86.3:** the README never says whether `.host` is a
    hostname string or a resolved address — the *same* ambiguity, in published prior art.
    Evidence that this is a widespread class, not a local slip.

## [H] ADVERSARIAL / DISSENTING EVIDENCE (deliberately sought)

**H1. Range-comparison has its own enumeration problem.** HackerOne report #3634400
against `arkadiyt/ssrf_filter` (2026): *SSRF filter bypass via unblocked NAT64 local-use
IPv6 prefix `64:ff9b:1::/48`* — "unblocked NAT64 local-use addresses remain in
public_addresses, so the private-IP guard is bypassed". (Snippet-only: hackerone.com is
JS-rendered; `curl` returned 3823 bytes containing only *"It looks like your JavaScript is
disabled"* — recorded honestly as NOT read in full.) **The lesson is real and it cuts
against a naive reading of option (a):** if you resolve and then compare against a
hand-maintained list of *ranges*, you have moved the enumeration problem one layer down,
not solved it. The 2026 CVEs corroborate: CVE-2026-27730 (esm.sh, *"string-based hostname
validation"*) and the `dssrf-js` resolver-fallback bypass.
**Why it does not overturn the recommendation for 86.27:** this step's comparison set is
not "all private ranges of the internet" — it is **this machine's own addresses**, which
is a *closed, locally enumerable* set (`psutil.net_if_addrs()` + loopback + unspecified).
And the residual is closed by polarity: on port 8000, **refuse unless provably remote**
(option c). An address family nobody anticipated — NAT64 included — fails "provably
remote" and is therefore refused, not allowed. An inverted default converts an unknown
unknown from a bypass into a false positive.

**H2. CWE-184 nominally prescribes "use an allowlist" — which sounds like what 86.3 has.**
This is the subtlety worth stating precisely, because it is where a careless reading goes
wrong. `_LOOPBACK_HOSTS` *looks* like an allowlist (a positive, enumerated set) but its
**role in the predicate is a denylist**: `refuse if host ∈ LIST`, so the *permitted* set
is `everything else` — unbounded. CWE-184's warning ("too many variants ... high
likelihood of missing some") therefore applies verbatim. Applying CWE-184's actual remedy
means inverting the polarity so the *enumerated* set is the permitted one: **allow only
what is provably not this machine**. That is an independent, standards-body derivation of
option (c), reached without reference to the socket seam.

**H3. Raising from an audit hook is not free (gh-146245), and PEP 578 disclaims sandboxing.**
PEP 578: *"This proposal does not attempt to restrict functionality"* / *"Hooks cannot be
removed or replaced."* `conftest.py:225-228` already records both caveats for the
filesystem hook. Measured on this box:
```
python 3.14.4 (main, Apr  7 2026)
gh-146245 getaddrinfo-raise leak probe: delta over 1000 raises = 116 objects
   (report: ~657 / 1000 on the UNFIXED build)
```
116 over 1000 (much of it ordinary allocation churn) is consistent with the backport being
present in 3.14.4. Reported as a measured delta, not as a definitive attribution.

**H4. The honest limit of the whole approach.** None of this survives a *determined*
caller: a test can `sys.modules`-patch, spawn a child, or use `os.system("curl -X POST ...")`.
PEP 578 says so plainly. The guard's job is to stop **accidents** — which is exactly what
the 2026-08-09 incident was (8 audit rows, the live armed book paused 4x). Any 86.27
contract should state that scope rather than imply containment.

## [I] ROUND 6 — the two most on-point precedents in the whole literature

17. **Oligo Security — "0.0.0.0 Day: Exploiting Localhost APIs From the Browser"** (2024,
    18-year-old flaw, actively exploited) —
    `https://www.oligo.security/blog/0-0-0-0-day-exploiting-localhost-apis-from-the-browser`
    (2026-08-10, WebFetch). Researchers *"ran a dummy HTTP server on localhost (127.0.0.1)"*
    and reached it from an external page *"using 0.0.0.0"*. Root cause, verbatim:
    **"we noticed that `0.0.0.0` was not on this list"** — 0.0.0.0 was simply *omitted from
    the Private Network Access private-IP list*. Affects macOS and Linux; *"Windows is not
    impacted."* RFC 1122 *"prohibits 0.0.0.0 as a destination address in IPv4"*, yet it has
    *"multiple uses"* — "all the IPs on this host", "all the network interfaces on this
    host", or "localhost". Fixes: Chromium 128+ blocks it outright; **WebKit "add[ed] a
    check to the destination host IP address. If it is all zeros, the request is blocked"**.
    **Why this matters more than any other source here:** it is the *identical* defect —
    an enumerated list of local-address spellings that omitted `0.0.0.0` — found
    industry-wide, unnoticed for **18 years**, across three browser engines, and exploited
    in the wild (ShadowRay). phase-86.6 lived the same bug in miniature three months ago.
    The industry's fix was *not* "add 0.0.0.0 to the list": Chrome blocks the destination,
    WebKit checks the **resolved destination IP** for all-zeros. Both moved the check off
    the string.

18. **CVE-2026-49857 — auth-fetch-mcp SSRF via IPv4-mapped IPv6 loopback** (published
    **2026-07-01**, CVSS 3.1 **7.4 High**) —
    `https://advisories.gitlab.com/npm/auth-fetch-mcp/CVE-2026-49857/` (2026-08-10,
    WebFetch). `assertSafeUrl()`'s `isPrivateV6()` *"fails to detect IPv4-mapped IPv6
    loopback addresses in their hex-normalized form."* `http://[::ffff:127.0.0.1]:PORT/`
    is silently normalised by the **WHATWG URL parser** to `[::ffff:7f00:1]`, so
    `net.isIPv4('7f00:1')` returns `false` and the private-IP check is skipped. Fixed in
    3.0.2. **This is a 2026 CVE for exactly row 10 of the [F] table.** Companion:
    CVE-2026-54272 (`ip-address` npm — `isLoopback()`/`isLinkLocal()`/`isUnspecified()`
    all return false for `::ffff:127.0.0.1` and NAT64 forms), Coturn GHSA-j8mm-mpf8-gvjg
    (`::ffff:127.0.0.1` defeats `denied-peer-ip 127.0.0.0/8`), and dotnet/runtime#28740
    (`IPAddress.IsLoopBack` returns false for `::ffff:127.0.0.1`). *(These four are
    snippet-only — see the snippet table.)*

    **Cross-language contrast worth recording, and it is GOOD news for the fix:**
    Python's `ipaddress` gets this right where Node and .NET do not — MEASURED above,
    `IPv6Address('::ffff:127.0.0.1').is_loopback is True` **and**
    `.ipv4_mapped == IPv4Address('127.0.0.1')`. So a Python address-level check does not
    inherit the CVE-2026-49857 class. The *string-level* check in `conftest.py:129` does
    fail on that spelling (measured, [F] row 10) — the defect is the layer, not the
    language.

## [J] ROUND 8 — the source that REFUTES the naive form of option (a)

19. **AutoGPT GHSA-wvjg-9879-3m7w / CVE-2025-31490 — "SSRF due to DNS Rebinding in
    requests wrapper"** (High, CVSS 7.5, published **2025-04-11**) —
    `https://github.com/Significant-Gravitas/AutoGPT/security/advisories/GHSA-wvjg-9879-3m7w`
    (2026-08-10, WebFetch). **[ADVERSARIAL — read specifically to attack the emerging
    recommendation.]**

    The broken code **already did what OWASP prescribes**: it resolved with
    `socket.getaddrinfo()` and checked every resulting IP against blocked ranges —
    `if _is_ip_blocked(ip_str): raise ValueError`. **It was still vulnerable**, because
    the check resolved once and `req.request()` resolved *again*; the attacker's DNS
    returned `1.2.3.4` at check time and `169.254.169.254` (AWS metadata) at use time,
    both TTL=0. Fix: *"Replace the hostname with a validated IP address in the URL itself
    and set the `Host` HTTP header to the original hostname"*, via a `HostHeaderSSLAdapter`
    that *"connects to an IP address but validates TLS for a different host."*

    **This is the most important adversarial finding in the brief.** It means:
    **option (a) — "resolve the host string and compare against my addresses" — is
    NOT sufficient on its own. There is a High-severity CVE for exactly that design.**
    Only two shapes are sound: (i) resolve, then **connect to the pinned address** (the
    AutoGPT fix; heavy, and it would mean 86.27 rewriting how tests issue requests), or
    (ii) **check at the socket layer**, where there is no second resolution to race
    ([D5]). For a test-suite guard, (ii) is strictly cheaper and strictly stronger.

## [K] ROUND 9 — measured reachability, and a CORRECTION to the [F] table

**READ-ONLY probe** (TCP open + immediate close; **no HTTP bytes sent, nothing mutated**):
```
127.0.0.1        -> TCP CONNECT OK      (reaches the live backend)
127.0.0.2        -> TimeoutError
127.9.9.9        -> TimeoutError
0.0.0.0          -> TCP CONNECT OK      (reaches the live backend)
192.168.86.85    -> TCP CONNECT OK      (reaches the live backend)
::1              -> ConnectionRefusedError
::ffff:127.0.0.1 -> TCP CONNECT OK      (reaches the live backend)
```
* **macOS is not Linux here:** only `127.0.0.1` of `127.0.0.0/8` is bound; `127.0.0.2`
  and `127.9.9.9` time out. A `127.0.0.0/8` range check is therefore *conservative*
  (refuses more than is strictly reachable) — the safe direction, and worth keeping so
  the guard is not silently platform-dependent.
* **`::1` is REFUSED** — `uvicorn --host 0.0.0.0` binds IPv4 only (`lsof`: `IPv4 TCP
  *:8000`). So `::1` is in `_LOOPBACK_HOSTS` but does **not** reach the backend, while
  `::ffff:127.0.0.1` (which is NOT in the list) **does**. The existing list is wrong in
  *both* directions.
* `ipaddress` classifies all of `127.0.0.0/8` as `is_loopback=True`, and
  `::ffff:127.0.0.2` as loopback with `.ipv4_mapped` → `127.0.0.2`. `0.0.0.0` is
  `is_unspecified=True`, **`is_loopback=False`** — the [C2] trap, restated.

**CORRECTION to the [F] table (stated rather than quietly left standing).** The "ACTUAL
socket addr" column records the **first** address Python attempted, because the hook
raised there. For `http://localhost.:8000` that was `('::1', 8000, 0, 0)` — and `::1` is
*refused* on this box, so the real request would have fallen through to `127.0.0.1:8000`
and succeeded. The bypass conclusion is unchanged; the mechanism is one hop longer.
**This is an argument FOR the socket seam, not against it:** the hook fires on **every**
candidate in the address-list walk (measured in [D1], where `localhost` produced two
`socket.connect` events), so it refuses at the first local candidate regardless of which
one would ultimately have connected. A string-level guard sees one string and never
learns there were two destinations.

**RFC 6890 / IANA special-purpose registry** (snippet-only): `0.0.0.0/8` is
*"This host on this network"* (RFC 1122 §3.2.1.3) — usable as a source, **not** as a
destination, not forwardable, not global; `127.0.0.0/8` is Loopback, valid as neither
source nor destination on the wire. Both are protocol-reserved. Useful as the
*specification* basis for a range predicate; it is not a substitute for the measured
reachability above.

---
# CONSOLIDATED FINDINGS

## Recency scan (2024-2026) — MANDATORY SECTION, and it is NOT empty

Searched 2024-2026 literature on SSRF/loopback filter bypass, URL-parser confusion, and
socket-level test egress guards. **Result: 6 findings from the window that materially
change the design, not merely complement the canonical sources.**

| Finding | Date | Effect on the plan |
|---|---|---|
| **Oligo "0.0.0.0 Day"** — 18-yr-old flaw, exploited in the wild; `0.0.0.0` was *omitted from the PNA private-IP list*; Chromium 128+ blocks it, WebKit checks the destination IP for all-zeros | 2024-08 | The *identical* omission as phase-86.6, industry-wide. Industry fix moved the check **off the string** onto the destination address. **Strongest precedent in the brief.** |
| **CVE-2026-49857** (auth-fetch-mcp) — `http://[::ffff:127.0.0.1]:PORT/` bypasses `isPrivateV6()` after WHATWG normalises to `[::ffff:7f00:1]`; CVSS 7.4 | **2026-07-01** | A 2026 CVE for exactly row 10 of the [F] table. Confirms v4-mapped is a live, current bypass class. |
| **CVE-2026-54272** (`ip-address` npm) — `isLoopback()/isLinkLocal()/isUnspecified()` all false for `::ffff:127.0.0.1` and NAT64 | 2026 | Same class, different library. Python's `ipaddress` is measured correct here — a cross-language contrast worth citing. |
| **CVE-2026-27730** (esm.sh) — SSRF because of *"string-based hostname validation"* | 2026 | Names the anti-pattern in a 2026 CVE description. |
| **HackerOne #3634400** — NAT64 `64:ff9b:1::/48` bypasses `ssrf_filter`'s private-IP guard | 2026 | **Adversarial**: range lists have their own enumeration tail. Drives the fail-safe polarity. |
| **CVE-2025-31490 / GHSA-wvjg-9879-3m7w** (AutoGPT) — a wrapper that DID `getaddrinfo` + range-check was still SSRF-able via DNS rebinding; CVSS 7.5 | 2025-04 | **Refutes naive option (a).** Drives the socket-layer seam. |
| **CPython gh-146245** — audit-hook raises leaked refs at `getaddrinfo`/`sendto`; fixed, backported to 3.13/3.14 | 2025-26 | Adversarial to the socket-hook seam; measured non-reproducible at reported magnitude on 3.14.4. |

Older canonical sources (Orange Tsai 2017, Stenberg 2022, Claroty 2022, CWE-180/184,
RFC 6890/1122) remain valid; the 2024-2026 window did not supersede them, it **confirmed
them with fresh CVEs and moved the recommended remediation to the address/socket layer.**

### Search queries run (three-variant discipline, made visible)
* **Current-year frontier (2026):** "SSRF protection localhost bypass research 2026 resolve address compare instead of hostname allowlist"; "python ... audit hook sys.addaudithook socket.connect 2025 2026"; "audit hook OR monkeypatch pytest guard mutating requests production ... 2026".
* **Last-2-year (2024-2025):** "0.0.0.0 day vulnerability Oligo Security"; "IPv4-mapped IPv6 ::ffff:127.0.0.1 bypass loopback check ... CVE".
* **Year-less canonical:** "OWASP SSRF Prevention Cheat Sheet block localhost allowlist bypass"; "URL parser confusion SSRF Orange Tsai 'A New Era of SSRF'"; "pytest-socket allow-hosts how it works"; "denylist ... anti-pattern canonicalization CWE-180 CWE-41"; "RFC 6890 special-purpose address registry"; "hermetic test isolation prevent tests hitting production endpoint".

## Key findings (the answer to the objective's question)

**1. Parse-and-match on the host string is NEVER sound here — and not for the reason the
step assumed.** The step frames it as "the list is too short". Measured, the deeper reason
is that **the guard's parser is not the requester's parser**. `urlsplit()` returns
`%31%32%37%2e%30%2e%30%2e%31` while urllib connects to `127.0.0.1:8000` ([F2]), because
CPython documents *"Percent-encoded sequences are not decoded"*
(docs.python.org/3/library/urllib.parse.html). Even a *perfect* canonicaliser applied to
`.hostname` fails on that input. Stenberg: *"if you parse a URL with parser A ... and then
pass the exact same URL to parser B ... it opens up ... for downright security
vulnerabilities"* (daniel.haxx.se). CPython itself warns: *"`urlsplit()` does not perform
validation ... code defensively ... Is there anything strange about that `hostname`?"*

**2. The population is open, and one measurement round proved it.** Known spellings went
**8 → 11** in a single pass: `0x7f000001` (hex), `017700000001` (octal — *named verbatim
by PortSwigger*), `%31%32%37%2e%30%2e%30%2e%31` (percent-encoded). CWE-184: *"Do not rely
exclusively on detecting disallowed inputs. There are too many variants to encode a
character ... high likelihood of missing some variants."*

**3. The defect's CWE identity is CWE-184 + CWE-180, and the polarity is the trap.**
`_LOOPBACK_HOSTS` *looks* like an allowlist but functions as a **denylist** — the
permitted set is "everything else", which is unbounded. And validation happens **before**
canonicalisation (`getaddrinfo` runs later, inside the requester) — textbook CWE-180:
*"Inputs should be decoded and canonicalized ... before being validated."*

**4. The stated latency objection dissolves at the right layer — measured.**
`socket.getaddrinfo` genuinely has **no timeout parameter** (confirmed against the CPython
signature) and worst case measured **5003.3 ms** on an unresolvable `.local` name. But at
the socket layer the DNS **has already happened**: the `socket.connect` audit event
carries `('127.0.0.1', 8000)` for `127.1`, `2130706433` and `0x7f000001` alike ([D1]).
Guard cost = a tuple unpack + an `AI_NUMERICHOST` parse (**0.016-0.123 ms, no network**)
+ a set lookup. **The guard never calls DNS at all.**

**5. Resolve-and-compare on the URL string has a CVE. The socket layer does not.**
CVE-2025-31490 (AutoGPT, CVSS 7.5): code that already resolved with `getaddrinfo` and
range-checked every result was **still** SSRF-able, because the requester resolved a
second time. At the socket layer there is no second resolution — the address checked *is*
the address about to be connected. **TOCTOU window = zero.**

**6. Port-only keying is safe in this repo but breaks one existing assertion.**
MEASURED: **nothing anywhere in the repo binds 8000** (the only non-ephemeral local
listener is `anthropic_max_bridge.py` on **18797**), and macOS's ephemeral range here is
**49152-65535**, so `bind(("127.0.0.1", 0))` can never return 8000. But
`test_phase_86_6_subprocess_channel.py:130` asserts `("https://example.com:8000", False)`
— right port, **wrong host, must be allowed**. So the invariant to key on is not the port
alone: it is the **destination socket** = (resolved address ∈ my addresses) **AND**
(port == 8000).

**7. The answer is (a)+(b)+(c) collapsed into one predicate, evaluated at the socket.**
Not three competing options — the sound design satisfies all three simultaneously:
compare against *this machine's own addresses* (a), key on the *destination socket*
rather than the host string (b), and default to **refuse-unless-provably-remote** on port
8000 (c) so an unanticipated address family — NAT64 included — becomes a false positive
instead of a bypass.

**8. The two-predicate drift alarm is structurally incapable of catching this, and no URL
table can fix it.** Both predicates return `False` on all 11 bypasses: they **agree and
are both wrong**. This is a common-mode fault — two implementations of the *same flawed
specification*. Equality oracles detect divergence only. Stenberg/Snyk's remedy ("single
parsing point", *"use as few different parsers as possible"*) says the fix is to delete
one of the two copies, not to compare them.

## Internal code inventory (all claims file:line anchored)

| File | Lines | Role | Status |
|---|---|---|---|
| `conftest.py` | 81 | `_LOOPBACK_HOSTS` 4-string set | **DEFECTIVE** — 11 measured bypasses |
| `conftest.py` | 125-133 | `_is_live_backend()` string predicate | **DEFECTIVE**; also fails OPEN on `ValueError` |
| `conftest.py` | 151-164 | `_guarded_urlopen` — stdlib seam | Sound seam, wrong predicate |
| `conftest.py` | 174-193 | urllib3 leg — keys on `self.host` (a *connection-pool* host string) | Same defect, second copy |
| `conftest.py` | 238-244 | `LiveStateWriteRefused(BaseException)` | **Correct precedent to reuse** — an `Exception` is swallowed by `kill_switch._append_audit` |
| `conftest.py` | 317-364 | PEP-578 `open`-event hook + `addaudithook` | **The seam to extend** — in-repo idiom already present |
| `scripts/qa/live_backend_origin.py` | 39,45-63 | 2nd copy of the predicate (`.lower()` is a no-op) | Same defect; **child-process** guard |
| `backend/tests/test_phase_86_6_subprocess_channel.py` | 120-136 | Frozen behaviour table incl. `example.com:8000 → False` | **Binding constraint** |
| `backend/tests/test_phase_86_6_subprocess_channel.py` | 139-175 | `..._predicates_AGREE` drift alarm | **Structurally blind** to common-mode fault |
| `backend/tests/test_phase_4000_2_cc_rail_smoke.py` | 176 | `ThreadingHTTPServer(("127.0.0.1", 0))` | **22 collected tests** — must stay allowed |
| `backend/tests/test_phase_76_9_2_max_bridge.py` | 130,137,347 | `_free_port()` + 2 stubs | **12 collected** — must stay allowed |
| `backend/tests/test_phase_86_3_live_egress_guard.py` | 271-274 | Stub + `assert port != 8000` | 12 collected; the assert is vacuous (kernel range) |
| `scripts/qa/smoke_cc_rail_e2e.py` | 468,482-525 | Child-process guard; imports the flawed predicate | Inherits the defect **outside** any audit hook |
| `scripts/ops/anthropic_max_bridge.py` | 43,57,179 | `127.0.0.1:18797` (loopback-only) | Not 8000 — irrelevant to port keying |
| `backend/main.py` | 531-532,552,601 | `_TAILSCALE_ORIGIN_RE` CORS allowlist | Same shape, **fail-CLOSED** — not this defect |
| `backend/api/auth.py` | 150-153 | `DEV_LOCALHOST_BYPASS` on `request.client.host` | **The SOUND pattern** — keys on the socket peer |
| `backend/api/auth.py` | 152 | `"localhost"` in the peer-address tuple | **DEAD** — a peer address is never a hostname |
| `scripts/qa/mutation_matrix_86_6.py` | 65 | Mutates the `addaudithook` install line | Existing mutation harness for this seam |
| `scripts/start_services.sh` | 52 | `uvicorn --host 0.0.0.0 --port 8000` | The wildcard bind (`lsof`: `TCP *:8000`) |
| `.claude/rules/security.md` | CORS/auth | Doc says bypass is "client-is-127.0.0.1" | Minor drift vs the 3-tuple at auth.py:152 |

## Consensus vs debate (external)

**Consensus (unanimous across OWASP, PortSwigger, CWE-184, CWE-180, Stenberg, Snyk,
Claroty, Oligo):** never make a security decision on an un-canonicalised host string;
canonicalise first, then validate; minimise the number of parsers; a list of forbidden
spellings for an open-ended population is an anti-pattern.

**Genuine debate — where to canonicalise.** OWASP/PortSwigger/the 2026 CVE fixes say
*resolve the name and check the resolved IP*. AutoGPT's CVE-2025-31490 and the DNS-
rebinding literature say *that is not enough — you must also connect to the address you
checked* (DNS pinning). The AutoGPT fix pins by rewriting the URL and moving the hostname
to the `Host` header — heavy, and inappropriate for a test guard. **The socket-layer check
is the third position, and it dominates both for this use case:** it is post-
canonicalisation *by construction* and pin-free because there is no later resolution.
Prior art exists (`thesis/sockfilter`, `pytest-socket`) but **both key on an ambiguous
`.host`** and pytest-socket has **no port dimension at all** — so neither is adoptable
off the shelf here.

## Pitfalls (from the literature + measured)

1. **`ipaddress.ip_address()`'s `except ValueError: return False` IS the bypass.** It
   rejects `127.1`/`0`/`2130706433`/`0x7f000001`/`017700000001`; the obvious except-clause
   maps exactly the dangerous set to "allow". ([C2])
2. **`0.0.0.0` is `is_unspecified`, NOT `is_loopback`.** A loopback-only test silently
   re-opens the hole 86.6 just closed — and Oligo shows the whole industry made this exact
   omission for 18 years.
3. **`::1` is in the list but does NOT reach the backend** (uvicorn binds IPv4 only,
   measured `ConnectionRefusedError`), while `::ffff:127.0.0.1` **does** and is not in the
   list. The list is wrong in both directions.
4. **A guard that raises must not raise an `Exception`** — `kill_switch._append_audit`'s
   `except Exception` swallows it (`conftest.py:210-215`). Reuse the `BaseException`
   precedent at `conftest.py:238`.
5. **Never call unrestricted `getaddrinfo` in the guard** — no timeout exists; 5003.3 ms
   measured on a `.local` miss. Use `AI_NUMERICHOST` (pure parser) or the already-resolved
   socket address.
6. **Don't let the predicate raise on junk** — `urlsplit("http://::1:8000/x").port` raises
   `ValueError`; both current predicates catch it and fail OPEN. Keep totality, but move
   the *default* to refuse for port 8000.
7. **The subprocess channel has no audit hook unless the child installs one.**
   `smoke_cc_rail_e2e.py` runs in a child that loads no conftest — it must install the
   same guard in its own `main()`, from the same shared module.
8. **Don't add a second predicate.** Two copies is the drift the 86.6 alarm was built for
   and cannot detect; Snyk's *"single parsing point"* says collapse to one.
9. **A drift/agreement test is not a correctness test.** Any new test must use an oracle
   independent of the spelling list — reachability or the resolved address — or it will
   pass a newly invented spelling exactly as today's does.

## Application to pyfinagent (external findings → file:line anchors)

**The recommended design — one predicate, evaluated on the destination socket.**
(Research only; Main owns PLAN. Presented as the evidence-supported direction, not a
contract.)

1. **ONE shared module** replaces both copies (`conftest.py:81,125` and
   `live_backend_origin.py:39,45`). Snyk *"single parsing point"* / Stenberg *"use a
   single parser"*. This deletes the common-mode-fault surface that
   `test_..._predicates_AGREE` (`test_phase_86_6_subprocess_channel.py:139`) cannot police.

2. **The predicate keys on `(address, port)`, never on a spelling.**
   `is_live_backend_socket(addr, port) := port == LIVE_BACKEND_PORT and _is_this_machine(addr)`
   where `_is_this_machine` = `ipaddress` classification (`is_loopback` **OR**
   `is_unspecified`, plus `.ipv4_mapped` unwrapping — measured correct on Python, unlike
   the Node/.NET CVEs) **union** this machine's interface addresses from
   `psutil.net_if_addrs()` (measured available; `netifaces` is **not** installed).
   Computed **once** at import — zero per-call cost.

3. **Install it at the PEP-578 `socket.connect` event**, alongside the existing hook at
   `conftest.py:364`. Measured ([D1]): the event carries the **resolved** address for
   every spelling, fires **before** the connect ([D2]), and aborts it when the hook raises.
   Raise a `BaseException` subclass, reusing the `LiveStateWriteRefused` precedent at
   `conftest.py:238` — an `Exception` is swallowed (`conftest.py:210-215`).

4. **Canonicalise the hook's `address[0]` with `AI_NUMERICHOST`** for the raw-hostname
   case measured in [D3] (`socket.connect(("127.1", 9))` hands the hook the raw string).
   0.016-0.123 ms, **no network**. This is CWE-180 satisfied: canonicalise, *then* validate.

5. **Keep the verb gate exactly as it is.** GETs must still pass —
   `conftest.py:23-31` constraint 1 is real: `_backend_is_up()` runs inside a
   `@pytest.mark.skipif` at module import, so a raised GET breaks collection of
   `test_phase_23_2_4_audit_log_clean_transitions`. **CAVEAT the socket layer introduces:**
   `socket.connect` fires *before* the HTTP verb is on the wire, so the verb is **not
   visible at that event**. The design must therefore keep the existing verb-aware
   `urlopen`/urllib3 wrappers as the *policy* layer and use the socket hook as the
   *address-truth* layer — i.e. the wrappers ask the shared predicate, and the socket hook
   is the backstop for paths the wrappers do not cover (raw sockets, `httpx`/`httpcore`,
   both listed as known gaps at `conftest.py:52-58`). **This is the one place where the
   socket seam does not simply replace what exists**, and any contract should say so
   rather than imply a clean swap.

6. **Preserve every row of the frozen table** (`test_phase_86_6_subprocess_channel.py:120-136`).
   Under the proposed predicate: `example.com:8000` → public address → not this machine →
   **allowed** ✓; `127.0.0.1:59999` → local but wrong port → **allowed** ✓ (59999 ∈
   49152-65535, measured); `127.0.0.1:3000` → **allowed** ✓; `localhost` (port 80) →
   **allowed** ✓; `""`/`"not a url"` → total, no raise ✓.

7. **The child process must install the same guard.** `smoke_cc_rail_e2e.py:468` imports
   the predicate today; it should also install the socket hook in its own `main()` before
   `:512`, because a child loads no conftest (`live_backend_origin.py:3-8`).

8. **The mutation test that would actually prove this.** Not another URL table — an oracle
   independent of the list: generate a spelling **at test time** that appears in no source
   file (e.g. the integer/hex/octal/percent-encoded rendering of `127.0.0.1`, or a
   `getaddrinfo`-derived local address), assert it is refused, and assert
   `grep -c '<that spelling>'` over the repo is **0**. That is the only test shape that
   can distinguish "the guard understands addresses" from "someone added another string".

9. **Sibling sites — the CLASS answer.** `backend/main.py:531` (`_TAILSCALE_ORIGIN_RE`)
   shares the *shape* but fails **closed** (a missed spelling denies a legitimate origin);
   it is a usability risk, not this defect — **do not "fix" it in this step**.
   `backend/api/auth.py:150-153` is the **sound** pattern already (keys on the socket peer
   address); its only flaw is the dead `"localhost"` element at `:152`, worth a one-line
   note but not a behaviour change. **No third instance of the fail-open defect was found**
   after sweeping every `bind`, every `HTTPServer`, every `8000` reference, and both
   documented allowlists.

## Read in full (>=5 required; 19 read; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how |
|---|-----|----------|------|-------------|
| 1 | https://raw.githubusercontent.com/OWASP/CheatSheetSeries/refs/heads/master/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md | 2026-08-10 | official doc | WebFetch |
| 2 | https://docs.python.org/3/library/socket.html | 2026-08-10 | official doc | WebFetch |
| 3 | https://docs.python.org/3/library/ipaddress.html | 2026-08-10 | official doc | WebFetch |
| 4 | https://peps.python.org/pep-0578/ | 2026-08-10 | standard/PEP | WebFetch |
| 5 | https://portswigger.net/web-security/ssrf | 2026-08-10 | authoritative | WebFetch |
| 6 | https://claroty.com/team82/research/exploiting-url-parsing-confusion | 2026-08-10 | research | WebFetch |
| 7 | https://github.com/miketheman/pytest-socket | 2026-08-10 | tool doc | WebFetch |
| 8 | https://daniel.haxx.se/blog/2022/01/10/dont-mix-url-parsers/ | 2026-08-10 | maintainer blog | WebFetch |
| 9 | https://snyk.io/blog/url-confusion-vulnerabilities/ | 2026-08-10 | research | WebFetch |
| 10 | https://docs.python.org/3/library/urllib.parse.html | 2026-08-10 | official doc | WebFetch |
| 11 | https://aydinnyunus.github.io/2026/03/14/ssrf-dns-rebinding-vulnerability/ | 2026-08-10 | CVE writeup | WebFetch |
| 12 | https://github.com/arkadiyt/ssrf_filter | 2026-08-10 | tool doc | WebFetch |
| 13 | https://cwe.mitre.org/data/definitions/184.html | 2026-08-10 | standard (MITRE) | WebFetch |
| 14 | https://cwe.mitre.org/data/definitions/180.html | 2026-08-10 | standard (MITRE) | WebFetch |
| 15 | https://github.com/python/cpython/issues/146245 | 2026-08-10 | upstream issue **[ADVERSARIAL]** | WebFetch |
| 16 | https://github.com/thesis/sockfilter | 2026-08-10 | tool doc | WebFetch |
| 17 | https://www.oligo.security/blog/0-0-0-0-day-exploiting-localhost-apis-from-the-browser | 2026-08-10 | research | WebFetch |
| 18 | https://advisories.gitlab.com/npm/auth-fetch-mcp/CVE-2026-49857/ | 2026-08-10 | CVE advisory | WebFetch |
| 19 | https://github.com/Significant-Gravitas/AutoGPT/security/advisories/GHSA-wvjg-9879-3m7w | 2026-08-10 | CVE advisory **[ADVERSARIAL]** | WebFetch |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://hackerone.com/reports/3634400 | NAT64 bypass report | **Attempted twice** — WebFetch returned only "HackerOne"; `curl` returned 3823 bytes of "enable JavaScript". JS-rendered, unreadable. |
| https://www.thehackerwire.com/esm-sh-ssrf-via-dns-alias-bypass-cve-2026-27730/ | CVE-2026-27730 | **Attempted** — HTTP 403 Forbidden |
| https://blackhat.com/docs/us-17/thursday/us-17-Tsai-A-New-Era-Of-SSRF-Exploiting-URL-Parser-In-Trending-Programming-Languages.pdf | Orange Tsai slides | Binary slide-deck PDF; content covered in full via Claroty + Snyk + Stenberg, which all cite it |
| https://advisories.gitlab.com/npm/ip-address/CVE-2026-54272/ | CVE advisory | Same class as #18, already read in full |
| https://github.com/coturn/coturn/security/advisories/GHSA-j8mm-mpf8-gvjg | advisory | Duplicate mechanism (`::ffff:` vs `denied-peer-ip`) |
| https://github.com/dotnet/runtime/issues/28740 | upstream issue | Cross-language contrast only |
| https://cheatsheetseries.owasp.org/.../Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html | rendered OWASP | Read the raw markdown instead (#1) |
| https://www.rfc-editor.org/rfc/rfc6890.html | RFC | Registry values obtained; not design-deciding |
| https://cwe.mitre.org/data/definitions/692.html | CWE | Child of CWE-184, XSS-specific |
| https://github.com/urllib3/urllib3/issues/2757 | upstream issue | urllib3 audit-event gap; noted, not design-deciding |
| https://github.com/miketheman/pytest-socket/issues/412 | issue | DNS-resolution limitation; covered by #7 |
| https://dailycve.com/dssrf-js-...-CVE-2026-54729-...| CVE | resolver-fallback bypass, same class |
| https://www.sciencedirect.com/science/article/abs/pii/S0957417426016386 | paper (SSRFinder) | Paywalled; detection tooling, not design doctrine |
| https://github.com/TandoorRecipes/recipes/security/advisories/GHSA-j6xg-85mh-qqf7 | advisory | Same rebinding class as #19 |
| https://github.com/SolaceLabs/solace-agent-mesh/issues/1517 | issue | Same v4-mapped class as #18 |
| https://github.com/slab/safeurl-elixir | tool | Cross-language, same doctrine |
| https://gofastmcp.com/python-sdk/fastmcp-server-auth-ssrf | tool doc | Same doctrine |
| https://lwn.net/Articles/984838/ | coverage | Secondary coverage of #17 |
| https://bugs.python.org/issue32085 | CPython issue | Orange Tsai's Python report; superseded by #8/#9 |
| https://github.com/python/cpython/issues/117566 | CPython issue | ipaddress-related, not design-deciding |
| https://docs.pytest.org/en/stable/how-to/monkeypatch.html | official doc | Generic; no new mechanism |
| https://abseil.io/resources/swe-book/html/ch23.html | book chapter | Hermetic-testing generality |
| https://www.bleepingcomputer.com/news/security/18-year-old-security-flaw-in-firefox-and-chrome-exploited-in-attacks/ | news | Secondary coverage of #17 |
| https://thehackernews.com/2024/08/0000-day-18-year-old-browser.html | news | Secondary coverage of #17 |

**URLs collected: 43** (19 read in full + 24 snippet-only).

## Research Gate Checklist

Hard blockers:
- [x] **>=5 authoritative external sources READ IN FULL via WebFetch** — 19
- [x] **10+ unique URLs total** — 43
- [x] **Recency scan (last 2 years) performed + reported** — 7 findings, all material
- [x] **Full papers / pages read (not abstracts)** — every row of the read-in-full table
      is a full WebFetch; the 2 that could not be read are recorded as attempts in the
      snippet table, not counted
- [x] **file:line anchors for every internal claim** — see the inventory table

Adaptive coverage (audit-class):
- [x] Rounds run: **10**; last 2 consecutive rounds surfaced **0** new read-in-full
      findings → `coverage.dry = true` (K=2 satisfied)
  - R1-R6, R8: new findings each round (4/5/3/1/3/2/1)
  - **R7 DRY** (hermetic-testing search — generic tutorial tier, no new mechanism)
  - R9 DRY (RFC 6890 — snippet-level corroboration only)
  - **R10 DRY** (pytest audit-hook/monkeypatch patterns — community tier, restates PEP 578)
  - *(R9 + R10 are the two consecutive dry rounds; R8 was the last productive round)*

Soft checks:
- [x] Internal exploration covered every module in scope + the 3 requested measurements
- [x] Contradictions / consensus noted — incl. **2 sources read specifically to attack
      the recommendation** (gh-146245, CVE-2025-31490) and 1 unreadable adversarial
      (HackerOne #3634400, reported honestly as snippet-only)
- [x] All claims cited per-claim with URL + access date, or file:line, or a pasted
      measurement

### Honest limitations
1. **HackerOne #3634400 could not be read in full** (JS-rendered; 2 methods attempted).
   Its argument is reconstructed from the search snippet and is flagged as such. It is
   the single most adversarial source to option (a) and deserves a retry with a
   JS-capable fetch if Main wants it load-bearing.
2. The "my addresses" set computed at import is itself a mild TOCTOU (a DHCP change
   mid-session would not be seen). Fails **open** in that narrow case; the
   refuse-unless-provably-remote polarity is what covers it.
3. `gh-146245` leak was probed by an object-count delta (116/1000 vs a reported 657/1000),
   which is consistent with the fix being present in 3.14.4 but is **not** a definitive
   attribution.
4. `psutil` availability was measured on this box only; a CI image without it needs a
   fallback path.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 19,
  "snippet_only_sources": 24,
  "urls_collected": 43,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": true,
    "rounds": 10,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.27.md",
  "gate_passed": true
}
```
