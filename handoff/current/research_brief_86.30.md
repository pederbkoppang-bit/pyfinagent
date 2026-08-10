# Research Brief -- phase-86.30

**Topic:** Fail-safe direction for a degraded-mode host-identity predicate --
when a program cannot enumerate its own network interfaces, should an address
of unknown provenance be treated as LOCAL (refuse) or REMOTE (allow)?

**Tier:** simple (caller-specified). **Audit-class:** NO -- `coverage` is
reported for information; `coverage.dry` is not required for this step.
**Date accessed for all sources:** 2026-08-10.
**Answer in one line:** treat it as **LOCAL / refuse**. The current
`return not ip.is_global` at `scripts/qa/live_backend_origin.py:186` does the
opposite for the exact address class a modern host actually carries, and this is
**measured on this machine**, not inferred.

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

1. **Year-less canonical** -- `Saltzer Schroeder fail-safe defaults protection of
   information in computer systems design principles`
2. **Current-year frontier (2026)** -- `fail closed vs fail open security control
   degraded mode default deny 2026`
3. **Last-2-year window** -- `CVE-2024-4032 Python ipaddress is_global is_private
   incorrect globality private address ranges` and `IPv6 privacy extensions
   temporary addresses rotation host holds multiple global unicast addresses 2025`

The read-in-full table below mixes year-less canonical (Saltzer 1975, RFC 4291
1998/2006, RFC 4193 2005, RFC 7136 2014), standing standards (RFC 8981 2021),
and last-2-year hits (CVE-2024-4032 vendor record; Python 3.13 `is_private`
changelog).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.cs.virginia.edu/~evans/cs551/saltzer/ | 2026-08-10 | paper (canonical, 1975) | WebFetch, full | Fail-safe defaults: *"Base access decisions on permission rather than exclusion... the default situation is lack of access."* And the **direction** argument, which is the whole of this step: *"A design or implementation mistake in a mechanism that gives explicit permission tends to fail by refusing permission, a safe situation, since it will be quickly detected. On the other hand, a design or implementation mistake in a mechanism that explicitly excludes access tends to fail by allowing access, a failure which may go unnoticed in normal use."* Also *"Every access to every object must be checked for authority"* (complete mediation). |
| https://cwe.mitre.org/data/definitions/636.html | 2026-08-10 | official doc (MITRE CWE) | WebFetch, full | CWE-636 *Not Failing Securely ('Failing Open')*: *"When the product encounters an error condition or failure, its design requires it to fall back to a state that is less secure than other options that are available"*; *"By entering a less secure state, the product inherits the weaknesses associated with that state, making it easier to compromise."* Consequence: **Bypass Protection Mechanism** (scope: Access Control). Root cause named as preferring *"fail functional"* over *"fail safe"*. Explicitly traced to Saltzer & Schroeder. |
| https://www.rfc-editor.org/rfc/rfc4291.html | 2026-08-10 | standard (IETF, IPv6 addressing architecture) | WebFetch, full | *"A single interface may also have multiple IPv6 addresses of any type (unicast, anycast, and multicast) or scope."* §2.8 (A Node's Required Addresses): *"A host is required to recognize the following addresses as identifying itself: its required Link-Local address for each interface; **any additional Unicast and Anycast addresses configured for interfaces**; the loopback address; ..."* -- the host's own identity set is defined to **include its global unicast addresses**, not merely loopback. |
| https://www.rfc-editor.org/rfc/rfc8981.html | 2026-08-10 | standard (IETF, temporary addresses) | WebFetch, full | Temporary addresses are *"global-scope addresses"*; hosts *"SHOULD generate new temporary addresses over time"*; `TEMP_PREFERRED_LIFETIME` *"Default value: 1 day"*, `TEMP_VALID_LIFETIME` *"Default value: 2 days"*; *"at most one temporary address per prefix should be in a nondeprecated state at any given time"* but valid (still-bound) ones overlap. **So the rotating addresses are precisely the `is_global == True` ones, and any cached enumeration is stale by design on a ~1-day clock.** |
| https://docs.python.org/3/library/ipaddress.html | 2026-08-10 | official doc (CPython) | WebFetch, full | `is_global` = *"True if the address is defined as globally reachable by iana-ipv4-special-registry (for IPv4) or iana-ipv6-special-registry (for IPv6)"*. It is a statement about **routability in the global registry**, never about **ownership by this host**. `is_global` is the near-exact complement of `is_private` (*"has value opposite to is_private, except for the shared address space (100.64.0.0/10) where they are both False"*). *"Changed in version 3.13: Fixed some false positives and false negatives"* -- the registry-derived classification is a **moving target**. |
| https://www.rfc-editor.org/rfc/rfc4193.html | 2026-08-10 | standard (IETF, ULA) | WebFetch, full | `FC00::/7`. *"By default, the scope of these addresses is global... Their limitation is in the routability of the prefixes, which is limited to a site."* *"These addresses are not expected to be routable on the global Internet."* -- i.e. **IPv6 scope and "globally reachable" are different axes**, so a single boolean cannot carry both, and neither carries "mine". |
| https://www.rfc-editor.org/rfc/rfc7136.html | 2026-08-10 | standard (IETF, IID semantics) | WebFetch, full | *"the bits in an interface identifier have no meaning and that the entire identifier should be treated as an opaque value"*; *"the whole IID value MUST be viewed as an opaque bit string by third parties"*; *"no reliable deductions can be made from the state of the 'u' and 'g' bits."* The general rule: **you may not infer a property of an address from the bits of the address.** Ownership is such a property. |
| https://ubuntu.com/security/CVE-2024-4032 | 2026-08-10 | vendor security advisory (recency) | WebFetch, full | Verbatim: *"The 'ipaddress' module contained incorrect information about whether certain IPv4 and IPv6 addresses were designated as 'globally reachable' or 'private'. This affected the is_private and is_global properties... where values wouldn't be returned in accordance with the latest information from the IANA Special-Purpose Address Registries."* CVSS 3.0 **7.5 High** (Ubuntu priority Low). Fixed in CPython 3.12.4 / 3.13.0a6; upstream PR python/cpython#113179. **`is_global` has itself been a CVE'd security-decision input.** |
| https://authzed.com/blog/fail-open | 2026-08-10 | engineering blog (authorization) | WebFetch, full | *"A fail-open state can inadvertently grant access to unauthorized users during unexpected failures, posing significant security risks"*; *"a fail-closed state ensures that, in the event of an error, access is denied."* Also the honest cost, which bears on how to WRITE the branch: *"Fail-closed code can end up awful to read: sometimes so awful that you might be more likely to write bugs because it's too hard to read."* (Dated 2021-01-16 -- canonical, not recency.) |

**9 sources read in full via WebFetch.** One further attempt failed and is
recorded honestly: `https://nvd.nist.gov/vuln/detail/CVE-2024-4032` returned
**HTTP 502 Bad Gateway**; the Ubuntu vendor record was substituted and read in
full instead.

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://nvd.nist.gov/vuln/detail/CVE-2024-4032 | official (NIST) | **Attempted, HTTP 502**; substituted by the Ubuntu record |
| https://handwiki.org/wiki/Saltzer_and_Schroeder's_design_principles | encyclopedia | Secondary to a primary already read in full |
| https://www.researchgate.net/publication/260635314_A_Contemporary_Look_at_Saltzer_and_Schroeder's_1975_Design_Principles | paper | Paywalled/JS-gated |
| https://www.opensecurityarchitecture.org/foundations/design-principles/ | industry | Tertiary restatement |
| https://nocomplexity.com/documents/securityarchitecture/architecture/saltzer_designprinciples.html | blog | Tertiary restatement |
| https://cydrill.com/cyber-security/secure-design-principles/ | blog | Community tier |
| https://www.jeremyjordan.me/security-design/ | blog | Community tier |
| https://www.ojp.gov/ncjrs/virtual-library/abstracts/protection-information-computer-systems | abstract | Abstract only -- would not count as read-in-full by rule |
| https://security.snyk.io/vuln/SNYK-AMZN2-PYTHONIPADDRESS-8747268 | vuln DB | Duplicate of CVE-2024-4032 |
| https://www.miggo.io/vulnerability-database/cve/CVE-2024-4032 | vuln DB | Duplicate |
| https://www.suse.com/security/cve/CVE-2024-4032.html | vendor | Duplicate |
| https://www.cvedetails.com/cve/CVE-2024-4032/ | vuln DB | Duplicate |
| https://vulert.com/vuln-db/CVE-2024-4032 | vuln DB | Duplicate |
| https://www.cybersecurity-help.cz/vdb/SB2024080203 | vuln DB | Duplicate |
| https://explore.alas.aws.amazon.com/CVE-2024-4032.html | vendor | Duplicate |
| https://vigilance.fr/vulnerability/Python-ipaddress-ingress-filtrering-bypass-via-is-private-is-global-44530 | vuln DB | Duplicate; paywalled |
| https://socradar.io/labs/app/cve-radar/CVE-2024-4032 | vuln DB | Duplicate |
| https://www.ituonline.com/comptia-securityx/comptia-securityx-4/mitigations-implementing-fail-secure-and-fail-safe-strategies-for-robust-security/ | training | Community tier |
| https://trainingcamp.com/glossary/fail-open/ | glossary | Community tier |
| https://1seal.org/protocols/ | blog | Community tier, off-domain |
| https://www.deepinspect.ai/blog/ai-gateway-fail-open-vs-fail-closed | vendor blog | Community tier; useful "fail-partial" framing only |
| https://devsecopsschool.com/blog/fail-closed/ | blog | Community tier |
| https://redeagle.tech/eaglepedia/fail-open-vs-fail-closed | wiki | Community tier |
| https://arxiv.org/pdf/2510.11837 | preprint | Off-topic (LLM security architecture); `/pdf/` URL -- not fetched per the arXiv rule |
| https://datatracker.ietf.org/doc/rfc8981/ | standard | Same document as the rfc-editor copy read in full |
| https://www.rfc-editor.org/rfc/rfc4941 | standard | **Obsoleted by RFC 8981** (read in full instead) |
| https://datatracker.ietf.org/doc/html/rfc4941 | standard | Duplicate of the above |
| https://datatracker.ietf.org/doc/rfc3041/ | standard | Obsoleted twice over |
| https://tools.ietf.org/html/draft-ietf-6man-rfc4941bis-06 | draft | Superseded by the published RFC 8981 |
| https://arxiv.org/pdf/2102.00542 | preprint | `/pdf/` URL; prefix-rotation attack paper, tangential |
| https://oneuptime.com/blog/post/2026-03-20-ipv6-privacy-extensions-android/view | blog (2026) | Community tier; corroborates rotation only |
| https://sid-500.com/2017/11/05/discovering-temporary-ipv6-global-unicast-addresses-privacy-extensions-with-powershell/ | blog | Community tier |
| https://ipaddresslocation.net/articles/ipv6-privacy-extensions-what-they-are-and-how-to-enable | blog | Community tier |

**Total unique URLs collected: 42** (9 read in full + 33 snippet-only).

---

## Recency scan (2024-2026)

Performed. Searched the 2024-2026 window on three axes (`...2026` fail-open /
fail-closed; `CVE-2024-4032` Python `ipaddress`; `2025` IPv6 temporary-address
rotation). **Two findings that COMPLEMENT rather than supersede the canonical
sources, and one that is directly load-bearing:**

1. **[LOAD-BEARING, 2024] CVE-2024-4032** -- `ipaddress.is_global` / `is_private`
   returned answers not matching the IANA Special-Purpose Address Registries;
   CVSS 3.0 7.5; fixed in CPython 3.12.4 / 3.13.0a6
   (https://ubuntu.com/security/CVE-2024-4032). Python 3.13 then *"Fixed some
   false positives and false negatives"* again, moving `192.0.0.0/24`,
   `64:ff9b:1::/48` and `2002::/16` into `is_private`
   (https://docs.python.org/3/library/ipaddress.html). **Consequence for this
   step: `is_global` is a registry-tracking value that has changed under
   security pressure twice in two years. It is not a stable predicate to hang a
   safety decision on, independent of the semantic mismatch below.** This
   machine runs Python 3.14.4, so it carries the fixed table -- the point is the
   class of dependency, not a present defect.
2. **[2026] No new standards work supersedes RFC 8981.** RFC 4941 and RFC 3041
   are obsoleted BY it; `draft-ietf-6man-rfc4941bis` is what became RFC 8981.
   Rotation behaviour is unchanged and now default-on across modern OSes
   (Android 8.0+ stable-privacy SLAAC + temporary addresses;
   https://oneuptime.com/blog/post/2026-03-20-ipv6-privacy-extensions-android/view).
3. **[2026] No new consensus against fail-closed for authorization.** The
   2026-dated frontier material restates the same rule (deny when trust cannot
   be established) and adds only one refinement worth carrying: a *fail-partial*
   pattern, and the instruction that **"the degraded path needs to be loud.
   Alert on it, count it, and make the number visible"**
   (https://www.deepinspect.ai/blog/ai-gateway-fail-open-vs-fail-closed,
   community tier -- recorded as a design hint, not as authority).

---

## Key findings

1. **The degraded branch must fail toward REFUSE, and this is the single
   oldest rule in the field.** *"A design or implementation mistake in a
   mechanism that explicitly excludes access tends to fail by allowing access, a
   failure which may go unnoticed in normal use"* (Saltzer & Schroeder 1975,
   https://www.cs.virginia.edu/~evans/cs551/saltzer/). CWE-636 names the
   opposite choice as a weakness with the consequence *Bypass Protection
   Mechanism* (https://cwe.mitre.org/data/definitions/636.html). **Answer to the
   objective's headline question: unknown provenance -> treat as LOCAL ->
   refuse.**

2. **`not ip.is_global` answers a different question than the one asked.**
   `is_global` is defined as *"globally reachable by iana-ipv4-special-registry
   ... or iana-ipv6-special-registry"*
   (https://docs.python.org/3/library/ipaddress.html) -- a **routability**
   property of the address in the global registry. `_is_this_machine` asks an
   **ownership** question: is this address bound to an interface on this host?
   The two are orthogonal: RFC 4291 §2.8 requires a host to recognise *"any
   additional Unicast ... addresses configured for interfaces"* as identifying
   itself, and those are ordinarily global unicast
   (https://www.rfc-editor.org/rfc/rfc4291.html). RFC 7136 states the general
   prohibition: *"no reliable deductions can be made"* from an address's bits
   (https://www.rfc-editor.org/rfc/rfc7136.html). RFC 4193 shows the two axes
   are genuinely independent -- ULAs have *"global"* scope yet are *"not
   expected to be routable on the global Internet"*
   (https://www.rfc-editor.org/rfc/rfc4193.html).

3. **The inversion hits exactly the addresses a SLAAC host carries, and it is
   MEASURED here, not argued.** RFC 8981 temporary addresses are *"global-scope
   addresses"* that rotate on a `TEMP_PREFERRED_LIFETIME` of *"1 day"* /
   `TEMP_VALID_LIFETIME` of *"2 days"*
   (https://www.rfc-editor.org/rfc/rfc8981.html) -- so the rotating set is the
   `is_global == True` set. On this machine, 2026-08-10, offline enumeration
   over `psutil.net_if_addrs()` returns **17 own addresses, of which 6 have
   `is_global == True`** -- six `2001:4654:6451:0:*` GUAs on `en1`, the
   signature of one stable SLAAC address plus accumulated non-deprecated
   temporaries. For all six, `not ip.is_global` is `False`, so the degraded
   branch answers *"not this machine" -> ALLOW*. **The branch's own docstring
   at `scripts/qa/live_backend_origin.py:181-186` says "Over-refusal, which is
   the safe direction". For 6 of this host's 17 addresses it under-refuses.
   The comment states the intent; the expression does the opposite.**

4. **`is_global` is additionally an unstable input.** CVE-2024-4032 (CVSS 7.5)
   was exactly "this property disagreed with the registry"
   (https://ubuntu.com/security/CVE-2024-4032), and 3.13 changed the answer
   again. A safety branch keyed on it inherits a moving classification.

5. **Documenting a degraded branch so its direction cannot be mis-stated.**
   The literature's shape, applied: (a) name the **direction** in the branch's
   first line, in the vocabulary of the decision (`refuse` / `allow`), not in
   the vocabulary of the implementation (`not is_global`); (b) state the
   **cost asymmetry** that justifies it (false-refuse = one test fails loudly;
   false-allow = a mutating request reaches the operator's live book, and per
   Saltzer *"may go unnoticed in normal use"*); (c) make the degraded path
   **loud and countable** rather than silent
   (https://www.deepinspect.ai/blog/ai-gateway-fail-open-vs-fail-closed);
   (d) **prove the direction with an executable cell**, because a comment that
   contradicts its expression is precisely what shipped here -- and the
   mutation matrix in this repo already has the machinery for that.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/qa/live_backend_origin.py` | 386 (read in full) | The single authority for "is this the live backend" | **CARRIES THE DEFECT** at `:181-186` |
| `scripts/qa/live_backend_origin.py:98-122` | `_enumerate_interface_addresses` | psutil-only enumerator; returns `None` on any failure | Correct; docstring at `:107-108` already flags psutil as transitive and undeclared |
| `scripts/qa/live_backend_origin.py:130-143` | `own_addresses` / `interfaces_enumerable` | Cached set + enumerability flag (module globals `_own_cache`, `_own_enumerable`, `_own_lock`) | Cache is refreshed on miss at `:178` -- correct given RFC 8981 rotation |
| `scripts/qa/live_backend_origin.py:146-188` | `_is_this_machine` | Ownership predicate | `:186` `return not ip.is_global` -- **the wrong-direction degraded branch** |
| `scripts/qa/live_backend_origin.py:199-223` | `_canonical_addresses` | `AI_NUMERICHOST`-first canonicalisation, no network for numerics | Correct |
| `scripts/qa/live_backend_origin.py:226-246` | `targets_this_machine` | **Already fails safe**: `verdict = True if addrs is None else ...` (`:243`) | Correct -- and is the in-file precedent for the fix |
| `scripts/qa/live_backend_origin.py:316-340` | `address_is_live_backend` | PEP-578 sockaddr decision; calls `_is_this_machine` at `:335` | Inherits the defect through `:335` |
| `scripts/qa/live_backend_origin.py:343-371` | `install_socket_guard` | `sys.addaudithook`; refuses only when `current_request_is_mutating()` | Correct |
| `conftest.py` | 430 | Sole in-process consumer | Imports the authority at `:87-95`; guard installed at `:220`; predicate used at `:204` and `:243` |
| `conftest.py:96-110` | import-failure fallback | **Degrades to PORT-ONLY refusal** -- *"That over-refuses ... which is the safe direction, and it is loud in the logs rather than silent"* | **The correct direction, in this same repo.** The two degraded paths disagree with each other -- `conftest` over-refuses, `_is_this_machine:186` under-refuses |
| `scripts/qa/mutation_matrix_86_27.py` | 179 (read in full) | Criterion-7 mutation harness | **Has SEVEN cells M1-M7, not M1-M3** (the caller's scope statement says M1-M3 -- correcting it). Hermetic: mutates a COPY in `tempfile.TemporaryDirectory`, loads it in a child (`:130-147`), asserts the tracked source is unchanged (`:160`) |
| `scripts/qa/mutation_matrix_86_27.py:41-103` | `MUTATIONS` list | Tuple shape `(id, anchor, replacement, probe_expr, description)` | A new cell is a single appended tuple. **Two documented traps to honour:** (i) anchor uniqueness is asserted at `:139-143` -- `original.count(anchor)` must be exactly 1; (ii) the probe must DISCRIMINATE -- see the `:54-61` and `:85-89` comments, where two first drafts survived because the control answer and the mutant's fail-safe answer coincided |
| `backend/tests/test_phase_86_27_live_origin_class.py` | 500 | The step's test module | **Contains ZERO references to `psutil`, `interfaces_enumerable`, `is_global` or "degraded" (grep, 0 hits). The degraded branch is entirely untested, and no M-cell targets it.** |
| `backend/tests/test_phase_86_6_subprocess_channel.py` | 189 | Child-process channel tests | Holds the frozen row `https://example.com:8000 -> allow` referenced in the module docstring at `:62-66` |
| `scripts/qa/smoke_cc_rail_e2e.py`, `scripts/qa/reproduce_86_27_spellings.py` | -- | Other referencing scripts (grep hits) | No degraded-branch dependency |
| `requirements.txt`, `backend/requirements.txt`, `functions/{ingestion,earnings,quant}/requirements.txt`, `scripts/autoresearch/requirements-autoresearch.txt` | 6 files | Dependency declarations | **`psutil` appears in NONE of them (grep, 0 hits).** Installed transitively at **psutil 7.2.2**. So the degraded branch is reachable by a routine `pip install -r` in a fresh venv -- it is not a theoretical path |

**MEASURED on this host, 2026-08-10 (offline; `psutil.net_if_addrs()` +
`ipaddress`, no network I/O):** 17 own addresses; 6 with `is_global == True`
(`2001:4654:6451:0:15d6:967b:7384:d0a`, `...:1cc2:d390:9d51:a7f1`,
`...:31:6467:1ea6:1852`, `...:5091:a4bd:92b5:4f19`, `...:c87f:2f22:e75e:85b9`,
`...:d4e8:c623:66d4:a689`, all on `en1`). Under the degraded branch all six
evaluate to `REMOTE(allow)`. Python 3.14.4.

**Prior-art anchor inside this repo:** the finding is not new -- Q/A recorded it
as note **N1** in `handoff/current/evaluator_critique_86.27.md:69`, including the
same measurement (*"this machine has SIX globally-routable IPv6 addresses"*) and
the same one-line remedy (*"in the `not interfaces_enumerable()` branch of
live_backend_origin.py:181-186, return True unconditionally"*). That note also
bounds materiality honestly and that bound should be carried into the contract:
psutil IS installed today, and `lsof -nP -iTCP:8000` showed uvicorn bound
IPv4-only, so no IPv6 spelling reaches the book **at present**. The defect is
**latent**, not live -- it arms the moment either condition changes (fresh venv
without psutil; or a dual-stack uvicorn bind).

---

## Consensus vs debate (external)

**Consensus, and it is close to unanimous** across 1975 (Saltzer & Schroeder),
2006-2026 (CWE-636), and the practitioner literature: an authorization/identity
check fails toward DENY. No source read argues for fail-open in an
authorization context.

**The only genuine debate** is availability-vs-security, and it does not reach
this decision: the fail-open case is argued only where *"core business revenue
is at stake AND [a] safe degraded mode exists"*. Here the cost of a false refuse
is one over-strict test-suite refusal on a developer machine; the cost of a
false allow is a mutating HTTP request reaching the operator's live trading
book. The asymmetry is not close.

**One real dissent worth recording** (AuthZed, https://authzed.com/blog/fail-open):
fail-closed code *"can end up awful to read: sometimes so awful that you might
be more likely to write bugs because it's too hard to read."* That is an
argument about **how to write** the branch, not which direction to choose --
and it favours `return True  # cannot prove remote` over any expression built
from `is_global`.

---

## Pitfalls (from literature)

1. **Silent failure.** Saltzer: an exclusion-mechanism mistake *"tends to fail by
   allowing access, a failure which may go unnoticed in normal use."* A degraded
   branch that under-refuses produces green tests. It looks like success.
2. **Fail-functional dressed as fail-safe.** CWE-636's stated root cause. The
   present branch is literally this: a comment asserting "over-refusal, which is
   the safe direction" above an expression that under-refuses.
3. **Inferring a property from an address's bits.** RFC 7136 forbids it in
   general; RFC 4193 shows scope != routability; RFC 4291 §2.8 shows ownership
   != either. Ownership is knowable only by enumeration -- which is exactly the
   capability that is missing in the degraded case, so no substitute predicate
   exists. That is the argument for an unconditional constant.
4. **Trusting a registry-derived boolean.** CVE-2024-4032 + the 3.13 changes.
5. **A cached address set is stale by design.** RFC 8981's 1-day/2-day
   lifetimes. `own_addresses(refresh=True)` at `:178` already concedes this.
6. **A mutation cell whose probe cannot discriminate.** Documented twice inside
   `mutation_matrix_86_27.py` itself (`:54-61`, `:85-89`): a mutant survives when
   the control answer and the mutant's fail-safe answer coincide. **A cell for a
   fix that returns `True` unconditionally must be probed with an address whose
   CONTROL answer is `True` for a reason the mutant destroys -- i.e. probe a
   global address of THIS machine with enumeration forced off, where control =
   True (post-fix) and mutant (`not ip.is_global`) = False.**

---

## Application to pyfinagent

- **The change is one expression** at `scripts/qa/live_backend_origin.py:186`:
  `return not ip.is_global` -> `return True`, with the comment rewritten to state
  the direction in decision vocabulary ("cannot enumerate -> cannot prove remote
  -> treat as this machine -> refuse") plus the cost asymmetry. This is exactly
  the remedy Q/A proposed at `handoff/current/evaluator_critique_86.27.md:69`.
- **It makes the two degraded paths agree.** `conftest.py:96-110` already
  over-refuses on import failure and says so; `targets_this_machine` at
  `live_backend_origin.py:243` already returns `True` when it cannot
  canonicalise. `_is_this_machine:186` is the one dissenter. The general pattern
  from the objective -- *a guard's degraded branch must err in the same
  direction as its primary branch* -- is already honoured twice in this file and
  broken once.
- **Blast radius is bounded and should be stated, not hand-waved.** The branch
  is only reached when `interfaces_enumerable()` is False, i.e. psutil absent or
  raising. psutil 7.2.2 is installed but declared in **no** requirements file, so
  the reachable population is fresh/CI venvs. In that population the change makes
  `_is_this_machine` return `True` for every parseable address, so
  `address_is_live_backend` refuses **every** mutating request on port 8000 --
  including `https://example.com:8000`, the frozen row in
  `test_phase_86_6_subprocess_channel.py`. **That row is graded and must be
  checked against the fix**; note the module docstring at `:62-66` already
  discusses this exact trade and chose to preserve the row. Either the test must
  pin psutil-present, or the row must be re-derived. This is the one place the
  contract needs a decision rather than a copy of this brief.
- **Cover it with an M8 cell** in `scripts/qa/mutation_matrix_86_27.py`
  (append one tuple to `MUTATIONS`, `:41-103`), honouring both documented traps:
  anchor must appear exactly once (asserted at `:139-143`), and the probe must
  force `interfaces_enumerable()` False AND use one of this host's own
  `is_global` addresses so control and mutant differ. Without such a cell the
  fix is untested -- today `test_phase_86_27_live_origin_class.py` has zero hits
  for `psutil` / `is_global` / `degraded`.
- **Do not "fix" it by widening the predicate.** No expression over the address
  can recover ownership without enumeration (RFC 7136 / RFC 4291 §2.8). This is
  the same lesson as 86.27's own docstring at `:32-43`: extending a list of
  spellings is not a fix for an open-ended population.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
      (1 paper, 1 MITRE CWE, 4 IETF RFCs, 1 CPython official doc, 1 vendor
      security advisory, 1 engineering blog). Hierarchy respected: 8 of 9 are
      tier 1-2.
- [x] 10+ unique URLs total -- **42** (9 full + 33 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- see section above;
      CVE-2024-4032 is a load-bearing 2024 finding
- [x] Full pages read (not abstracts) for the read-in-full set. One failure
      disclosed: NVD returned HTTP 502 and was replaced, not counted.
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope, plus the
      `psutil`-declaration question (answered: absent from all 6 requirements
      files; installed 7.2.2 transitively)
- [x] Contradictions / consensus noted -- incl. the in-repo contradiction
      between two degraded paths, and the AuthZed readability dissent
- [x] All claims cited per-claim with URL or file:line
- [x] **Two corrections to the caller's own framing, recorded rather than
      silently absorbed:** (1) `mutation_matrix_86_27.py` has cells **M1-M7**,
      not M1-M3; (2) the finding is **not new** -- it is Q/A note N1 at
      `handoff/current/evaluator_critique_86.27.md:69`, with the same
      measurement and the same proposed one-line remedy.

---

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 33,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 1,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.30.md",
  "gate_passed": true
}
```
