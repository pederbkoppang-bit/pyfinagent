---
name: project-test-suite-live-egress-86-3
description: phase-86.3 research — the urlopen guard seam already exists (slack.com denylist only); no published tool has VERB granularity; skipif runs at collection so no fixture can cover it; the caller's live-host list was 4/8 false positives
metadata:
  type: project
---

Researching 86.3 (pytest suite POSTing pause/resume to the operator's live
kill switch), four things were non-obvious and cost real search time:

1. **The interception seam was already built and already in-path.**
   `backend/tests/conftest.py:47-61` monkeypatches `urllib.request.urlopen` at
   conftest IMPORT time — but denies exactly one host (`slack.com`), by design
   ("this is not a general network jail"). The offending module resolves
   `urlopen` by module-attribute lookup at call time, so the existing patch sits
   in its path. The fix is a predicate widening, not a new mechanism. Look for
   an existing guard before designing one.

2. **No published tool has HTTP-VERB granularity.** pytest-socket,
   pytest_network, pytest-recording, pytest-test-categories ADR-001, httpretty,
   responses, respx, MSW, Laravel `preventStrayRequests` — every one gates on
   HOST (or on LIBRARY), never on METHOD. When the required policy is
   "GET host:port yes, POST host:port no", the answer is necessarily
   hand-rolled. Confirmed across a 2024-2026 recency pass.

3. **`@pytest.mark.skipif(not _probe())` runs at COLLECTION time**, so no
   fixture — session-scoped included — can intercept it. Only an import-time
   conftest patch is early enough. And if the probe's `except` tuple doesn't
   include the guard's exception type, raising there turns the whole module into
   a COLLECTION ERROR, silently killing sibling tests the criteria require to
   stay green. Always read the probe's `except` clause before choosing what the
   guard raises.

4. **A grep for `localhost:8000` massively over-counts.** Of 8 modules handed to
   me as "reaches a live host", 4 were false positives: a URL in a docstring, a
   curl string passed as an ARGUMENT to a pure path-resolver under test, an AST
   analysis of `requests.get` calls in another file's source, and an in-process
   `TestClient`. One genuine live-host module was MISSING from the list, and it
   POSTs to its own ephemeral-port `127.0.0.1` stub server — so a host-level
   block would have broken it. Classify from executed CALL SITES
   (`grep -rn "urlopen("` then read the surroundings), state the rule, and check
   the port, not just the host.

**Why:** each of these would have produced a wrong or breaking fix design if
assumed rather than measured.

**How to apply:** on any "stop the tests touching prod" task — grep the
conftest tree for an existing egress patch first; check whether the policy needs
verb granularity (if so, no off-the-shelf plugin works); read decorator-level
probes for collection-time evaluation; and derive the live-host set from call
sites with a written-down rule. See [[feedback_measure_dont_assert_claims]] and
[[project_kill_switch_deadlock_85_6]].
