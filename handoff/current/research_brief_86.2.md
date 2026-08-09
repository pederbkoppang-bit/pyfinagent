# Research Brief -- phase-86.2

**Topic:** Fail-safe replay of a safety-critical audit log: a single malformed
row (`OverflowError: int too large to convert to float`) aborts
`kill_switch._load_from_audit`, stranding EVERY protective limit (total disarm).

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage is
informational only for this step).

**Status:** COMPLETE -- gate PASSED (9 sources read in full, 39 URLs, recency
scan performed). Written incrementally under write-first discipline.
**Read-only session:** no production code, tests, or config were modified.

---

## Search queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| 1 | `event sourcing poison pill corrupt record replay skip or abort dead letter queue` | year-less canonical |
| 2 | `IEC 61508 fail-safe state corrupt data safety-critical software error handling 2025` | last-2-year |
| 3 | `Python float() OverflowError int too large to convert to float exception hierarchy ArithmeticError` | year-less canonical |
| 4 | `2025 2026 event sourcing replay corrupt event fail-safe state reconstruction resilience research` | current-year frontier |
| 5 | `"graceful degradation" OR "fail-safe" safety instrumented system bypass corrupt input IEC 61511 selective bypass` | year-less canonical |
| 6 | `Python 2026 robust numeric coercion catch OverflowError ValueError TypeError decimal InvalidOperation numpy overflow best practice` | current-year frontier |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://event-driven.io/en/rebuilding_read_models_skipping_events/ | 2026-08-09 | blog (Oskar Dudycz, event-sourcing authority) | WebFetch, full | Skipping is acceptable ONLY if the skip is RECORDED: "If we can't prevent skips from happening, let's make them visible... it could record that it skipped in the same transaction as the event append." And the warning: "A DLQ is only helpful if it's monitored, processed, and understood why messages end up there. Otherwise, it's just a fancy way to lose data slowly rather than immediately." |
| 2 | https://docs.python.org/3/library/exceptions.html | 2026-08-09 | official docs (CPython) | WebFetch, full | Hierarchy tree confirms `Exception -> ArithmeticError -> {FloatingPointError, OverflowError, ZeroDivisionError}` -- `OverflowError` shares NO ancestor with `TypeError`/`ValueError` below `Exception`. Verbatim: "Raised when the result of an arithmetic operation is too large to be represented. This cannot occur for integers (which would rather raise MemoryError than give up). However, for historical reasons, OverflowError is sometimes raised for integers that are outside a required range." |
| 3 | https://cwe.mitre.org/data/definitions/396.html | 2026-08-09 | official taxonomy (MITRE CWE) | WebFetch, full | CWE-396 *Declaration of Catch for Generic Exception*, and Python is explicitly listed as an applicable language. Verbatim: "'condensing' catch blocks by catching a high-level class like Exception can obscure exceptions that deserve special treatment or that should not be caught at this point in the program." Consequence class: **Hide Activities**. Parents: CWE-755, CWE-705, CWE-221. |
| 4 | https://www.postgresql.org/docs/current/runtime-config-developer.html | 2026-08-09 | official docs (PostgreSQL) | WebFetch, full | **[ADVERSARIAL to naive skip-and-continue]** The DEFAULT for redo/WAL recovery is ABORT, not skip: `ignore_invalid_pages` -- "If set to off (the default), detection of WAL records having references to invalid pages during recovery causes PostgreSQL to raise a PANIC-level error, aborting the recovery." Skipping is opt-in, superuser-gated, **server-start-only**, and carries the verbatim warning "This behavior may cause crashes, data loss, propagate or hide corruption, or other serious problems." Same shape for `zero_damaged_pages` ("This behavior will destroy data") and `ignore_checksum_failure`. |
| 5 | https://www.confluent.io/blog/kafka-connect-deep-dive-error-handling-dead-letter-queues/ | 2026-08-09 | vendor engineering blog (Confluent/Kafka) | WebFetch, full | `errors.tolerance = none` is the **default**: one bad message aborts the connector into FAILED. `= all` is an explicit opt-in, and the author warns it is silent by default: "When it does, by default it won't log the fact that messages are being dropped. If you do set errors.tolerance = all, make sure you've carefully thought through if and how you want to know about message failures that do occur." Recommended posture: "My starting point would always be the use of a dead letter queue and close monitoring of the available JMX metrics." Companion knobs `errors.log.enable` / `errors.log.include.messages`. |
| 6 | https://cwe.mitre.org/data/definitions/390.html | 2026-08-09 | official taxonomy (MITRE CWE) | WebFetch, full | CWE-390 *Detection of Error Condition Without Action*. Mitigation verbatim: "Properly handle each exception. This is the recommended solution. **Ensure that all exceptions are handled in such a way that you can be sure of the state of your system at any given moment.**" And: "If a function returns an error, it is important to either fix the problem and try again, alert the user that an error has happened and let the program continue, or alert the user and close and cleanup the program." Testing guidance explicitly names **mutation** testing. Consequence: "place the system in unexpected states". |
| 7 | https://www.sciopta.com/wp-content/uploads/2019/02/RTOSandIEC61508.pdf | 2026-08-09 | industry/standards paper (Mike Medoff, exida) | `curl` + `pdfplumber` (5 pp, 16 047 chars extracted; the primary IEC 61508-7 mirror returned HTTP 468) | IEC 61508 software-hazard-analysis method: for each failure mode "a measure must exist in the product to ensure that the failure is **safe**", and "A safe failure is defined as one where the outputs can be placed in the state that shuts down the process which is normally de-energized." His worked hazard table lists the "Corrupt ... parameters are corrupted" row with consequence "In the worst case a safety related shutdown will not occur in a timely manner" and reaction "Outputs set to failsafe state" -- i.e. corrupt safety *parameters* must drive the system TOWARD the trip, never away from it. |
| 8 | https://risknowlogy.com/articles/detail/17309/ | 2026-08-09 | industry (functional-safety consultancy) | WebFetch, full | Graceful degradation is legitimate only when designed: "the priorities, triggers, and behaviours for degraded operation are designed up front and verified -- so the degraded state is deliberate and auditable, not accidental." And the annunciation rule: "**Degraded mode must be visible, not silent**"; "Long periods in degraded mode hide failures that accumulate undetected." Bound: "degradation buys time and preserves safety functions, but if conditions continue to deteriorate or integrity targets are not met, the system must transition to a defined safe state." |
| 9 | https://github.com/alecthomas/voluptuous/pull/291 | 2026-08-09 | code/real-world prior art (voluptuous validation lib) | WebFetch, full | The IDENTICAL defect class in a widely used Python validator: `Coerce`'s except tuple was `(ValueError, TypeError)` and missed `decimal.InvalidOperation`. "For some reason Python std lib has inconsistency in handling invalid values for numeric types" -- `float('abc')`/`int('abc')`/`Fraction('abc')` raise `ValueError` but `Decimal('abc')` raises `decimal.InvalidOperation`. Fix = widen the tuple. Merged 2017-05-31. Evidence that `(TypeError, ValueError)` is a *folk* tuple, not a complete one. |

---

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.cechina.cn/eletter/standard/safety/iec61508-7.pdf | standard (IEC 61508-7 mirror) | **ATTEMPTED TWICE and FAILED** -- WebFetch and `curl -A <browser UA>` both returned HTTP 468; the 14 862-byte body is not a PDF (`pdfplumber`: "No /Root object!"). Substituted source #7 (exida). |
| https://www.javacodegeeks.com/2026/05/the-dead-letter-queue-problem-why-your-async-systems-silently-lose-data.html | blog, **May 2026** | ATTEMPTED -- WebFetch returned HTTP 403. Snippet retained as a 2026 recency data point. |
| https://swehb.nasa.gov/display/SWEHBVC/SWE-134+-+Safety-Critical+Software+Design+Requirements | official (NASA handbook) | ATTEMPTED -- HTTP 404 (handbook URL scheme has moved). |
| https://mdcpp.com/doc/standard/IEC61511-2-2003.pdf | standard (IEC 61511-2) | Superseded 2003 edition; the 61511 bypass material is covered by the two industry sources below. |
| https://www.waitingforcode.com/apache-spark-structured-streaming/corrupted-records-poison-pill-records-apache-spark-structured-streaming/read | blog | Spark `badRecordsPath` / `PERMISSIVE` mode -- same pattern as Kafka Connect, already covered by #5. |
| https://www.conduktor.io/blog/dead-letter-topics-handling-poison-pills | vendor blog | Duplicates #5's DLQ semantics. |
| https://dzone.com/articles/causes-and-remedies-of-poison-pill-in-apache-kafka | community | Community tier; duplicates #5. |
| https://github.com/confluentinc/ksql/issues/6563 | issue tracker | "Handle poison pill records by placing them on a dead letter queue" -- corroborates that abort-on-poison-pill is a *known live gap*, not a hypothetical. |
| https://www.abstractalgorithms.dev/dead-letter-queue-pattern-poison-message-recovery | blog | Bounded-retry + triage-ownership guidance; duplicates #1/#5. |
| https://softwarepatternslexicon.com/event-driven-architecture-patterns/reliability-and-delivery/dead-letter-queues/ | pattern catalogue | Reference catalogue, no new claim. |
| https://instrumentationtools.com/iec-61511-standard-requirements-for-safety-bypass-and-override/ | industry | IEC 61511 bypass rules (time-limited, tracked, returned to service) -- corroborates #8. |
| https://www.kenexis.com/bypass-safety-instrumented-functions/ | industry | Same. |
| https://silsafe.net/glossary/graceful-degradation/ | glossary | Definition only. |
| https://risknowlogy.com/articles/detail/17305/ | industry | IEC 61508 fault-detection overview; #8 is the on-point one. |
| https://www.ketryx.com/blog/navigating-iso-26262-and-iec-61508-3-functional-safety-standards | industry | ISO 26262 / 61508-3 overview; no replay-specific claim. |
| https://promwad.com/industries/industrial/iec-61508-safety-software | vendor marketing | Lowest tier. |
| https://realpython.com/ref/builtin-exceptions/overflowerror/ | reference | Superseded by the official docs (#2). |
| https://www.geeksforgeeks.org/python/overflowerror-convert-int-large-to-float-in-python/ | community | Superseded by #2. |
| https://thelinuxcode.com/overflowerror-convert-int-too-large-to-float-in-python-what-it-means-why-it-happens-and-what-to-do-instead/ | community | Superseded by #2. |
| https://tutorialreference.com/python/examples/faq/python-error-overflowerror-int-too-large-to-convert-to-float | community | Superseded by #2. |
| https://rcoh.me/posts/python-float-overflow/ | blog | "PSA: Python Float Overflow" -- corroborates the `int -> float` boundary. |
| https://docs.python.org/3/library/decimal.html | official docs | `DecimalException` -> `ArithmeticError` verified DIRECTLY by execution instead (see the measured table). |
| https://codegym.cc/groups/posts/python-parse-string-to-int-float | community | The `(ValueError, TypeError)` folk-tuple, stated as best practice -- useful as evidence of how the defect propagates. |
| https://event-driven.io/ (site root, other posts) | blog | Only the rebuild/skip post was on-topic. |
| https://www.axoniq.io/blog/disaster-recovery-why-event-sourcing-enhances-resilience-of-any-system | vendor blog | Recency-scan hit; DR framing, no corrupt-record rule. |
| https://zylos.ai/research/2026-02-17-event-sourcing-architecture-pattern/ | secondary, **2026-02** | Recency-scan hit; AI-generated survey, low weight. |
| https://intuitionlabs.ai/articles/event-sourcing-vs-queue-systems | secondary | Recency-scan hit; low weight. |
| https://www.ijsat.org/papers/2025/1/2447.pdf | paper, **2025** | Recency-scan hit; low-tier journal, not on the corrupt-record question. |
| https://arxiv.org/abs/2606.30306 | preprint, **2026** | "Always-On Agents: A Survey of Persistent Memory, State, and Governance in LLM Agents" -- adjacent (durable agent state), not about corrupt-record replay. |
| https://github.com/alecthomas/voluptuous (repo) | code | Repo root; the PR (#9) is the on-point artefact. |

**Unique URLs collected: 39** (9 read in full + 30 snippet-only/attempted).

---

## Recency scan (2024-2026)

Searched explicitly in the 2024-2026 window (queries #2, #4 and #6 above).

**Result: 4 new findings in the window; NONE supersede the canonical sources, and
one of them REINFORCES the loudness requirement.**

1. *The Dead Letter Queue Problem: Why Your Async Systems Silently Lose Data*
   (javacodegeeks, **May 2026**) -- fetch blocked (HTTP 403), snippet only. Its
   thesis restates finding #2 below: "a dead letter queue protects throughput by
   moving repeatedly failing messages out of the hot path, but it only works if
   retries are bounded, triage has an owner, and replay is a deliberate workflow
   instead of a panic button." No change to the design.
2. Event-sourcing tooling matured 2025-2026 (Kurrent, ex-EventStoreDB; Marten;
   EventSourcingDB) with first-class replay primitives, and vendors now quote
   replay throughputs and state-reconstruction SLAs. **Nothing in that maturation
   changes the corrupt-record rule** -- the newer literature still assumes the
   skipped record is *recorded*, not silently dropped.
3. Zylos (**2026-02**) and IntuitionLabs event-sourcing surveys -- secondary,
   AI-authored, no primary claim. Logged for completeness, weighted near zero.
4. `arXiv:2606.30306` (**2026**) *Always-On Agents: A Survey of Persistent
   Memory, State, and Governance in LLM Agents* -- adjacent domain (durable state
   for long-running agents) and confirms the general framing, but says nothing
   specific about malformed-record replay. Not read in full; out of scope.

**Nothing found in the window contradicts the canonical guidance.** The oldest
source in the read-in-full set (voluptuous PR #291, 2017) is still the most
precisely on-point prior art, and the 2026 material is a restatement.

---

## Key findings

**F1. `OverflowError` is not reachable from `(TypeError, ValueError)` -- by
construction, not by accident.** The official hierarchy is
`Exception -> ArithmeticError -> {FloatingPointError, OverflowError,
ZeroDivisionError}`, a sibling branch of `TypeError` and `ValueError`
(https://docs.python.org/3/library/exceptions.html, 2026-08-09). Measured in this
repo's own interpreter (Python 3.14.4):
`isinstance(OverflowError(), (TypeError, ValueError)) == False`; MRO
`(OverflowError, ArithmeticError, Exception, BaseException, object)`.

**F2. The complete escape set for `float(x)` where `x` came from `json.loads`, MEASURED.**
Every JSON-decodable value was enumerated against `_coerce_nav`'s exact logic:

| JSON value | `float()` result | Outcome under today's code |
|---|---|---|
| `1e400`, `-1e400`, `Infinity`, `NaN` | `inf`/`-inf`/`nan` | **safe** -- rejected by `math.isfinite` at `:139` |
| `"1e400"` (string), 401-digit *string* | `inf` | **safe** -- same guard |
| **integer literal, 310..4300 digits (+/-)** | **`OverflowError`** | **ABORTS THE WHOLE REPLAY** |
| integer literal, >4300 digits | -- | **safe** -- `json.loads` itself raises `ValueError` (`int_max_str_digits=4300`), caught by the per-line handler at `:249` |
| `null`, `[1]`, `{"a":1}` | `TypeError` | handled -> `None` |
| `"abc"` | `ValueError` | handled -> `None` |
| `true` | `1.0` | **accepted as a NAV of 1.0** (already documented at `kill_switch.py:340-347`) |
| `" 12 "` | `12.0` | accepted (whitespace-tolerant) |

So the exposure is a **precisely bounded window: a JSON integer of 310 to 4300
decimal digits.** That is the entire reachable set for this call site -- a fact
the fix's test can assert rather than hand-wave.

**F3. The exception classes a robust numeric coercion must handle** (the answer to
research question 4). `(TypeError, ValueError)` is a *folk* tuple. Measured /
documented additions:
- `OverflowError` (`ArithmeticError`) -- `float(huge_int)`. **The live defect.**
- `decimal.InvalidOperation` -- MRO measured:
  `(InvalidOperation, DecimalException, ArithmeticError, Exception, ...)`;
  `issubclass(decimal.DecimalException, ArithmeticError) is True`, and
  `issubclass(InvalidOperation, (TypeError, ValueError)) is False`. This is
  exactly the bug voluptuous PR #291 fixed. Not reachable here today (nothing
  constructs a `Decimal`), but it is the same class of miss.
- `ValueError` from `int(s)`/`repr(int)` above `sys.get_int_max_str_digits()`
  (4300) -- *is* in the tuple, but note it can fire from a `logger.error("%r", raw)`
  on a huge int, not only from the coercion.
- `MemoryError` -- the docs' own note: an integer too large "would rather raise
  MemoryError than give up". Not `Exception`-adjacent in intent; do not swallow it.
- `RecursionError` / `numpy` `FloatingPointError` under `np.seterr(over='raise')`
  -- not reachable at this call site (no numpy on the replay path), listed for
  completeness.
Recommended shape for a *coercion* helper: **catch broadly and return the
sentinel** (`except Exception: return None`) rather than chase the tuple -- but
see F6 for why the loudness must move with it.

**F4. Industry practice for replay is split, and the split is principled.**
- **Skip-and-continue is the streaming/event-sourcing default**, but *only with a
  record of the skip*: "If we can't prevent skips from happening, let's make them
  visible... it could record that it skipped in the same transaction as the event
  append" (Dudycz, event-driven.io, 2026-08-09).
- **Abort is the DEFAULT in both of the systems closest to this one.** Kafka
  Connect: `errors.tolerance = none` is the default and one bad message fails the
  connector (Confluent, 2026-08-09). PostgreSQL redo: "If set to off (the
  default), detection of WAL records having references to invalid pages during
  recovery causes PostgreSQL to raise a PANIC-level error, aborting the recovery"
  (postgresql.org, 2026-08-09).
- The discriminator is **what the replayed state is used for**. A read-model /
  sink can be rebuilt, so skipping is cheap. A *recovery* of authoritative state
  cannot, so PostgreSQL prefers to refuse to start over starting wrong -- and it
  makes the skip superuser-gated, restart-only, and labelled "may cause crashes,
  data loss, propagate or hide corruption."

**F5. "Aborted halfway" is strictly worse than both endpoints, and that is the
actual defect here.** PostgreSQL's abort is a *refusal to start*: the server does
not come up, so nothing operates on partial state. Kafka Connect's abort is a
*FAILED* connector: it stops, visibly. `_load_from_audit` does neither -- it
aborts mid-loop and then **returns normally**, leaving a `KillSwitchState` object
that is fully constructed, callable, and silently reflecting only the prefix of
history before the bad row. That is the third state neither reference system
allows. CWE-390's mitigation names the invariant it breaks: "Ensure that all
exceptions are handled in such a way that **you can be sure of the state of your
system at any given moment**" (cwe.mitre.org/data/definitions/390.html,
2026-08-09). CWE-396 names the mechanism: catching `Exception` "can obscure
exceptions that deserve special treatment", consequence class **Hide Activities**
(cwe.mitre.org/data/definitions/396.html, 2026-08-09).

**F6. Safety standards require the degraded state to be ANNOUNCED, and the
current log level fails that.** "Degraded mode must be visible, not silent"
(risknowlogy.com/articles/detail/17309/, 2026-08-09); degradation is legitimate
only when "designed up front and verified -- so the degraded state is deliberate
and auditable, not accidental." The exida IEC 61508 hazard-analysis worked
example puts corrupt safety parameters in the row whose consequence is "a safety
related shutdown will not occur in a timely manner" and whose required reaction
is "Outputs set to failsafe state"
(sciopta.com/.../RTOSandIEC61508.pdf, 2026-08-09). Today `kill_switch.py:395`
logs the total disarm of a live book at **WARNING**, while the far less severe
"I ignored one authoritative peak assignment" at `:422` logs at **ERROR**. The
severity ordering in this module is inverted.

---

## Internal code inventory (ALL line numbers RE-DERIVED 2026-08-09, not trusted from the step text)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/kill_switch.py` | 1082 | the defect + every anchor below | LIVE, read in full |
| `scripts/diagnostics/measure_sod_date_reachability.py` | 191 | 86.2 immutable verification command | LIVE, read in full, RUN |
| `backend/tests/test_book_safety_69.py` | 432 | phase-86.1 idioms new tests must follow | read in full |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | 1063 | `ks_tmp_audit` / `isolated_state` fixtures | inspected |
| `backend/utils/json_io.py` | (see below) | `parse_json_line`, the FIRST coercion layer | inspected |
| `.claude/masterplan.json` | -- | 86.2 verification block | extracted verbatim |

### Re-derived anchors in `backend/services/kill_switch.py`

| Symbol | Anchor | Note |
|---|---|---|
| `_AUDIT_PATH` | `kill_switch.py:48` | `parents[2]/handoff/kill_switch_audit.jsonl`; `.parent.mkdir` at `:49` |
| `_AUDIT_ARCHIVE_SUBDIR` / `_AUDIT_ARCHIVE_GLOB` | `:74-75` | `"audit"` / `"kill_switch_audit*.jsonl"` |
| `_audit_archive_dir()` | `:89-91` | DERIVED from `_AUDIT_PATH.parent` -- one redirect moves everything |
| `_audit_source_paths()` | `:94-111` | archives first (sorted glob), live file LAST; reads `_AUDIT_PATH` at CALL time |
| **`_coerce_nav()`** | **`:114-141`** | **the except tuple is at `:127`: `except (TypeError, ValueError): return None`** |
| `_coerce_nav` non-finite reject | `:139-140` | `if not math.isfinite(nav): return None` -- this is why `1e400` (-> `inf`) is safe |
| `_read_audit_rows()` | `:208-270` | static; returns `(rows, complete)` |
| per-line parse guard | `:247-258` | `except Exception: complete = False; continue` -- **the correct idiom already exists one layer up** |
| non-dict row guard | `:259-261` | `complete = False; continue` |
| ts merge-sort | `:262` key build, `:268` `keyed.sort(key=lambda t: (t[0], t[1], t[2]))` | `(ts, src_idx, line_idx)`; missing `ts` -> `""` -> collates FIRST |
| **`_load_from_audit()`** | **`:272-395`** | the whole `for row in rows:` loop is inside ONE `try:` at `:273` |
| **the broad swallow** | **`:394-395`** | `except Exception as e: logger.warning(f"kill_switch: audit load failed: {e}")` -- WARNING, not ERROR, and no re-raise |
| `sod_snapshot` branch | `:298-316` | `_coerce_nav(row.get("nav"))` at `:299` -- **escape site #1** |
| `peak_update` anchor branch | `:362-364` | `_coerce_nav(row.get("prior_peak"))` at `:363` -- **escape site #2**; `_apply_authoritative_peak(row.get("nav"), ...)` at `:364` |
| `peak_update` ratchet branch | `:366-368` | `nav = _coerce_nav(row.get("nav"))` at `:366` -- **escape site #3 (the measured one)** |
| `peak_reset` branch | `:369-387` | `_apply_authoritative_peak(row.get("new_peak"), "peak_reset")` at `:382` -- **escape site #4** |
| `_apply_authoritative_peak()` | `:397-430` | `value = _coerce_nav(raw)` at `:420`; `logger.error(...)` + `return` at `:422-429` on None |
| `_append_audit()` | `:432-443` | the ONLY writer; `except Exception` -> warning at `:442-443` |
| `_state = KillSwitchState()` | `:704` | module-level singleton -- **importing the module replays the live journal** |
| `_BASELINE_EVENTS` | `:709` | `frozenset({"sod_snapshot", "peak_update", "peak_reset"})` |
| `baseline_history_exists()` | `:712-739` | its own `except Exception` at `:736-738` returns **True** (fail-toward-blocking) |
| `evaluate_breach()` | `:749-882` | per-leg arming guards at `:797-811` |
| `daily_baseline_missing` | `:797` | `not (sod is not None and sod > 0)` |
| `trailing_baseline_missing` | `:798` | `not (peak is not None and peak > 0)` |
| `daily_baseline_stale` | `:809` | `_sod_date_is_stale(...)` |
| `armed` | `:811` | `not (daily_leg_unevaluable or trailing_baseline_missing)` |
| daily leg fire | `:859-861` / trailing leg fire `:865-867` | each gated on its OWN missing flag |
| `baselines_present_in()` | `:885-902` | presence-only predicate shared with paper_trader's BUY gate |
| `_sod_date_is_stale()` | `:905-930` | `except (ValueError, TypeError)` at `:928` -- same narrow tuple, but on `date.fromisoformat` |
| `_log_disarmed_once()` | `:933-960` | one-shot `logger.error`, no network |

### The defect, stated mechanically

`_load_from_audit` at `:273` opens ONE `try:` around the ENTIRE `for row in rows:`
loop. `_coerce_nav:127` catches only `(TypeError, ValueError)`. `OverflowError`
is **not** in that tuple -- its MRO is
`OverflowError -> ArithmeticError -> Exception` (measured, see below), so it
propagates out of `_coerce_nav`, out of the branch, out of the loop, and lands
in the `:394` `except Exception`, which logs at **WARNING** and returns. Every
row after the malformed one in `ts` order is never applied.

Note the asymmetry that makes this a one-line-class bug rather than a design
gap: **the per-row skip idiom already exists**, one layer up at `:247-258`, and
it even records `complete = False` so a later anchor cannot claim authority. The
replay loop simply does not have the equivalent.

### Measured today (2026-08-09), verbatim from the verification command

```
CASE E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)
  replayed snapshot : sod_nav=None sod_date=None peak_nav=None
  current_nav       : 80.0   (drop vs sod: None%)
  armed             : False
  daily_baseline_missing=True daily_baseline_stale=False
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : False  (0.0%)
  >>> any_breached      : False
...
  E  oversized int    : sod_nav=None peak=None, any_breached=False  <-- BOTH legs stranded
  HEALTHY control     : daily=True trailing=True any=True
```

Contrast with every OTHER degraded case in the same run -- C (UTC rollover),
A/B (legacy row), F (startup) -- all three printed `any_breached=True`, because
the date-independent trailing leg still fires. **Case E is the only one where
`any_breached` is False on a real 20% drawdown.** That is the step's claim,
re-derived, not assumed.

### What case E does TODAY, and what it does NOT do

- **It exists** at `measure_sod_date_reachability.py:130-146`, built by
  `_measure("E -- OVERSIZED INT ...", ...)`.
- The malformed row is constructed by raw string interpolation at `:138-139`
  (`'{"ts": "%sT00:00:00+00:00", "event": "peak_update", "nav": %s}' % (_today(), "1" + "0"*400)`)
  -- **not** via `_row(**kw)`/`json.dumps`, because `json.dumps` of a 401-digit
  Python int would work but the author wanted the literal on the wire. It is
  FIRST in `ts` order (`T00:00:00`), and the two good rows follow at `T00:01:00`
  (`sod_snapshot`) and `T00:02:00` (`peak_update`). The in-code comment at
  `:135-137` records that the author's first construction put it LAST and
  produced the opposite (wrong) result.
- **It currently PASSES the script.** `main()` returns `1` only when the HEALTHY
  control fails to breach (`:184-186`); case E's outcome is *printed and
  narrated* (`:167-171`, `:178-183`) but never asserted. Measured: the command
  exits **0** today, with the defect fully present. **So the immutable
  verification command CANNOT be the RED-test vehicle for criterion 1** -- it is
  a measurement harness, and its case-E block is a before/after *witness*. The
  RED test must be a new pytest.
- **It writes nothing.** Every case runs inside `tempfile.TemporaryDirectory()`
  (`:48`), redirects `ks._AUDIT_PATH` (`:52`), and asserts isolation twice
  (`:54` prefix check, `:59-61` "no live source in the replay"), restoring
  `_AUDIT_PATH` in a `finally` (`:72`) and `ks._state` in a nested `finally`
  (`:69-70`). Verified empirically: `handoff/kill_switch_audit.jsonl` sha256 was
  `90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653` (62 lines)
  before AND after the run.
- Caveat worth knowing: `import backend.services.kill_switch` builds the
  module singleton at `:704`, which **reads** the live journal on import. Read
  only; no write path is reached.

### masterplan 86.2 -- verbatim

`verification.command`:

```
bash -c 'source .venv/bin/activate && python scripts/diagnostics/measure_sod_date_reachability.py'
```

`verification.success_criteria` (verbatim, 5 items):

1. "the failure mode is reproduced FIRST as a red test (malformed row first in ts order -> both legs stranded -> any_breached False on a real breach), so the fix has something to turn green"
2. "_coerce_nav no longer lets a malformed value abort the replay: a bad row is skipped and logged, and every WELL-FORMED row before AND after it still applies -- proven by a test with rows on both sides"
3. "the replay's fail-safe direction is stated and tested: whatever a malformed row does, it must never leave a leg reporting armed=True on a baseline it did not actually load"
4. "a mutation reverting the widened except makes the reproduction test fail again"
5. "no threshold changed, no leg re-armed on an unloadable baseline; fresh Q/A PASS"

`verification.live_check`: "live_check_86.2.md with the verbatim before/after
output of measure_sod_date_reachability.py case E, and the verbatim mutation
transcript"

### phase-86.1 idioms in `backend/tests/test_book_safety_69.py` (new tests MUST follow both)

1. **The function-scoped autouse journal guard**, `:34-43`:
   `_live_kill_switch_journal_is_byte_identical` reads
   `REPO_ROOT/handoff/kill_switch_audit.jsonl` bytes before the yield and
   asserts equality after, with a line-count delta in the message. Its own
   docstring (`:20-29`) is explicit that it is a **DETECTOR, not a preventer** --
   the bytes are already on disk when it fires; the preventer is step 86.6.
   `REPO_ROOT`/`_LIVE_AUDIT` at `:30-31`.
2. **The detached-state idiom**, `test_peak_reset_dark_by_default:211-268`:
   redirect `ks._AUDIT_PATH` to `tmp_path` BEFORE constructing anything
   (`:245`), then **assert the derivation still holds** (`:246-249`:
   `ks._audit_archive_dir() == tmp_path/"audit"`), then assert no live path is
   in `ks._audit_source_paths()` (`:250-253`), then **pin** the settings flag
   rather than inherit it (`:256-257`), then build `st = ks.KillSwitchState()`
   which is DETACHED -- never `ks.get_state()`. The docstring records why the
   old form was **vacuous by identity** (`monkeypatch.setattr(ks,"get_state",lambda: st)`
   where `st` was already the singleton) and why a redirect ALONE is a HALF fix
   (`reset_peak` assigns `self._peak_nav` at `:697` BEFORE auditing).
   `test_valid_nav_still_breaches:103-156` shows the same shape plus a
   `monkeypatch.setattr(ks, "_state", st)` when the module-level
   `evaluate_breach` must see the detached state, plus
   `monkeypatch.setattr(ks, "_disarmed_logged", False)` to un-latch the one-shot
   log, plus an explicit PRECONDITION assert before the real assertions
   (`:149-152`).
   `test_stale_anchor_disarms_the_daily_leg_but_the_trailing_leg_still_fires:159-208`
   is the closest template for an 86.2 test: it asserts the disarm AND that the
   surviving leg still fires. For 86.2 the analogous assertion is stronger --
   after the fix the well-formed rows on BOTH sides must apply.

---

## Consensus vs debate (external)

**Consensus (all 9 sources agree):**
- A malformed record must never leave the system in a state whose trustworthiness
  is unknown to its callers (CWE-390, CWE-396, risknowlogy, exida).
- Whatever is done with the bad record, it must be **loud and recorded**
  (Dudycz: "let's make them visible"; Confluent: `errors.log.enable`;
  risknowlogy: "visible, not silent"; PostgreSQL: "but still report a warning").
- Narrow except-tuples on numeric coercion are a recognised recurring defect, not
  a novel finding (voluptuous PR #291).

**Genuine debate -- skip vs abort:**
- *Skip-and-continue* wins where the replayed state is a **derived projection**
  that can be rebuilt (Dudycz; Kafka Connect with `tolerance=all` + DLQ).
- *Abort/refuse-to-start* wins where the replay reconstructs **authoritative
  state** (PostgreSQL redo: PANIC by default; Kafka Connect default
  `tolerance=none`). PostgreSQL is the **adversarial** source against the naive
  "just skip it" reading, and it is the closest analogue to a kill-switch replay:
  it would rather not come up than come up with a state it cannot vouch for.
- **The debate does NOT extend to "abort silently and keep serving."** No source
  endorses that, and it is precisely what `kill_switch.py:394-395` does.

## Pitfalls (from literature, mapped to concrete traps for this step)

1. **Silent skipping is the failure the DLQ literature warns about.** "A DLQ is
   only helpful if it's monitored... Otherwise, it's just a fancy way to lose data
   slowly rather than immediately" (Dudycz). Widening the tuple to
   `except Exception: return None` without raising the log level and without
   marking the replay incomplete converts a loud total disarm into a *quiet
   partial* disarm -- a smaller blast radius but a worse detection story.
2. **Trading one narrow tuple for another.** Adding `OverflowError` alone leaves
   `decimal.InvalidOperation` and any future arithmetic sibling. The measured
   escape set (F2) exists so the fix can be justified by enumeration rather than
   by imagination -- the same reasoning `kill_switch.py:350-358` already applies
   to the anchor rule ("the flag was NEGATIVELY derived, so it was only ever as
   good as the author's imagination").
3. **Fixing `_coerce_nav` only.** `_coerce_nav` is *one* of the things inside the
   `:273` try-block. `datetime.fromisoformat` at `:308` already has its own inner
   guard; `bool(row.get(...))` at `:316` cannot raise; but the loop as a whole is
   still one abort away from the same outcome for any future branch. Criterion 2
   is worded around `_coerce_nav`, but the durable fix is a **per-row** try in the
   replay loop, mirroring the idiom that already exists at `:247-258`.
4. **The `complete` flag must move with the skip.** `_read_audit_rows` sets
   `complete = False` for an unparseable line (`:250-258`) precisely so a later
   anchor-from-`None` cannot claim authority (`:362-363`, `:625-630`). A skipped
   row in `_load_from_audit` is the *same* class of unseen history. If the fix
   skips without recording incompleteness, it opens the exact hole five prior Q/A
   passes closed.
5. **Ordering.** The reproduction MUST place the malformed row FIRST in `ts`
   order. The sort key is `(ts, src_idx, line_idx)` at `:262`/`:268`; rows with no
   `ts` collate to `""` and sort FIRST, so an alternative reproduction is a
   malformed row with **no `ts` at all**. The step text and
   `measure_sod_date_reachability.py:135-137` both record that getting this
   backwards produced the opposite (wrong) conclusion.
6. **The verification command does not gate this.** Measured: it exits **0**
   today with the defect fully present (only the HEALTHY control can fail it,
   `:184-186`). Criterion 1's "red test" must be a **new pytest**; the diagnostic
   script is the before/after *witness* for `live_check_86.2.md`.
7. **Do not re-arm on an unloadable baseline (criterion 3).** The fail-safe
   direction here is already settled by this module's own doctrine at `:768-785`:
   absence must never read as health, the markers are PER LEG, and a missing
   baseline must NOT set `any_breached=True` (that would flatten a healthy book on
   a housekeeping event). So the correct fail-safe direction for 86.2 is: **load
   as much true history as possible, and where a row could not be loaded, the
   affected leg reports `armed=False` rather than a guessed baseline.**
8. **Journal integrity.** `import backend.services.kill_switch` builds the
   singleton at `:704` and replays the live journal. Any new test must use the
   phase-86.1 detached-state idiom + the autouse byte-identity fixture, and must
   redirect `_AUDIT_PATH` BEFORE constructing anything.

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication for the 86.2 contract |
|---|---|---|
| F1/F2 measured escape set | `kill_switch.py:127` | The except tuple is the proximate cause; the reachable trigger is exactly a 310-4300-digit JSON integer at `:299`, `:363`, `:366`, `:382`. |
| F5 "aborted halfway" | `kill_switch.py:273` (try) + `:394-395` (swallow) | The **structural** fix is a per-row `try` inside the loop so an abort can never span rows. The `:273` handler should remain as a last-resort net but must no longer be the thing that eats a per-row fault. |
| F4 PostgreSQL: authoritative-state replay refuses rather than half-loads | `kill_switch.py:279` `self._history_complete = complete` | The project's equivalent of "refuse to start" already exists as a *flag*, not a refusal. A skipped row should set `complete = False` on the same channel so `update_peak:625-630` and the anchor gate at `:362-363` keep working. |
| F6 "visible, not silent" + inverted severity | `kill_switch.py:395` (WARNING) vs `:422` (ERROR) | Raise the replay-failure log to ERROR and name the row index / source path, matching `_apply_authoritative_peak`'s precedent and `_log_disarmed_once:956`. |
| CWE-390 mitigation names mutation testing | criterion 4 | The mutation is well defined: revert the widened `except` at `:127` (or remove the per-row `try`) and the reproduction test must go RED again. Per `feedback_mutation_test_guards_and_fixtures`, mutate the **production call site**, not a helper. |
| F4 skip-with-DLQ | `measure_sod_date_reachability.py:130-146` | Case E becomes the before/after witness. AFTER the fix its expected line is `sod_nav=100.0 sod_date=<today> peak_nav=100.0`, `armed=True`, both legs breached, `any_breached=True` -- i.e. the two well-formed rows AFTER the bad one apply. Criterion 2 additionally demands a row BEFORE it, which case E does not currently have. |
| phase-86.1 idioms | `test_book_safety_69.py:34-43`, `:211-268` | New tests: autouse byte-identity fixture + detached `KillSwitchState` + redirect-then-assert-derivation + pin any settings flag. |
| 36.7 fixtures | `test_phase_36_7_kill_switch_rotation_rearm.py:118-141` (`ks_tmp_audit`) / `:143-184` (`isolated_state`) / `:186-207` (autouse write-protect) | `ks_tmp_audit` yields `(ks, live, archive)` and patches `_AUDIT_PATH` ALONE deliberately (archive is derived, asserted at `:140`). `isolated_state` builds a detached state via `object.__new__` and ALSO redirects `_AUDIT_PATH` because `resume()` appends -- and resets `_disarmed_logged`. **`isolated_state` does not set `_baseline_provenance` / `_sod_provisional` / `_history_complete`; those survive as CLASS-level defaults (`kill_switch.py:160/169/174`), which is exactly why they are class-level.** If 86.2 adds new instance state to the replay, it must be class-level too or these fixtures break. |
| Blast-radius framing | `kill_switch.py:797-811`, `evaluate_breach:859-867` | Confirmed by measurement: cases C / A/B / F all keep `any_breached=True` via the date-independent trailing leg; case E alone reaches `False`. After a per-row skip, case E would degrade to the *bounded* class rather than total disarm. |
| Residual (out of scope, worth queueing) | `kill_switch.py:611`, `:633`, `:697` | `update_peak` and `reset_peak` coerce with a bare `float(nav)` -- the SAME OverflowError is reachable there from the in-memory path, and `:406-408` already flags this as masterplan step 36.19. 86.2 should say explicitly whether it is in or out of scope rather than leave it ambiguous. |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (**9**: 8 via WebFetch, 1 via `curl` + `pdfplumber` per `.claude/rules/research-gate.md` step 3)
- [x] 10+ unique URLs total (**39**)
- [x] Recency scan (2024-2026) performed + reported (4 findings; none supersede)
- [x] Full pages read, not abstracts, for the read-in-full set
- [x] file:line anchors for every internal claim (all RE-DERIVED today; none trusted from the step text)

Soft checks:
- [x] Internal exploration covered every module the caller named, plus `backend/utils/json_io.py` and the masterplan block
- [x] Contradictions / consensus noted (skip-vs-abort split; PostgreSQL is the adversarial source)
- [x] Claims cited per-claim with URL + access date, or by measured execution transcript
- [ ] GAP: the primary IEC 61508-7 standard text (cechina mirror) was unreachable (HTTP 468 via two methods); the exida IEC 61508 paper + the risknowlogy graceful-degradation article stand in. The IEC 61511 *selective-bypass* framing used by phase-85.5.1 is carried here by snippet-level sources only.
- [ ] GAP: `measure_sod_date_reachability.py` was RUN read-only, but no fix was prototyped -- the post-fix case-E output above is a PREDICTION, and GENERATE must measure it.

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 30,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 1,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.2.md",
  "gate_passed": true
}
```

_Hard constraints honoured: the full backend test suite was NOT run; nothing was
POSTed to localhost:8000; `handoff/kill_switch_audit.jsonl` is byte-identical
(sha256 `90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653`,
62 lines) verified before and after the diagnostic run. Read-only throughout._
