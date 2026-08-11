---
name: absent-upstream-data-86-41
description: 86.41 — the NoneType crash is in the CF not this repo (/workspace/main.py:89); a retry-then-return-None erases the 429 cause; a classifier keyed on the wrapper message inherits that blindness
metadata:
  type: project
---

Step 86.41 (defensive handling of absent upstream financial data). Four findings
that generalise past this step.

**1. A traceback frame of `/workspace/main.py` is NOT this repo — it is a Cloud
Function's filesystem.** `grep "QuantAgent failed"` returns 0 hits in every
tracked file; the string exists only inside log payloads, streamed back verbatim
and re-logged by `orchestrator.py:1141` (`logger.info(f"Quant: {line}")`). Before
proposing a local fix for a crash, check whether the frame belongs to you. A local
`or {}` cannot stop an exception raised on the other side of an HTTP boundary; it
can only decide what you do about it.

**Why:** the step (and an older queued step, `scripts/add_phase_27_6_sub.py:59`)
was framed as "add defensive `.get()` / `or {}` on the upstream dep" — a code fix
aimed at a component this repo does not contain.

**How to apply:** grep the tracked tree for the error string FIRST. Zero hits +
a plausible-looking path = the path is someone else's.

**2. Retry-then-return-None destroys cause provenance one frame later.** The
measured chain: SEC 429 → CF retry ladder (2s/3s/5s) exhausts → CIK-map fetcher
returns `None` instead of raising → `cik_map.get(...)` raises `AttributeError` →
the 429 is now unrecoverable from the exception message. Google SRE's rule is the
opposite: an exhausted retry budget must "let the failure bubble up to the caller"
as an explicit typed condition. **The three-way question "provider returned
nothing / legitimately null / code assumed a dict" collapses to case 1 wearing
case 3's clothes** whenever a sentinel return sits between them.

**3. A classifier keyed on the WRAPPER message inherits that blindness — and its
own coverage assertion cannot catch it.**
`scripts/qa/derive_lite_fallback_census_86_38.py` asserts parsed == raw per file
and withholds the census on a shortfall (`:248-261`) — rigorous *line accounting*.
But `classify()` (`:39-53`) tests `429` against the wrapper reason only, and the
wrapper for this class reads `ERROR: QuantAgent failed for X: 'NoneType' ...` with
no `429` in it. So 17-of-18 rate-limit events file under "code defect". The 429
evidence sits in the preceding INFO stream lines, which the population rule
(`:136-137`) never reads. **An instrument can be internally airtight and still be
wrong about cause.** Coverage assertions prove you counted every line; they prove
nothing about attribution.

**4. Measure the population before freezing an absence-of-string criterion.**
Current `backend.log` holds **0** of these events (all 18 are in rotated `.gz`).
The pre-existing queued live_check — "fresh Claude cycle has zero
`QuantAgent failed.*NoneType` log lines" (`add_phase_27_6_sub.py:87`) — would pass
vacuously. Same class as [[feedback_immutable_criteria_must_be_green_able]] and
[[feedback_zero_assertion_guard_passes_vacuously]].

**Repo-shape note worth keeping:** `orchestrator.py:1792` is the ONLY unguarded
sub-agent call in the pipeline — RAG (`:1160-1167`), ingestion (`:1775-1787`) and
phase-32.3 (`:1828-1839`) all fail open with stated rationales, and
`autonomous_loop.py` already has BOTH a typed degradation exception
(`SynthesisDegradedError`, `:2156-2164`) and a persisted `_degraded` marker row
(`:2245-2252`). Any new degradation mechanism should reuse those, and must have a
consumer — `_intended_path` was deleted at `:2221-2233` for being write-only.

**External anchor for the design debate:** Fowler's Tolerant Reader and RFC 9413
genuinely contradict each other. Reconciliation that held up:
**be tolerant about SHAPE, strict about ABSENCE.** See
[[project_research_gate_discipline]] for the gate mechanics.
