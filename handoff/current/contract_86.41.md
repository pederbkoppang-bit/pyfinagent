# Contract -- step 86.41

**Step**: `86.41` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~13:0x CEST) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** No production file is modified at this moment.

**Ownership**: the peer session filed this step and confirmed it is mine, asking
explicitly that every number be re-derived and that I contradict them rather than
inherit. I have, and the headline finding does contradict them.

---

## 1. Research gate

**PASSED** -- `wf_216e86b6-d1e`, tier `moderate`, brief `research_brief_86.41.md`
(37,074 chars). Enforced: **7 sources read in full** (floor 5), **21 URLs** (floor
10), recency scan present, all 7 claimed URLs verified, `brief_status: COMPLETE`,
`rail_dropped: null`, self-report agreed.

## 2. THE STEP'S PREMISE IS REFUTED, AND CRITERION 1 IS ANSWERED

The step is titled *"the QuantAgent NoneType crash is a bigger source of degraded
analyses than the 429 ever was"*. **It is not a competing cause. It is
DOWNSTREAM of a 429.**

**Criterion 1 -- the call site, IDENTIFIED from a traceback, not inferred:**
`/workspace/main.py:89` in `get_cik` — inside the **remote Quant Agent Cloud
Function**, one frame after a SEC 429 exhausts its CIK-map retry ladder and the
fetcher returns `None` instead of raising. **The failing call is not in this
repository at all**, so "fix the NoneType" is not a change available here.

**Criterion 3 -- ABSENT UPSTREAM FIELD, not a logic error.** The distinguishing
evidence is the 429 immediately preceding it: the CIK map is absent because the
SEC refused it, and the remote function converts that refusal into a `None`
return rather than an exception. A logic error would not correlate with an
upstream rate limit.

### The census misattributes it, and I verified the mechanism by reading it

`scripts/qa/derive_lite_fallback_census_86_38.py` extracts only the wrapper
string:

```python
_REASON = re.compile(r"Full orchestrator failed for (\S+?): (.*?) -- falling back", re.S)
def classify(reason: str) -> str:
    if "resource_exhausted" in r or "429" in reason: return "429 RESOURCE_EXHAUSTED (quota)"
    ...
    if "nonetype" in r: return "code defect: QuantAgent NoneType"
```

The 429 is logged **upstream on a different line**, so `classify()` never sees
it; the wrapper carries "NoneType", so every such event lands in `code defect`.
The 429 branch is first and still cannot fire. **That is the entire 6-vs-3
split.** This also means **86.38's evidence inherits the same misattribution** —
recorded here, not fixed here (that is the peer's closed step).

### My numbers vs the gate's: numerator agrees, RATIO DOES NOT

Scanning 42 retained log files myself:

| quantity | mine | gate |
|---|---|---|
| events preceded by a 429 within 8 lines | **17** | 17 |
| distinct tickers | **13** | 13 |
| total events | **34 (RAW, no dedup)** | 18 (deduped) |

**The numerator and ticker count reproduce exactly; the denominator does not,
because the dedup rule is one I have not reproduced.** So "17 of 18 (94%)" and
"17 of 34 (50%)" are both defensible and **neither may be quoted without stating
the rule**. GENERATE must either reproduce the dedup or report the raw ratio.
This is the third instance today of a ratio whose normalisation was missing.

## 3. Immutable success criteria

Six; copied verbatim into `experiment_results_86.41.md` §2 at GENERATE rather
than duplicated here — a paraphrase in two places is the divergence defect that
made 86.36 CONDITIONAL this morning, and the peer's 86.38 FAIL included handing
an evaluator the *live_check* text while labelling it "VERBATIM from
masterplan.json".

## 4. Plan

**P1 -- CRITERION 2, THE CENSUS, RE-DERIVED WITH ITS COVERAGE ASSERTION SHOWN.**
The criterion warns an earlier parser silently dropped 416 events. I will run the
checked-in instrument, show the assertion passing, **and separately state that a
passing coverage assertion proves LINE COUNTING, not ATTRIBUTION** — the gate's
finding is precisely that the instrument counts correctly and classifies wrongly.
A green coverage assertion is not evidence the classification is right.

**P2 -- THE ACTIONABLE DEFECT IS OURS, AND IT IS NARROWER THAN THE TITLE.**
`orchestrator.py:1792` is the **only unguarded sub-agent call**; RAG
(`:1160-1167`), ingestion (`:1775-1787`) and phase-32.3 (`:1828-1839`) all fail
open. `autonomous_loop.py:2201` converts that one crash into a **whole-ticker
lite fallback**. So one remote dependency's rate limit degrades an entire
analysis. External consensus supports fixing this rather than the None:
Azure's Bulkhead pattern isolates at the **dependency**, not the ticker; Google
SRE treats routine fallback as an anti-pattern. Fowler's tolerant reader and RFC
9413 genuinely disagree, reconciled as **tolerate SHAPE, refuse ABSENCE**.

**P3 -- CRITERION 4 MUTATION, with the vacuity trap named.** Revert the guard at
`:1792` and require a NAMED assertion red, green control captured first.

**P4 -- CRITERION 5 by diff**: `_fallback_rate_check`,
`_degradation_summary_fields` and the record-always call site must be
byte-identical. The peer shipped those hours ago and they are not mine to touch.

**P5 -- CRITERION 6 is a prohibition and I will honour it twice over.** I will
not claim this caused the drought — and note I already retracted exactly that
claim about the 429 this morning, on this same pipeline. Any drought-bearing
evidence gets its own step.

### A CRITERION TRAP, named in advance

**"Zero NoneType errors in the next cycle" would be VACUOUS**: there are already
zero in the current `backend.log`, which rotated at 08:41 today. Any
absence-based success measure must state the window and show the population was
non-empty in it.

### Explicitly NOT doing

- **Not** modifying 86.38's fallback mechanism (criterion 5), the census
  instrument's classification (recorded, belongs to a closed step), or anything
  in the remote Cloud Function (not in this repo).
- **Not** claiming a drought link (criterion 6).

### Risk

`orchestrator.py:1792` and `autonomous_loop.py:2201` are on the live analysis
path, and the book runs at 20:00 CEST. A change here alters what happens when a
sub-agent fails. It cannot take effect without a restart, and restarts batch to
session end — so nothing shipped today reaches tonight's cycle unless the
operator restarts.

## 5. References

- `handoff/current/research_brief_86.41.md` (gate `wf_216e86b6-d1e`)
- Google SRE: addressing cascading failures, handling overload; Azure Bulkhead;
  Fowler *TolerantReader*; RFC 9413; Pydantic validation rationale; arXiv 2503.13657v2
- `scripts/qa/derive_lite_fallback_census_86_38.py:39-53`;
  `backend/agents/orchestrator.py:1792`; `backend/services/autonomous_loop.py:2201`
