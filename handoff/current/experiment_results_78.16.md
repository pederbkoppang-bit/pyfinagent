# Experiment results — phase-78.16

**Step:** 78.16 — `make_client` DROPS `enable_prompt_caching`, so the documented
one-flag revert does NOT restore prior behaviour
**Date:** 2026-07-25 · Cycle 164 · Executor: Main (Opus, xhigh)
**Contract:** `handoff/current/contract_78.16.md`
**Research gate:** `handoff/current/research_brief_78.16.md` — `gate_passed: true`
(7 external sources read in full, 21 URLs, recency scan performed, 16 internal
files inspected, tier `moderate`)

---

## 1. What was built

### Decision taken: **option (a)** — `make_client` accepts and forwards the intent

The masterplan offered three options. **(a)** was chosen, and the research gate
independently recommended the same. The reasoning, stated explicitly because the
criterion allows an escape hatch that I deliberately did **not** take:

- Criterion 1 permits either *exact restoration* **or** *"the divergence is
  measured, justified and documented as intended"* (option (c)). Option (c)
  requires the divergence to be **measured harmless**, and it is not measurable
  right now: Anthropic's documented cache-write minimum for Haiku 4.5 is **4,096
  tokens** and the block's token count straddles that floor on every available
  heuristic (3,877 / 4,551 / 4,769 for 19,075 chars). Under the floor, caching is
  a **silent no-op**; over it, a **miss costs 2×** base input (1h TTL write) and
  Anthropic documents that concurrent requests all miss — which is precisely the
  shape of the three per-ticker services (`asyncio.gather` at
  `pead_signal.py:371`, `analyst_narrative_scorer.py:227`,
  `call_transcript_gpr.py:199`).
- So option (c) would have been "assert harmlessness we cannot measure", on the
  money path, inside a step whose entire purpose is to make an advertised revert
  honest. Option (a) needs no such assertion.
- **The deciding factor is revert fidelity, not dollars.** Magnitude either way is
  ~$0.005/call, single-digit cents per cycle. I am not claiming this step saves
  money; I am claiming a flag the operator is told is a safe revert now is one.

### Files changed

| File | Change |
|---|---|
| `backend/agents/llm_client.py:2044-2081` | `make_client` gains a **keyword-only** `enable_prompt_caching: bool \| None = None` (signature at `:2049`), documented in the docstring including the fact that it is ignored on the CC rail |
| `backend/agents/llm_client.py:2154-2171` | The single production `ClaudeClient(...)` construction site forwards the intent **only when it is not `None`** (`:2168-2171`), with the defect's history recorded inline |
| `backend/services/meta_scorer.py:242` | passes `enable_prompt_caching=False` |
| `backend/services/news_screen.py:288` | passes `enable_prompt_caching=False` |
| `backend/services/macro_regime.py:527` | passes `enable_prompt_caching=False` |
| `backend/services/pead_signal.py:300` | passes `enable_prompt_caching=False` |
| `backend/services/analyst_narrative_scorer.py:156` | passes `enable_prompt_caching=False` |
| `backend/services/call_transcript_gpr.py:135` | passes `enable_prompt_caching=False` |
| `backend/tests/test_phase_78_16_prompt_caching_intent.py` | **NEW**, 9 tests |

Each of the six carries the same comment block recording (i) that this restates a
pre-78.1 intent, (ii) that the *original* rationale is now stale, and (iii) that
re-deciding it is queued rather than silently taken here.

### Why `None` and not `False` as the default

`None` means *"the caller expressed no preference"* and leaves the `ClaudeClient`
class default untouched. The 7 `make_client` callers outside the C-block pass no
caching argument; defaulting to `False` would have silently changed **their**
behaviour, which is outside this step's boundary. There is a dedicated test for
this (`test_make_client_default_leaves_class_default_untouched`) and mutation M3
confirms it fires.

---

## 2. Verbatim verification output

**Immutable verification command:**

```
$ .venv/bin/python -m pytest backend/tests/ -q -k 'llm_client or make_client or prompt_caching'
19 passed, 2016 deselected, 1 warning in 4.64s
```

Baseline before this step's changes, same command: `10 passed, 2016 deselected`.

**Test-file naming note (disclosed, because it looks like gaming and is not).**
The file was first written as `test_phase_78_16_caching_intent.py`. Under that
name the immutable `-k 'llm_client or make_client or prompt_caching'` filter
selected only **2 of its 9 tests** (the two whose function names contain
`make_client`), leaving the six per-service revert-shape guards — the heart of
criterion 2 — **outside the step's own verification command**. The file was
renamed to `test_phase_78_16_prompt_caching_intent.py` so the module name matches
the filter and all 9 run. The immutable criteria and command were **not** touched;
only my own file name was, in the direction of *more* coverage under the gate.

**Regression sweep** (the six services, the 78.1 suite, and the request-shape
pins):

```
$ .venv/bin/python -m pytest backend/tests/ -q -k 'meta_scorer or news_screen or macro_regime or pead or analyst_narrative or call_transcript or claude_request_shapes or 78_1'
62 passed, 1973 deselected, 1 warning in 3.96s
```

**Syntax check** on all seven edited modules: `ast.parse` clean (7/7).

**Undisclosed side effect, now disclosed (Q/A finding N4).** Appending step 78.17
re-serialized `.claude/masterplan.json` with `ensure_ascii=False`, which rewrote
~148 unrelated lines across phases 4.5/12/16/17/23 by unescaping `—` → `—`
and `§` → `§`. Measured: `git diff --numstat` = `169 added / 150 removed`,
of which only ~21 lines are step 78.17 itself. The Q/A independently proved
**zero semantic drift** (structural compare: 971 → 972 steps, `ADDED=['78.17']`,
`REMOVED=[]`, zero differences on all 971 shared steps) and noted that this
encoding actually **restores** the phase-75.17 baseline: the masterplan diff
against that baseline shrinks from 106 non-artifact removed lines at HEAD to 11
in the working tree. So it is a net improvement rather than a corruption — but it
makes this commit's diff noisy, and it should not have gone unmentioned.

---

## 3. Criterion-by-criterion

| # | Criterion (verbatim) | Evidence |
|---|----------------------|----------|
| 1 | "Flipping PAPER_USE_CLAUDE_CODE_ROUTE=false returns the six to their PRE-78.1 request shape, proven by captured wire kwargs (system as a plain str, no cache_control) -- or the divergence is measured, justified and documented as intended" | **Met in the strong form** (exact restoration, not the escape hatch). `live_check_78.16.md` §1: on the flag-OFF path `type(system)` is `str`, len 19075, `cache_control ABSENT`; every other kwarg byte-identical to the PRE capture. Closed against the *actual* pre-78.1 construction: `ClaudeClient(..., enable_prompt_caching=False).enable_prompt_caching` is `False` and `make_client(...)` now yields `False` — `MATCH: True`. |
| 2 | "A test asserts the revert-path request shape, so this cannot regress silently again" | `test_revert_path_restores_pre_78_1_request_shape[…]`, parametrised over all six services, asserting on **captured wire kwargs**. It reads the intent each service *actually passes* out of the AST and pushes that through the real `make_client` path, so it fails both when a service drops the kwarg and when `make_client` stops forwarding it. |
| 3 | "MUTATION: drop the caching intent again -> that test goes red" | `live_check_78.16.md` §2. M1 (service drops it) → 1 failed. M2 (`make_client` drops it) → 6 failed. Plus M3 (default flip) → 1 failed and M4 (stub mutated) → 2 failed. All reverts SHA-verified. **M1's first run was GREEN and that is disclosed in full** — the mutation string hit the explanatory comment rather than the call line; re-targeted, it goes red. |

---

## 4. Scope honesty

**In scope and done:** `make_client` + the six callers + tests, per the
masterplan boundary (`llm_client.py make_client + tests`; the six were touched
only to restate an intent they already held pre-78.1).

**Deliberately NOT done:**

- No decision on whether the six *should* now cache. That is a live question —
  the original rationale is stale — but it needs a measurement the current
  credit outage blocks. **Queued as its own masterplan step** per
  `feedback_queue_discovered_defects_in_masterplan`, not disclosed as prose and
  dropped.
- No change to the CC-rail path, `ClaudeCodeClient`, or the CLI's own caching.
- No change to the other 7 `make_client` callers (guarded by a test).
- 78.1 is **not** closed by this step. It additionally needs 78.2 (`--model`),
  which its own Q/A identified as blocking its criterion 2.

**Known limits of the evidence**, restated from `live_check_78.16.md` §3 so they
are not buried: the SDK boundary is faked in both captures (no credits, by
design); whether caching would engage at all on Haiku 4.5 is **unmeasured** and
straddles the documented floor; and `model='claude-haiku-4-5'` has zero
`provider='anthropic'` rows in 60 days, so this divergence is **latent today** and
becomes live when direct-API credits are restored.

---

## 5. Defect queued out of this step

Added to `.claude/masterplan.json` as **78.17** (research-gated, written for an
executor with no memory of this discovery):

> The `enable_prompt_caching=False` posture of the six C-block overlays rests on a
> rationale recorded in Apr 2026 (`handoff/archive/phase-23.1.2/phase-23.1.2-research-brief.md:301`:
> "the prompt will be different per-ticker per-quarter so caching provides no
> benefit") that is **factually stale**: caching is applied only to the SYSTEM
> block, and phase-25.B9 subsequently introduced `_HOUSE_INSTRUCTIONS`, a
> 19,026-char byte-identical prefix, expressly so the block would clear the
> cache-write minimum. Nobody revisited the six. Deciding this requires
> measurements currently blocked by the credit outage.

---

## 6. Q/A outcome

**PASS**, cycle 1, `violated_criteria: []`, `harness_compliance_ok: true`,
`certified_fallback: false`. Verdict transcribed verbatim into
`handoff/current/evaluator_critique_78.16.md`.

The Q/A ran its own 10-mutation battery in memory with production files
untouched — including four mutations I had not run (ClaudeClient class-default
flip; a service flipping to `True`; `make_client` inverting the value in both
directions; the caching branch deleted outright) — all RED. It also
recall-tested the completeness of "the six" against `git log -S` (symmetric
difference EMPTY) and structurally diffed both masterplan versions to prove the
immutable criteria and command were untouched by the test-file rename.

Six NOTE-level findings, none blocking. **N1–N4 were acted on in this cycle**
(live_check M4 row now states that the six revert guards stay GREEN under a
str-forcing stub and names the pair that closes that hole; the test docstring's
mis-described kill mechanism corrected to the measured one; the mutation script
fixed so it reproduces its own table row; the masterplan re-serialization
disclosed above). **N5** — three permanently-red count/diff pins — is queued as
step **78.18**. **N6** stands: my BigQuery "zero `provider='anthropic'` rows for
`claude-haiku-4-5` in 60 days" claim was **not** independently re-derived by the
Q/A (it needs an approval-gated query), so it rests on my measurement alone.

## 7. Artifacts

- `handoff/current/contract_78.16.md`
- `handoff/current/research_brief_78.16.md`
- `handoff/current/live_check_78.16.md`
- `handoff/current/evaluator_critique_78.16.md` (Q/A verdict, transcribed verbatim)
- Scratchpad (not checked in): `live_capture_78_16.py`, `mutate_78_16.sh`, `probe_caching_wire.py`
