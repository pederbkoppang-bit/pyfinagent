# Research Brief -- phase-86.69

**Topic:** Root-cause a dated regression (2026-06-12..15) where 81% of LLM equity
analyses persist as empty `final_score=0.0` HOLD rows; design the fail-closed
persistence fix (a failed analysis must be recorded as an ABSENCE, never as the
most common valid verdict).

**Tier:** complex (caller-set; not self-selected)
**Audit-class:** NO (coverage reported for information only)
**Researcher:** Layer-3 researcher, Workflow rail, started 2026-08-17

---

## ENVELOPE (phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 22,
  "urls_collected": 31,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "Cause established by measurement, not inference: phase-60.1 (fa62b5fe, 2026-06-11) restored the full pipeline after gemini-2.0-flash's 2026-06-01 retirement had silently forced everything onto the healthy lite path. The _path provenance stamp 60.1 added dates the deploy exactly -- last unstamped row 2026-06-10 18:38Z, first stamped 2026-06-11 10:17Z -- and zero-scores start the same day, so the break is 2026-06-11, not 06-12..15. All 211 empty rows are _path=full; the lite path produced 0/19. Every one carries final_synthesis.error='Failed to parse final report.' (orchestrator.py:1687-1692) with no scoring_matrix, against 0/38 healthy rows -- so the DARK phase-61.2 guard's own predicate separates the live populations at 100% sensitivity and 100% specificity. Fabrication is at autonomous_loop.py:2179 (uppercase HOLD default) and :2190-2192, not the lite writer. The second half is the same cause: BUY conversion is 2.6% on full vs 36.8% on lite, a mixture effect, not a threshold change. Transport is the CC rail, whose --json-schema is post-hoc validated, not constrained.",
  "brief_path": "handoff/current/research_brief_86.69.md",
  "gate_passed": true
}
```

---

## Research Gate Checklist

**Hard blockers**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total -- **31** (9 full + 22 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 4 new findings + 1 qualifier
- [x] Full pages read (not abstracts) for the read-in-full set -- all 9 via
      `arxiv.org/html/` or native HTML; no `arxiv.org/pdf/` fetched
- [x] file:line anchors for every internal claim

**Soft checks**
- [x] Contradictions / consensus noted -- the SRE cascading-failures chapter
      argues AGAINST a naive fail-closed reading and is recorded as such
- [x] All claims cited per-claim
- [~] Internal exploration covered every relevant module -- **one real gap**:
      the LIVE value of `paper_synthesis_integrity_enabled` could not be read.
      `GET /api/settings/` exposes 45 keys and does not include it (measured);
      `backend/.env` is read-denied. Default is `False`. **Main must confirm the
      running value before assuming the flag is dark.**

**Other stated limits**
- The D4 transport mechanism is measured (rail tag, 257/259) but the *reason*
  the parse fails is not proven -- no finish reason is logged.
- The 2026-08 improvement (97.0% -> 51.5%) is **unexplained**; do not treat any
  single pre-window figure as a stable baseline.
- Steps 86.38 (lite-fallback census) and 86.7 (keychain credential) touch the
  same rail. Overlaps are noted in D3/D4; **their scope is not absorbed here**.

---

## Work log

- [x] Read `handoff/current/q1_binding_constraint_86.59.md` (given diagnosis)
- [x] (1) THE CAUSE -- section A: hypothesis (i) CONFIRMED, (ii)/(iii) refuted
- [x] (2) THE PERSIST CALL SITE -- section B, three fabrication sites
- [x] (3) THE SECOND HALF -- section A4: same cause, mixture effect
- [x] Existing partial fixes -- section C
- [x] Consumer set -- section F
- [x] External literature + recency scan -- 9 sources read in full

---

# A. THE CAUSE (criterion 1 of the internal scope) -- ESTABLISHED, and the break DATE MOVES

**Hypothesis (i) is CONFIRMED. Hypotheses (ii) and (iii) are REFUTED as the
primary cause.** The cause is a *restoration*, not a breakage: phase-60.1
brought the full/deep pipeline back into force, and the full path's synthesis
step fails silently while the lite path it replaced was healthy.

### A1. What phase-60.1 actually did

Commit `fa62b5fe`, `2026-06-11 13:06:26 +0200` (= 11:06:26 UTC),
*"phase-60.1: deep-pipeline restoration + honest-degradation alarm (AW-4)"*.
Its own message states the pre-existing condition verbatim:

> "Repin gemini-2.0-flash (discontinued on Vertex 2026-06-01) -> gemini-2.5-flash
> via live smoke"

and in `backend/config/model_tiers.py` (added at that commit):

> "gemini-2.0-flash was DISCONTINUED server-side 2026-06-01 (Google model
> lifecycle docs); every pinned reference silently 404'd from 06-02 and **the
> full pipeline fell back to the 2-call lite analyzer for the entire away
> week**."

So through 2026-06-10 the book was running **entirely on the lite path**. The
`GEMINI_WORKHORSE` repin (`model_tiers.py`, `GEMINI_WORKHORSE = "gemini-2.5-flash"`)
plus `AnalysisOrchestrator._GEMINI_FALLBACK = GEMINI_WORKHORSE`
(`backend/agents/orchestrator.py`) un-404'd the full pipeline and it started running again.

### A2. The deploy is dated FROM THE DATA, not from the commit

phase-60.1 also added the `_path` provenance stamp
(`autonomous_loop.py:3581-3582`, `full_report = {**full_report, "_path": ...}`).
That stamp is a **natural deploy marker**: no row before the deploy can carry it.

Measured (BigQuery, `financial_reports.analysis_results`, Python client + ADC,
2026-05-01..2026-08-16, this session):

| `full_report_json.$._path` | first row | last row | n |
|---|---|---|---|
| absent (`no_path_stamp`) | 2026-05-16 21:56:56Z | **2026-06-10 18:38:57Z** | 238 |
| present (`has_path_stamp`) | **2026-06-11 10:17:17Z** | 2026-08-14 19:32:26Z | 281 |

**A clean, gapless changeover at 2026-06-11 between 18:38Z (06-10) and 10:17Z.**
The first stamped row precedes the commit timestamp by ~49 min, consistent with
the commit message's own live-test note (*"LIVE: MU d1fbcc82 full pipeline
end-to-end (BQ `_path=full`, synthesis 46,661 chars)"*) -- the code was in force
in the running process before it was committed.

### A3. The break is 2026-06-11, one to four days EARLIER than the q1 diagnosis

Daily, re-measured (I did not restate the q1 dating; I re-derived it and it moved):

| Date | n | zero-score | `_path=full` | no stamp | BUY-class |
|---|--:|--:|--:|--:|--:|
| 2026-06-05 | 9 | **0** | 0 | 9 | 7 |
| 2026-06-08 | 5 | **0** | 0 | 5 | 4 |
| 2026-06-09 | 5 | **0** | 0 | 5 | 4 |
| 2026-06-10 | 5 | **0** | 0 | 5 | 4 |
| **2026-06-11** | 8 | **5** | **7** | **0** | 1 |
| 2026-06-12 | 5 | 3 | 3 | 0 | 2 |
| 2026-06-15 | 7 | 7 | 7 | 0 | 0 |
| 2026-06-17 | 5 | 4 | 4 | 0 | 0 |
| 2026-06-22..25 | 26 | 26 | 26 | 0 | 0 |

**The `_path` stamp and the zero-scores arrive on the SAME DAY, 2026-06-11.**
The q1 doc's "between 2026-06-12 and 2026-06-15" is off by one to four days
because it split on the BUY rate, and 06-11/06-12 still produced 1 and 2 BUYs
during the mixed changeover. The zero-score series is the sharper instrument: it
is exactly `0,0,0,0` then `5,3,7,4,...`. **Correct break date: 2026-06-11.**

### A4. The path, not the date, is the discriminator -- and it also explains THE SECOND HALF

| Regime | `_path` | n | zero-score | `HOLD` (upper) | `Hold` (title) |
|---|---|--:|--:|--:|--:|
| PRE (<=06-12) | absent | 238 | 87 | 55 | 72 |
| PRE | full | 10 | 8 | 6 | 2 |
| PRE | lite | 3 | 0 | 0 | 0 |
| **POST (>=06-15)** | **full** | **249** | **211** | **211** | 33 |
| **POST** | **lite** | **19** | **0** | 0 | 0 |

**All 211 zero-score POST rows are `_path=full`; the lite path produced ZERO
empties out of 19 rows.** And `HOLD`-uppercase count on POST-full equals the
zero count exactly (211 = 211), confirming section B2's line attribution
(`autonomous_loop.py:2179`) against an independent column.

BUY conversion **among real-score rows only** (`final_score > 0`), same windows:

| Regime | `_path` | real-score n | BUY | rate | mean score |
|---|---|--:|--:|--:|--:|
| PRE | absent (lite era) | 151 | 87 | **57.6%** | 6.14 |
| PRE | lite | 3 | 3 | 100.0% | 7.0 |
| POST | **lite** | 19 | 7 | **36.8%** | 5.89 |
| POST | **full** | 38 | 1 | **2.6%** | 5.79 |

**THE SECOND HALF IS THE SAME CAUSE, NOT A SEPARATE ONE.** The q1 doc treated
"BUY conversion among real-score analyses fell 3.5x" as an independent second
factor requiring a threshold/mapping explanation. It is not: the POST aggregate
(8 BUY / 57 real = 14.0% in my window; q1 reported 16.3% over a 3-day-shorter
window) is a **mixture** of a still-healthy lite path at 36.8% and a
near-dead full path at 2.6%. The lite path's own conversion fell 57.6% -> 36.8%,
a 1.6x drift, not 3.5x. **I searched for a June scoring/threshold change and
found none in the analysis pipeline** (`git log --since=2026-06-01
--until=2026-06-20 -- backend/` returns only away-ops/Slack/alerting commits
plus the 60.x series); the composition effect accounts for the gap without one.

Consequence for the fix: **arming the fail-closed guard is predicted to recover
BOTH halves**, because a `SynthesisDegradedError` routes to the lite fallback
(`autonomous_loop.py:2208-2243`), and the lite path measures 0/19 empty and
36.8% BUY in the SAME post-break window. That is a falsifiable prediction the
contract can state.

### A5. Hypotheses (ii) and (iii) -- REFUTED as the primary cause

- **(ii) model/provider change.** There WAS one -- `gemini-2.0-flash` ->
  `gemini-2.5-flash` in `fa62b5fe`. But it is not the mechanism: the model change
  is what made the full path *reachable*, and the failure is at the synthesis
  assembly, not at model resolution. A model outage would produce the lite
  fallback (which is healthy), not an empty full row. The 2.5-family retirement
  (2026-10-16) is a FUTURE trigger, not this one.
- **(iii) upstream `QuantAgent ... 'NoneType' object has no attribute 'get'`.**
  Real but far too small: 10 occurrences in 21 days against 104 analyses (~10%),
  and that error routes to `Full orchestrator failed ... falling back to lite`
  (`autonomous_loop.py:2208-2212`), i.e. it produces **lite** rows -- and lite
  rows are 0/19 empty. It cannot produce the 211 `_path=full` empties. The site
  is `functions/quant/main.py:264-266` (a Cloud Function, `logging.critical` then
  `yield f"ERROR: QuantAgent failed for {ticker_str}: {str(e)}"`).

---

# B. THE PERSIST CALL SITE (criterion 2 of the internal scope) -- ANSWERED

The q1 diagnosis inferred the writer from log co-occurrence. It is now read from
source. **There are THREE fabrication sites, not one, and the one that matches
the 211-row signature is NOT the one the diagnosis suspected.**

### B1. The row signature and which line produces each column

`211/211` POST rows: `final_score = 0.0`, `summary = ''`,
`debate_confidence = NULL`, `recommendation = 'HOLD'` (UPPERCASE).

| Column | Value | Produced at |
|---|---|---|
| `recommendation` | `'HOLD'` **uppercase** | `backend/services/autonomous_loop.py:2179` |
| `final_score` | `0.0` | `backend/services/autonomous_loop.py:2190-2192` |
| `summary` | `''` | `backend/services/autonomous_loop.py:3641-3645` |

**`autonomous_loop.py:2179` (the FULL path's return dict):**

```python
"recommendation": rec.get("action", "HOLD") if isinstance(rec, dict) else str(rec),
```

**`autonomous_loop.py:2190-2192`:**

```python
"final_score": synthesis.get(
    "final_weighted_score", synthesis.get("final_score", 0)
),
```

where `synthesis = report.get("final_synthesis", {})` (`:2156`) and
`rec = synthesis.get("recommendation", {})` (`:2172`), `risk =
synthesis.get("risk_assessment", {})` (`:2174`).

**`autonomous_loop.py:3639-3645` (`_persist_analysis`, the BQ writer):**

```python
final_score=None if _degraded else float(analysis.get("final_score") or 0.0),
recommendation=None if _degraded else (analysis.get("recommendation") or "Hold"),
summary=(
    ("DEGRADED: " + str(analysis.get("_degraded_reason") or "")[:400])
    if _degraded
    else (analysis.get("risk_assessment") or {}).get("reason", "") or ""
),
```

### B2. The UPPERCASE `HOLD` is the discriminator, and it exonerates the lite path

`_persist_analysis:3640` defaults to title-case `"Hold"`. The 211 rows carry
**uppercase `HOLD`**. Title-case is therefore NOT the source. Uppercase `HOLD`
is the literal default at `:2179` -- the **full-orchestrator return dict**. So
the empty rows are written by the **full path completing "successfully" with an
empty/errored `final_synthesis`**, not by the lite fallback and not by the
`_persist_analysis` `or`-default. This corroborates the q1 log split
independently: only 10 `Full orchestrator failed` lines in 21 days against 104
analyses (~10%), far too few to explain 81% empties via the fallback.

The q1 addendum's own single-cycle sample agrees: the healthy 2026-08-13 cycle
shows `009150.KS Hold 5.88`, `DELL Buy 6.58` (title-case, real scores) beside
`MRVL HOLD 0.0` (uppercase, zero). **Case tracks the failure**, exactly as
`_degraded_scoring_check` already assumes at `autonomous_loop.py:2801-2806`
(`rec.isupper()` is its "rail-down fallback tell").

### B3. Why the guard that exists does not fire here

`autonomous_loop.py:2163-2171` DOES contain a correct fail-closed guard:

```python
if getattr(settings, "paper_synthesis_integrity_enabled", False) and (
    not isinstance(synthesis, dict)
    or synthesis.get("error")
    or "scoring_matrix" not in synthesis
):
    ...
    raise SynthesisDegradedError(f"synthesis_error: {_syn_err}")
```

It is gated on `paper_synthesis_integrity_enabled`, which is
**`False` by default** (`backend/config/settings.py:206-208`, phase-61.2,
"DARK until operator promotion"), and its own description names this exact
defect: *"instead of persisting synthetic 0.0/HOLD (the defect that destroyed
two live BUY/0.62 consensuses on cycle 0725d2aa)"*. **The fix for 86.69 was
already written in phase-61.2 and has never been armed.**

Live-value caveat (`committed-is-not-in-force` class, inverted): I could NOT
read the running value. `GET /api/settings/` (measured this session, 45 keys)
does not expose it -- the same blind spot the q1 doc hit for the diversity
flags. Default is `False`; a live read needs another route.

---

# C. Existing partial fixes -- why they don't prevent the empty persist

| Guard | Anchor | Why it does not stop this |
|---|---|---|
| Degraded-scoring guard (phase-56.2) | `autonomous_loop.py:1300-1324` | **ALERT-ONLY.** It sets `summary["degraded"]`, logs, and pages P1. It runs at `:1301` -- *after* `dispatch_analyses` at `:1260`, which already called `_persist_analysis` inside `_run_and_persist_one` at `:1249`. The rows are in BigQuery before the guard reads them. |
| `_fold_degraded_for_trading` | `autonomous_loop.py:2772-2782` | Only drops analyses carrying `_degraded`, which is set **only** on the both-paths-failed branch (`:2252-2267`) and only when the integrity flag is ON. A fabricated `0.0/HOLD` has no `_degraded` key, so it passes through into `decide_trades`. |
| `_failforward_floor_ok` | `autonomous_loop.py:2661-2698` | Correct fail-closed shape (structural gate + degenerate-signature rejection), but it guards the phase-72.0.2 **fail-forward** path only, which is itself behind `paper_rail_failforward_enabled` (default `False`, `settings.py:198-200`). |
| `_degraded_scoring_check` predicate | `autonomous_loop.py:2785-2809` | Correctly identifies the signature (`final_score == 0`, or `confidence == 0` with an UPPERCASE rec). It is a **pure predicate with an alerting-only consumer** -- nothing routes its verdict back into the write. |

**The pattern:** every guard in this file detects the condition; none of them is
positioned upstream of the write. Detection is at `:1301`; the write is at
`:1249`. This is a Write-Audit-Publish ordering defect -- the audit runs after
the publish.

---

# D. WHY the full path produces an empty synthesis -- MEASURED, uniform, and it makes the dark guard a PERFECT discriminator

### D1. The 211 rows all carry one error string

BigQuery, POST window (>=2026-06-15), `_path='full'`, this session:

| bucket | n | `final_synthesis` absent | `.error` present | no `scoring_matrix` | no `recommendation` obj | mean blob length |
|---|--:|--:|--:|--:|--:|--:|
| **ZERO** (`final_score = 0.0`) | **211** | 0 | **211** | **211** | **211** | 205,850 |
| **REAL** (`final_score > 0`) | **38** | 0 | **0** | **0** | **0** | 346,708 |

Every zero row's `final_synthesis.error` is the same literal:
**`"Failed to parse final report."`** -- emitted at
`backend/agents/orchestrator.py:1687-1692`:

```python
logger.warning("Failed to parse final report, returning error.")
return {
    "error": "Failed to parse final report.",
    "synthesis_iterations": synthesis_iterations,
    "critic_degraded": critic_degraded,
}
```

reached when `_parse_json_with_fallback(draft_text, "Synthesis-Final")`
(`orchestrator.py:1681`, definition at `:308-316`) returns `None` on a
`json.JSONDecodeError`.

**The blob is ~206 KB, not empty.** The 28-agent pipeline RAN -- `bias_report`,
`conflict_report`, dissent registries are all present in the persisted JSON.
Only the FINAL SYNTHESIS PARSE failed. The analysis work was done and then
discarded at the last step, and the discard was recorded as a `HOLD`.

### D2. This makes the phase-61.2 guard a perfect discriminator -- measured, not assumed

The dark guard at `autonomous_loop.py:2163-2167` tests exactly
`synthesis.get("error") or "scoring_matrix" not in synthesis`. Against the real
production population that predicate is:

- **sensitivity 211/211 (100%)** -- every defective row matches;
- **specificity 38/38 (100%)** -- no healthy row matches.

I know of no stronger evidence that a fix is correct than the fix's own
predicate separating the live defective and healthy populations without error.
**The remedy for 86.69 is already written, already regression-tested, and has
never been switched on.**

### D3. The parse failure is SYSTEMIC to the Gemini structured-output path, not specific to synthesis

Counted over every rotated backend log (`handoff/logs/backend.log*` + the 7
`.gz` archives), `"<agent> returned invalid JSON"` (`orchestrator.py:315`):

| Agent | count |
|---|--:|
| Analyst | 926 |
| Critic | 602 |
| Moderator | 359 |
| Advocate | 342 |
| Judge | 314 |
| **Synthesis-Final** | **264** |
| Critic-Retry | 52 |

**2,859 structured-output parse failures in total.** Synthesis is only 9.2% of
them -- but it is the one that reaches the book, because synthesis is what
produces `final_score` and `recommendation`. The other 2,595 degrade the inputs
silently. **This is a much larger surface than 86.69's stated scope; it should
be QUEUED as its own step, not absorbed here.**

### D4. Mechanism: the synthesis asks for a schema guarantee the CC rail cannot give

**Correction, recorded rather than hidden.** My first hypothesis here was
Gemini-2.5-pro thinking-budget truncation -- the `-pro` exclusion in phase-60.1's
`llm_client.py` guard (`"-pro" not in bundle.model_name`) leaving the deep-think
model with default dynamic thinking under a 4,096-token output cap. **That
hypothesis is REFUTED and is replaced, not qualified.** It assumed the synthesis
runs on Gemini. Measured:

- `self.synthesis_client = make_client(deep_model_name, _synth_vertex, settings)`
  -- `backend/agents/orchestrator.py:658`.
- Live values read from the RUNNING process (`GET /api/settings/`, this session):
  `deep_think_model = 'claude-opus-5'`, `gemini_model = 'claude-sonnet-5'`.
  Neither is a Gemini model, so `GeminiClient`'s `ThinkingConfig` branch is never
  reached for synthesis.
- BQ `standard_model` on the full path is `claude-sonnet-4-6` for the entire
  regression window (243 of 249 POST rows) and `claude-sonnet-5` from 2026-08-14.

The real transport, measured from the persisted provenance tag phase-60.1 added
(`full_report["rail"]`, `autonomous_loop.py:2202`):

| `$.rail` | n (full path, >=2026-06-11) | zero-score | rate |
|---|--:|--:|--:|
| **`claude_code`** | **257** | **217** | **84.4%** |
| (absent) | 2 | 2 | 100% |

**The full pipeline runs on the Claude Code CLI rail.** And the phase-78.1
measurement of that rail (auto-memory `cc-rail-vs-claudeclient-78-1`,
measured 2026-07-25 against the CLI reference + Agent SDK doc) says:

> "`--json-schema` is POST-HOC validated with internal re-prompting, not
> constrained decoding... a run can end `subtype: success` with NO
> `structured_output` -- treat as failure."
>
> "No `--temperature` and no output-token flag exist... `temperature: 0.0` and
> `max_output_tokens` are silently unreachable on the rail (the repo already
> no-ops max_tokens at `claude_code_client.py:280`)."

So every field of `_SYNTHESIS_STRUCTURED_CONFIG` (`orchestrator.py:129-133`) --
`temperature: 0.0`, `top_k: 1`, `max_output_tokens: 4096`,
`response_mime_type: "application/json"`, `response_schema: SynthesisReport` --
is a **Gemini-shaped structured-output contract handed to a transport that
honours none of it.** The synthesis therefore has *no* schema guarantee, and
`_parse_json_with_fallback` is the only thing standing between a prose answer
and the book. That is a sufficient mechanism for a 84.4% parse-failure rate and
it needs no truncation story.

External corroboration that this is the expected outcome, not bad luck:
constrained decoding is what buys the guarantee, and post-hoc validation does
not -- "Structured output modes guarantee the schema, not the quality"
(https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/,
snippet); and where a grammar IS enforced the model pays for it in reasoning
(GSM8K natural-language 86.5% -> JSON 23.4% on Claude-3-Haiku,
https://arxiv.org/html/2408.02442v3, accessed 2026-08-17). **Both horns matter
for the contract: tightening the schema is not a free fix.**

Note this mechanism is *transport*-level, so it is **orthogonal to phase-60.1**.
60.1 did not create it; 60.1 made the full path reachable again, and the full
path had this defect waiting. See section E.

---

# E. 2026-06-11 is a RESUMPTION, not a first occurrence -- and the clean baseline is only 14 days

Daily, 2026-05-01..2026-06-12 (BQ, this session). This contradicts the framing
of the objective ("a dated regression"), so I state it plainly:

| Date | n | zero | `HOLD` | `Hold` | BUY | standard_model |
|---|--:|--:|--:|--:|--:|---|
| 2026-05-16 | 27 | **21** | 6 | 14 | 3 | gemini-2.5-flash |
| 2026-05-17 | 27 | 9 | 0 | 6 | 20 | claude-sonnet-4-6 |
| 2026-05-22 | 51 | **40** | 11 | 29 | 8 | gemini-2.5-pro |
| 2026-05-26 | 13 | 6 | 6 | 5 | 1 | (empty) |
| 2026-05-27 | 30 | 11 | 11 | 16 | 2 | claude-sonnet-4-6 |
| 2026-05-28 | 9 | **0** | 1 | 2 | 5 | claude-sonnet-4-6 |
| 2026-05-29 .. 2026-06-10 | 51 | **0** | 18 | 0 | 46 | claude-sonnet-4-6 |
| **2026-06-11** | 8 | **5** | 3 | 2 | 1 | claude-sonnet-4-6 |

**The empty-HOLD defect was running at ~78% on 2026-05-16 and 2026-05-22.** Its
May cause is *documented in the code itself* and is a DIFFERENT bug:
`autonomous_loop.py:2180-2189` records that the full path read
`synthesis["final_score"]` when the orchestrator writes `final_weighted_score`,
which *"cascaded into `_persist_analysis` writing 0 to
`analysis_results.final_score` for every full-path autonomous cycle since the
first clean run on 2026-05-22 (commit 29ab0ff6 phase-34.2)"*. That was fixed and
the zeros stop on 2026-05-28.

So the clean baseline is **2026-05-28..2026-06-10, 60 analyses, 0 zeros, 14
days** -- and 2026-06-01..06-10 of it is the accidental all-lite window created
by the `gemini-2.0-flash` retirement. **Only 2026-05-28..05-31 (4 days, 37
analyses) is a window in which the full path was both reachable and clean, and
its `_path` cannot be checked because the provenance stamp did not exist yet.**

Consequences the contract must absorb:
1. **Two independent bugs have produced the identical row signature** (`0.0` /
   `HOLD` / `''`) five weeks apart, with different root causes. A criterion that
   only asserts "the signature is gone" does not prove the mechanism was fixed.
2. **A pre/post comparison is confounded.** The full-path zero-rate is drifting
   without an identified change: 2026-06 **97.0%** (64/66), 2026-07 **96.0%**
   (120/125), 2026-08 **51.5%** (35/68), with daily values in August swinging
   0%-100% on n=3-12. `standard_model` is `claude-sonnet-4-6` throughout except
   2026-08-14 (`claude-sonnet-5`, 0/6 zero, n too small to attribute). **Do not
   size an improvement against a single pre-window number.**
3. Zero-rate is slightly worse for non-US listings (41/44 = 93.2%) than US
   (170/205 = 82.9%), consistent with the KR skipped-stages path but not
   explaining the bulk.

---

# F. CONSUMER SET -- the honest-absence shape is ALREADY plumbed end to end

Every writer of `analysis_results`:

| Writer | Anchor | Path |
|---|---|---|
| `_persist_analysis` | `autonomous_loop.py:3561-3661` | autonomous cycle (lite + full + degraded) |
| `save_report` | `db/bigquery_client.py:61,162` | the shared BQ sink |
| API-triggered rich write | `api/analysis.py:213-246`, `tasks/analysis.py:213` | manual /reports run -- a DIFFERENT path |

Readers of `final_score` / `recommendation` (the set criterion 3 needs):

| Reader | Anchor | Effect of a fabricated `0.0` / `HOLD` |
|---|---|---|
| `decide_trades` buy gate | `portfolio_manager.py:304` (`if rec not in _BUY_RECS`) | dropped -- can never become a candidate |
| buy-candidate ranking | `portfolio_manager.py:353,430` | `final_score = analysis.get("final_score", 0)`; sorts to the bottom |
| **signal-downgrade SELL** | `portfolio_manager.py:62,264` | **`_DOWNGRADE_RECS = {"HOLD","SELL","STRONG_SELL"}` -- a synthetic `HOLD` on a HELD position is a SELL trigger** |
| swap engine | `portfolio_manager.py:732,788,836` | a 0.0 holding looks like the weakest holding |
| Slack digest | `slack_bot/formatters.py:441-442,557-558` | publishes `0.0/10` + `rec or "DEGRADED"` |
| outcome tracking | `services/outcome_tracker.py:147,168` | trains reflections on a non-decision |
| signal attribution | `services/signal_attribution.py:185-195` | `str(...).upper() or "HOLD"` -- re-defaults again |
| API model | `api/models.py:99-100` | **already `Optional[float] = None`; comment: "NULL `final_score` <=> degraded row"** |
| Frontend types | `frontend/src/lib/types.ts:123-126` | **already `final_score: number \| null`; comment: "A null final_score identifies a degraded row"** |

**Two things follow, and both cheapen the fix:**

1. **The NULL representation is already plumbed** from BQ (nullable) through
   Pydantic (`api/models.py:100`) to TypeScript (`types.ts:125`). Writing
   `final_score=NULL, recommendation=NULL` for a failed analysis is a
   *supported* shape today, not a new contract. The only fabricating component
   is the writer.
2. **There is a live SEQUENCING HAZARD.** `_DOWNGRADE_RECS` contains `HOLD`, so
   a fabricated `HOLD` on a held position is a *sell* signal, not merely a
   non-buy. It is currently inert only because positions store trade reasons
   rather than analysis recommendations. `portfolio_manager.py:212-217` already
   emits a WARNING for this combination. **`paper_synthesis_integrity_enabled`
   MUST be armed before `paper_position_recommendation_fix_enabled` or
   `paper_recommendation_vocab_fix_enabled`**, or arming those converts 84% of
   analyses into SELL pressure on healthy holdings. This ordering belongs in the
   contract as an explicit constraint.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://arxiv.org/html/2606.08162 | 2026-08-17 | paper (2026) | WebFetch | *"Silent failures ... produce no exceptions or alarms -- detectible only through systematic measurement"*; "Data Consistency Decay" = individual ops succeed while aggregate state diverges; prescribes *"a monitoring layer that operates outside the probabilistic agent execution path"* |
| https://arxiv.org/html/2606.05806 | 2026-08-17 | paper (2026, ToolMaze) | WebFetch | 2x2 taxonomy: **implicit** failures = *"structurally valid but semantically corrupted outputs"*; agents show *"systemic over-trust in corrupted outputs"* and *"blindly propagate poisoned values"*; recovery rate on implicit vs explicit failures differs by **37.15 pts** |
| https://arxiv.org/html/2408.02442v3 | 2026-08-17 | paper (2024, EMNLP) | WebFetch | JSON-mode costs reasoning: GSM8K Claude-3-Haiku **86.5% -> 23.4%**; *"parsing failures aren't the primary cause"*; *"Overly restrictive schemas may hinder LLM performance"* |
| https://sre.google/sre-book/data-integrity/ | 2026-08-17 | official (Google SRE) | WebFetch | *"Bad data doesn't sit idly by, it propagates"*; the hard cases are *"low-grade corruption ... discovered weeks to months after"*; prescribes *"a system of out-of-band checks and balances"* |
| https://sre.google/workbook/data-processing/ | 2026-08-17 | official (Google SRE) | WebFetch | *"Avoid promoting a corrupt set of data to your low-latency frontends. Aim to catch these types of issues as early as possible, before they reach your users"*; job success != data correctness; separate data-correctness SLOs |
| https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-17 | official (Google SRE) | WebFetch | [ADVERSARIAL to a naive fail-closed] endorses *"Serve lower-quality, cheaper-to-compute results"*; and the warning that governs this fix: *"The code path you never use is the code path that (often) doesn't work"* |
| https://aws.amazon.com/blogs/big-data/build-write-audit-publish-pattern-with-apache-iceberg-branching-and-aws-glue-data-quality/ | 2026-08-17 | official vendor eng blog | WebFetch | WAP mechanics: staged writes *"aren't accessible to downstream users who can only access the main branch"*; audit runs on staging; only passing records are promoted |
| https://medium.com/@rosgluk/your-llm-json-is-valid-and-still-wrong-12dbbecf1fdc | 2026-08-17 | blog (2026-05) | WebFetch | *"A parse error fails loudly and immediately. A valid-but-wrong value fails downstream, hours later, in a place nobody expected"*; 4-layer mitigation ending in *"retry with specific errors, capped at 2-3 attempts, failing closed on refusal"* |
| https://ai.google.dev/gemini-api/docs/thinking | 2026-08-17 | official (Google) | WebFetch | `gemini-2.5-pro` and `-flash` thinking **"On"** by default; *"response pricing is the sum of output tokens and thinking tokens"*. Fetched to test the refuted D4 hypothesis; page has migrated to `thinking_level` and no longer carries the budget table |

## Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/pdf/2607.07405 | paper | "Reason Less, Verify More: Deterministic Gates ... Silent Policy-Violation" -- closest adjacent work; PDF-only, no `/html/` render; deferred |
| https://arxiv.org/pdf/2606.28733 | paper | "Agentic Abstention" -- abstention vs forced binary; PDF-only |
| https://arxiv.org/pdf/2603.23806 | paper | "Willful Disobedience: detecting failures in agentic traces"; PDF-only |
| https://arxiv.org/pdf/2606.22388 | paper | PlanBench-XL; tool-ecosystem scale, off-topic for persistence |
| https://arxiv.org/pdf/2604.10390 | paper | LLM-PRISM silent data corruption -- GPU faults in *training*, wrong layer |
| https://arxiv.org/html/2604.00726v1 | paper | SDC reliability in training -- same, wrong layer |
| https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/ | blog | *"Structured output modes guarantee the schema, not the quality"* -- quoted from snippet only, labelled as such |
| https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/ | blog | fallback chains; lower tier, covered by higher-tier sources |
| https://www.dremio.com/blog/streamlining-data-quality-in-apache-iceberg-with-write-audit-publish-branching/ | vendor blog | duplicate of the AWS WAP source |
| https://www.telm.ai/blog/what-is-write-audit-publish-in-apache-iceberg-and-why-it-matters-for-data-quality/ | vendor blog | duplicate WAP |
| https://bauplanlabs.com/post/write-audit-publish-ship-data-safely-move-faster | vendor blog | duplicate WAP |
| https://towardsdatascience.com/write-audit-publish-for-data-lakes-in-pure-python-no-jvm-25fbd971b17d/ | blog | duplicate WAP |
| https://www.y42.com/blog/gitops-for-data-2 | blog | duplicate WAP |
| https://medium.com/expedia-group-tech/chill-your-data-with-iceberg-write-audit-publish-746c9eb3db48 | blog | duplicate WAP |
| https://blog.dataengineerthings.org/how-does-netflix-ensure-the-data-quality-for-thousands-of-apache-iceberg-tables-76d3ef545085 | blog | Netflix WAP origin; covered |
| https://github.com/mem0ai/mem0/issues/5245 | issue tracker | exact analogue (partial embed failure logged at WARNING, no exception, caller cannot know); community tier |
| https://towardsdatascience.com/llm-fallbacks-break-agent-pipelines-i-built-the-missing-recovery-layer/ | blog | recovery-layer design; lower tier |
| https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk | community | lowest tier |
| https://devtoollab.com/blog/llm-structured-outputs-guide-2026 | blog | lowest tier |
| https://collinwilkins.com/articles/structured-output | blog | lowest tier |
| https://www.researchgate.net/publication/394297807_Understanding_Silent_Data_Corruption_in_LLM_Training | paywalled | training-layer SDC; wrong layer + paywall |
| https://projectsupply.in/blog/structured-output-llm-2026 | blog | lowest tier |

**URLs collected: 31** (9 read in full + 22 snippet-only).

### Search-query composition (three-variant discipline)

1. **Current-year frontier (2026):** *"LLM structured output parse failure
   fallback default value trading pipeline 2026 fail closed"*
2. **Last-2-year window (2024-2025):** *"agent tool failure silent default
   substitution 2025 abstention 'I don't know' null propagation evaluation"*
3. **Year-less canonical:** *"fail-closed persistence LLM pipeline failure
   recorded as default value silent data corruption"* and
   *"Write-Audit-Publish pattern data quality validate before publish Netflix
   Iceberg"* -- the WAP query is what surfaced the canonical prior art the
   codebase already names in a comment.

### Recency scan (2024-2026) -- PERFORMED

Result: **4 new findings in the window that materially shape the fix**, plus one
that qualifies it.

1. **ToolMaze (arXiv 2606.05806, 2026)** names this exact class: *implicit*
   failures -- "structurally valid but semantically corrupted" -- and measures
   that agents recover from them **37.15 points worse** than from explicit
   errors. pyfinagent's `0.0`/`HOLD` is a textbook implicit failure: the
   downstream consumer cannot tell it from a real neutral.
2. **The Entropy Principle (arXiv 2606.08162, 2026)** supplies the design rule
   the current guard violates: monitoring must sit *"outside the probabilistic
   agent execution path"*. The 56.2 guard is inside the cycle and downstream of
   the write.
3. **"Let Me Speak Freely?" (arXiv 2408.02442, 2024)** is the qualifier: hard
   schema constraints are not a free fix -- JSON-mode cost Claude-3-Haiku 63
   points on GSM8K. Tightening the synthesis schema to force parseability could
   degrade the judgment being parsed.
4. **Rost Glukhov (2026-05)** states the governing asymmetry in one line and
   arrives independently at *"failing closed on refusal"* after 2-3 retries.
5. Nothing in the window supersedes the WAP pattern (2017, Netflix) or the
   Google SRE data-integrity guidance; both remain the canonical prior art and
   are cited above as such.

### Consensus vs debate (external)

**Consensus** (all 9 sources): a failure recorded as a plausible value is worse
than a loud failure, because detection moves from seconds to weeks and the
corrupted value propagates. Google SRE and WAP agree the audit must run
**before** the data becomes consumer-visible.

**Genuine debate** -- and it is the one the contract must resolve:
`addressing-cascading-failures` **endorses serving degraded results**
(*"Serve lower-quality, cheaper-to-compute results to the user"*), which reads
as an argument FOR the current fabricate-and-continue behaviour. The
reconciliation is that Google's degraded results are *labelled and bounded*,
whereas pyfinagent's are *indistinguishable from a real verdict* -- and the same
chapter supplies the strongest argument for arming the dark flag now:
*"The code path you never use is the code path that (often) doesn't work."*
`paper_synthesis_integrity_enabled` has been dark since phase-61.2 and has never
run in production.

### Pitfalls (from the literature, mapped to this fix)

1. **Do not fix it by tightening the schema alone** -- arXiv 2408.02442 measures
   up to a 63-point reasoning cost from JSON-mode.
2. **Do not rely on the CC rail's `--json-schema`** -- it is post-hoc validated
   with re-prompting, not constrained decoding, and a run can end
   `subtype: success` with no structured output (phase-78.1 measurement).
3. **Do not alert on job success** -- Google SRE workbook: job completion is not
   data correctness. The cycle "succeeded" on every one of the 211 rows.
4. **Do not leave the new path unexercised** -- SRE cascading-failures; the
   honest-absence branch needs a live exercise, not only unit tests.
5. **Do not audit after publishing** -- WAP; the 56.2 guard's position at
   `autonomous_loop.py:1301` (after the write at `:1249`) is the defect.

### Application to pyfinagent

| External finding | pyfinagent anchor | Implication |
|---|---|---|
| Implicit failure / poisoned value (ToolMaze) | `autonomous_loop.py:2179,2190-2192` | The full-path return dict is the poisoning site; `rec.get("action","HOLD")` and `synthesis.get(...,0)` manufacture a valid-looking verdict from an error dict |
| Audit before publish (WAP, SRE workbook) | guard at `:1301` vs write at `:1249` | Move the assertion upstream of `_persist_analysis`, or make the write itself refuse -- the guard is correct and merely mis-ordered |
| Record absence, never a default | `_persist_analysis:3639-3645` `or 0.0` / `or "Hold"` | NULL is already legal end-to-end (`api/models.py:100`, `types.ts:125`); the `or`-defaults are the only fabrication |
| Monitoring outside the execution path (Entropy Principle) | `_degraded_scoring_check:2785-2809` | A correct pure predicate with an alerting-only consumer; give it a *blocking* consumer |
| Schema constraints cost reasoning (2408.02442) | `_SYNTHESIS_STRUCTURED_CONFIG:129-133` | Prefer fail-closed persistence over a stricter grammar |
| Fail closed after bounded retry (Glukhov) | `claude_code_empty_retry_max` (`settings.py:192-196`) | The retry knob exists and is *"effective only when `paper_synthesis_integrity_enabled` is True"* -- same flag |

### Recommended shape for the contract (research view; Main owns PLAN)

1. **Arm `paper_synthesis_integrity_enabled`.** Measured 100% sensitivity and
   100% specificity on the live 211/38 split. It routes the error to the lite
   fallback, which measures **0/19 empty and 36.8% BUY** in the same window.
2. **Delete the fabrication independently of the flag** at
   `autonomous_loop.py:2179` and `:2190-2192` -- `rec.get("action","HOLD")` and
   `synthesis.get("final_weighted_score", synthesis.get("final_score", 0))`
   should not manufacture a verdict from a synthesis with `error` set. A flag
   that must be ON for correctness is a flag that will be OFF one day.
3. **Reorder the guard** so the degraded check runs before `_persist_analysis`.
4. **Encode the sequencing constraint**: integrity flag before the vocab /
   position-recommendation flags (section F.2).
5. **Do not size success against a single pre-window number** (section E.2).
6. **Queue, do not absorb:** the 2,595 non-synthesis parse failures (D3), and
   the CC-rail-vs-Gemini-config mismatch (D4). Both are larger than 86.69.


