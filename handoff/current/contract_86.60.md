# Contract -- step 86.60

**Step:** 86.60 -- every alternative signal is computed only for an UNRANKED
head-of-universe slice, so none of them can promote a ticker into the pipeline.
**P1, money path.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_f9b57c37-7d8`; 14 sources read in full, 119 URLs collected
against 123 distinct in the brief, audit-class dry after 10 rounds; brief
`research_brief_86.60.md`, 64,380 chars). Every line number in the spawn prompt
was stale by 6-8 and has been re-derived.

**Four findings change the plan. The first two change what the defect IS.**

**1. The step's premise is confirmed but its EMPHASIS is wrong: the problem is
not the slice's WIDTH, it is that the slice is unranked.**
`screen_universe` returns `results` UNSORTED (`screener.py:147/:240/:246`; the
only sort is at `:515` inside `rank_candidates`). So
`screen_data[: 2 * paper_screen_top_n]` is the **alphabetically-first passers of
the Wikipedia constituent table**, not a top-20 by any score. Widening the slice
would buy coverage of more alphabetically-early names; it would not make the
slice meaningful.

**2. THE CENTRAL FINDING: the repo ALREADY contains both architectures, and the
split is an accident.** `rank_candidates` scores the ENTIRE filtered universe
and sorts (`:491-492`), so **any signal delivered as a `rank_candidates` kwarg
over the whole universe is already an ENTRY PATH**. Measured split of the
thirteen signals: five take **no ticker argument** and are therefore entry paths
today -- PEAD (`autonomous_loop.py:565`), news (`:575`), sector_events,
sector_momentum, defense_signal. The other eight slice `screen_data[:20]` and
can only reorder it. All thirteen are passed to the SAME `rank_candidates` call,
so the two classes look identical at the call site; **the difference is entirely
upstream, in whether the producer was handed a slice.** It was never a design
decision -- it is eight copy-pasted expressions.

**3. The defect is LATENT, not active.** Read from the running process (pid
41635): of the signals involved, only the two genuine entry paths (PEAD, news)
are ON. So no currently-enabled overlay is being wasted on an unranked slice --
which means **this step must not claim a live P&L harm it cannot show.**

**4. Two things are UNMEASURABLE and are reported as such rather than
estimated.** (a) Past slice contents were never persisted -- summaries and logs
carry counts only, so criterion 1's "measure the slice across at least 3 cycles"
**cannot be satisfied retrospectively**; it requires instrumentation first.
(b) The eight overlay flags' running state is only partially readable
(`backend/.env` is permission-denied to the researcher, and `GET /api/settings/`
exposes 45 keys, none of them) -- the same blind spot 86.108 owns.

**External bound on the obvious fix:** widening the candidate pool naively
measures **-3.3 / -5.8 nDCG** in the retrieval literature, and cascades can lose
to routing. The recall ceiling is a theorem, so a fast signal is needed -- but
"compute everything for everyone" is measurably the wrong shape.

**Boundary with 86.108, stated so neither step claims the other's fix:** 86.60
owns the **empty-response diagnosis at the news-screen site** (`news_screen.py`)
and the instrumentation that makes it decidable. 86.108 owns the pipeline-wide
schema/transport contract. If the diagnosis lands on "the CC rail returned an
object with no `.text`", that is **handed to 86.108 as evidence, not fixed
twice**; if it lands on `refusal`, it is 86.60's alone. The brief's own
empty-response triage already rules OUT `max_tokens` and rate-limit (char-0),
leaving refusal / empty `end_turn` / missing `.text`.

## Hypothesis

The eight sliced overlays cannot promote a ticker because their producers are
handed a 20-name slice of an unsorted list before ranking happens. The fix is
architectural and cheap in principle -- move a producer above the slice so it
feeds `rank_candidates` over the full universe -- but it is **cost-bounded**,
because five of the eight are paid API or LLM calls per ticker. The step's first
duty is to make the currently-unmeasurable measurable, and to prove the
promotion barrier by DRIVING it rather than by reading the source.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the slice's contents are MEASURED across at least 3 cycles: log or reconstruct the actual tickers in screen_data[:2*paper_screen_top_n] per cycle and report how many are common to all of them, with the command used -- do NOT assert stability
2. the claim that overlays cannot promote an outside ticker is proven by DRIVING the pipeline, not by reading: construct a candidate ranked outside the slice with a maximal overlay signal and show it does not enter the analysis set on today's code
3. the news-screen parse failure is diagnosed to a cause and reported: an empty model response is either a prompt, a model-routing, or a rate-limit failure, and which one it is must be stated with evidence; a retry that also returns empty is not a diagnosis
4. if the slice is widened or the overlays are moved before ranking, the COST is measured and stated -- these are paid API and LLM calls, and the per-cycle cost delta must be reported alongside the coverage gain
5. flag-OFF / unchanged-config parity is proven: with the change disabled the candidate list is byte-identical to today's, demonstrated against an oracle
6. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/autonomous_loop.py\").read()); ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'`

**Immutable live_check:** `live_check_86.60.md` with the per-cycle slice
contents, the driven proof that an outside ticker cannot be promoted, and the
news-screen parse-failure diagnosis.

## Plan

**P1 -- criterion 1, and the honesty problem it contains.** Past slice contents
are **not reconstructable** -- the researcher established this rather than
assuming it. So criterion 1 is met in two parts, and the split is declared:
(a) ship the instrumentation that logs the slice per cycle, and (b) report
whatever cycles it has captured by evaluation time, stating n explicitly. **If
fewer than 3 cycles have elapsed, that is reported as "not yet satisfiable, n=k"
-- not padded, not estimated, not back-filled from counts.** A criterion that
needs data the system never recorded cannot be retro-satisfied, and saying so is
the correct outcome.

**P2 -- criterion 2, DRIVEN.** Construct a candidate ranked outside the slice
carrying a maximal overlay signal and show it does not enter the analysis set,
by executing `screen_universe` -> slice -> `rank_candidates` on fixture data.
Source inspection is explicitly excluded by the criterion. Include the
DISCRIMINATING control: the same fixture routed through one of the five genuine
entry-path producers (no ticker arg) DOES promote -- otherwise the test proves
only that the fixture never promotes anything.

**P3 -- criterion 3, diagnosed not retried.** Instrument the news-screen call
site to capture `stop_reason`/`stop_details` and whether the response object
carried `.text` at all. `char 0` already rules out `max_tokens` and rate-limit.
State which of the remaining causes it is, **with the captured evidence**; if
the answer is a transport-shaped one, hand it to 86.108 rather than fixing it
here.

**P4 -- criterion 4, cost before coverage.** Any move of a producer above the
slice must report the per-cycle call-count and dollar delta beside the coverage
gain. Five of the eight are paid API/LLM per ticker, so a naive lift from 20 to
~500 names is a ~25x fan-out. The literature's -3.3/-5.8 nDCG result is the
argument against doing it naively.

**P5 -- criterion 5, parity against an ORACLE.** With the change disabled, the
candidate list must be byte-identical to today's -- demonstrated against a
recorded oracle, not against two hand-picked examples.

**P6 -- criterion 6, mutations** with the control observed GREEN first and a
byte-identical restore; each cell scored, UNSCORABLE if its control was not
green.

## Scope honesty -- what this step does NOT do

- **It does not claim a live P&L harm.** The eight sliced overlays are OFF in
  the running process, so the defect is LATENT. Claiming the trade drought as
  its consequence would be unsupported -- 86.47 owns the drought cause, and two
  prior steps were already filed on drought theories their own research gates
  refuted.
- **It does not promote any flag and writes no `.env`.** Enabling an overlay is
  an operator-gated change and is recorded as a numbered ask.
- **It does not fix 86.108's schema/transport contract**, and it does not
  rebuild 86.59's momentum score.
- **It does not widen the slice as its remedy** without paying criterion 4's
  cost measurement first.

## References

`research_brief_86.60.md` (findings I-1..I-6, the thirteen-signal architecture
table, the unmeasurables, the 86.108 boundary);
`research_brief_86.59_rerun.md` (the momentum lock-in this compounds);
`contract_86.108.md` (the sibling boundary as this step accepts it).
