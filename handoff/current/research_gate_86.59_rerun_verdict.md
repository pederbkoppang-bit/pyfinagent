# Research gate — step 86.59 RE-RUN — **PASSED**

**Run:** `wf_ff8717e8-ccf` | **Date:** 2026-08-14 ~05:55 CEST
**Brief:** `handoff/current/research_brief_86.59_rerun.md` (37,733 chars)
**2 agents, 0 errors, 0 empty returns, 185,412 subagent tokens, 606s**

> Verdict transcribed from the captured return in the same turn it landed. The prior brief
> (`research_brief_86.59.md`) is **preserved unmodified** — the re-run wrote to a new path.

---

## Why the first gate failed, measured before the re-run

Not merely the URL-count mismatch the goal described. **The envelope carried no
`sources_read_in_full` array at all**, so `enforceGate` could not corroborate a single
claimed URL — that alone fails, independent of counts. It also claimed
`urls_collected: 30` / `snippet_only_sources: 24` while **13** distinct URLs appeared
anywhere in the 28,878-byte file.

**I did not patch it.** The 17 missing URLs are not in the artifact and inventing them is
fabrication; editing a brief so its own gate passes would be Main authoring research
evidence. The spawn prompt said explicitly: *"DO NOT INFLATE and DO NOT INVENT URLs: if you
can only corroborate 13, report 13 and let the floors decide honestly."*

## The recomputed result

| check | value |
|---|---|
| `sources_floor_ok` | **8 ≥ 5** |
| `urls_floor_ok` | **54 ≥ 10** |
| `urls_collected_corroborated` | **54 ≤ 54 distinct URLs in the brief** |
| `all_8_claimed_sources_present_in_brief` | yes, 0 missing |
| `brief_status_in_brief` | `COMPLETE` |
| `recency_section_present` | yes *(structural, not a judgement of substance)* |
| **`self_report_disagreed`** | **false** — agent said `true`, script recomputed `true` |
| `rail_dropped` | `null` |

**Nothing was loosened.** The floors are unchanged at 5 and 10; the difference is that the
envelope is now corroborated by its own text.

---

## The finding that changes the fix

**Z-scoring will NOT stop the slate repeating, and that is the decisive result.**

Standardising each trailing-return horizon is a **per-horizon affine transform**. It
corrects declared-vs-effective weights — real, and worth fixing — but leaves the ranking a
**monotone function of the same slow state**, so the slate stays sticky. Subtracting a
common factor component is **not** affine cross-sectionally: measured, FF5F residualisation
flips a weekly raw-return *contrarian* result into momentum (**Sharpe 1.3392**).

**So the weights bug and the repetition are different defects, and fixing the first does
not touch the second.**

### Adversarial finding, recorded because it cuts against the step's own premise

**No residual-momentum source endorses a daily-varying slate.** Alkshaik (2025 / FoFI-2026)
deliberately rebalances residual momentum **semi-annually**.

### The premise splits three ways

| | defect? |
|---|---|
| (a) weights hit raw returns of differing dispersion | **yes — fix** |
| (b) no orthogonal/fast signal exists | **yes — fix** |
| (c) the same 4–6 names daily | **NOT a defect** — correct for a slow predictor; amplified only by `paper_analyze_top_n=5` |

**Turnover buys more budget than it spends:** break-even cost **0.93–1.49** vs **0.87**
traditional (Novy-Marx/Velikov framing). Graef/Hoechle/Schmid: the firm-specific component
drives momentum; industry-neutral performs similarly.

---

## Internal path corrections — INDEPENDENTLY VERIFIED BY ME

The gate contradicted a path I had been repeating from the goal. I checked rather than
accepted it, and **the gate is right**:

| claim | verified |
|---|---|
| `backend/services/screener.py` | **DOES NOT EXIST** |
| real path | `backend/tools/screener.py` (34,595 bytes) |
| `candidate_picker.py` | **absent from the repo** |
| `_zscore` | defined `:532`, called **only** at `:607-610` |
| gating flag | `settings.py:478` `multidim_momentum_enabled: Field(False)` — **default OFF** |
| override? | **not in `backend/.env`** — positive control: `.env` has 52 override lines incl. `paper_risk_judge_reject_binding` |
| wiring | `autonomous_loop.py:997` `getattr(settings, "multidim_momentum_enabled", False)` |
| `paper_analyze_top_n` | `settings.py:407` = **5** |

**Consequence:** the only `_zscore` call sites sit behind a dark flag, so the live composite
does hit raw returns. The prior brief's line anchors were **stale by ~10 lines** (composite
`:301-305`, sort/slice `:491-492`).

**I propagated the wrong path into `goal_next_2026-08-14.md` and into two messages.**
Corrected in the goal.

---

## What this does NOT license

- **It does not reopen the ranking work.** Q1's answer stands: the binding constraint is
  upstream analysis emptiness (**86.69**), and finding (c) above independently says the
  daily repetition is *not* the defect. This gate closed a **research** gap; it did not
  change what should be built next.
- **No code was changed.** No contract written, no GENERATE started.
- **`coverage.dry` is false** (3 rounds, 0 dry) — informational only, since the step is not
  audit-class. A future audit-class step on this surface would need the loop-until-dry pass.
- The recency check is **structural** — a section exists; the gate does not judge whether
  the scan was substantive.
