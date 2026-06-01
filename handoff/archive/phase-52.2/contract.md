# Contract -- phase-52.2: wire the 52wh momentum tilt LIVE (config-gated, default OFF)

**Step id:** 52.2 | **Priority:** P1 | **depends_on:** 52.1 | **Date:** 2026-06-01
**harness_required:** true | **$0 LLM** | no pip | **flag DEFAULT OFF -> byte-identical live; ENABLE deferred**

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `a4a60b262ce3351b7`: gate_passed=true, tier moderate, 6 sources read in full, 15 URLs, recency scan, 4 internal files). Decisive (all file:line CONFIRMED against the current tree):
- **Insertion point: `screener.py:473`** -- the gated 52wh-tilt post-pass goes AFTER the `sector_neutral` block (ends :472) and IMMEDIATELY BEFORE the final `scored.sort` (:474). It runs after every composite + RSI/vol multiplier + the multidim/sector_neutral passes; the single final sort honours it. Cross-sectional centered tilt (universe mean over the same `scored` set).
- **Flag/kwarg pattern (mirror `multidim_momentum`):** add `momentum_52wh_tilt: bool = False, momentum_52wh_tilt_k: float = 0.5` to `rank_candidates` (signature :249-273, beside multidim at :261); guard `if momentum_52wh_tilt and scored:` -> skipped when False. Helper `_apply_52wh_tilt(scored, k)` mirroring `_apply_multidim_momentum` (:491), FAITHFUL to the 52.1 reference `scripts/ablation/sector_neutral_replay.py hi52_tilt_basket` (same `composite*(1+k*(p-mean))`, mean over non-None pct_to_52w_high, missing -> tilt 1.0 no-op) so the LIVE ranking == the 52.1-MEASURED ranking.
- **settings (`settings.py` :334-338):** `momentum_52wh_tilt_enabled: bool = Field(False, ...)` + `momentum_52wh_tilt_k: float = Field(0.5, ...)`.
- **call-site (`autonomous_loop.py:638-654`, the ONLY live caller):** add `momentum_52wh_tilt=getattr(settings,"momentum_52wh_tilt_enabled",False)` + `momentum_52wh_tilt_k=getattr(settings,"momentum_52wh_tilt_k",0.5)` beside the multidim args (:649).
- **`pct_to_52w_high` is ALREADY at rank time** -- computed in `screen_universe`'s per-ticker loop (screener.py:210-214), set on EVERY row (:228) for all screened names; flows screen_universe -> rank_candidates. NO threading needed. `min_periods=20` -> None for short windows -> helper treats as tilt 1.0.
- **Byte-identity when OFF:** the post-pass is the last mutation before the sort, fully behind `if momentum_52wh_tilt and scored:`, kwarg default False. The helper writes `composite_score_raw` ONLY when it runs -> its ABSENCE on OFF-path rows witnesses the OFF path never touched `scored`.
- **Regression risk when OFF (this step): NONE** (byte-identical by construction). The enable-decision risk (momentum alpha decay; the +0.05 is 1 of 5 configs -> must be DSR-deflated + pass a post-Monday OOS gate) belongs to the DEFERRED enable, NOT this step. Reassuring: crowded MOMENTUM has LOWER crash risk (0.38x), unlike reversal factors (arXiv 2512.11913).

## Hypothesis
Adding a config-gated `_apply_52wh_tilt` post-pass to `rank_candidates` (kwarg default OFF, settings flag default False, plumbed from autonomous_loop) makes the 52.1-measured +0.05-Sharpe 52wh tilt a production-ready, single-flag-reversible LIVE overlay -- while the live engine stays BYTE-IDENTICAL until the flag is flipped (which is a SEPARATE, post-Monday-baseline, operator-gated action). The live ranking under the flag == the 52.1-measured ranking (same tilt logic).

## Success criteria (IMMUTABLE -- verbatim from masterplan step 52.2)
1. screener.rank_candidates gains a config-gated 52wh-tilt post-pass (centered multiplicative tilt, k from settings) that reproduces the 52.1 replay's ranking logic; the live momentum composite is otherwise unchanged
2. with the flag OFF (default), rank_candidates is BYTE-IDENTICAL to today (the tilt post-pass is skipped) -- proven by a test asserting identical ranked order with the flag off; the working US engine is NOT regressed
3. the flag is plumbed from settings (momentum_52wh_tilt_enabled default False) through autonomous_loop's rank_candidates call; NO flag flip in this step (enable is a separate operator-gated action)
4. live_check_52.2.md records the byte-identity proof (flag OFF) + the tilt-active behavior (flag ON) + the deferred-enable plan

**Verification command:** `pytest backend/tests/test_phase_52_2_live_tilt.py` + `ast.parse(screener.py, autonomous_loop.py)` + `test -f live_check_52.2.md`.
**live_check:** REQUIRED -- byte-identity (flag OFF) + tilt-active (flag ON) + the deferred-enable plan; NO flag flip.

## Plan steps (GENERATE)
1. **screener.py:** add `_apply_52wh_tilt(scored, k)` (mirror `_apply_multidim_momentum`): centered multiplicative tilt on `composite_score`, mean over non-None `pct_to_52w_high`, missing -> 1.0; writes `composite_score_raw` before tilting (the OFF-witness field). Add the two kwargs to `rank_candidates` (:261-ish). Insert `if momentum_52wh_tilt and scored: _apply_52wh_tilt(scored, momentum_52wh_tilt_k)` at :473 (after sector_neutral, before the sort).
2. **settings.py:** `momentum_52wh_tilt_enabled` (False) + `momentum_52wh_tilt_k` (0.5), beside multidim (:334).
3. **autonomous_loop.py:638-654:** pass `momentum_52wh_tilt`/`momentum_52wh_tilt_k` from settings (default OFF).
4. **test** `backend/tests/test_phase_52_2_live_tilt.py`: (a) BYTE-IDENTITY -- `rank_candidates(data)` ranked order == `rank_candidates(data, momentum_52wh_tilt=False)` AND no `composite_score_raw` on any OFF-path row; (b) ON flips the ranking (a high-52wh name with equal composite ranks above a low-52wh one) + matches the 52.1 `hi52_tilt_basket` logic; (c) missing pct -> tilt 1.0 no-op.
5. **Verify:** pytest; ast.parse(screener.py, autonomous_loop.py); capture the byte-identity (OFF) + tilt-active (ON) proofs into `live_check_52.2.md` + the deferred-enable plan. Confirm the live default path is byte-identical.
6. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 52.2 -> done.

## Safety / scope notes
- **Flag DEFAULT OFF -> the live engine is BYTE-IDENTICAL after this step** (no behavior change; the +20% engine is untouched). The ENABLE (flag flip via settings/.env + restart) is a SEPARATE operator-gated action, deferred to AFTER Monday's multi-market baseline is measured.
- The live tilt logic must MATCH the 52.1 `hi52_tilt_basket` (same formula) so the measured +0.05 is what we'd get live.
- Enable-decision caveats (for the deferred step, NOT now): DSR-deflate the +0.05 (1 of 5 configs); post-Monday OOS confirm; k=0.5 (the milder/plateau choice).
- $0 LLM; no pip; no spend; no DROP/DELETE; no flag flip in this step.

## References
- handoff/current/research_brief.md (52.2 gate) + .claude/agent-memory/researcher/project_52wh_tilt_live_wiring.md
- backend/tools/screener.py:249-273 (rank_candidates sig), :434-440 (multidim pass), :448-472 (sector_neutral pass), :473 (insert), :474 (sort), :491-550 (_apply_multidim_momentum mirror), :210-214/:228 (pct_to_52w_high on every row)
- backend/config/settings.py:324/334 (gated-flag pattern); backend/services/autonomous_loop.py:638-654 (the only live caller)
- scripts/ablation/sector_neutral_replay.py hi52_tilt_basket (the 52.1 reference logic); George-Hwang 2004; arXiv 2512.11913 (momentum lower crash risk); Bailey-LdP DSR (deferred enable)
