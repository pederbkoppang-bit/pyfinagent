# live_check_59.1 — Fable 5 adoption: evidence

**Step:** 59.1. **Date:** 2026-06-11. **Required shape (masterplan):** the pin diff map + unchanged-roles list + the restart/roster-verify instruction + test output.

## A. Pin diff map (old → new per role)

| Layer | Role / file | Old | New | Rationale |
|---|---|---|---|---|
| L3 | `.claude/agents/researcher.md` | `model: opus` (→4.8), maxTurns 30 | **`model: fable`**, **maxTurns 40** | rare-event gate role; 2 observed stalls at 30; effort: max retained (documented over-spec; Fable baseline is `high`) |
| L3 | `.claude/agents/qa.md` | `model: opus` (→4.8), maxTurns 12 | **`model: fable`**, **maxTurns 30** | evaluator gate; FIVE mid-evaluation stalls at 12 on 2026-06-10/11 (tool-use 20-26/run); effort: max retained |
| L2 | `model_tiers.py::mas_main` | claude-opus-4-8 | **claude-fable-5** | operator-paced Slack/iMessage orchestrator (rare-event; agent_definitions.py:181) |
| L2 | `model_tiers.py::autoresearch_strategic` | claude-opus-4-8 | **claude-fable-5** | nightly 2am cron memo (rare-event; long-horizon synthesis = Fable's strongest delta) |
| L2 | `ticket_queue_processor.py::main, q-and-a` | claude-opus-4-8 | **claude-fable-5** | operator-paced tickets ~1-2/day → ~$0.18/day at $10/$50 (researcher cost math; decision recorded in the map comment) |
| L2 | `model_tiers.py::EFFORT_SUPPORTED_MODELS` | (no fable) | **+ claude-fable-5** | REQUIRED: without it `model_supports_effort()` returns False and `llm_client.py:1481` silently drops the effort param |
| L2 | `model_tiers.py::MODEL_EFFORT_FALLBACK` | (no fable) | **+ ("claude-fable-5", "xhigh")** | researcher-grounded tier (doc baseline `high`; project posture xhigh; EFFORT_DEFAULTS role overrides still win) |
| docs | `CLAUDE.md` effort policy | phase-29.2 flat-fee rationale | **additive Fable 5 block** ($10/$50; June-23 Max-credit change SUPERSEDES flat-fee; classifier-fallback note; alias + version floor; /model fable) — Opus 4.8 history preserved | criterion 3 |

## B. Unchanged roles (cost discipline — explicitly NOT repinned)

`mas_qa` (per-ticker analyst — the metered volume role), `mas_communication`, `mas_research`, `autoresearch_fast`, `autoresearch_smart`, ticket `research` (Sonnet 4.6), all `gemini_*` roles (Gemini-locked: Vertex Search/Grounding/schemas), the lite analyzers (`settings.gemini_model`), and the 28 per-ticker pipeline agents. Unit-tested: `test_metered_roles_unchanged`.

## C. Restart / roster-verify instruction (Layer-3 snapshot semantics)

The Agent-tool roster snapshots at session start: **the `model: fable` + maxTurns pins take effect at the NEXT Claude Code session** (or `/clear`). This session's remaining researcher/qa spawns run the OLD snapshot (Opus 4.8 / old caps) — acceptable: the pins change models, not protocol. **Next session: run `scripts/qa/verify_qa_roster_live.sh`** to confirm the new frontmatter is in the live snapshot (retry-on-FAIL doctrine in `docs/runbooks/per-step-protocol.md` §4). Governance: the operator's 8 in-session answers (2026-06-11) are the pre-approval of record for these agent-file edits (separation-of-duties satisfied).

## D. Version + syntax validation (researcher-grounded)

- `claude --version` → **2.1.172** (floor for the `fable` alias is 2.1.170 — clears).
- Frontmatter alias `model: fable` documented in the sub-agents + model-config docs; `effort: max` valid on Fable (effort table low→max); `maxTurns` is the documented stall lever.
- Fable 5 facts recorded: `claude-fable-5`, $10/$50 per Mtok, Max-free June 9-22 then usage credits, classifier fallback to Opus 4.8 (>95% sessions unaffected).

## E. Test output (verbatim)

```
$ python -m pytest backend/tests -k 'fable or model_tiers or phase_59' -q
7 passed, 773 deselected, 1 warning in 2.85s
$ python -m pytest backend/tests/test_agent_map_live_model.py -q        # the test the repin would have broken
7 passed in 0.07s
$ python -m pytest backend/tests -q                                      # FULL suite (the -k false-green guard)
762 passed, 12 skipped, 6 xfailed, 1 warning in 76.81s
```

New tests (`test_phase_59_1_fable_adoption.py`, named to enter the `-k` net per the researcher's false-green finding): rare-event resolution, metered-roles-unchanged, effort-not-silently-dropped (`model_supports_effort` + fallback tier + role override), ticket-map pins, Layer-3 frontmatter (model/caps/effort + economics/restart annotations), CLAUDE.md additive check. `test_agent_map_live_model.py` updated mas_main → claude-fable-5 in the same change.
