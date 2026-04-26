# Per-Model Skill Optimization -- Forward Design

**Phase:** 21.3 (design doc, no implementation this cycle)
**Date:** 2026-04-26
**Status:** Design north star for a future cycle. Operator request:
"in the future we would have different LLM models handling different
tasks using hooks, skills optimized for that specific model chosen in
settings."

---

## Problem statement

Today every Layer-1 skill prompt at `backend/agents/skills/*.md` is
written generically -- the same prompt is sent to whichever model the
orchestrator routes to. But Sonnet, Opus, Gemini-Flash, and GPT-4 each
have different strengths: token economy, JSON discipline, reasoning
depth, instruction-following style. A "one prompt fits all models"
strategy leaves quality on the table for whichever model the operator
picks in Settings.

When phase-21.1's `apply_model_to_all_agents` flag is on, this gap
becomes more visible: 28 enrichment skills all routed to (say)
`claude-haiku-4-5` get the same prompt that was tuned for
`gemini-2.0-flash`.

## Goal

When the operator picks a model in Settings, every skill executes the
prompt variant best-tuned for that model -- automatically, transparently,
with no per-skill manual override.

## Non-goals

- Manual per-skill model overrides (already possible today via
  `agent_definitions.py`; this design is about prompt-content variation,
  not model-routing variation).
- A/B testing framework -- separate cycle.
- LLM-as-judge for prompt-variant selection -- separate cycle (10.7.7
  evaluator gate already addresses one slice).

## Recommended architecture

### 1. Prompt-variant directory layout

```
backend/agents/skills/
  bull_agent/
    base.md              # default; used when no model-specific variant exists
    claude.md            # tuned for any claude-* model
    gemini.md            # tuned for any gemini-* model
    haiku.md             # tuned specifically for claude-haiku (smaller token budget)
    gpt.md               # tuned for any gpt-* model
  bear_agent/
    base.md
    claude.md
    ...
```

Convention: filename = model family or specific model ID. Resolution
falls back from most-specific (model ID) to family (provider) to base.
Most skills only need `base.md` and one or two variants; rare skills
that need fine-grained tuning can add specific `model-id.md` files.

### 2. Resolution layer

New module: `backend/agents/skill_loader.py`

```python
def load_skill(skill_name: str, model_id: str) -> str:
    """Return the best-matching prompt variant for the given model.

    Resolution order (first hit wins):
      1. backend/agents/skills/{skill}/{model_id}.md   (e.g. claude-haiku-4-5.md)
      2. backend/agents/skills/{skill}/{family}.md     (e.g. claude.md, gemini.md)
      3. backend/agents/skills/{skill}/base.md         (always exists)
    """
```

Family extraction: prefix-match `claude-*` -> "claude", `gemini-*` -> "gemini",
`gpt-*` / `o1-*` / `o3-*` -> "gpt", else "base".

### 3. Hook integration point

The orchestrator's existing `_load_skill_prompt(name)` call site is the
single integration point. Replace:

```python
prompt = Path(f"backend/agents/skills/{name}.md").read_text()
```

with:

```python
from backend.agents.skill_loader import load_skill
prompt = load_skill(name, resolve_model(role))
```

This is a 2-line change at each call site. Backward compat: if no
variant directory exists for a skill, fall back to the legacy single
.md file path.

### 4. Migration plan

Phase A (1 cycle): build `skill_loader.py` + tests + migrate ONE skill (`bull_agent`) as proof.

Phase B (1 cycle per skill batch): migrate skills in batches of 4-6, each
batch gets its own A/B against base.md to confirm the variant produces
better output.

Phase C (1 cycle): once 80% of skills are migrated, deprecate the
legacy single-file fallback; require all skills to have at least
`base.md`.

## Hook scaffolding (already in place)

phase-21.1 + 21.2 give us the prerequisites:
- `apply_model_to_all_agents` toggle resolves a single chosen model
- `resolve_model(role)` is the single chokepoint
- `_GEMINI_LOCKED_ROLES` already documents which roles can't switch

The skill loader plugs into the same chokepoint -- it asks
`resolve_model(role)` for the active model, then loads the matching
prompt variant.

## Risks

1. **Variant drift** -- if `bull_agent/claude.md` and `bull_agent/gemini.md` diverge in semantic intent (not just style), debugging becomes hard. Mitigation: each variant must declare in a YAML front-matter what it asks for; a CI check compares the front-matter across variants and fails if the "question being asked" differs.
2. **Maintenance overhead** -- 4 variants × 28 skills = 112 files. Mitigation: only most-impactful skills need variants; the resolution falls back to base.md so partial migration is fine.
3. **Cost-tier confusion** -- if the operator switches from Sonnet to Haiku via the toggle, are they getting the Haiku-tuned variant automatically? YES, by design (this is the point).
4. **A/B measurement** -- knowing whether a variant actually performs better requires an A/B framework. That's out of scope for the loader cycle but a prerequisite before building >2 variants per skill.

## Engineering cost

| Cycle | Scope | Effort |
|-------|-------|--------|
| 21.3.1 | `skill_loader.py` + tests + migrate `bull_agent` as proof | 1 day |
| 21.3.2 | Migrate 5 most-used skills (bear, devil's advocate, synthesis, risk_judge, deep_dive) | 2 days |
| 21.3.3 | A/B framework for variant evaluation | 2 days |
| 21.3.4 | Migrate remaining skills batch-by-batch | 4-6 days (split across cycles) |
| 21.3.5 | Deprecate legacy single-file fallback | 0.5 day |

Total: ~10 days for full migration. Can be shipped incrementally; phase
A unblocks the path even if phases B-E are deferred.

## Decision

This is the design north star. **No implementation this cycle.**
Phase-21.3 is the design-doc deliverable; phases 21.3.1-21.3.5 (above)
are the implementation arc that gets opened when operator decides to
prioritize.

## Cross-references

- `backend/agents/skills/` (28 active skills today)
- `backend/agents/orchestrator.py:806-870` (skill prompt load + compaction)
- `backend/config/model_tiers.py` (resolve_model + EFFORT_DEFAULTS pattern)
- `.claude/agents/skills/` (Claude Code skill loader pattern -- inspiration)
- phase-21.1 (apply_model_to_all_agents flag)
- phase-21.2 (Settings UI toggle)
