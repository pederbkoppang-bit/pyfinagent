# Research Brief — Step 75.4.2 (Researcher, Layer-3)

**Step:** 75.4.2 — Audit75 S4 follow-up: skill_optimizer post-write delivery invariant
**Tier:** moderate | **audit_class:** false | **Priority:** P1
**Executor tag:** sonnet-4.6/high (spell everything out; executor has no session memory)
**Assumption:** tier=moderate per spawn prompt.
**Status:** IN PROGRESS (write-first, appended incrementally)

---

## 1. Problem statement (from masterplan 75.4.2, verbatim intent)

`backend/agents/skill_optimizer.py::apply_modification` autonomously rewrites
`backend/agents/skills/*.md` via an `old_text -> new_text` string replacement,
guarded ONLY by an "occurs-exactly-once" check + a flag-gated (currently DARK)
phase-71.4 independent review. After writing, it re-loads the skill and reverts
on exception — BUT `load_skill()` succeeding does NOT prove the *delivered*
prompt (via `load_skill`/`format_skill`) still contains the required sections.

75.4 moved previously-undelivered content INSIDE the `## Prompt Template` region
of 9 skill files, which NEWLY exposes that content to this auto-optimizer. An
autonomous run can silently re-break the heading levels 75.4 just fixed
(e.g. a `### ` heading promoted back to `## ` loads fine but truncates the
delivered template), re-truncating delivered prompts with no operator signal.

**Fix required:** add a post-write DELIVERY invariant — capture the delivered
template before the write, and reject-and-revert if the write drops any
`{{placeholder}}` or shrinks the delivered length beyond a threshold. **Fail
CLOSED (revert) on violation.**

### Immutable success criteria (copied verbatim — DO NOT EDIT)
1. New `backend/tests/test_phase_75_4_2_optimizer_invariant.py` passes offline and
   calls the REAL `apply_modification` against a temp copy of a skill file (not a
   stub): a modification that promotes a `### ` heading back to `## ` inside the
   Prompt Template region is REVERTED and `apply_modification` returns False
2. A modification that drops a `{{placeholder}}` from the delivered template is
   REVERTED; the file on disk is byte-identical to its pre-call content
3. A legitimate modification that changes only body prose is ACCEPTED and written
   — proving the guard is not blanket-refusing (negative control)
4. Mutation matrix recorded in experiment_results.md: removing the invariant
   check, and weakening it to a `load_skill()`-succeeds check only, each fail at
   least one test

**verification.command:**
`cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py -q`

**verification.live_check:** `handoff/current/live_check_75.4.2.md` — verbatim
command output (exit 0) + `git diff --stat` + evidence the guard fails CLOSED (a
rejected write leaves the file byte-identical, shown by hash before/after).

---

## 2. Internal audit — the optimizer write path (file:line)

**File:** `backend/agents/skill_optimizer.py` — `apply_modification` at **:406-479**
(step said ~:397-459; exact anchors below).

| Anchor | What happens | Relevance |
|--------|--------------|-----------|
| `:412` | `skill_path = SKILLS_DIR / f"{agent_name}.md"` | write target; `SKILLS_DIR` imported from `backend.config.prompts` at `:21` |
| `:416` | `content = skill_path.read_text(encoding="utf-8")` | **pre-write snapshot** of the RAW file — this is the exact byte string used for revert |
| `:417-421` | `if old_text not in content: return False` | existing guard |
| `:424-429` | `if content.count(old_text) > 1: return False` | the "occurs-exactly-once" guard (the ONLY structural guard today) |
| `:431-450` | phase-71.4 independent LLM review — **flag-gated `skill_modification_review_enabled` (default `False`, settings.py:463) → DARK today** | separate, pre-write, metered-LLM gate; NOT the deterministic check we add |
| `:452` | `new_content = content.replace(old_text, new_text, 1)` | the mutation |
| `:453` | `skill_path.write_text(new_content, encoding="utf-8")` | **THE WRITE** |
| `:459-460` | `reload_skills()` + `SkillFileIdCache.invalidate(agent_name)` | clears mtime cache so the next `load_skill` re-reads |
| `:463-470` | `try: load_skill(agent_name) except Exception: revert to content; return False` | **the ONLY post-write validation today — "loads without raising"**. This is exactly what the step says is insufficient: a truncated template still loads fine. |
| `:472-478` | `_git add` + `_git commit` (non-fatal on RuntimeError) | commits the (possibly broken) skill |
| `:479` | `return True` | |

**Key facts for the executor:**
- `import re` is **NOT** at module top (top imports are csv/json_io/logging/subprocess/time/datetime/Path/typing). `re` is imported locally in `think_harder` (:536) and `_extract_json` (:903). **The executor must add `import re` at module top.**
- `revert_modification` (:481-493) is a **git-based** revert (`git checkout HEAD~1`) — do NOT use it for the invariant. The correct fail-closed revert is the in-function `skill_path.write_text(content)` pattern already used at :467 (restores the exact pre-write bytes, no git dependency).
- The invariant should be **UNCONDITIONAL (always on)**, unlike the phase-71.4 review which is flag-gated DARK. It is a cheap deterministic structural check with zero LLM cost.

### Invocation path / silent-degradation today
- `backend/api/skills.py:52` `POST /optimize` → background task → `optimizer.run_loop(...)` (`:47`).
- `backend/agents/meta_coordinator.py:178-184` — MetaCoordinator emits `action="skill_opt"` with `target_agents` (MDA→Agent bridge) on a `MIN_SKILL_OPT_INTERVAL_DAYS` cadence; `_run_one_iteration(target_agents=...)` (skill_optimizer.py:707) then calls `apply_modification` (:782).
- **What happens today if the optimizer writes a broken skill (heading `### `→`## ` promotion):** `load_skill` still succeeds (returns the truncated ~190-char template) → `apply_modification` returns `True` → `_git commit` records the break → `reload_skills` makes it live → **every subsequent pipeline analysis for that agent silently delivers the truncated prompt** (missing `{{quant_model_data}}` + Instructions + Uncertainty/Code-Exec sections). No crash, no log. The metric-driven DISCARD (:836-848) only reverts *after* live analyses ran on the broken prompt, and the immediate delta is usually 0 → PENDING (:813-835), so the break can persist. The invariant stops it at write time.

## 3. The loader delivery contract (75.4 fix)

**File:** `backend/config/prompts.py`.

- **`load_skill(agent_name)` (:178-206)** extracts the delivered template via
  `re.search(r"^## Prompt Template\s*\n(.*?)(?=^## |\Z)", content, re.MULTILINE|re.DOTALL)`
  then `.strip()`. **The region ends at the FIRST subsequent `^## ` heading** (level-2)
  or EOF. Caches by mtime (:188-191). Raises `ValueError` if no `## Prompt Template`
  section (:201-202); `FileNotFoundError` if missing file (:185-186).
- **This is the exact truncation mechanism (gap5-01):** `^## ` matches `## Foo` (two
  hashes + space) but NOT `### Foo` (three hashes). 75.4 DEMOTED in-template section
  headings from `## ` to `### ` so they stay inside the region. A modification that
  PROMOTES a `### ` heading back to `## ` re-inserts a region terminator → the loader
  truncates everything after it → delivered template collapses (7532→190 chars for
  quant_model_agent in the original bug), and any `{{placeholder}}` past that heading
  is dropped.
- **`format_skill(template, **kwargs)` (:209-250)** only SUBSTITUTES `{{key}}`→value on
  the delivered template (:243). It cannot add/remove sections. phase-75.4 item f added
  a WARNING when a kwarg has no matching placeholder (:244-249) — i.e. the builder had
  data but the placeholder was truncated away (silent discard). This warning is the
  *symptom*; the invariant is the *prevention*.
- **`reload_skills()` (:253-272)** clears `_skill_cache`; with no client arg it is a
  pure cache-clear (safe in tests).
- **The "delivered prompt" the invariant must protect = the string `load_skill` returns**
  (the extracted `## Prompt Template` section with `{{placeholders}}` intact).
  Checking at the `load_skill` layer is the correct granularity: `format_skill` is a pure
  substitution, so if every placeholder + the section body survive `load_skill`, delivery
  is intact. (No `critic_degraded`-style flag lives in the loader; that flag is in the
  orchestrator's critic loop, unrelated to this step.)

## 4. Invariant design recommendation (the core deliverable)

**Add a deterministic, always-on, fail-CLOSED post-write DELIVERY invariant to
`apply_modification`.** Snapshot the delivered template *before* the write, compare
*after*, and revert-to-`content` on violation.

### 4a. Module-level additions (top of skill_optimizer.py)
```python
import re  # add to the module-top import block

# Reject a self-modification that shrinks the delivered template below this
# fraction of its pre-write length -- a structural truncation (heading promotion),
# never a 2-10 line prose edit. Tunable; 0.80 leaves generous headroom for real edits.
DELIVERY_MIN_RETAIN_RATIO = 0.80
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")   # same pattern the 75.4 test uses


def _delivery_invariant_ok(delivered_before: str, delivered_after: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=False -> the write broke delivery; caller must revert.

    Two independent guards (both must hold):
      1. Placeholder-subset: every {{placeholder}} in the pre-write delivered template
         is still present after (exact; no threshold). Catches a dropped runtime var.
      2. Length-retention: the delivered template did not shrink below
         DELIVERY_MIN_RETAIN_RATIO of its prior length. Catches heading-promotion
         truncation that drops non-placeholder body (Uncertainty/Code-Exec sections).
    """
    before_ph = set(_PLACEHOLDER_RE.findall(delivered_before))
    after_ph = set(_PLACEHOLDER_RE.findall(delivered_after))
    dropped = before_ph - after_ph
    if dropped:
        return False, f"dropped placeholders: {sorted(dropped)}"
    if len(delivered_after) < DELIVERY_MIN_RETAIN_RATIO * len(delivered_before):
        return (
            False,
            f"delivered template shrank {len(delivered_before)}->{len(delivered_after)} "
            f"chars (< {DELIVERY_MIN_RETAIN_RATIO:.0%} retained)",
        )
    return True, "ok"
```

### 4b. Insertion into `apply_modification` (two edits)

**(i) Baseline capture — after `content = skill_path.read_text(...)` (:416), before the
occurs-once guard.** Fail closed if the current file can't be delivered:
```python
try:
    delivered_before = load_skill(agent_name)
except Exception as exc:
    logger.warning(
        "[skill-opt] cannot baseline delivered template for %s (%s) -- skipping (no write)",
        agent_name, exc,
    )
    return False
```

**(ii) Post-write check — replace the existing `try: load_skill(...) except` block at
:463-470** with a version that ALSO runs the invariant and reverts on violation:
```python
# Validate the modified skill still LOADS *and* still DELIVERS its full template.
try:
    delivered_after = load_skill(agent_name)
except Exception as e:
    logger.error("Modified skill for %s failed to load: %s. Reverting.", agent_name, e)
    skill_path.write_text(content, encoding="utf-8")
    reload_skills()
    SkillFileIdCache.invalidate(agent_name)
    return False

ok, reason = _delivery_invariant_ok(delivered_before, delivered_after)
if not ok:
    logger.error(
        "[skill-opt] DELIVERY INVARIANT violated for %s (%s). Reverting (fail closed).",
        agent_name, reason,
    )
    skill_path.write_text(content, encoding="utf-8")
    reload_skills()
    SkillFileIdCache.invalidate(agent_name)
    return False
```
The subsequent `_git add/commit` (:472-478) then runs ONLY when both load + invariant
pass — so a rejected write is never committed AND the on-disk file is byte-identical to
`content` (the criterion-2 hash-equality requirement).

**Design rationale / decisions:**
- **Check at `load_skill` layer, not raw file:** the raw file may legitimately carry
  `{{tokens}}` in `## Data Inputs` prose (documentation) that are NOT real placeholders —
  the 75.4 test explicitly guards against this false-positive (:114-116). Comparing
  `load_skill`-before vs `load_skill`-after is symmetric and only sees real delivered
  placeholders, so the doc-token issue cannot cause a false reject.
- **Two guards, not one:** placeholder-subset is exact (kills a placeholder-drop with no
  length change); length-retention catches a heading promotion that truncates a
  placeholder-free tail. Each guard is independently mutation-killable (criterion 4).
- **Fail CLOSED = revert-and-return-False**, matching the existing load-exception path.
- **Always on** (not flag-gated) — deterministic, $0, and it directly re-closes the 75.4
  regression surface.

## 5. Fixture patterns from 75.4 tests (reuse these)

**Source:** `backend/tests/test_phase_75_skill_delivery.py` (read in full).

- **Placeholder extraction pattern (reuse verbatim):** `re.findall(r"\{\{(\w+)\}\}", text)`
  (:112-113) — identical to the invariant's `_PLACEHOLDER_RE`.
- **Real-`load_skill`, never-a-stub doctrine (:22-27):** the suite's binding rule — "a
  guard that cannot fail does not count. Every assertion goes through the REAL
  `load_skill()`, never a string stub." The 75.4.2 test MUST likewise drive the REAL
  `apply_modification` (criterion says "not a stub").
- **`quant_model_agent.md` is the canonical truncation fixture.** Its `## Prompt Template`
  (line 72) contains `{{fact_ledger_section}}` (73), `{{ticker}}` (74), then
  `### Quant Model Data` (78) + `{{quant_model_data}}` (79), `### Instructions` (81),
  `### Uncertainty Permission` (91), `### Code Execution Tasks` (113), ending at
  `## Experiment Log` (123). Promoting `### Quant Model Data` (78) → `## Quant Model Data`
  truncates the delivered template to ~lines 73-77 (~190 chars) AND drops
  `{{quant_model_data}}` — tripping BOTH invariant guards. This is the ideal criterion-1
  fixture.
- **Per-case, not `any()`-shaped (:124-127):** the 75.4 suite parametrizes per file so a
  single relocated section fails exactly one case. Mirror this: separate test functions
  per criterion, not one combined assertion.
- **Negative control is mandatory (:193-198, :334-343):** the suite always pairs a
  "must-fire" test with a "must-stay-silent" happy-path test. Criterion 3 is exactly this.
- **How to point `apply_modification` at a temp copy** (the "not a stub" mechanism):
  `apply_modification` uses `skill_optimizer.SKILLS_DIR` for the write and
  `prompts.load_skill` (which reads `prompts.SKILLS_DIR`) for delivery. The test must
  `monkeypatch.setattr` BOTH `backend.agents.skill_optimizer.SKILLS_DIR` and
  `backend.config.prompts.SKILLS_DIR` to a `tmp_path`, copy a real skill file
  (e.g. `quant_model_agent.md`) into it, and construct/`__new__` a `SkillOptimizer`.
  Also monkeypatch `backend.agents.skill_optimizer._git` to a no-op (the accept path
  calls `_git add/commit`; the temp file is outside the repo so a real `git add` is
  noise) and, for hermeticity, patch `backend.config.prompts._SKILL_FILE_ID_CACHE_PATH`
  to `tmp_path/".skill_file_ids.json"` so `SkillFileIdCache.invalidate` cannot touch the
  real cache. Stubbing `_git`/the cache path is fine — the criterion's "not a stub" refers
  to the SKILL FILE (must be a real temp copy), not to infra side-effects.
- **Constructing `SkillOptimizer` without live clients:** `SkillOptimizer.__init__` builds
  `BigQueryClient` + `OutcomeTracker` (needs creds). `apply_modification` uses none of
  `self.*` except nothing (it's effectively static over `agent_name`/`proposal`). Build
  via `SkillOptimizer.__new__(SkillOptimizer)` (the 75.4 suite uses the same `__new__`
  trick for the orchestrator at :281, :445) and call `apply_modification` on it — avoids
  the BQ/creds dependency entirely and keeps the test offline.

## 6. External research

### Read in full (>=5 required; 6 fetched — counts toward the gate)
| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://arxiv.org/html/2312.13382 (DSPy Assertions, Singhvi/Khattab et al.) | 2026-07-24 | paper | `dspy.Assert` = HARD constraint: on failure after R retries "transitions to an error state σ⊥ and raises an AssertionError...halting execution" (**fail-closed**). `dspy.Suggest` = SOFT: "logs a warning...and continues execution" (**fail-open**). Constraints are checked AFTER a module produces output (post-condition), then backtrack+regenerate. **Our invariant = the `Assert` (fail-closed) discipline, minus the retry — we revert instead.** |
| 2 | https://ar5iv.labs.arxiv.org/html/2309.03409 (OPRO, Yang et al.) | 2026-07-24 | paper | Optimizer keeps only candidates scored higher on an objective set; "optimization terminates when the LLM is unable to propose new solutions with better optimization scores." A candidate is **scored before it is retained** — a measured keep/discard, never blind adoption. Maps to skill_optimizer's own keep/discard, and argues the write itself needs a pre-adoption structural check. |
| 3 | https://ar5iv.labs.arxiv.org/html/2211.01910 (APE, Zhou et al.) | 2026-07-24 | paper | APE proposes candidates, **filters by a score function above a threshold**, then "Return instruction with the highest score." Adaptive filtering discards low-quality candidates cheaply. Prior-art for "validate/score a proposed prompt before adopting it." |
| 4 | https://www.eiffel.org/doc/solutions/Design_by_Contract_and_Assertions | 2026-07-24 | doc (canonical) | Meyer's DbC: **postcondition** = "conditions that will be true always if s works correctly", checked AFTER the routine; a postcondition/invariant violation is a **supplier (implementation) failure**; fail-fast "early detection...rather than permitting corrupted states to propagate." **This is the exact frame: the invariant is a POSTCONDITION on `apply_modification`.** |
| 5 | https://arxiv.org/html/2510.05156 (VeriGuard, LLM agent safety) | 2026-07-24 | paper (2025) | "correct-by-construction": generate an action AND its verification; a runtime monitor "validate[s] each proposed agent action against the pre-verified policy **before execution**." Uses Hoare triples `{Cpre} p {Cpost}`. On failure → block/terminate/re-plan. Caveat: soundness depends on the generated constraints matching intent — argues for a **deterministic** structural invariant, not an LLM judge. |
| 6 | https://arxiv.org/html/2604.08059 (Governed Capability Evolution + rollback) | 2026-07-24 | paper (2026) | Treats a new capability version as a "governed deployment candidate," not an immediate replacement; **Recovery Compatibility (κR)** checked at admission; "any interface incompatibility yields immediate rejection"; "rollback...restore[s] previous active version"; "governed upgrade...maintain[s] zero unsafe activations." Directly models: a self-modification must prove it does not break delivery, else revert. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/pdf/2603.15691 (VibeContract, 2026) | paper | Recency: contracts as QA for AI-generated code — same shape (LLM writes code, contract guards it); snippet sufficed |
| https://arxiv.org/pdf/2605.15665 (PRISM, 2026) | paper | Recency: prompt reliability via iterative simulation/monitoring |
| https://arxiv.org/abs/2406.11695 (MIPRO) | paper | DSPy multi-stage optimizer; candidate scoring corroboration |
| https://dspy.ai/learn/programming/7-assertions/ | doc | Official DSPy assertions page; corroborates #1 |
| https://www.promptingguide.ai/techniques/ape | doc | APE technique summary; corroborates #3 |
| https://arxiv.org/pdf/2512.02840 (promptolution, 2025) | paper | Modular prompt-optimization framework |
| https://arxiv.org/pdf/2606.05743 (Membrane, 2026) | paper | Self-evolving safety memory for agents |
| https://arxiv.org/pdf/2606.05805 (Guardrail Feedback, 2026) | paper | Risk→remediation guardrail loop |
| https://learn.adacore.com/courses/intro-to-ada/chapters/contracts.html | doc | Ada DbC tutorial; corroborates #4 |
| https://www.digitalapplied.com/blog/llm-guardrails-production-safety-layers-reference-2026 | blog | Prompt versioning + rollback practice |
| https://futureagi.com/blog/top-10-prompt-optimization-tools-2025 | blog | Landscape of prompt-optimizer tooling |

## 7. Recency scan (2024-2026)

**Performed.** Searched 2024/2025/2026-scoped variants (see §8). Findings: the last-2-year
window has clearly SUPERSEDED the older prompt-optimizer papers (APE 2022, OPRO 2023) on the
specific question of *validating a self-modification before adoption*:
- **VeriGuard (arXiv:2510.05156, 2025)** — "correct-by-construction" + verify-before-execute
  with Hoare-triple pre/post-conditions; fail-closed action blocking. Newer than DbC/DSPy and
  directly about *agent-generated artifacts*.
- **Governed Capability Evolution (arXiv:2604.08059, 2026)** — the strongest match: a
  self-evolving component version is a *candidate* that must pass compatibility (incl. recovery)
  checks or be **rejected/rolled back to the previous version**; "zero unsafe activations."
- **VibeContract (arXiv:2603.15691, 2026)** and **PRISM (arXiv:2605.15665, 2026)** — 2026 work
  putting contracts / reliability monitoring around AI-authored code and prompts (snippet-only).
- The canonical prior-art (DbC/Eiffel, APE, OPRO, DSPy Assertions) remains valuable for the
  *mechanism* (postcondition, keep/discard, fail-closed), but the 2025-26 work confirms the
  design direction: **deterministic verify-before-adopt with fail-closed revert.** No newer
  source contradicts a simple structural post-write invariant; if anything they validate it
  (VeriGuard explicitly warns LLM-generated constraints may not match intent → prefer a
  deterministic check, which is exactly what this step builds).

## 8. Queries run (3-variant discipline)

- **Topic A — prompt-optimizer candidate validation:**
  frontier `automatic prompt optimization validate candidate before adoption OPRO DSPy 2026`;
  last-2-yr `DSPy assertions constraints LM program verify output prompt optimizer`;
  year-less `OPRO large language models as optimizers validation set prompt scoring` +
  `automatic prompt engineer APE candidate selection filtering scoring`.
- **Topic B — post-condition / invariant after mutation:**
  year-less `design by contract postcondition invariant verification software`.
- **Topic C — self-modifying agent safety rails:**
  frontier `self-improving LLM agent guardrails rollback verification safety 2026`.
Mix spans current-year (2026), last-2-year (2025), and year-less canonical (2022-2023 papers).

## 9. PLAN recommendations (sized for a SONNET executor — no session memory assumed)

**Scope:** ONE code file + ONE new test file. Do NOT touch `prompts.py`, the loader, the
skill `.md` files, or the phase-71.4 review. The invariant is additive and always-on.

**PLAN step 1 — `backend/agents/skill_optimizer.py`:**
1. Add `import re` to the module-top import block (it is NOT there today).
2. Add module constants + helper immediately after imports (verbatim in §4a):
   `DELIVERY_MIN_RETAIN_RATIO = 0.80`, `_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")`,
   and `def _delivery_invariant_ok(delivered_before, delivered_after) -> tuple[bool, str]`.
3. In `apply_modification` (:406): after `content = skill_path.read_text(...)` (:416) and
   before the occurs-once guard, capture `delivered_before = load_skill(agent_name)` inside a
   try/except that returns `False` if it raises (§4b(i)).
4. Replace the post-write `try: load_skill(...) except` block (:463-470) with the version in
   §4b(ii): keep the load-exception revert, and ADD the `_delivery_invariant_ok` check that
   reverts (`skill_path.write_text(content, ...)` + `reload_skills()` +
   `SkillFileIdCache.invalidate(agent_name)`) and returns `False` on violation — placed BEFORE
   the `_git` commit so a rejected write is never committed.

**PLAN step 2 — `backend/tests/test_phase_75_4_2_optimizer_invariant.py` (new):**
Reuse the 75.4-suite doctrine (real `load_skill`, no string stub; per-case functions; a
must-fire + a must-stay-silent negative control). Harness (spell out for the executor):
- `tmp = tmp_path`; copy `SKILLS_DIR/"quant_model_agent.md"` → `tmp/"quant_model_agent.md"`.
- `monkeypatch.setattr("backend.agents.skill_optimizer.SKILLS_DIR", tmp)` AND
  `monkeypatch.setattr("backend.config.prompts.SKILLS_DIR", tmp)` AND
  `monkeypatch.setattr("backend.config.prompts._SKILL_FILE_ID_CACHE_PATH", tmp/".skill_file_ids.json")`.
- `monkeypatch.setattr("backend.agents.skill_optimizer._git", lambda *a, **k: "")` (no-op).
- `opt = SkillOptimizer.__new__(SkillOptimizer)` (skip `__init__`, avoids BQ/creds).
- Helper `def call(old, new): return opt.apply_modification("quant_model_agent", {"old_text": old, "new_text": new, "description": "test"})`.

Test cases (each maps to a success criterion):
- **T1 (criterion 1) heading promotion REVERTED:** `old_text="### Quant Model Data"`,
  `new_text="## Quant Model Data"`. Assert `call(...) is False`; assert file bytes/sha256
  unchanged vs the pre-call copy. (Trips BOTH guards: truncates ~190 chars AND drops
  `{{quant_model_data}}`.)
- **T2 (criterion 2) placeholder drop REVERTED, byte-identical:** capture
  `before = (tmp/"quant_model_agent.md").read_bytes()` and its sha256.
  **CAUTION — measured 2026-07-24: `{{quant_model_data}}` occurs TWICE in the file**
  (once in `## Data Inputs` prose at line 27, once in the template at line 79). A bare
  `old_text="{{quant_model_data}}"` would be rejected by the **pre-existing occurs-once
  guard (:424), NOT the new invariant** → a VACUOUS test that passes for the wrong reason.
  Use a UNIQUE multi-line `old_text` that contains the placeholder:
  `old_text="### Quant Model Data\n{{quant_model_data}}"`,
  `new_text="### Quant Model Data\nthe provided factor data"` (keeps the `###` heading so no
  truncation; drops the placeholder; ~same length so ONLY the placeholder guard fires).
  Assert `call(...) is False`; assert `read_bytes() == before` and sha256 equal. The
  `## Data Inputs` occurrence is BEFORE `## Prompt Template`, so `load_skill` only sees the
  line-79 one — the invariant logic is correct; only the raw `old_text` must be made unique.
- **T3 (criterion 3) NEGATIVE CONTROL — body prose accepted:** pick a prose line inside the
  template that is NOT a heading and carries no placeholder (e.g. a line from the `### Instructions`
  body); `old_text=<that line>`, `new_text=<reworded same-length-ish line>`. Assert
  `call(...) is True`; assert the file CHANGED (sha256 differs) and `load_skill` still contains
  `{{quant_model_data}}` and all pre-existing placeholders (guard is not blanket-refusing).
- (optional but recommended per mutation-test doctrine) **T4 length-only:** truncate a
  placeholder-free tail so ONLY the length guard fires → pins the length half independently.

**PLAN step 3 — mutation matrix (criterion 4), record verbatim in experiment_results.md:**
Run the suite, then apply each mutation to a scratch copy and show the named test flips RED:
- **M1 remove the invariant entirely** (delete the `_delivery_invariant_ok` call, keep only
  the load-succeeds check) → **T1 and T2 FAIL** (apply_modification returns True on a truncated
  write). ← this is the exact "load_skill()-succeeds only" mutation the criterion names.
- **M2 weaken to load-succeeds-only** (identical effect to M1 but by neutering the helper to
  `return True, "ok"`) → **T1 and T2 FAIL**.
- **M3 drop the placeholder guard** (keep length) → **T2 FAIL** (small-length placeholder drop
  slips through).
- **M4 drop the length guard** (keep placeholder) → **T4 FAIL** (placeholder-free truncation
  slips through). (If T4 is omitted, note M4 is not independently killed and add T4.)
- Also mutate the TEST fixture itself (75.4 doctrine): swap T3's prose edit for a no-op
  `old==new`-shaped change and confirm it would be rejected by `old_text not in content`/
  occurs-once — proving T3 actually exercises the accept path.

**PLAN step 4 — live_check_75.4.2.md:** paste the verbatim `pytest -q` output (exit 0),
`git diff --stat`, and the sha256-before == sha256-after evidence from T1/T2 showing a rejected
write leaves the file byte-identical (fail-closed proof).

**Watch-outs for the executor:**
- Do NOT call `revert_modification` (git `checkout HEAD~1`) for the invariant — use the
  in-function `write_text(content)` revert (git-free, byte-exact).
- The occurs-once guard (:424) means `old_text` must appear exactly once in the file — pick
  fixture strings that are unique and **verify `content.count(old_text)==1` in the test setup
  for EVERY fixture**. Measured counts in quant_model_agent.md (2026-07-24):
  `### Quant Model Data` → 1 (OK for T1), `### Instructions` → 1, but
  `{{quant_model_data}}` → **2** (do NOT use it bare as `old_text` — see T2's unique 2-line
  form). This is the exact vacuous-guard trap the mutation-test doctrine warns about: a T2 that
  rejects via the pre-existing occurs-once guard would still be GREEN after M1/M2 mutate away the
  new invariant, so it would prove nothing.
- `DELIVERY_MIN_RETAIN_RATIO=0.80` is a chosen threshold; the placeholder guard is exact. Keep
  both; do not lower the ratio below what T3's legit edit needs (a ~same-length prose swap has
  ratio ~1.0, so 0.80 has ample headroom).

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "skill_optimizer.apply_modification (skill_optimizer.py:406-479) rewrites skills/*.md via old->new replace, guarded only by an occurs-once check + a DARK flag-gated LLM review; its ONLY post-write check is that load_skill() does not raise (:463-470) -- which a heading '###'->'##' promotion passes while silently truncating the delivered template (the loader regex prompts.py:196 stops at the first '## '). Fix: add a deterministic, always-on, fail-CLOSED delivery postcondition: snapshot load_skill(agent) BEFORE the write, compare AFTER; revert-to-content + return False if any {{placeholder}} is dropped OR delivered length < 80% of prior. External prior-art: Meyer DbC postconditions (fail-fast, supplier-fault), DSPy Assert (fail-closed halt vs Suggest fail-open), APE/OPRO score-before-adopt, and 2025-26 VeriGuard (verify-before-execute) + Governed Capability Evolution (candidate must pass recovery-compat or roll back). Test drives REAL apply_modification against a temp copy of quant_model_agent.md; mutation matrix M1/M2 (remove/weaken to load-only) each fail T1/T2.",
  "brief_path": "handoff/current/research_brief_75.4.2.md",
  "gate_passed": true
}
```
