---
step: phase-29.6
tier: complex
written: 2026-05-19
author: researcher
---

## Research: phase-29.6 — Extract qa.md code-review-heuristics to `.claude/skills/code-review-trading-domain/SKILL.md`

Assumption: `complex` tier stated by caller.

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://code.claude.com/docs/en/skills | 2026-05-19 | Official doc | WebFetch | Full frontmatter reference; `disable-model-invocation` blocks preload into subagents; `user-invocable: false` hides from `/` menu only |
| https://code.claude.com/docs/en/sub-agents | 2026-05-19 | Official doc | WebFetch (persisted output) | Exact `skills:` YAML-list syntax for subagent frontmatter; "full skill content is injected at startup" |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices | 2026-05-19 | Official doc | WebFetch | Naming: lowercase/numbers/hyphens only, ≤64 chars; body ≤500 lines; description third-person; progressive disclosure |
| https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview | 2026-05-19 | Official doc | WebFetch | Three-tier loading model: metadata always loaded, body only on invoke, resources on demand |
| https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills | 2026-05-19 | Anthropic engineering blog | WebFetch | Design intent: progressive disclosure, on-demand loading, subagent specialisation via skills |
| https://github.com/anthropics/claude-code/issues/29441 | 2026-05-19 | GitHub issue | WebFetch | Bug: `skills:` preload fails silently for team-spawned (process-level) agents; in-process subagents (Task/Agent tool) work correctly. Status: closed as duplicate |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/anthropics/claude-code/issues/26251 | GitHub issue | Rate-limited; key content: `disable-model-invocation: true` also blocks user slash-command invocation (separate bug) |
| https://github.com/anthropics/claude-code/issues/19141 | GitHub issue | Rate-limited; content: `user-invocable` is UI-only, does NOT block model invocation — docs now clarify |
| https://github.com/anthropics/claude-code/issues/22345 | GitHub issue | Snippet; plugin skills don't honour `disable-model-invocation` — separate plugin bug |
| https://code.claude.com/docs/en/changelog | Official changelog | Fetched via claudeupdates.dev instead for v2.1.140–144 detail |
| https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/ | Blog | Snippet; confirms invocation table |
| https://allahabadi.dev/blogs/ai/claude-code-skills-frontmatter-complete-guide/ | Blog | Snippet; consistent with official docs |
| https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md | Example SKILL.md | Snippet; confirms directory-per-skill layout |
| https://www.claudeupdates.dev/version/2.1.142 | Changelog service | Rate-limited on second fetch; key info extracted from first response |
| https://gist.github.com/yurukusa/88bad9331aa4ec50175ce559957d94da | Gist | Snippet; confirms v2.1.140 agent-type matching fix |
| https://www.trensee.com/en/blog/explainer-claude-code-skills-fork-subagents-2026-03-31 | Blog | Snippet; confirms skills+fork subagent relationship |

Total URLs collected: 16

---

### Recency scan (2024-2026)

Searched for 2026 frontier ("Claude Code skills SKILL.md frontmatter 2026"), 2025 window ("skills: subagent preload 2025"), and year-less canonical ("Claude Code SKILL.md specification"). Result:

**v2.1.140 (2026-05-12):** Agent tool subagent-type resolution improved (case/separator insensitive). No skills-system change.

**v2.1.141 (2026-05-13):** `Skill(name *)` permission wildcard fixed — prefix match now works. Impacts permission rules, not preload semantics.

**v2.1.142 (2026-05-14):** Plugin skills surfacing improvement (root-level `SKILL.md` without `skills/` subdir). Does not affect project-level skills or subagent `skills:` preload.

**v2.1.143 (2026-05-15):** Plugin dependency enforcement. No skills-system change.

**v2.1.144 (2026-05-19):** MCP paginated tool list fix. No skills change.

The `disable-model-invocation` blocks-preload rule is stable across all recent versions. The `user-invocable: false` field was documented as UI-only since its introduction; issue #19141 (closed) confirmed the docs now match. No new findings supersede the canonical docs for the preload mechanism.

---

### Key findings

1. **`skills:` YAML-list syntax in subagent frontmatter** — canonical form is a YAML block list, one item per line with `- ` prefix (Source: sub-agents docs, WebFetch 2026-05-19):

   ```yaml
   skills:
     - code-review-trading-domain
   ```

   The inline JSON-array form `skills: ["code-review-trading-domain"]` is NOT shown in any official example. Use the block-list form only.

2. **`disable-model-invocation: true` blocks preload** — the official skills doc states verbatim: "You cannot preload skills that set `disable-model-invocation: true`, since preloading draws from the same set of skills Claude can invoke." The correct combination for background-knowledge auto-preload is `user-invocable: false` + `disable-model-invocation: false` (or omit `disable-model-invocation` since default is `false`). (Source: code.claude.com/docs/en/skills, WebFetch 2026-05-19)

3. **`user-invocable: false` semantics** — hides the skill from the `/` slash-command menu (UI only). Claude can still auto-invoke and the skill CAN be preloaded. Correct field for a background reference block that users should not invoke directly. (Source: skills doc invocation table, WebFetch 2026-05-19)

4. **Preload semantics** — when a subagent lists a skill in `skills:`, the full body of SKILL.md is injected at spawn, not just the description. Distinct from a regular session where only the description is pre-loaded and the body loads on invocation. (Source: sub-agents docs §"Preload skills into subagents", WebFetch 2026-05-19)

5. **Naming convention** — directory name = skill name = lowercase letters, numbers, hyphens only, max 64 chars. `code-review-trading-domain` is 28 chars, all valid characters, accurately scopes the heuristics as trading-domain-specific. Preferred over phase-29.0 draft `code-review-heuristics`. (Source: best-practices doc, WebFetch 2026-05-19)

6. **Body size ceiling** — official guidance: "Keep SKILL.md under 500 lines." The extracted block (qa.md lines 208-437) is approximately 230 lines, well within the limit. (Source: best-practices doc, WebFetch 2026-05-19)

7. **Known bug: team-spawned subagents** — `skills:` preload works for in-process subagents (Agent/Task tool) but fails silently for process-level team teammates (GitHub issue #29441, closed as duplicate). pyfinagent's Q/A is spawned via the Agent tool from Main's session (in-process), so this bug does NOT affect this project.

8. **v2.1.141 skill permission wildcard fix** — `Skill(name *)` now works as prefix match; no settings.json change needed for this extraction. (Source: changelog WebFetch 2026-05-19)

9. **What stays in qa.md** — lines 1-206 (frontmatter + all operational sections through the 3rd-CONDITIONAL constraint text) remain unchanged. Line 207 is a `---` horizontal rule separator. Lines 207-437 (separator + code-review block) are replaced with a one-line cross-reference. Nothing in lines 208-437 makes back-references to qa.md's other sections; the block is self-contained.

10. **Existing `.claude/skills/` pattern** — `ls` confirmed `.claude/skills/masterplan/SKILL.md` is present, confirming project uses the skill pattern. The masterplan skill uses `allowed-tools: Bash(python3 *)` and `!`-injection. The new skill requires neither — pure markdown reference content, no bash injection needed.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa.md` | 437 | Q/A subagent; lines 1-206 operational, 207-437 code-review block to extract | Active |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/skills/masterplan/SKILL.md` | ~180 | Existing skill — template shape reference | Active |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/skills/` | dir | Project skills root; only `masterplan/` present | Active |
| `.claude/skills/code-review-trading-domain/SKILL.md` | (new, ~230 lines) | Target — extracted code-review-heuristics body | To create |

---

### Consensus vs debate (external)

**Consensus:** `user-invocable: false` + omitting `disable-model-invocation` = auto-preloadable background reference. This is the documented pattern for "background knowledge users shouldn't invoke directly" (skills doc invocation table).

**Resolved debate:** Issue #19141 was opened because docs were unclear about whether `user-invocable: false` hides from Claude's auto-invoke or only from the menu. Resolution (now in docs): UI-only. Current docs are unambiguous.

**Open bug (not blocking):** Team-spawned agent preload failure (issue #29441). Does not affect in-process Task/Agent tool path used by pyfinagent.

---

### Pitfalls (from literature)

1. **Using `disable-model-invocation: true` when you want preload** — silently prevents preload. Correct combo: `user-invocable: false` + `disable-model-invocation: false` (or omit `disable-model-invocation`).

2. **Inline JSON array syntax** — `skills: ["code-review-trading-domain"]` is not shown in any official example. Use YAML block-list form only.

3. **Session-restart requirement** — after adding `skills:` frontmatter to qa.md, the next Q/A spawn must happen in a new Claude Code session or after `/clear`. CLAUDE.md already documents this.

4. **SKILL.md body >500 lines** — extracted block is ~230 lines; no current risk. Future additions should use progressive disclosure (supporting files) rather than expanding the main body.

5. **Skill body in context for every turn** — once preloaded at Q/A spawn, the full 230-line body is in context for every Q/A turn. Intended behaviour; token cost is paid once per spawn.

---

### Application to pyfinagent (mapping findings to file:line anchors)

| Finding | File:line |
|---------|-----------|
| YAML block-list `skills:` syntax — add after `permissionMode: plan` | `qa.md:16` (insert after) |
| `user-invocable: false` in SKILL.md frontmatter | New file:3 |
| `disable-model-invocation` absent (default false) | New file — omit field |
| Lines 207-437 are self-contained, no back-references | `qa.md:207-437` |
| Line 207 is `---` separator — cut boundary | `qa.md:207` |
| Skill directory name `code-review-trading-domain` | `.claude/skills/code-review-trading-domain/SKILL.md` |
| Session restart required after qa.md frontmatter change | CLAUDE.md "Agent definition changes require session restart" |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (qa.md full read, skills/masterplan/ as template)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
