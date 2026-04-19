# Research Gate -- How-to (phase-4.16)

This is the HOW-TO guide for every researcher spawn. The REFERENCE
record lives in `ARCHITECTURE.md::Research Gate Discipline (phase-4.16)`.
The agent-facing prompt lives in `.claude/agents/researcher.md`.
Do not duplicate rules across the three files -- cross-link.

## Floor: at least 5 sources read in full

Every researcher spawn, at every tier, must fetch and read IN FULL
at least 5 sources via `WebFetch`. Search snippets do NOT count.
This applies to `simple`, `moderate`, and `complex` tiers alike;
the tier only sets the depth of analysis and the length of the
brief, not the 5 sources floor.

If fewer than 5 sources were fetched in full, the researcher MUST
return `gate_passed: false` and list what was attempted. Padding a
brief to mask an under-fetch is a protocol breach.

## Last-2-year recency scan (mandatory)

Every brief must include a dedicated "Recency scan (last 2 years)"
section reporting either:

- N new findings from the last-2-year window that complement or
  supersede older sources, OR
- No relevant new findings in the window.

The section must be present even when empty. An older canonical
source is still valuable; newer work just needs to be evaluated
against it.

## Search-query composition (mandatory)

Every research session MUST run at least three search-query variants
per topic to cover both the current frontier and the canonical prior
art. A single year-locked query is a protocol breach.

1. **Current-year frontier** -- append `2026` to the topic. Example:
   `"agent skill optimization 2026"`. Catches the latest published
   work in the current calendar year.
2. **Last-2-year window** -- append `2025` (and optionally `2024`).
   Used alongside #1 for the "Recency scan" section. Do NOT rely on
   this alone; see #3.
3. **Year-less canonical** -- the bare topic with NO year suffix.
   Example: `"agent skill optimization"`. This surfaces well-known
   prior-art (textbooks, the founding paper, classic blog posts) that
   a year-locked query will miss because search engines heavily bias
   to recent results when any year is present.

The brief must make the three-variant discipline visible: either by
listing the queries run in a short subsection, or by ensuring the
source table has a mix of current-year, last-2-year, and year-less
hits. If the topic is genuinely too new for year-less prior-art
(e.g., "phase-4.14.27 Anthropic effort param"), say so explicitly;
don't silently skip the year-less query.

Snippet-only hits from the year-less canonical search are valuable
evidence of "what prior art exists" even when not read in full, and
belong in the snippet-only table.

## Source quality hierarchy

1. Peer-reviewed (arXiv, ACM, IEEE, Journal of Finance)
2. Official docs (Anthropic, Google, IETF, NIST engineering blogs)
3. Authoritative blogs (OpenAI, DeepMind, named academic researchers)
4. Industry practitioner (Two Sigma, AQR, quant firms)
5. Community (StackOverflow, forums) -- lowest weight

Enforce this hierarchy in the fetched-in-full set. A brief with 5
community-tier URLs read in full does NOT clear the gate.

## URL collection

Collect 10+ unique candidate URLs before pruning to the 5 sources
read-in-full set. The snippet-only set is the remaining URLs --
recorded in its own table so auditors can see what was evaluated
vs what was read.

## JSON envelope (always emit)

Every brief ends with this envelope, even when the caller does not
ask for it:

```json
{
  "tier": "simple|moderate|complex",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "gate_passed": false
}
```

`gate_passed: true` iff `external_sources_read_in_full >= 5` AND
`recency_scan_performed == true` AND every hard-blocker checklist
item is satisfied.

## Handoff folder convention

The `handoff/` tree is strictly partitioned:

| Directory | Purpose | Writer |
|-----------|---------|--------|
| `handoff/current/` | In-flight step's files + `_templates/` | Main (cycle in progress) |
| `handoff/archive/phase-<sid>/` | Completed-step snapshots | `.claude/hooks/archive-handoff.sh` on masterplan status flip |
| `handoff/audit/` | Append-only JSONL audit streams | PreToolUse / ConfigChange / InstructionsLoaded hooks |
| `handoff/logs/` | Runtime process logs | Gitignored |

Invariants (verified by `scripts/housekeeping/verify_handoff_layout.py`):

- `handoff/current/` contains NO files belonging to `status=done`
  steps. Rolling top-level files (`contract.md`,
  `experiment_results.md`, `evaluator_critique.md`,
  `research_brief.md`) are allowed; they snapshot into the archive
  dir on each step close.
- No `*_audit.json*` at `handoff/` root (they live in `handoff/audit/`).
- No `*.log` at `handoff/` root (they live in `handoff/logs/`).

Backfill script: `scripts/housekeeping/backfill_handoff_archive.py`
(idempotent; safe to re-run).

## Cross-references

- `.claude/agents/researcher.md` -- agent prompt (system message).
- `ARCHITECTURE.md::Research Gate Discipline (phase-4.16)` -- MADR
  record.
- `CLAUDE.md` -- harness protocol (cycle-2 flow, session restart,
  stress-test doctrine). Authoritative for cycle semantics; defers
  to this file for research-gate mechanics.
- `docs/runbooks/per-step-protocol.md` -- operator runbook.
