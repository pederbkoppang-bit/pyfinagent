# Research Brief — step 86.84

**Topic:** Remedy design for Workflow `agent({schema})` drops:
`agent({schema}): subagent completed without calling StructuredOutput (after in-conversation nudge)`.
Root cause taken as GIVEN from the caller's measurement (not re-derived here): the
subagent exhausts its `.md` frontmatter `maxTurns` budget with no turn left to emit
the schema call.

**Tier:** complex · **Audit-class:** false · **Started:** 2026-08-14

```json
{
  "brief_status": "INCOMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 4,
  "urls_collected": 12,
  "recency_scan_performed": false,
  "internal_files_inspected": 3,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 9,
    "dry": false
  },
  "summary": "in progress",
  "brief_path": "handoff/current/research_brief_86.84.md",
  "gate_passed": false
}
```

## Status log

- Skeleton + born-inert envelope written first.
- Round 1: 7 sources read in full. The turn/StructuredOutput arithmetic is now
  DOC-GROUNDED (see F-1..F-4). Pending: Workflow `agent()` opts reference,
  recency scan, internal file:line anchors.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|---|
| 1 | https://code.claude.com/docs/en/agent-sdk/agent-loop | 2026-08-14 | Official doc (tier 2) | WebFetch, full | Defines a turn; `max_turns` "counts tool-use turns only"; **default "No limit"**; `error_max_turns` result has **no `result` field**. |
| 2 | https://code.claude.com/docs/en/sub-agents | 2026-08-14 | Official doc (tier 2) | WebFetch, full | Frontmatter table: `maxTurns` = "Maximum number of agentic turns before the subagent stops". **No default documented.** Partial-output-on-cutoff applies to API errors only. |
| 3 | https://code.claude.com/docs/en/agent-sdk/subagents | 2026-08-14 | Official doc (tier 2) | WebFetch, full | `AgentDefinition.maxTurns` is a per-agent-definition field; depth/concurrency/spend caps; workflow pointer. |
| 4 | https://code.claude.com/docs/en/workflows | 2026-08-14 | Official doc (tier 2) | WebFetch, full | `agent()` resolves to `null` on stop/unrecoverable API error; workflow subagents run `acceptEdits` + inherit allowlist; cache prefix keyed on agent type. |
| 5 | https://code.claude.com/docs/en/agent-sdk/structured-outputs | 2026-08-14 | Official doc (tier 2) | WebFetch, full | "the SDK validates the output against it, **re-prompting on mismatch**"; `error_max_structured_output_retries`; "success but no `structured_output` ... **Treat that case as a failure**". |
| 6 | https://github.com/anthropics/claude-code/issues/65500 | 2026-08-14 | Vendor issue tracker (tier 2) | WebFetch, full | **[ADVERSARIAL to the local retry design]** Exact error string with "(after **2** in-conversation nudges)"; reporter measured the throw as **not catchable** at script level. OPEN, v2.1.162. |
| 7 | https://github.com/anthropics/claude-code/issues/41143 | 2026-08-14 | Vendor issue tracker (tier 2) | WebFetch, full | **[ADVERSARIAL to "raise maxTurns"]** `maxTurns: 10` not enforced on Agent-tool dispatch (72-75 turns, status SUCCESS). **Closed as not planned**, v2.1.84. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://code.claude.com/docs/llms.txt | Doc index | Index used to discover exact page URLs; not a source. |
| https://github.com/anthropics/claude-code/issues/65731 | Vendor issue | Adjacent (`/deep-research` rate-limit banner), different failure mode. |
| https://github.com/anthropics/claude-code/issues/8501 | Vendor issue | Request for authoritative frontmatter docs; superseded by the live doc table. |
| https://platform.claude.com/docs/en/build-with-claude/structured-outputs | Official doc | Referenced by #5 for JSON-Schema limitations. |

## Recency scan (2024-2026)

_pending — round 2._

## Key findings

**F-1. A turn is a tool-use round trip, and `max_turns` counts ONLY those.**
"A turn is one round trip inside the loop: Claude produces output that includes
tool calls, the SDK executes those tools, and the results feed back to Claude
automatically." And: "You can cap the loop with `max_turns` / `maxTurns`, which
counts tool-use turns only." (Source: agent-loop doc, URL 1.)

**F-2. StructuredOutput is itself a tool call — so emitting the schema COSTS a
tool-use turn.** Claude Code "wraps the JSON schema as a fake tool definition
called 'StructuredOutput' and forces the model to 'call' it with valid JSON
arguments, using the model's tool-calling machinery" (URL 6 thread analysis).
Combined with F-1 this makes the remedy arithmetic, not a heuristic: the budget
must be `work_turns + 1`. A cap sized to the work is a cap that cannot terminate.

**F-3. The documented default for an ABSENT `maxTurns` is NO LIMIT.** The
agent-loop "Turns and budget" table lists Max turns default as `No limit`
(URL 1). This corroborates the caller's measurement that `general-purpose` and
`Explore` — which carry no `maxTurns` frontmatter — run to 63/56 turns and have
never dropped. It is a documented *absence of a cap*, not a high default.

**F-4. At `maxTurns` there is NO result to salvage — the graceful-degradation
question is answered NO by the docs.** The result-subtype table (URL 1) gives
`error_max_turns` → "`result` field available? **No**". The one documented
partial-return path is different: it applies when "a rate limit, overload, or
server error cuts off a subagent that already produced text output" (URL 2) —
an API error, not a turn-limit stop. So the schema path returning nothing at the
cap is the *expected* documented behaviour, and the non-schema Agent path does
not do meaningfully better at the same boundary.

**F-5. Structured outputs already have their OWN retry ladder, distinct from the
nudge.** "the SDK validates the output against it, re-prompting on mismatch. If
validation does not succeed within the retry limit, the result is an error"
(URL 5), surfacing as `error_max_structured_output_retries`. That ladder handles
*malformed* output. The pyfinagent failure is *absent* output — a different
branch, and one the docs explicitly tell you to treat as a failure: "A result can
also end with subtype `success` but no `structured_output` value ... Treat that
case as a failure as well."

**F-6. The nudge count is 2, per the only public report of the exact string.**
Issue #65500 quotes `agent({schema}): subagent completed without calling
StructuredOutput (after 2 in-conversation nudges)`. The string pyfinagent
observes says "(after in-conversation nudge)" — singular, no count — so the
runtime message has changed between v2.1.162 and the version in use here. No
documentation of the nudge exists on any official page fetched; it is
undocumented runtime behaviour. **Whether a nudge consumes a turn is not
documented anywhere** and cannot be answered from public sources.

**F-7. `maxTurns` enforcement has a public contradiction, and it matters for
sizing.** Issue #41143 (v2.1.84, **closed as not planned**) reports `maxTurns:
10` producing 72-75 tool calls with status SUCCESS on the Agent-tool path. The
caller's measurement shows exact-30 and exact-40 stops on the *Workflow* path.
Both can be true: enforcement differs by dispatch path and/or version. The
consequence for remedy design is that `maxTurns` is a *load-bearing but
historically unreliable* control, and a fix that depends on its exact value
should be paired with a check that does not.

**F-8. `agent()` is documented to resolve to `null` for two causes only.** "An
`agent()` call resolves to `null` if you stop it mid-run or it hits an
unrecoverable API error" (URL 4). A StructuredOutput drop is not in that list —
consistent with #65500's report that the throw escaped a `.catch()`.

**F-9. Workflow subagents inherit `acceptEdits` + the session allowlist.** "The
subagents the workflow spawns always run in `acceptEdits` mode and inherit your
tool allowlist, regardless of your session's mode" (URL 4). This is the doc that
bears on whether the researcher could get `Write` without `agentType:'researcher'`
— see the Application section.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/agents/qa.md` | 835 | Q/A system prompt; frontmatter `maxTurns: 30` | LIVE |
| `.claude/agents/researcher.md` | 404 | Researcher system prompt; frontmatter `maxTurns: 40` | LIVE |
| `.claude/workflows/qa-verdict.js` | 491 | Q/A Workflow rail; retry loop added 2026-08-14 | LIVE |
| `.claude/workflows/research-gate.js` | 873 | Research-gate Workflow rail; same retry shape | LIVE |
| `scripts/qa/rail_drop_rate.py` | 254 | Drop-rate measurement over run records | LIVE |

_file:line anchors pending — round 2._

## Consensus vs debate (external)

_pending._

## Pitfalls (from literature)

_pending._

## Application to pyfinagent

_pending._

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (12)
- [ ] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim

Soft checks:
- [ ] Internal exploration covered every relevant module
- [ ] Contradictions / consensus noted
- [ ] All claims cited per-claim
