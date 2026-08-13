# Research Brief -- step 86.64

**Topic:** The qa-write-guard cannot see the write channel that would be used to evade it.
**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Researcher:** Layer-3 Researcher (Workflow rail). **Started:** 2026-08-14.
**Constraint:** RESEARCH ONLY -- nothing implemented, no hook edited, no production code touched.

---

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 30,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.64.md",
  "gate_passed": true
}
```

Sources read in full (all 9 appear in the R1-R9 table below):
`https://web.mit.edu/Saltzer/www/publications/protection/Basic.html`,
`https://arxiv.org/html/2407.05710`,
`https://cwe.mitre.org/data/definitions/638.html`,
`https://cwe.mitre.org/data/definitions/424.html`,
`https://cwe.mitre.org/data/definitions/693.html`,
`https://code.claude.com/docs/en/hooks`,
`https://code.claude.com/docs/en/permissions`,
`https://code.claude.com/docs/en/tools-reference`,
`https://arxiv.org/html/2607.21642v2`.

`coverage.dry` is `false` and that is CORRECT for this step: the caller set
`audit_class: NO`, so `coverage` is informational and round 2 was still producing new
findings when the tier budget was reached. It does not gate.

---

## Work log (append-only)

- [t0] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full (binding STEP 0).
- [t0] Created this brief with born-inert envelope (binding STEP 0b).
- [t1] Internal: read `qa-write-guard.sh` (145 lines), `settings.json` (213 lines), `qa.md` frontmatter.
- [t2] External round 1: 5 sources read in full (Saltzer&Schroeder 1975; arXiv 2407.05710; CWE-638; CWE-424; CC hooks doc).

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md` "Search-query composition")

| # | Variant | Query | Purpose |
|---|---------|-------|---------|
| 1 | year-less canonical | `Saltzer Schroeder complete mediation principle reference monitor incomplete mediation bypass` | founding prior art |
| 2 | year-less canonical | `fail-open vs fail-closed security control availability tradeoff formal analysis` | criterion-3 prior art |
| 3 | last-2-year | `LLM agent sandbox escape shell command bypass file write guardrail 2025` | recency window |
| 4 | current-year | `agent guardrail bypass measurement 2026 incomplete mediation AI agent tool call` | 2026 frontier |
| 5 | year-less canonical | `security theater ineffective security control false sense of security measurement study` | criterion-4 prior art |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| R1 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-14 | peer-reviewed (Proc. IEEE 63(9), 1975) | WebFetch (HTML, full) | **Complete mediation**: "Every access to every object must be checked for authority." Requires "system-wide implementation across initialization, recovery, shutdown, and maintenance", "reliable methods for identifying request sources", and "skeptical examination of caching authority-check results". **Fail-safe defaults**: "Base access decisions on permission rather than exclusion" -- a mistake in a permission-based scheme "fail[s] safely by refusing access -- quickly detected"; a mistake in an exclusion-based scheme "fail[s] by granting unauthorized access, potentially going unnoticed." **Open design**: "The design should not be secret", enabling "skeptical users to verify system adequacy." |
| R2 | https://arxiv.org/html/2407.05710 | 2026-08-14 | preprint (Patnaik, Hallett, Rashid, Univ. of Bristol; Intl. Workshop on SE in 2030, Nov 2024) | WebFetch (arXiv native HTML per the /html chain) | Complete mediation applied to LLM systems "requires trust"; an LLM's own claim that everything "is mediated through" a control is not evidence that it is. Core 2030 risk named: "developers mistake advertised security for actual security." LLMs have "no Fail-Safe Defaults of [their] own." |
| R3 | https://cwe.mitre.org/data/definitions/638.html | 2026-08-14 | official taxonomy (MITRE CWE) | WebFetch (full entry) | CWE-638 *Not Using Complete Mediation*. Parents: CWE-657 Violation of Secure Design Principles, CWE-862 Missing Authorization. **Child: CWE-424.** Mitigation: "Identify all code paths accessing sensitive resources and create a **centralized access check interface**." |
| R4 | https://cwe.mitre.org/data/definitions/424.html | 2026-08-14 | official taxonomy (MITRE CWE) | WebFetch (full entry) | CWE-424 *Improper Protection of Alternate Path*: the product fails to "sufficiently protect all possible paths that a user can take to access restricted functionality or resources." Consequence: **"Bypass Protection Mechanism."** Parents: CWE-638 **and CWE-693 Protection Mechanism Failure**. Mitigation: "Deploy different layers of protection to implement security in depth." CAPEC-554 Functionality Bypass. |
| R5 | https://code.claude.com/docs/en/hooks | 2026-08-14 | official vendor doc (Anthropic) | WebFetch (full page) | PreToolUse `matcher` filters on `tool_name`; exit 2 blocks the tool call. `agent_type` = "Agent name (for example, `\"Explore\"` or `\"security-reviewer\"`)". **Vendor's own guidance, verbatim: "Because the `if` filter is best-effort, use the [permission system](/docs/en/permissions) rather than a hook to enforce a hard allow or deny."** The doc carries **no** claim that a Write/Edit matcher observes Bash writes -- so the hook's own `:18` KNOWN GAP citation ("permissions doc") is directionally right but the load-bearing vendor sentence is this one. |
| R6 | https://code.claude.com/docs/en/permissions | 2026-08-14 | official vendor doc (Anthropic) | WebFetch (full page, 59.8KB persisted; extracted with `/usr/bin/grep`) | **The premise-correcting quote:** "When Claude Code makes a tool call, PreToolUse hooks run before the permission prompt, **for every tool except `EndConversation`**." Also: "A hook that exits with code 2 stops the tool call **before permission rules are evaluated**." And: "Hook decisions don't bypass permission rules. Claude Code evaluates **deny** and **ask** rules regardless of what a PreToolUse hook returns." And: "Permission rules are enforced by Claude Code, not by the model. Instructions in your prompt or `CLAUDE.md` shape what Claude tries to do, but they don't change what Claude Code allows." `bypassPermissions` "Skips permission **prompts**... including for writes to protected paths such as `.git` and `.claude`". |
| R7 | https://code.claude.com/docs/en/tools-reference | 2026-08-14 | official vendor doc (Anthropic) | WebFetch (full page, 87.7KB persisted; extracted with `/usr/bin/grep`) | The canonical tool list -- **45 tools**. "The tool names are the exact strings you use in permission rules, subagent tool lists, and hook matchers." Verbatim rows: `Write` = "Creates or overwrites files"; `Edit` = "Makes targeted edits to specific files"; `NotebookEdit` = "Modifies Jupyter notebook cells"; `Bash` = "Executes shell commands in your environment". |
| R8 | https://arxiv.org/html/2607.21642v2 | 2026-08-14 | preprint (Liu, Zhang, Yang, Zhang, Feng, Wang, Qiu, Liu, Poczos, Hong -- CARE, 2026) | WebFetch (arXiv native HTML per the /html chain) | State of the art on **exactly this problem** (mediating the shell channel for an LLM agent). "CARE reaches **85.64% F1 with a 0.91% false-positive rate at 2.32 ms** mean latency." **"CVE-2025-66032 showed that Claude Code's read-only command validation could be bypassed through `$IFS` rewriting and short-flag abbreviation."** Fail-behaviour, verbatim: **"If parsing fails and the raw or normalized command still contains strong high-risk indicators, CARE fails closed"** -- i.e. NARROW fail-closed conditioned on positive danger evidence. Self-described as "a complementary pre-execution defense rather than a replacement for sandboxing, host hardening, or TOCTOU-safe enforcement", operating under a "**limited observability model**". |
| R9 | https://cwe.mitre.org/data/definitions/693.html | 2026-08-14 | official taxonomy (MITRE CWE) | WebFetch (full entry) | CWE-693 *Protection Mechanism Failure*, the other parent of CWE-424. Extended description names three modes verbatim: "**missing**", "**insufficient**", and "an '**ignored**' mechanism occurs when a mechanism is available and in active use within the product, but the developer has not applied it in some code path." Also: "protection mechanism failures have not been studied comprehensively." |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/pdf/2510.06445 (Survey on Agentic Security) | preprint | superseded for this question by R8, which is specific to the shell channel |
| https://arxiv.org/pdf/2601.10440 (AgentGuardian: learned access-control policies for agents) | preprint | learned-policy direction is out of scope -- this step must not add an LLM to a hook |
| https://arxiv.org/pdf/2606.01991 (SafeMCP: proactive power regulation) | preprint | MCP-scoped, not shell-channel |
| https://arxiv.org/pdf/2606.26479 (Adaptive eval of out-of-band prompt-injection defenses) | preprint | adjacent threat model (injection, not self-evasion) |
| https://arxiv.org/pdf/2606.12320 (Five-Plane Reference Architecture for Runtime Governance of AI Agents) | preprint | architecture-scale, no measurement bearing on this control |
| https://arxiv.org/pdf/2510.11837 (Countermind multi-layer LLM security architecture) | preprint | defense-in-depth framing already covered by CWE-424 mitigation |
| https://arxiv.org/pdf/2606.06767 (Custody Envelope Threshold) | preprint | off-topic hit from the Saltzer query |
| https://beerkay.github.io/cs529/content/papers/saltzerschroeder.pdf | course-hosted PDF of R1 | duplicate of R1; PDF path banned as primary per the arXiv/PDF rule |
| https://web.mit.edu/Saltzer/www/publications/protection/ | canonical index page | fetched, returned only abstract+glossary+TOC; the principles live in `Basic.html` = R1. Recorded as an attempt, not a read. |
| https://code.claude.com/docs/en/settings | official doc | fetched; the tools table is not on this page -- it is on R7. Recorded as an attempt. |
| https://authzed.com/blog/fail-open | practitioner blog | community tier; R8 supplies the measured fail-behaviour instead |
| https://redeagle.tech/eaglepedia/fail-open-vs-fail-closed | wiki | community tier |
| https://www.reach.security/blog/compensating-controls-the-unsung-heroes-of-cyber-resilience | industry (2026-02) | used only for the compensating-control criterion, cited as snippet |
| https://www.isaca.org/resources/news-and-trends/isaca-now-blog/2026/the-security-metrics-that-lie | industry (2026) | activity-metrics-create-false-confidence framing; corroborates R2, not load-bearing |
| https://www.group-ib.com/resources/knowledge-hub/security-theater/ | industry | definitional only |
| https://www.hiddenlayer.com/research/same-model-different-hat | industry research | self-policing bypass; adjacent |
| https://slavadubrov.github.io/blog/2026/04/20/ai-agent-security/ | blog (2026-04) | community tier |
| https://dwheeler.com/secure-programs/Secure-Programs-HOWTO/follow-good-principles.html | book chapter | secondary restatement of R1 |
| https://nocomplexity.com/documents/securityarchitecture/architecture/saltzer_designprinciples.html | reference arch | secondary restatement of R1 |
| https://www.jeremyjordan.me/security-design/ | blog | secondary restatement of R1 |
| https://www.usna.edu/Users/cs/choi/it430/lec/l07/lec.html | lecture notes | secondary restatement of R1 |
| https://info-savvy.com/cissp-saltzer-and-schroeders-principles-bk1d3t1st1/ | cert-prep | lowest tier |
| https://cwe.mitre.org/data/definitions/425.html (Direct Request / Forced Browsing) | taxonomy | child of R4; the web-specific specialisation, not applicable |
| https://cwe.mitre.org/data/definitions/862.html (Missing Authorization) | taxonomy | parent of R3; covered transitively |
| https://cwe.mitre.org/data/definitions/657.html (Violation of Secure Design Principles) | taxonomy | parent of R3; covered transitively |
| https://futureagi.com/blog/llm-jailbreak-step-by-step-2026/ | blog | jailbreak-focused, wrong threat model |
| https://dev.to/aws/ai-agent-guardrails-rules-that-llms-cannot-bypass-596d | community | lowest tier |
| https://arxiv.org/pdf/2604.13954 (HINTBench) | preprint | non-attack trajectory benchmark, off-question |
| https://arxiv.org/pdf/2410.15413 (Cognitive biases in LLMs) | preprint | off-topic hit from the security-theater query |
| https://community.cisco.com/t5/security-knowledge-base/fail-open-amp-fail-close-explanation/ta-p/5012930 | vendor community | lowest tier |

**URLs collected: 39** (9 read in full + 30 snippet-only/attempted).

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Searched the 2024-2026 window on all three sub-questions (queries 3, 4, 5 above).
**Result: 3 new findings that MATERIALLY change the analysis, and they do not merely
complement the canonical sources -- one of them partially SUPERSEDES the step's own premise.**

1. **R8 / CARE (arXiv:2607.21642v2, 2026)** is the direct state of the art on mediating
   the shell channel for an LLM agent, and it is *measured*: **85.64% F1, 0.91% FPR,
   2.32 ms**. It also supplies the fail-behaviour pattern that reconciles criterion 3
   with Saltzer's fail-safe defaults (see Finding 3).
2. **CVE-2025-66032** (reported in R8): **Claude Code's own** read-only-command validation
   "could be bypassed through `$IFS` rewriting and short-flag abbreviation." A *first-party*
   demonstration that string-level shell mediation on this exact platform has already been
   defeated in the wild. This is the strongest single argument for criterion 4's conclusion.
3. **R2 (2024)** names the failure mode this step is about, in an AI context, as a
   headline risk: "developers mistake advertised security for actual security."

Canonical prior art (R1, 1975; R3/R4/R9, MITRE) is **not** superseded -- it supplies the
vocabulary (complete mediation, alternate path, ignored mechanism) that the 2026 work
assumes. No 2024-2026 source contradicts Saltzer.

---

## Key findings

**F1 -- DISAGREEMENT WITH THE ESTABLISHED PREMISE (flagged as instructed; this one changes the design space).**
The prompt states, and the hook says at `:18`, that "Write/Edit hooks do not intercept Bash".
That sentence is **true as written but its natural reading is false**. The vendor's
permissions doc says verbatim: *"When Claude Code makes a tool call, PreToolUse hooks run
before the permission prompt, **for every tool except `EndConversation`**"* (R6). A
PreToolUse hook registered with `matcher: "Bash"` **does** fire on every Bash call, and
*"A hook that exits with code 2 stops the tool call before permission rules are evaluated"*
(R6). So the Bash channel is **not uninterceptable at the platform level -- it is
unintercepted by this hook's registration** (`.claude/settings.json:35`). In CWE-693's own
words this is the **"ignored"** mode, not the "missing" mode: *"a mechanism is available and
in active use within the product, but the developer has not applied it in some code path"*
(R9). Consequence for the contract: criterion 4's "cannot be made sound" conclusion is
**not forced by platform capability**. If it is reached, it must be reached on
**decidability/assurance** grounds (F2), which is a different and more honest argument.

**F2 -- What actually makes the channel hard: decidability, not interception.**
Mediating Bash means deciding, from a command string, whether it writes to a guarded path.
CARE (R8) is a purpose-built, path-sensitive, multi-layer verifier for exactly this and
still reports **85.64% F1** -- roughly **one in seven** dangerous commands not caught. It
concedes *"The goal of canonicalization is bounded normalization, not full semantic
recovery"*. And **CVE-2025-66032** defeated Claude Code's own command validator via `$IFS`
rewriting (R8). A `grep`-shaped Bash guard in this repo would therefore be a
**probabilistic control**, never a boundary. That is the real, defensible basis for
criterion 4.

**F3 -- Fail-open vs fail-closed: the literature does NOT force the choice criterion 3 forbids.**
Saltzer's fail-safe defaults argues for exclusion-by-default: *"Base access decisions on
permission rather than exclusion"*, because a permission-scheme bug *"fail[s] safely by
refusing access -- quickly detected"* while an exclusion-scheme bug *"fail[s] by granting
unauthorized access, potentially going unnoticed"* (R1). Taken flat, that contradicts
criterion 3. **CARE resolves the tension** and gives the transferable pattern: it fails
closed **only** *"if parsing fails **and** the raw or normalized command still contains
strong high-risk indicators"* (R8). That is **narrow fail-closed conditioned on positive
evidence of danger** -- not a blanket deny-on-error. A guard built that way blocks on a
matched dangerous pattern and **allows on its own internal error**, which satisfies
criterion 3 exactly while still honouring fail-safe defaults where it counts.

**F4 -- The vendor's named hard boundary is `permissions.deny`, and it is STILL LIVE here.**
The hooks doc says *"use the permission system rather than a hook to enforce a hard allow or
deny"* (R5). `.claude/settings.json:171` sets `defaultMode: bypassPermissions`, which per R6
*"Skips permission **prompts**"* -- **prompts, not deny rules**. R6 is explicit: *"Claude
Code evaluates **deny** and **ask** rules regardless of what a PreToolUse hook returns."*
The repo already depends on this (the 27-entry deny list at `settings.json:183-210` is what
keeps `mcp__alpaca__place_stock_order` unreachable). **But** deny rules are
**session-global**: they cannot be conditioned on `agent_type`, so they cannot express "the
Q/A may not write here, but the Researcher must." *That is the structural reason this
control exists as a hook and not as a permission rule*, and it should be stated in the
contract rather than rediscovered.

**F5 -- The evasion channel is not hypothetical. It has already been exercised, first-party, in this repo.**
The phase-82.39 Q/A wrote in its own critique, verbatim:
> "the qa-write-guard hook correctly blocked my one attempted scratchpad Write, so I moved everything to stdin"

(`handoff/archive/phase-82.39/evaluator_critique.md:68`.) A Q/A blocked on `Write`
**immediately channel-switched to the Bash/stdin path** -- benign intent, but a measured
instance of CWE-424 *Improper Protection of Alternate Path* (R4) in this codebase. The
blocked agent finds the alternate path **by default**, without adversarial motivation.

**F6 -- The guard is DOUBLY blind, so a matcher change alone would not fix it.**
Even if `.claude/settings.json:35` were widened to include `Bash`, the decision leg at
`qa-write-guard.sh:124` gates on `tool_name in ("Write","Edit")`, so a Bash event reaching
the Python body **falls through to `print("allow ok")` at `:131`**. Two independent changes
would be required, not one.

**F7 -- MEASURED: zero platform-delivered Bash events have ever reached this hook.**
`handoff/logs/qa_write_guard.log` (10,097 lines; 10,065 `LOG {` records) contains
`"tool_name": "Bash"` **307 times** -- and **all 307 are synthetic checker fixtures**:
153 carry the probe marker `"agent_id": "aX"`, and the other 154 are 9 fixture paths
(`/tmp/evil.md`, `../../../etc/x`, `frontend/src/lib/api.ts`, ...) repeated exactly 17
times each, plus one blank. **Positive control for that negative:** real
platform-delivered records carry a **12-key** `payload_keys` set
(`agent_id, agent_type, cwd, effort, hook_event_name, permission_mode, prompt_id,
session_id, tool_input, tool_name, tool_use_id, transcript_path`) and a 17-hex `agent_id`
such as `a0daa10a3d87d331b`; the probes carry a **5-key** set and `agent_id: "aX"`. The
same `/usr/bin/grep` invocation that returns 0 real Bash events returns **5,443** `Write`
and **3,906** `Edit` records, so the probe is demonstrably not blind. *(Binary pinned to
`/usr/bin/grep` per the caller's constraint -- the interactive shell's `grep` is a function
wrapping `ugrep`.)*

**F8 -- Doc drift inside the guard.** `qa-write-guard.sh:68` justifies keys-only logging
with *"this log is committed to the repo"*. It is **not**: `.gitignore:76` ignores
`handoff/logs/`, and `git ls-files --error-unmatch` reports **NOT TRACKED**. The keys-only
decision is still correct (values could carry prompt text), but its stated rationale is
false and a future editor could "correct" the wrong half.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/hooks/qa-write-guard.sh` | 145 | The control. Matcher-dependent; `is_qa_role()` at `:94-121`; decision at `:124-131`; fail-open at `:38-44` (`print("allow malformed-payload")`) and `:134-145` (`case ... *) exit 0`); self-declared KNOWN GAP at `:18-20`; `agent_id` read but **log-only** at `:55` (`:69` "it feeds NO decision below"); apostrophe warning at `:70-73`. | READ-ONLY here. Sound as far as it reaches; blind past `Write`/`Edit`. |
| `.claude/settings.json` | 213 | `:34-43` registers the guard under `"matcher": "Write|Edit"` -- **this line, not the hook body, is what excludes Bash**. `:40` statusMessage is the advertised claim. `:171` `defaultMode: bypassPermissions`. `:183-210` the live deny list (27 entries). | The matcher at `:35` is the true locus of the gap. |
| `.claude/agents/qa.md` | 4 (frontmatter `tools:`) | `tools: Read, Bash, Glob, Grep, SendMessage, mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot, mcp__playwright__browser_take_screenshot, mcp__playwright__browser_console_messages`. **`Bash` is present and load-bearing** (deterministic-reproduction leg). `Write`/`Edit` are **absent** from this list yet demonstrably available -- auto-injected by `memory: project` per the guard's own `:5-11`. | Confirms the evasion channel is one the role MUST have. |
| `.claude/agents/researcher.md` | 4 (`tools:`) | `Read, Grep, Glob, Bash, WebSearch, WebFetch, SendMessage`. Write likewise auto-injected via `memory: project`; write-first is mandatory for this role. | Why a blanket `agent_type` match cannot be used (86.33's disclosed residual). |
| `docs/runbooks/per-step-protocol.md` | 460 | `:155-174` **POST-VERDICT CLEANLINESS (phase-75.20.1, MANDATORY)** -- Main runs `git status --short` and diffs after every verdict. | The named compensating control. It observes the **outcome**, not the channel -- see F11. |
| `handoff/logs/qa_write_guard.log` | 10,097 | The measurement substrate. **Gitignored** (`.gitignore:76`), NOT tracked. | Evidence source for F7; contradicts `qa-write-guard.sh:68`. |
| `.claude/masterplan.json` (86.33 entry, `status: done`) | -- | Establishes `agent_type` is caller-chosen; explicitly leaves `workflow-subagent` (80 events) and `general-purpose` (22) unmatched as a **disclosed** choice. | Independent hole (c). Do not re-litigate here. |

### Criterion 2 -- EVERY file-writing tool, enumerated from the platform's own tool list (R7)

R7 is the canonical list: **45 tools** -- `Agent, Artifact, AskUserQuestion, Bash,
CronCreate, CronDelete, CronList, Edit, EndConversation, EnterPlanMode, EnterWorktree,
ExitPlanMode, ExitWorktree, Glob, Grep, ListAgents, ListMcpResourcesTool, LSP, Monitor,
NotebookEdit, PowerShell, PushNotification, Read, ReadMcpResourceTool, RemoteTrigger,
ReportFindings, ScheduleWakeup, SendMessage, SendUserFile, ShareOnboardingGuide, Skill,
TaskCreate, TaskGet, TaskList, TaskOutput, TaskStop, TaskUpdate, TodoWrite, ToolSearch,
WaitForMcpServers, WebFetch, WebSearch, Workflow, Write` (+ `protocols`/`url` doc artefacts).

| Tool | Can it write a file on disk? | Intercepted by this hook? | Can the Q/A role invoke it? (`qa.md:4`) |
|---|---|---|---|
| `Write` | **Yes** -- "Creates or overwrites files" | **YES** (matcher + `:124`) | Yes (auto-injected by `memory: project`) |
| `Edit` | **Yes** -- "Makes targeted edits to specific files" | **YES** | Yes (auto-injected) |
| `NotebookEdit` | **Yes** -- "Modifies Jupyter notebook cells" | **NO** -- absent from the matcher AND from `:124` | **No** -- not in `tools:` |
| `Bash` | **Yes** -- "Executes shell commands"; heredoc / `>` / `tee` / `python -c` | **NO** (both legs) | **YES -- and it is load-bearing** |
| `PowerShell` | **Yes** (same class as Bash) | **NO** | No -- not in `tools:` (and not a macOS path) |
| `Agent` | **Indirectly** -- spawns a subagent whose own writes fire the hook under the *subagent's* `agent_type` (identity laundering; ties to 86.33) | Partially -- only if the child's `agent_type` matches `is_qa_role` | **No** -- not in `tools:` |
| `Workflow` | **Indirectly** -- runs a script that can call `agent()` with an arbitrary `agentType` | Same as `Agent` | No -- not in `tools:` |
| `Skill` | **Indirectly** -- "Executes a skill within the main conversation"; a skill's writes still surface as `Write`/`Edit`/`Bash` | Inherits the channel's status | No -- not in `tools:` |
| `mcp__playwright__browser_take_screenshot` | **Yes** -- writes an image to a caller-supplied path | **NO** (matcher is `Write|Edit`) | **YES** -- explicitly in `tools:` |
| `SendUserFile` | No -- sends an existing file *out*; does not create one in the tree | n/a | No |
| `Artifact` | No -- publishes to claude.ai, not the local tree | n/a | No |
| `TodoWrite` | No -- session checklist; "Disabled by default as of v2.1.142" | n/a | No |
| `MultiEdit` | **DOES NOT EXIST** on the current platform | n/a | n/a |
| all remaining 30 (`Read`, `Grep`, `Glob`, `WebFetch`, `TaskGet`, ...) | No | n/a | -- |

*Positive control for the `MultiEdit` negative:* the same `/usr/bin/grep` over R7 returns
**0** `MultiEdit` rows and **1** `Write` row, so the pattern works and the absence is real.
Any project doc still naming `MultiEdit` is stale.

**Net for criterion 2: of the 45 canonical tools, 6 can write a file on disk
(`Write`, `Edit`, `NotebookEdit`, `Bash`, `PowerShell`, `browser_take_screenshot`) plus 3
indirect channels (`Agent`, `Workflow`, `Skill`). The hook intercepts 2 of 6. The Q/A role
can invoke 3 of the 6 (`Write`, `Edit` via injection, `Bash`, `browser_take_screenshot` --
4 counting the MCP one), of which 2 are unguarded.**

---

## Consensus vs debate (external)

**Consensus.** Every source agrees a monitor that does not observe every path to the
resource is not an access-control boundary. R1 requires "every access to every object";
R3 requires a "centralized access check interface"; R4 names the exact failure ("Bypass
Protection Mechanism"); R9 classes it as "ignored". No source dissents.

**Debate -- and it lands squarely on criterion 3.** R1 argues fail-*closed* for access
control on the grounds that a fail-open bug goes unnoticed. R8 (2026, measured) and the
practitioner literature argue that a fail-closed *pre-execution* guard on a noisy channel
is an availability hazard, and adopt **narrow** fail-closed instead. The project's own
criterion 3 sits with R8. **This brief takes R8's side and records the disagreement
explicitly**: Saltzer is right about the *default*, R8 is right about the *error path*,
and CARE's formulation ("fails closed [only] if parsing fails **and** ... strong high-risk
indicators") is the reconciliation.

---

## Pitfalls (from the literature + this repo)

1. **Do not read "hooks cannot intercept Bash" into the `:18` comment** (F1). It would
   make criterion 4 look forced when it is a judgement call.
2. **Do not build a string-matching Bash guard and then call it a boundary** (F2 /
   CVE-2025-66032). If one is built, its own docs must state the F1 measurement.
3. **Do not make the existing hook fail-closed** -- criterion 3, and R8's own design
   agrees: blanket deny-on-parse-failure is what CARE deliberately avoids.
4. **Do not register a new Bash guard inside the existing hook entry.** Project memory
   `reference_claude_code_hooks_run_in_parallel`: hooks under one matcher run in parallel
   with no ordering guarantee. A separate entry keeps a bug in the new leg from touching
   the `Write`/`Edit` leg.
5. **Do not treat the guard log's 307 Bash records as platform evidence** (F7). They are
   fixtures. Any future census must apply the 12-key/5-key discriminator.
6. **Watch the apostrophe trap** (`qa-write-guard.sh:70-73`): the Python body is inside a
   bash single-quoted block; one apostrophe degrades the guard to allow-everything,
   silently. Project memory `feedback_fail_open_guards_hide_their_own_breakage`.
7. **`Edit` with a non-matching `old_string` never reaches the hook** -- the 86.31 Q/A
   measured this and disclosed it (`handoff/archive/phase-86.31/evaluator_critique.md:80`:
   "the Edit tool's own `old_string` validation short-circuits BEFORE the PreToolUse hook
   fires"). Any demonstration for criterion 1 must not use a bad `old_string`, or it will
   prove nothing.

---

## Application to pyfinagent

**F9 -- What the control actually is, in CWE-693's vocabulary.** `.claude/settings.json:40`
advertises *"the qa evaluator is read-only outside `.claude/agent-memory/qa/`"*. What is
enforced is: *the qa **role**, on `Write`|`Edit` **only**, when `agent_type` happens to be
truthful*. That is CWE-693 **"insufficient"** (partial defense), compounded by
**"ignored"** (the hook mechanism works for `Bash` and was not applied there). It is
**not** "missing" -- the control does real work: F5 is a *success* record, the guard did
block that Write. The accurate description is **an advisory convention check plus an
audit log**, and criterion 4's "accuracy of the claim is itself an acceptable outcome" is
directly supported by R1's Open design principle: the design must enable *"skeptical users
to verify system adequacy"*, which an overstated statusMessage defeats.

**F10 -- Is the guard's stated purpose achievable at all?** Split the question:
- *As a **boundary** against a motivated evader:* **No.** Two independent holes
  (uninterceptable-in-practice channel + unauthenticated identity), and even a
  best-in-class shell verifier is ~85% F1 with a first-party CVE against this very
  platform's validator (F2). Closing the identity hole (86.33) does not help, because
  `Bash` bypasses the identity check entirely.
- *As a **defence against accident and drift**, which is the actual threat model
  (a well-intentioned evaluator reaching for the nearest tool):* **Yes, and it already
  works** -- F5 is the proof, and it shows the residual risk precisely: the agent went to
  `Bash` next, without being told to.

**F11 -- The structural answer the literature points to: mediate the RESOURCE, not the CHANNEL.**
The channel set is open-ended (6 direct + 3 indirect today, and R7 grows every release).
CWE-424's own mitigation is *"Deploy different layers of protection to implement security
in depth"* (R4), and R8 calls itself *"a complementary pre-execution defense rather than a
replacement for ... TOCTOU-safe enforcement"*. The repo already has the
channel-independent layer: `docs/runbooks/per-step-protocol.md:155-174` POST-VERDICT
CLEANLINESS, where Main diffs the tree after every verdict. **A post-hoc tree diff observes
the object (Saltzer's "every access to every object") rather than the path, so it is
complete over an unbounded channel set in a way no matcher list can be.** Its weakness is
that it is *detective, not preventive*, and it depends on Main executing it. If this step
does one high-value thing, hardening that check (a script rather than a prose instruction)
buys more than any matcher widening -- and it cannot brick a session, so criterion 3 is
satisfied trivially.

**Contract-shaping summary (Main owns PLAN; these are options, not a plan).**
Three moves, none of which weakens an existing guard:
(i) **Correct the claim** (criterion 4) -- the statusMessage at `settings.json:40` and the
`:18` comment, with F1's correction so the next reader is not misled about capability.
(ii) **Optionally** add a *separate* `matcher: "Bash"` hook entry using CARE's narrow
fail-closed shape: deny only on a positive high-risk match (qa-role `agent_type` **and** a
redirect/`tee`/heredoc/`python -c` targeting outside the memory dir), fail-open on parse
failure, on internal error, and on everything else. Ship it labelled as probabilistic.
(iii) **Harden the compensating control** (F11) -- the 2026 compensating-controls
literature's test is that it must "reduce real risk and adapt to change"; a scripted
post-verdict diff meets that, a prose instruction does not.

**Criterion 1 note (task boundary):** criterion 1 asks for the Bash evasion to be
*demonstrated*, not argued. F7 is strong circumstantial evidence (zero platform-delivered
Bash events in 10,065 records) and F5 is a first-party instance, but the live drive is a
**GENERATE-phase** action. I did not perform it -- this session is research-only.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total (incl. snippet-only) -- **39**
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, one premise-correcting
- [x] Full papers / pages read (not abstracts); arXiv via `/html/` per the chain, never `/pdf/`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE (hook, settings.json, qa.md, masterplan 86.33) plus researcher.md, per-step-protocol.md and the guard log
- [x] Contradictions / consensus noted -- the Saltzer-vs-CARE fail-open disagreement is recorded rather than smoothed over, and I flagged disagreement with the caller's own premise (F1)
- [x] All claims cited per-claim
- [x] Every zero paired with a positive control (F7 Bash-vs-Write counts; MultiEdit-vs-Write rows; per-step-protocol grep vs file length)
- [x] `/usr/bin/grep` pinned throughout per the caller's constraint
- [x] Task boundaries honoured: nothing implemented, no hook or production code edited, no contract written, paper-trading untouched
