# Research Brief -- phase-86.33

**Topic:** Identity and authorization for sandboxed sub-agents -- how a policy
enforcement point decides WHO a caller is when the caller supplies its own label.
**Tier:** moderate (caller-specified)
**Audit-class:** NO (coverage reported for information only)
**Researcher:** Layer-3 researcher via Workflow rail
**Started:** 2026-08-11

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 12,
  "snippet_only_sources": 10,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "gate_passed": true
}
```

10 of the 12 read-in-full sources were fetched via `WebFetch`; the 2 PDFs
(Anderson 1972, NIST SP 800-162) via the documented `curl` + `pypdf` chain,
because `research-gate.md:107-155` forbids `WebFetch` on binary PDFs. The
WebFetch-only count is therefore **10**, still >= the floor of 5.

**Disclosed shortfall (soft check, does not clear silently):** `WebSearch`
was unavailable session-wide (200/200 spent before spawn), so the mandatory
three-variant search discipline could not be executed and no adversarial
source was surfaced. See B0. Hard blockers are all met; Main should weigh
B0 before treating the external half as exhaustive.

---

## Work log (append-only)

- [t0] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [t0] Brief created with born-inert envelope.
- [t1] INTERNAL half complete: guard read in full, log distribution derived by script.
- [t2] External searches + fetches.

---

# PART A -- INTERNAL MEASUREMENT (the Explore half)

Everything below is MEASURED on the installed tree at
`/Users/ford/.openclaw/workspace/pyfinagent`, Claude Code **v2.1.227**
(`claude --version`, 2026-08-11). Derivation script:
`/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/75941db6-ec67-4375-8394-f7886d027a40/scratchpad/analyze_guard_log.py`
(re-runnable; reads `handoff/logs/qa_write_guard.log` only).

## A1. What the guard reads, and its match rule

`.claude/hooks/qa-write-guard.sh` -- 127 lines, bash wrapper around an
embedded `python3 -c` heredoc.

Fields read from the PreToolUse payload (all via `d.get(X) or ""`, so an
absent field is indistinguishable from an empty one):

| Field | Line | Used in the decision? |
|---|---|---|
| `agent_type` | `qa-write-guard.sh:46` | **YES** -- the sole authorization input |
| `agent_id` | `qa-write-guard.sh:55` | **NO** -- log-only, added by phase-86.33 P0 (commit `8a9a4293`) |
| `tool_name` | `qa-write-guard.sh:57` | YES -- gates on `in ("Write","Edit")` |
| `tool_input.file_path` | `qa-write-guard.sh:58-60` | YES -- the target path |

Match rule after the **86.31 widening** (`is_qa_role`, `qa-write-guard.sh:76-103`):

```python
n = (name or "").strip().lower()
return n == "qa" or n.startswith("qa-") or n.startswith("qa_")
```

Deny condition (`qa-write-guard.sh:106-112`): `is_qa_role(agent_type)` AND
`tool_name in ("Write","Edit")` AND `os.path.normpath(file_path)` does not
contain the segment `.claude/agent-memory/qa/`. `normpath` is applied first so
a `../` traversal cannot smuggle the memory-dir substring (`:109-110`).

**This is a NAME-prefix allowlist, and 86.31's own docstring says so.** The
predicate matches a *string the spawner chose*. Anything not starting `qa`
is unguarded by construction.

## A2. Where it fails open (three places, one of them MEASURED firing)

1. `qa-write-guard.sh:39-44` -- any exception parsing stdin prints
   `allow malformed-payload`.
2. `qa-write-guard.sh:46,55,57` -- `d.get(...) or ""`: a **missing
   `agent_type` allows**. Main's own calls carry no `agent_type` and are
   therefore always allowed (by design, per the header comment `:15-16`).
3. `qa-write-guard.sh:116-127` -- the bash `case` default arm `*) exit 0`.
   If python is missing, crashes, or emits nothing, the hook allows.

**Fail-open #3 has demonstrably fired.** The log contains **32 non-JSON
lines** that are Python `SyntaxError` tracebacks (`File "<string>", line 32
/ def is_qa_role(name): / ^ / SyntaxError: invalid syntax`). During that
window the guard allowed **everything** while still appearing installed.
This is the same class as the 2026-08-10 apostrophe breakage the P0 commit
message cites.

Documented-but-unclosed gap (header `:18-20`): Write/Edit hooks do not
intercept **Bash** subprocess writes. `echo > file` bypasses the guard
entirely; the named covering control is Main's post-verdict `git status`
rule.

## A3. The agent_type population (DERIVED, not transcribed)

`handoff/logs/qa_write_guard.log`: 7,757 lines -> **7,727 parseable
records**, 32 unparseable (the SyntaxError traces above), 1 stray fragment.
Range `2026-07-24T11:57:29Z` .. `2026-08-11T12:33:44Z`.

**72 distinct `agent_type` values all-time.** Top of the distribution:

| n | agent_type | what it is |
|---:|---|---|
| 2056 | `<EMPTY>` | Main's own tool calls (no subagent) |
| 1473 | `qa` | project agent definition `.claude/agents/qa.md` |
| 617 | `qa-80-2-c2` | caller-chosen NAME |
| 612 | `qa-36-12-cycle6` | caller-chosen NAME |
| 611 | `QA-80-2` | caller-chosen NAME (note the case) |
| 611 | `qa-85-5-c3` | caller-chosen NAME |
| 611 | `qa_86_31` | caller-chosen NAME (underscore form) |
| 607 | `researcher` | project agent definition `.claude/agents/researcher.md` |
| 82 | `workflow-subagent` | Workflow `agent()` with NO `agentType` pin |
| 38 | `res-78-1` | caller-chosen NAME -- **does not start with `qa`/`research`** |
| 24 | `general-purpose` | built-in Claude Code agent type |
| 11 | `quality-auditor` | caller-chosen NAME -- semantically Q/A, matches NOTHING |
| 11 | `QA-Upper` | caller-chosen NAME |

Split at the P0 commit `8a9a4293` (UTC **2026-08-11T07:55:14**;
`git show` confirms author date `2026-08-11T09:55:14+02:00`):

- **BEFORE:** 7,557 rows, **72 distinct** agent_type values.
- **AFTER:** 170 rows, **3 distinct** -- `<EMPTY>` (77), `qa` (68),
  `researcher` (25). The collapse is not a platform change: every recent
  spawn happens to come through the Workflow rail, which **pins**
  `agentType`. It is the pinning that narrows the set, not the runtime.

`quality-auditor` (11 events) is the sharpest single datum for criterion 2:
a Q/A-role agent whose name shares no prefix with `qa`, so the widened
predicate misses it entirely.

## A4. agent_id -- populated, opaque, per-spawn, and NOT self-asserted

- **First appearance of the key:** `2026-08-11T07:54:07.979061Z` -- about
  67 seconds BEFORE the commit timestamp, because the hook takes effect on
  file save, not on commit.
- **Exactly ONE pre-cutover non-empty value:**
  `{"agent_type":"qa","agent_id":"a-probe-123","file_path":"backend/main.py"}`.
  That is **Main's own synthetic probe payload**, hand-piped to the hook.
  It is direct proof that *the hook's stdin is forgeable by anything that
  can invoke the script* -- but NOT evidence that a subagent can influence
  the value the **runtime** supplies. Those are different threat models
  (see Part C).
- **Post-change population.** 170 AFTER rows: 63 carry `agent_id`, 107 do
  not. Broken out:

  | agent_type | with agent_id | without |
  |---|---:|---:|
  | `<EMPTY>` (Main) | 0 | 77 |
  | `qa` | 44 | 24 |
  | `researcher` | 19 | 6 |

- **CORRECTION / EXTENSION of Main's "23 of 33".** The 30 role-typed rows
  lacking `agent_id` are **all synthetic probe fixtures** -- their targets
  are exclusively `/tmp/evil.md` (6), `/tmp/x.md` (6), `backend/main.py`
  (6), `.claude/agent-memory/qa/../../../etc/x` (6), and
  `.claude/agent-memory/qa/MEMORY.md` (6). Every row with a REAL runtime
  target (`handoff/current/research_brief_*.md`,
  `.claude/agent-memory/qa/verdicts/verdict_wip_*.md`,
  `.claude/agent-memory/researcher/MEMORY.md`) carries an `agent_id`. So
  the accurate statement is: **`agent_id` is populated on 63/63 real
  subagent Write/Edit calls observed post-change, and on 0/77 of Main's
  own calls.** It is a reliable *subagent-vs-Main* discriminator on this
  version.
- **Shape:** exactly 17 chars, `a` + 16 lowercase hex
  (`[0-9a-f]`), e.g. `a77dfb7250e0af1df` (this session's researcher).
  18 distinct values post-change.
- **Does it join to anything authoritative?** **No.** Zero occurrences of
  any observed `agent_id` anywhere in the repo outside this log. It is an
  opaque per-spawn handle: stable within one spawn (1-8 writes per id),
  never reused, and **zero agent_ids appear under more than one
  `agent_type`**. It carries NO role information by itself -- you can only
  learn a role from it by joining it to the `agent_type` in the same
  payload, which is the very field in question. It is an *identifier*, not
  an *attribute*.

## A5. Is there any other field carrying the TYPE?

**No.** The payload exposes `agent_type` and `agent_id` and nothing else
identity-bearing. Critically, `agent_type` is a **union of two namespaces**
with no separator:

- **Platform/definition-assigned:** `general-purpose` and `Explore`
  (built-ins), plus `qa` / `researcher` -- and `.claude/agents/` contains
  **only** `qa.md` and `researcher.md` (`ls .claude/agents/`), so those two
  are the *entire* project-definition namespace.
- **Caller-chosen:** every one of the other ~68 values
  (`qa-80-2-c2`, `QA-Upper`, `quality-auditor`, `res-78-1`, ...).

When a spawn supplies a name, the **name wins** and overwrites the type in
the same field. There is therefore **no field on this version from which a
hook can recover the agent-definition file a subagent was spawned from**
once a name is given. That is the direct answer to the central question:
**NO -- the installed payload does not let a hook distinguish TYPE from
NAME.** `agent_id` distinguishes *subagent from Main*, and distinguishes
*instances from each other*, but not *role from role*.

## A6. What `workflow-subagent` and `general-purpose` actually are

- `general-purpose` -- a **built-in Claude Code agent type**, not a project
  file. Its 24 events (2026-08-04 .. 2026-08-10) wrote
  `handoff/current/evaluator_critique_82.5.md` (8) and `_82.7.md` (7) plus
  `backend/main.py` and `frontend/src/lib/api.ts`. Those critique writes are
  a Q/A-role agent writing the artifact Main is contractually the scribe
  for -- i.e. **the 86.31 breach class recurring under a non-`qa` label.**
- `workflow-subagent` -- the **default label for a Workflow `agent()` call
  with no `agentType` pin** (82 events, 2026-07-25 .. 2026-08-10). It wrote
  production code: `backend/services/kill_switch.py` (12),
  `backend/tests/test_phase_36_7_*.py` (7),
  `frontend/src/components/KillSwitchPanel.tsx` (5),
  `backend/api/paper_trading.py` (4).

Neither is caller-*chosen* in the naming sense, but neither is an
authenticated role assertion either -- both are labels the **spawn site**
selects. Every workflow in `.claude/workflows/` now pins one explicitly:
`qa-verdict.js:232` -> `'qa'`; `research-gate.js:670` -> `'researcher'`;
`research-gate.js:734` -> `'Explore'`; `harness-self-audit.js:84,90` ->
`'Explore'`; `probe-qa-tool-surface.js:42` -> `'qa'`.

## A7. Does the researcher rail write through this guard? YES

**821 rows** have an `agent_type` beginning `research` (31 distinct
variants: `researcher`, `research-82-0`, `res-78-1`, `researcher-80-4-death`,
...). Their top write targets are exactly the write-first deliverables:
`handoff/current/research_brief_86.31.md` (96),
`.claude/agent-memory/researcher/MEMORY.md` (77), and a long tail of
`research_brief_*.md`. **This very session** appears as
`{"agent_type":"researcher","agent_id":"a77dfb7250e0af1df","tool_name":"Write",
"file_path":".../handoff/current/research_brief_86.33.md"}`.

**A fail-CLOSED change would break this rail** unless `researcher` and every
`research*` naming variant is permitted. Write-first is non-negotiable
(`researcher.md:97-105`) and `research-gate.js:43-48` states the researcher
"legitimately NEEDS Write".

**STALE COMMENT FOUND.** `research-gate.js:47-48` says "the qa-write-guard
PreToolUse hook matches `agent_type == 'qa'` only, so it does not block this
path." After the 86.31 widening the predicate is the `qa`/`qa-`/`qa_`
**prefix** rule, not `== 'qa'`. The conclusion (researcher is not blocked)
still holds, but the stated reason is wrong -- and it would become an active
trap if anyone ever named a researcher spawn something beginning `qa`.

---

# PART B -- EXTERNAL RESEARCH

## B0. METHOD LIMITATION -- DISCLOSED, NOT HIDDEN

**`WebSearch` was unavailable for this entire session.** All three search
variants returned: *"Web search was not performed: this session has used its
web search budget (200 of 200 WebSearch calls)."* The budget is
session-shared and was exhausted **before this researcher was spawned**.

Consequence: the **three-variant search discipline**
(`research-gate.md:33-56`) could **NOT** be executed as written. I did not
silently skip it -- I could not run it. Mitigation actually used: I went
directly to **canonical primary-source URLs** (which is what the discipline
is *for*: reaching Anderson 1972, Saltzer & Schroeder 1975, the SPIFFE
spec, the Kubernetes docs, NIST SP 800-162, RFC 9700, and Anthropic's own
docs -- every one of which the caller named explicitly). `WebFetch` worked
normally, so the >=5-read-in-full floor is met with primary sources rather
than search-surfaced ones. **Treat "no adversarial/serendipitous source was
surfaced by search" as an open gap, not as a null result.**

## B1. Sources READ IN FULL (12; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key verbatim finding |
|---|---|---|---|---|---|
| 1 | https://csrc.nist.gov/csrc/media/publications/conference-paper/1998/10/08/proceedings-of-the-21st-nissc-1998/documents/early-cs-papers/ande72.pdf | 2026-08-11 | paper (Anderson 1972) | curl + pypdf 6.10.2, 142pp, 321,746 chars | "A) The reference validation mechanism must be tamper proof. B) The reference validation mechanism must always be invoked. C) The reference validation mechanism must be small enough to be subject to analysis and tests to assure that it is correct." |
| 2 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-11 | paper (Saltzer & Schroeder 1975) | WebFetch | "Complete mediation: Every access to every object must be checked for authority." / "Fail-safe defaults: Base access decisions on permission rather than exclusion." / "Economy of mechanism: Keep the design as simple and small as possible." |
| 3 | https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.sp.800-162.pdf | 2026-08-11 | standard (NIST SP 800-162, ABAC) | curl + pypdf 6.10.2, 47pp, 137,241 chars | "These subject attributes are assigned and managed by an authority within the organization" / "Subject attributes are provisioned by attribute authorities--typically authoritative for the type of attribute that is provided and managed through an attribute administration point." |
| 4 | https://code.claude.com/docs/en/hooks | 2026-08-11 | official vendor doc | WebFetch | `agent_type`: "Agent name (for example, `"Explore"` or `"security-reviewer"`)... For subagents, the subagent's type takes precedence over the session's `--agent` value." `agent_id`: "Unique identifier for the subagent... **Use this to distinguish subagent hook calls from main-thread calls.**" |
| 5 | https://code.claude.com/docs/en/sub-agents | 2026-08-11 | official vendor doc | WebFetch (93.8KB) | `name`: "Unique identifier using lowercase letters and hyphens. **Hooks receive this value as `agent_type`.**" / "identity comes only from the `name` frontmatter field" / `disallowedTools`: "Tools to deny, removed from inherited or specified list" |
| 6 | https://code.claude.com/docs/en/security | 2026-08-11 | official vendor doc | WebFetch | "Claude Code uses strict read-only permissions by default." / "**Fail-closed matching**: Unmatched commands default to requiring manual approval" |
| 7 | https://raw.githubusercontent.com/spiffe/spiffe/main/standards/SPIFFE.md | 2026-08-11 | spec (SPIFFE) | WebFetch | "A SPIFFE ID is a structured string (represented as a URI) which serves as the 'name' of an entity." / the Workload API "explicitly does not include an authentication handshake or authenticating token from the workload. Implementors can verify the authenticity of the caller via an out-of-band method." |
| 8 | https://spiffe.io/docs/latest/spire-about/spire-concepts/ | 2026-08-11 | official doc (SPIRE) | WebFetch | "Workload attestation asks the question: 'Who is this process?'" -- the agent works by "interrogating locally available authorities (such as the node's OS kernel, or a local kubelet running on the same node) in order to determine the properties of the process calling the Workload API." |
| 9 | https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/ | 2026-08-11 | official doc | WebFetch | "An admission controller is a piece of code that intercepts requests to the Kubernetes API server prior to persistence of the resource, but after the request is authenticated and authorized." / "If any of the controllers in either phase reject the request, the entire request is rejected immediately" |
| 10 | https://kubernetes.io/docs/concepts/security/service-accounts/ | 2026-08-11 | official doc | WebFetch | "To assign a ServiceAccount to a Pod, you set the `spec.serviceAccountName` field... Kubernetes then automatically provides the credentials" / "Kubernetes gets a short-lived, automatically rotating token using the `TokenRequest` API" |
| 11 | https://www.rfc-editor.org/rfc/rfc9700.html | 2026-08-11 | IETF BCP (Jan 2025) | WebFetch | s2.6: "authorization servers SHOULD NOT allow clients to influence their `client_id` or any other claim that could cause confusion with a genuine resource owner." |
| 12 | https://docs.aws.amazon.com/rolesanywhere/latest/userguide/introduction.html | 2026-08-11 | official doc | WebFetch | "your workloads must use X.509 certificates issued by your certificate authority (CA). You register the CA with IAM Roles Anywhere as a **trust anchor**... Your workloads outside of AWS authenticate with the trust anchor using certificates issued by the trusted CA in exchange for temporary AWS credentials." |

## B2. Fetched in full but NON-CONTRIBUTING / snippet-only (does NOT count)

| URL | Kind | Why it does not count |
|---|---|---|
| https://web.mit.edu/Saltzer/www/publications/protection/ | paper index | Index + glossary + TOC only; the principles live in `Basic.html` (source #2, read in full). |
| https://spiffe.io/docs/latest/spiffe-about/spiffe-concepts/ | official doc | Fetched in full; returned a NEGATIVE result -- "workload attestation" does not appear on that page. Recorded as a null finding; the substance is on the SPIRE page (source #8). |
| https://code.claude.com/docs/en/authentication | official doc | Reached as the 301 target of `/docs/en/iam`; fetched in full but covers login/credentials, not tool-permission precedence. Off-topic. |
| https://docs.claude.com/en/docs/claude-code/hooks | redirect | 301 -> `code.claude.com/docs/en/hooks` (source #4). |
| https://code.claude.com/docs/en/permissions | official doc | Referenced by source #6 for allow/deny/ask rule precedence; not fetched (budget). |
| https://code.claude.com/docs/en/sandboxing | official doc | Referenced by source #6 for filesystem/network isolation; not fetched. |
| https://code.claude.com/docs/en/sandbox-environments | official doc | Referenced by source #6 ("compare isolation approaches"); not fetched. |
| https://claude.com/blog/ciso-guide-to-agentic-ai | vendor blog | Named in source #6 as the agentic-AI security framework; not fetched. |
| https://kubernetes.io/docs/reference/access-authn-authz/authentication/ | official doc | The authN half of the K8s pipeline; admission-control half sufficed. |
| https://trust.anthropic.com | vendor portal | Compliance artifacts (SOC 2 / ISO 27001), not technical. |

**Unique URLs collected: 22.**

## B3. Recency scan (2024-2026) -- PERFORMED, method substituted

Because `WebSearch` was dead (B0), the recency scan was done by fetching
**version-dated current editions** of the primary docs rather than by dated
search queries. Findings:

1. **RFC 9700 is itself the recency finding (January 2025).** It is the
   current OAuth 2.0 Security Best Current Practice and **obsoletes RFC
   6819**. Its s2.6 / s4.15 recommendation -- servers "SHOULD NOT allow
   clients to influence their `client_id`" -- is 2025-vintage guidance that
   lands directly on this step's question, and it is the citation the
   phase-86.33 P0 commit message already leans on.
2. **The Claude Code docs are live-current (2026)** and carry
   version-stamped behaviour notes as recent as **v2.1.223** (installed:
   **v2.1.227**). The `agent_id` field and the "distinguish subagent hook
   calls from main-thread calls" wording are current-generation, not legacy
   -- i.e. **the platform added the very field this step needs**, and 86.31
   predates its adoption here.
3. **Kubernetes bound-token hardening is in-window.** The
   `ServiceAccountNodeAudienceRestriction` gate (beta, 2024-2025) further
   narrows which audiences a kubelet may request -- the trend is
   monotonically *away* from long-lived, self-presented identity.
4. **Claude Code sandboxing (`/sandbox`, `denyRead` rules) is
   2025-2026 material** and is the platform's own answer to
   filesystem isolation -- relevant to (d) because it is enforced *below*
   the agent, not by a hook the agent's own label steers.
5. **No source found that argues FOR name-based authorization.** I did not
   surface a genuine adversarial source; given B0 I record this as an
   **unsearched gap**, not as evidence of consensus.

## B4. Key findings (per-claim cited)

**F1 -- A self-asserted identity is not an authenticated one; the check
must be unbypassable.** Anderson's three criteria are the canonical frame:
"A) The reference validation mechanism must be tamper proof. B) The
reference validation mechanism must always be invoked. C) The reference
validation mechanism must be small enough to be subject to analysis and
tests to assure that it is correct." (Anderson 1972, p.16;
`ande72.pdf.txt:795-797`, extracted via pypdf). Anderson adds the operating
condition: "The continuous invocation of the reference validation mechanism
reflects that it must be applied to all programs including the operating
system itself." (`:806`).

**F2 -- Complete mediation and fail-safe defaults.** "Every access to every
object must be checked for authority" and "Base access decisions on
permission rather than exclusion" (Saltzer & Schroeder 1975,
https://web.mit.edu/Saltzer/www/publications/protection/Basic.html). The
second is literally the allowlist-vs-denylist rule: a *permit* list
fails closed on anything unrecognised; an *exclusion* list fails open.
The guard is built the wrong way round -- it excludes `qa*` rather than
permitting a known set.

**F3 -- Authorization must key on an attribute an AUTHORITY assigns, not
one the subject supplies.** NIST SP 800-162: "These subject attributes are
assigned and managed by an authority within the organization" and "Subject
attributes are provisioned by attribute authorities" (`sp800-162.pdf.txt:868,1044`).
The same rule in the OAuth setting: "authorization servers SHOULD NOT allow
clients to influence their `client_id` or any other claim that could cause
confusion with a genuine resource owner" (RFC 9700 s2.6).

**F4 -- The comparable systems all mint identity rather than accept it.**
- SPIRE: "Workload attestation asks the question: 'Who is this process?'",
  answered by "interrogating locally available authorities (such as the
  node's OS kernel...) in order to determine the properties of the process
  calling the Workload API" -- properties the workload cannot choose. The
  SPIFFE spec is explicit that the Workload API "does not include an
  authentication handshake or authenticating token **from the workload**."
- Kubernetes: the Pod does not name itself; "you set the
  `spec.serviceAccountName` field... Kubernetes then automatically provides
  the credentials", and the token is minted by the control plane via
  `TokenRequest`, "short-lived, automatically rotating".
- AWS IAM Roles Anywhere: identity is an X.509 cert chained to a registered
  **trust anchor**; the workload proves possession of a key, it does not
  send a role name.

The invariant across all three: **the caller transmits a credential the
platform can verify; it never transmits the authorization-relevant label.**

**F5 -- Enforcement point placement (Kubernetes admission).** An admission
controller "intercepts requests to the Kubernetes API server prior to
persistence of the resource, but **after** the request is authenticated and
authorized", and "If any of the controllers in either phase reject the
request, the entire request is rejected immediately." Two lessons for a
PreToolUse hook: (i) admission runs *after* identity is already
established -- it consumes an authenticated identity, it does not
manufacture one; (ii) rejection is hard and immediate -- there is no
fail-open arm.

**F6 -- Anthropic already ships fail-closed defaults AND a
non-forgeable enforcement point.** "Claude Code uses strict read-only
permissions by default" and "**Fail-closed matching**: Unmatched commands
default to requiring manual approval"
(https://code.claude.com/docs/en/security). More important for this step:
the subagent `tools` allowlist / `disallowedTools` denylist is enforced by
the **runtime**, keyed on the loaded definition -- the built-in `Explore`
and `Plan` agents are documented as "read-only tools; Write and Edit are
denied" (https://code.claude.com/docs/en/sub-agents). That is a policy
decision the spawned agent's *label* cannot influence.

**F7 -- The documented model vs the measured reality (the crux).** The docs
say `agent_type` IS a definition attribute: `name` is a "Unique identifier
... Hooks receive this value as `agent_type`", and "identity comes only from
the `name` frontmatter field". If that held universally, a name-keyed
allowlist would be sound. **It does not hold on the installed build**: only
TWO definitions exist (`.claude/agents/qa.md` name `qa`,
`researcher.md` name `researcher`) yet the guard log contains **72 distinct
`agent_type` values**. The extra ~70 arrive from invocation-time
labels/dynamic `--agents` JSON, occupying the same field. So `agent_type`
is a **union of a platform-derived attribute and a caller-chosen label,
with no discriminator between them** -- exactly the RFC 9700 anti-pattern.

## B5. Consensus vs debate

**Consensus (unanimous across all 12 sources):** authorization keys on an
attribute bound by an authority; the enforcement point must be
always-invoked and small; defaults deny. No source in the read-in-full set
supports authorizing on a caller-supplied name.

**Debate / genuine tension:** *where* to put the control. Kubernetes and
Anderson favour a single mandatory chokepoint; Saltzer & Schroeder's
economy-of-mechanism warns that a chokepoint you cannot verify is worse
than none. The Claude Code product resolves this two ways at once --
runtime tool-restriction (verifiable, non-forgeable) *and* hooks
(flexible, but user-authored and fail-open by local choice). **The
literature clearly prefers the former.** I did not find a source arguing
the opposite, but per B0 that absence is unsearched, not established.

## B6. Pitfalls the literature predicts (all three already realised here)

1. **Exclusion-list drift** (violates fail-safe defaults). Every new
   spelling of a role is a new bypass. MEASURED: `quality-auditor`,
   `QA-Upper`, `general-purpose` writing `evaluator_critique_82.*.md`.
2. **A guard that is not tamper-proof / not always-invoked** (violates
   Anderson A+B). MEASURED: 32 `SyntaxError` traces in the log = a window
   where the guard allowed everything; plus the documented Bash-subprocess
   bypass.
3. **Confusing an identifier with an attribute** (NIST 800-162). `agent_id`
   is a *name for an instance*, not a *role attribute*. Authorizing on it
   directly would need a registry mapping id -> role, which does not exist.

---

# PART C -- APPLICATION TO PYFINAGENT

**C1. Answer to the central question (criterion 2).** On Claude Code
**v2.1.227**, the PreToolUse payload does **NOT** let a hook distinguish a
subagent's TYPE from its caller-chosen NAME. `agent_type` carries both, and
the docs call it "Agent name". There is no second field carrying the
definition. What `agent_id` *does* give you -- and the docs state this as
its purpose -- is a reliable **subagent-vs-main-thread** discriminator,
confirmed by measurement (63/63 real subagent writes carry it; 0/77 of
Main's do). `agent_id` is **not forgeable by the subagent in-band** (the
runtime writes the payload), but it is also **not authoritative for role**:
it is opaque, joins to nothing, and 18 distinct values map to roles only
via the same untrusted `agent_type` in the same record. The `a-probe-123`
row proves only that anything able to *invoke the script directly* controls
its stdin -- a different threat model from a subagent under the runtime.

**C2. The design consequence.** A name-keyed allowlist cannot be repaired by
widening the pattern -- 86.31 already widened `== "qa"` to the `qa*` prefix
and `quality-auditor` still walks past. Per F3/F4 the fix must key on
something the spawn site cannot choose. Three candidate shapes, in
descending strength:
- **(i) Platform-enforced tool restriction** -- `disallowedTools: Write,
  Edit` in `.claude/agents/qa.md`, enforced by the runtime, not the hook
  (F6). Strongest, and it is the pattern the built-in `Explore` agent
  already uses (and which `harness-self-audit.js:84,90` already relies on).
  **Blocker to check in PLAN:** qa.md's `memory: project` (`qa.md:26`)
  exists precisely to re-inject Write/Edit for memory curation; a blanket
  `disallowedTools` would kill the `verdict_wip_*.md` write-first mechanism
  that phase-86.31 built (measured: 44 post-change `qa` writes, nearly all
  into `.claude/agent-memory/qa/verdicts/`). This is a real conflict, not a
  formality.
- **(ii) Invert the hook to a narrow PERMIT list** (fail-safe defaults,
  F2): deny Write/Edit for **any** non-empty `agent_type` outside an
  explicitly enumerated set of (role, allowed-path) pairs. This closes the
  `quality-auditor` class.
- **(iii) Keep the exclusion list.** Rejected by F2/F3; documented here so
  the contract can say why.

**C3. THE FAIL-CLOSED RISK IS REAL AND MEASURED (criterion d).** Shape (ii)
would break the **researcher rail** unless `researcher` (and every
`research*`/`res-*` variant -- 31 distinct spellings, 821 events) is
permitted to write `handoff/current/research_brief_*.md` and
`.claude/agent-memory/researcher/MEMORY.md`. **This very brief was written
through that path** (`agent_type: "researcher"`, `agent_id:
a77dfb7250e0af1df`). Write-first is non-negotiable
(`.claude/agents/researcher.md:97-105`) and `research-gate.js:43-48` says
the researcher "legitimately NEEDS Write". A fail-closed change that
forgets the `research*` name-space converts a *silent authorization gap*
into a *loud gate outage* -- and note the irony: a permit list keyed on
names inherits the SAME spelling-drift weakness on the allow side.
`workflow-subagent` (82 events, wrote `backend/services/kill_switch.py`)
and `general-purpose` (24) must also be classified before any flip.

**C4. Bug found in passing (not in scope, queue it).**
`.claude/workflows/research-gate.js:47-48` states the guard "matches
`agent_type == 'qa'` only". That is **stale since the 86.31 widening** --
the live predicate is the `qa`/`qa-`/`qa_` prefix rule
(`qa-write-guard.sh:102-103`). The conclusion it supports (researcher is
not blocked) still holds, so nothing is broken today, but the comment
would actively mislead the next editor.

**C5. Also worth the contract's attention.** The guard's fail-open arm has
demonstrably fired (32 SyntaxError traces). Anderson's criterion A
(tamper-proof) and B (always invoked) are both currently unmet, independent
of the identity question. A liveness assertion -- prove the guard is ALIVE
before asserting what it decides -- belongs in the same step.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/hooks/qa-write-guard.sh` | 127 | PreToolUse Write/Edit guard; the PEP under study | LIVE; name-keyed; 3 fail-open arms, one measured firing |
| `.claude/settings.json` | :37-41 | Registers the guard under the `Write\|Edit` matcher | LIVE |
| `.claude/agents/qa.md` | :4 `tools`, :26 `memory: project`, :28 `permissionMode: plan` | Q/A definition; `tools` omits Write/Edit but `memory` re-injects them | The root cause the hook exists to compensate for |
| `.claude/agents/researcher.md` | :4 `tools`, :27 `memory: project`, :97-105 write-first | Researcher definition; legitimately needs Write | Would be broken by a careless fail-closed flip |
| `.claude/workflows/research-gate.js` | :43-48 comment, :670 `agentType:'researcher'`, :734 `agentType:'Explore'` | Researcher rail launch | LIVE; **comment at :47-48 is STALE** |
| `.claude/workflows/qa-verdict.js` | :207-210 rationale, :232 `agentType:'qa'` | Q/A rail launch | LIVE |
| `.claude/workflows/harness-self-audit.js` | :84, :90 `agentType:'Explore'` | Read-only auditors via platform tool-restriction | LIVE; the F6 pattern already in use |
| `.claude/workflows/probe-qa-tool-surface.js` | :42 `agentType:'qa'` | Tool-surface probe | LIVE |
| `handoff/logs/qa_write_guard.log` | 7,757 lines / 7,727 parsed | The empirical record | 72 distinct agent_type; 18 agent_id |

Internal files inspected: **9** (plus the derivation script and two
extracted PDFs).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **12** (10 via
      WebFetch, 2 via the documented curl+pypdf PDF chain since
      `research-gate.md:107-143` forbids WebFetch on `arxiv/pdf`-class PDFs)
- [x] 10+ unique URLs total -- **22**
- [x] Recency scan (last 2 years) performed + reported -- B3, **with the
      substituted method disclosed**
- [x] Full papers / pages read, not abstracts -- Anderson 142pp / 800-162
      47pp fully extracted; all web sources fetched whole
- [x] file:line anchors for every internal claim -- Part A + inventory

Soft checks:
- [x] Internal exploration covered every relevant module (guard, settings,
      both agent defs, all 4 workflow launch sites, the log)
- [x] Contradictions / consensus noted -- B5, and the doc-vs-measurement
      contradiction is F7
- [x] Claims cited per-claim
- [ ] **GAP: the mandatory three-variant search discipline could not be
      run** (WebSearch 200/200 exhausted before spawn -- B0). No
      adversarial source was surfaced. Primary-source substitution covered
      every entity the caller named, but serendipitous discovery did not
      happen. Main should weigh this when deciding whether the external
      half is sufficient for the contract.

