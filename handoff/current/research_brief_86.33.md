# Research Brief -- step 86.33

**Topic:** Authorization when the subject identifier is unreliable or shared --
enforcing least privilege on a caller whose identity field is self-reported,
absent, or reused by unrelated principals.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported
for information only; `coverage.dry` not required).

**Accessed / written:** 2026-08-11. Researcher = Layer-3 pyfinagent MAS (Workflow rail).

---

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 8,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.33.md",
  "gate_passed": false
}
```

`brief_status: COMPLETE` means the run finished and this brief is assessable -- it
does NOT mean the gate passed. `gate_passed` is **false**: the run did not drop, but
one caller-named floor (the three-variant search discipline) could not be executed
because the session's WebSearch budget was already exhausted at spawn time. See the
Research Gate Checklist at the foot of this brief.

---

## Status log (append-only)

- [t0] Brief created; envelope written born-inert. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full.
- [t1] Internal: read the guard, both checkers, the rails' `agentType` pins, and
  tabulated all 7,224 records of the guard log.
- [t2] **CONSTRAINT DISCLOSED:** `WebSearch` returned *"Web search was not performed:
  this session has used its web search budget (200 of 200 WebSearch calls)"* on the
  FIRST search attempt of this run. The cap is session-level and was already spent
  before this researcher was spawned. Consequence for the mandatory three-variant
  search discipline (`.claude/rules/research-gate.md` "Search-query composition"):
  **I could not run the current-year / last-2-year / year-less query triple.**
  `WebFetch` is a separate tool and still works, so the >=5-read-in-full and >=10-URL
  floors are met by fetching canonical URLs directly. The recency scan is performed by
  *fetching* dated 2025-2026 primary sources rather than by *searching* for them. That
  is a weaker discovery method -- it can only surface work I already knew of, so a
  2025-2026 unknown-unknown would be missed. Disclosed, not papered over.
- [t3] Read in full: kernel.org seccomp_filter, RFC 9700, SPIFFE concepts,
  arXiv:2501.09674, Claude Code hooks reference. FAILED to fetch:
  `http://cap-lore.com/CapTheory/ConfusedDeputy.html` ("unable to verify the first
  certificate") and `http://www.erights.org/elib/capability/duals/myths.html`
  ("connect ECONNREFUSED 209.59.210.181:443") -- both are the canonical
  capability-security primary sources; substitutes fetched instead.
- [t4] Read in full: SELinux Notebook domain transitions, no_new_privs,
  credentials(7), confused-deputy article. Round-2 internal measurement separating
  real agent writes from checker-synthetic probes.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/hooks | 2026-08-11 | official vendor doc | WebFetch, full | `agent_type` is documented as *"Agent name (for example `"Explore"` or `"security-reviewer"`). Present when the session uses `--agent` or the hook fires inside a subagent."* Also documents a SEPARATE common field **`agent_id`** the guard never reads. Exit 2 blocks; 1 does not. All matching hooks under one matcher run in parallel. |
| https://en.wikipedia.org/wiki/Confused_deputy_problem | 2026-08-11 | encyclopedia (community tier) | WebFetch, full | Hardy 1988 compiler/(SYSX)BILL example. Root cause: *"the designator for the file does not carry the full authority needed to access the file"* and *"the program's own permission to access the file is used implicitly"*. ACLs *"separate object designation from authorization, leaving the ambient authority vulnerability intact."* |
| https://raw.githubusercontent.com/SELinuxProject/selinux-notebook/main/src/domain_object_transitions.md | 2026-08-11 | official project doc | WebFetch, full | A domain transition needs THREE permissions: `process transition` on the source, `file execute` on the binary, and **`file entrypoint` granted by the TARGET domain**. *"Processes cannot arbitrarily choose their own new domain."* |
| https://docs.kernel.org/userspace-api/no_new_privs.html | 2026-08-11 | official kernel doc | WebFetch, full | *"execve() promises not to grant the privilege to do anything that could not have been done without the execve call."* Once set it **"cannot be unset"** and persists across fork/clone/execve. Setuid bits stop changing uid; LSMs stop relaxing constraints post-execve. |
| https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html | 2026-08-11 | official kernel doc | WebFetch, full | *"Prior to use, the task must call `prctl(PR_SET_NO_NEW_PRIVS, 1)` or run with CAP_SYS_ADMIN"* -- because *"this requirement ensures that filter programs cannot be applied to child processes with greater privileges than the task that installed them."* Filters are inherited: *"any child processes will be constrained to the same filters."* |
| https://man7.org/linux/man-pages/man7/credentials.7.html | 2026-08-11 | official man page | WebFetch, full | Real UID *"determine[s] who owns the process"*; effective UID is *"used by the kernel to determine the permissions."* Across `execve(2)` *"real user and group ID and supplementary group IDs are preserved; the effective and saved set IDs may be changed."* PID/PPID persist across execve. |
| https://www.rfc-editor.org/rfc/rfc9700.html | 2026-08-11 (pub. Jan 2025) | IETF BCP | WebFetch, full | S4.15: AS *"SHOULD NOT allow clients to influence their `client_id` or any other claim that could cause confusion with a genuine resource owner."* S2.3: RS *"obliged to verify, for every request, whether the access token ... was meant to be used for that particular resource server. If it was not, the resource server MUST refuse."* |
| https://spiffe.io/docs/latest/spiffe-about/spiffe-concepts/ | 2026-08-11 | official project doc | WebFetch, full (page is thin) | *"the Workload API does not require that a calling workload have any knowledge of its own identity, or possess any authentication token when calling the API."* Identity is ASSIGNED by an attestor, never presented by the caller. (Attestation mechanics live on a further page -- see snippet-only.) |
| https://arxiv.org/html/2501.09674v1 | 2026-08-11 (pub. Jan 2025) | preprint (South, Marro, Hardjono, Mahari, Whitney, Greenwood, Chan, Pentland) | WebFetch, full (arXiv native HTML per the html->ar5iv chain) | Agent authority must be bound cryptographically, not asserted: the delegation token contains *"references to (e.g., hash of) the corresponding user's ID token and the agent's Agent-ID token."* Advocates *"enforcing resource scoping with structured permissions"* because *"structured permissions are unambiguous and deterministic."* |

## Identified but snippet-only / not fetched (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| http://cap-lore.com/CapTheory/ConfusedDeputy.html | Hardy 1988, canonical primary | **Fetch attempted and FAILED**: "unable to verify the first certificate" |
| http://www.erights.org/elib/capability/duals/myths.html | Miller/Yee/Shapiro, canonical primary | **Fetch attempted and FAILED**: "connect ECONNREFUSED 209.59.210.181:443" |
| https://srl.cs.jhu.edu/pubs/SRL2003-02.pdf | peer-reviewed PDF (Capability Myths Demolished) | Binary PDF; per auto-memory `reference_webfetch_pdf_summaries_fabricate_quotes`, WebFetch PDF summaries have twice fabricated quotes in this project. Not fetched rather than risk a fabricated quote. |
| https://www.rfc-editor.org/rfc/rfc8693.html | IETF RFC (OAuth Token Exchange; `act`/`may_act` delegation claims) | Adjacent; RFC 9700 covered the confused-deputy leg |
| https://www.rfc-editor.org/rfc/rfc9207.html | IETF RFC (`iss` parameter) | Named BY RFC 9700 as the mix-up countermeasure; not independently needed |
| https://csrc.nist.gov/pubs/sp/800/207/final | NIST SP 800-207 Zero Trust | PDF; same fabrication risk |
| https://man7.org/linux/man-pages/man5/apparmor.d.5.html | official man page (AppArmor Px/Ux/ix transitions) | SELinux notebook already established the mediated-transition principle |
| https://spiffe.io/docs/latest/spire-about/spire-concepts/ | official project doc (SPIRE node/workload attestors) | The concepts page pointed here for attestation mechanics; budget |

**Search-variant disclosure:** the three-variant discipline (current-year /
last-2-year / year-less) could NOT be executed -- see [t2]. The read-in-full set
does span the three eras by construction (year-less canonical: Hardy 1988,
credentials(7), SELinux, seccomp, no_new_privs; 2025: RFC 9700, arXiv:2501.09674;
current: the Claude Code hooks doc), but that spread was achieved by direct
fetching, not by search. This is a **hard-blocker gap** and is scored as such below.

## Recency scan (last 2 years, 2024-2026)

**Performed** -- by direct fetch of dated primary sources, not by search (see [t2]).
Result: **2 new findings that complement, and do not supersede, the canonical prior art.**

1. **RFC 9700 (Jan 2025)** restates the confused-deputy defence in modern protocol
   terms and lands exactly on this step's question: an authorization server *"SHOULD
   NOT allow clients to influence their `client_id`"*. A self-chosen identifier is
   treated as an attack surface, not an input.
2. **arXiv:2501.09674 (Jan 2025)** is the closest published work to pyfinagent's
   actual situation -- authorization for AI agents whose identity is self-reported.
   Its answer is *not* a better name-matching rule: it is a cryptographically bound
   delegation token plus structured scopes.

Nothing found in the window overturns Hardy (1988) or the capability-vs-ACL
analysis; the 2025 work applies them to agents. **Caveat:** because search was
unavailable, absence of a superseding 2025-2026 result is NOT established -- only
that none of the sources I could reach reports one.

## Key findings

1. **The defect class has a name and a 1988 citation.** A confused deputy is
   *"a computer program that is tricked by another program (with fewer privileges)
   into misusing its authority"*; the root cause is that *"the designator ... does not
   carry the full authority needed to access the file"* and *"the program's own
   permission ... is used implicitly"* (Wikipedia/Hardy 1988,
   https://en.wikipedia.org/wiki/Confused_deputy_problem).
2. **Allowlisting on a mutable identity string is the anti-pattern, explicitly.**
   RFC 9700 S4.15: servers *"SHOULD NOT allow clients to influence their `client_id`
   or any other claim that could cause confusion"* (https://www.rfc-editor.org/rfc/rfc9700.html).
   The guard's predicate matches on a string the spawner freely chooses.
3. **The systems that solve this never let the subject name itself.** SPIFFE:
   *"the Workload API does not require that a calling workload have any knowledge of
   its own identity, or possess any authentication token"* -- identity is derived by an
   attestor from properties the workload cannot forge
   (https://spiffe.io/docs/latest/spiffe-about/spiffe-concepts/).
4. **A rename boundary is enforceable only if a third party mediates the rename.**
   SELinux requires THREE permissions for a domain transition, and the decisive one is
   `entrypoint`, granted by the *target* domain over the binary -- so *"processes cannot
   arbitrarily choose their own new domain"*
   (https://raw.githubusercontent.com/SELinuxProject/selinux-notebook/main/src/domain_object_transitions.md).
   The kernel, not the process, performs the relabel.
5. **The alternative to mediating the rename is to make privilege non-increasing
   across it.** `no_new_privs`: *"execve() promises not to grant the privilege to do
   anything that could not have been done without the execve call"*, and it
   *"cannot be unset"* (https://docs.kernel.org/userspace-api/no_new_privs.html).
   Seccomp requires it precisely so *"filter programs cannot be applied to child
   processes with greater privileges than the task that installed them"*
   (https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html).
   **This is the directly transplantable idea: default-deny + monotonic restriction
   beats name-matching, because it does not need to identify anyone.**
6. **Unix already separates "who you claim to be" from "where you came from."**
   Effective UID is the authorization input, but PID/PPID persist across `execve(2)`
   while *"the effective and saved set IDs may be changed"*
   (https://man7.org/linux/man-pages/man7/credentials.7.html). Ancestry is the
   unforgeable-by-the-subject channel; the credential is the mutable one.
7. **The 2025 agent-specific answer is binding, not naming.** arXiv:2501.09674 binds
   agent authority by *"references to (e.g., hash of) the corresponding user's ID token
   and the agent's Agent-ID token"* and argues for structured scopes because they are
   *"unambiguous and deterministic"* (https://arxiv.org/html/2501.09674v1).

## Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/hooks/qa-write-guard.sh` | `:66-93` `is_qa_role()`; `:96-102` decision; `:39-44,:107-117` fail-open | The guard. Predicate = `n == "qa" or n.startswith("qa-") or n.startswith("qa_")` on a lowercased `agent_type` | LIVE. Predicate is a **prefix match on a self-chosen string** |
| `.claude/hooks/qa-write-guard.sh` | `:13-16` header vs `:66-91` docstring | Header still cites the hooks doc as *"for custom subagents, this is the frontmatter name"*; the `is_qa_role` docstring corrects it to the SPAWN NAME | **Internal contradiction, unresolved.** The upstream doc I fetched says *"Agent name (for example `"Explore"` or `"security-reviewer"`)"* -- the docstring is right, the header is stale |
| `.claude/hooks/qa-write-guard.sh` | `:46` `agent_type = d.get("agent_type")`; `:57-58` log line | Reads and logs ONLY `agent_type` | **Gap.** The hooks doc lists `agent_id` as a common PreToolUse field. It is never read, never logged -- so there is zero project data on whether a non-self-chosen discriminator is even populated |
| `.claude/hooks/qa-write-guard.sh` | `:18-20` | Discloses that Bash subprocess writes are not intercepted | **Corroborated externally** -- the hooks doc has no subprocess-write event |
| `handoff/logs/qa_write_guard.log` | 7,224 parsed records (32 unparsed), 2026-07-24 -> 2026-08-11 | The empirical identity census | **72 distinct `agent_type` values** (the docstring's "3110 records / 27 identities" is now stale) |
| `.claude/workflows/qa-verdict.js` | `:232` | Pins `agentType: 'qa'` | Matched by the predicate |
| `.claude/workflows/research-gate.js` | `:670` | Pins `agentType: 'researcher'` | NOT matched (correct -- write-first is mandatory) |
| `.claude/workflows/research-gate.js` | `:734` | Pins `agentType: 'Explore'` for the stage-2 brief verifier | **Third Layer-3 agentType.** ZERO events in the log -- it has never issued a Write/Edit, consistent with a read-only verifier at `effort: 'low'` |
| `scripts/qa/prove_qa_write_separation_86_31.py` | `:59-112` 15 cases; `:127-140` liveness; `:157-164` liveness control | Drives the REAL hook; includes false-positive cases | Green. `:109-111` case `quality-auditor` -> ALLOW asserts the predicate *must not* over-match |
| `scripts/qa/verify_qa_write_first_86_31.py` | `:170` `QA_IDENTITIES`; `:211` researcher idents; `:400` R2 residual text | The checker that must keep passing | Green. Both checkers hard-code identity LISTS -- so neither can see an identity not on its list |
| `.claude/settings.json` | `:37-41` | Hook registration + statusMessage | LIVE under a `Write|Edit` matcher |

### Measured: is the residual gap theoretical or exercised?

The log conflates real agent activity with checker-driven synthetic probes (both
drive the same hook). Separating by timestamp spread:

- **Synthetic**: e.g. 5 different identities (`researcher`, `research-82-0`,
  `res-78-1`, `workflow-subagent`, `general-purpose`) all "wrote"
  `frontend/src/lib/api.ts` within the SAME second, `2026-08-10T10:07:32`. That is a
  probe sweep, not five agents.
- **REAL and load-bearing**: `general-purpose` issued **15 Write/Edit events on
  `handoff/current/evaluator_critique_82.7.md` and `evaluator_critique_82.5.md`** on
  2026-08-04, spread across two multi-hour windows (11:16-11:48 and 17:34-18:39).
  Those are evaluator artifacts -- the exact artifact class the guard blocks when the
  writer happens to be named `qa*` (`prove_qa_write_separation_86_31.py:72-74`).
- **REAL and legitimate**: `workflow-subagent` issued ~50 production-tree writes
  (28 `backend/`, 13 `frontend/`, 5 `scripts/`, 4 `handoff/`) spread over 2026-07-25
  to 2026-08-10 -- plausible GENERATE-phase implementation.

**So the guard's docstring is half-right.** Its claim that the unmatched identities
are *"indistinguishable from LEGITIMATE writers"* is corroborated for
`workflow-subagent`. But it is **not** the case that nothing evaluator-shaped ever
wrote under an unmatched name: `general-purpose` really did author evaluator
critiques. Whether that agent was acting as an evaluator or as Main's drafting helper
**cannot be determined from `agent_type`** -- and that indistinguishability is
precisely the finding, not a reason to dismiss it.

## Consensus vs debate (external)

**Consensus** (Hardy 1988; SPIFFE; SELinux; no_new_privs; RFC 9700): do not
authorize on a name the subject controls. Either (a) bind authority to an unforgeable
handle, (b) have a third party mediate every identity transition, or (c) make the
authority set monotonically non-increasing so identity stops mattering.

**Debate:** capability purists hold that ACL/identity systems are *structurally*
confused-deputy-prone; the RFC 9700 lineage keeps identity but hardens it with
audience restriction and sender-constrained tokens. Both agree the *name* is not the
authority. arXiv:2501.09674 sits with the second camp for agents (OAuth-extension),
which is the more transplantable of the two here.

## Pitfalls (from literature)

1. **Denylisting a name-shape.** Any predicate over a self-chosen string is bypassed
   by choosing another string -- the checkers at `verify_qa_write_first_86_31.py:170`
   and `prove_qa_write_separation_86_31.py:83-92` enumerate a LIST, so they structurally
   cannot detect the identity that is not on it (cf. auto-memory
   `feedback_guard_from_instance_not_class`).
2. **Over-matching breaks the other rail.** `researcher.md` "Write-first
   (non-negotiable)"; `prove_qa_write_separation_86_31.py:103-111` already encodes
   both directions. Widening the prefix is not free.
3. **Fail-open guards hide their own breakage** -- `qa-write-guard.sh:39-44,107-117`;
   the liveness leg at `:127-140` exists for this and must survive any edit.
4. **Parallel hooks.** The hooks doc confirms *"All matching hooks run in parallel"* --
   do not add a second `Write|Edit` hook that assumes ordering relative to this one.
5. **Ambient authority.** Every subagent inherits the session's write authority by
   default; the guard subtracts from it for one name-shape only. Default-deny is the
   literature's answer, default-allow-minus-a-prefix is the current shape.

## Application to pyfinagent

- The step's design question -- "enforce a rule at a boundary the subject can rename
  itself across" -- has exactly three answers in the literature, and only two are
  available here. **Mediated transition** (SELinux `entrypoint`) is unavailable: nothing
  in the harness can force the runtime to assign `agent_type`. **Monotonic
  restriction** (`no_new_privs`) is available in spirit and is the strongest candidate:
  invert the guard from *deny-if-named-qa* to *allow-only-if-on-a-narrow-permit*, so an
  unrecognised name gets LESS authority rather than more. That directly removes the
  rename incentive, because renaming buys nothing.
- **Before any redesign, close the observability gap first.** `qa-write-guard.sh:46,57-58`
  logs only `agent_type`. The hooks doc documents `agent_id` as a common PreToolUse
  field. Adding `agent_id` (plus `session_id`, `permission_mode`) to the log line at
  `:57-58` is a pure log-only change with no decision impact, and it is the only way to
  answer the objective's literal question -- *how do you distinguish two callers that
  present the same identity* -- with measurement instead of assertion. **Do not assume
  `agent_id` is populated or stable; measure it.**
- **Do NOT widen the prefix** to catch `general-purpose` / `workflow-subagent`:
  measured above, `workflow-subagent` does ~50 legitimate production writes. Blocking
  it breaks GENERATE. The `quality-auditor` -> ALLOW case at
  `prove_qa_write_separation_86_31.py:109-111` is the codified warning.
- **Fix the stale header** at `qa-write-guard.sh:13-16` -- it contradicts the
  `is_qa_role` docstring at `:66-91` and the upstream doc. Per auto-memory
  `feedback_matching_totals_hide_contradictory_content`, two copies of one fact that
  disagree are worse than one.
- **Refresh the docstring's numbers** at `:72-73`: "3110 records ... 27 DISTINCT qa-*
  identities" is now 7,224 records / 72 distinct `agent_type` values. State the
  predicate next to the number (auto-memory `feedback_count_the_class_not_your_list`).
- **The covering control named at `:88-90`** (Main's post-verdict `git status`
  cleanliness rule) is a *detective* control, and it did not detect the 15
  `general-purpose` evaluator-critique writes. Any contract should say so rather than
  cite it as sufficient.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total -- **17** (9 read in full + 8 identified/failed)
- [x] Recency scan (last 2 years) performed + reported -- see section; method
      limitation disclosed
- [x] Full pages read (not abstracts) for the read-in-full set -- arXiv via native
      HTML per the html->ar5iv chain; no `arxiv.org/pdf/` fetch attempted
- [x] file:line anchors for every internal claim
- [ ] **Three-variant search discipline (current-year / last-2-year / year-less)** --
      **NOT MET.** `WebSearch` was budget-exhausted (200/200) before this run began;
      zero searches executed. `.claude/rules/research-gate.md` calls a single
      year-locked query a protocol breach, so zero queries cannot be scored as met.

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE (all 7
      named files, plus `.claude/settings.json` and both `.claude/rules` docs)
- [x] Contradictions / consensus noted (capability-purist vs hardened-identity;
      guard header vs its own docstring)
- [x] All claims cited per-claim with URL or file:line

**Gate verdict: `gate_passed: false`.** Nine sources read in full and the recency scan
was performed, but one hard-blocker -- the mandatory three-variant search discipline --
could not be executed at all because the session's WebSearch budget was already spent.
The substantive findings above stand on their own and are usable by PLAN; what is NOT
established is that no relevant 2024-2026 work was missed. Per
`.claude/rules/research-gate.md`, returning `false` honestly is the correct outcome and
padding the brief to mask the gap would be a protocol breach. **Remedy for a re-run:**
raise `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` or re-spawn in a fresh session, then
run the three query variants over "confused deputy", "capability vs identity access
control", and "AI agent identity authorization 2026".
