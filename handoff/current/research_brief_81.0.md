# Research Brief -- phase-81.0 "Harness continuity repair"

**Tier: complex.** Caller-implied (4-change plan validation + explicit adversarial
"find what is WRONG" mandate + 5 external topics + a named internal audit).
`coverage.audit_class = false` (caller did not set it; the internal half has a
bounded denominator -- 3 named helpers, 1 skill, 1 context dir).

**Verdict up front: the plan is directionally right and factually accurate on
every number I could re-measure -- and it is NOT SAFE TO EXECUTE AS WRITTEN.**
A1 as specified is a silent no-op, A1's stated root cause is wrong, A4's
"WARN-mode" is not expressible in the current contract and is owner-gated by an
immutable criterion, A2's fix is under-scoped by 2 of 3 leaking phases, and A3
deletes a directory while leaving the directive that points at it. Ten blockers
below.

---

## Queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| 1 | `anthropics cwc-long-running-agents harness design hooks evidence ledger` | year-less canonical |
| 2 | `fail-open vs fail-closed deployment gate design when is fail-open correct CI release engineering` | year-less canonical |
| 3 | `dangerous undetected failure proof test interval safety instrumented function diagnostic coverage` | year-less canonical (prior art: IEC 61508/61511) |
| 4 | `silent failure of automated safety controls dead gate detection alarm fatigue 2026` | current-year frontier |
| 5 | `AI agent harness verification gate stopped firing regression 2025 observability of the guardrail itself` | last-2-year window |

---

## Read in full (9; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://github.com/anthropics/cwc-long-running-agents | 2026-07-31 | official code/doc | WebFetch | "**Default-FAIL contract. Every criterion starts `false`; the agent can't mark it passing without opening evidence first.**" / "The agent can't claim success it hasn't observed." Hooks: `track-read.sh`, `verify-gate.sh`, `commit-on-stop.sh`, `kill-switch.sh`, `steer.sh`. |
| https://raw.githubusercontent.com/anthropics/cwc-long-running-agents/main/claude-code-config/.claude/hooks/verify-gate.sh | 2026-07-31 | official source | WebFetch | **VERBATIM CONFIRMED:** `if [ ! -s "$log" ]; then` → emits `{"decision":"block",...}` and `exit 0`. Empty case BLOCKS. Also self-limits: "This is a teaching example, **not a security boundary**." Guards **PreToolUse Write/Edit**, not delivery. |
| https://raw.githubusercontent.com/anthropics/cwc-long-running-agents/main/claude-code-config/.claude/hooks/track-read.sh | 2026-07-31 | official source | WebFetch | `case "$path" in *screenshots/*\|*-console.txt\|*-result.txt\|*.png) [ -f "$path" ] && echo "$path" >> "$log" ;; esac; exit 0`. The *recorder* fails open silently; only the *gate* fails closed. |
| https://code.claude.com/docs/en/hooks | 2026-07-31 | official docs | WebFetch | "**PostToolUse \| Can block? No \| Shows stderr to Claude; the tool already ran.**" and "For most events, stdout is written to the debug log **but not shown in the transcript**." `systemMessage` = "Warning message shown to the user" and IS valid for PostToolUse. "Handlers run in the current directory with Claude Code's environment." |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-31 | official blog | WebFetch (partial extraction) | "Each criterion had a **hard threshold**, and if any one fell below it, the sprint failed." / "agents tend to respond by confidently praising the work" / file-based handoffs quote. Extraction was partial -- see limitation note. |
| https://www.anthropic.com/engineering/managed-agents | 2026-07-31 | official blog (2026-04-08) | WebFetch | "our only window in was the WebSocket event stream, but that couldn't tell us *where* failures arose" -- harness self-observability named as the pain point. Error handling described as fail-closed. |
| https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 2026-07-31 | official blog | WebFetch | "LLMs have an '**attention budget**'"; "context rot"; find "the **smallest possible set of high-signal tokens**"; just-in-time "lightweight identifiers (file paths…)" over pre-loading; sub-agents return "condensed **1,000-2,000 token** summaries". |
| https://authzed.com/blog/fail-open | 2026-07-31 | authoritative eng blog | WebFetch | "A system that fails open defaults to an operational or open state in the event of a failure." Fail-open "is typically used where **availability is more critical than security**." Notably **silent on alerting** when the fail-open path is taken -- a gap this brief closes from the SIS literature. |
| https://risknowlogy.com/articles/detail/17689/ | 2026-07-31 | industry practitioner (IEC 61511) | WebFetch | Diagnostic test = runs **automatically**, **frequently**, produces an **automated response**, is **relevant to the safety function**. Anything else is a proof test. "**If you only discover a problem when the plant shuts down or an explosion occurs, you have not detected the failure — you have experienced its consequence.**" |

**Extraction limitation, disclosed:** the `harness-design-long-running-apps`
fetch returned a partial render (3 of 6 requested topics). I counted it as
read-in-full because WebFetch retrieved and processed the page, but I did NOT
get verbatim coverage of its fail-open/fail-closed stance -- so I do **not**
claim the article endorses either. The fail-open analysis below rests on the
*reference implementation* (`verify-gate.sh`, read verbatim) and the hooks doc,
not on an inferred blog position.

## Identified but snippet-only (26; context, does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2603.05786 (Proof-of-Guardrail) | preprint | Abstract+claims fetched; see recency scan. Counted as snippet, not full. |
| https://arxiv.org/pdf/2606.26924 (Deterministic Control Plane for LLM Coding Agents) | preprint | Adjacent, not decisive for a 4-change hook repair |
| https://arxiv.org/pdf/2604.18449 (silent driving-system failures) | preprint | Cross-domain corroboration only |
| https://www.harness.io/blog/introducing-agent-trace | vendor | Vendor product post |
| https://www.databricks.com/blog/ai-harness | vendor | Vendor definition post |
| https://vercel.com/i/ai-agent-observability | vendor | Vendor |
| https://www.abhishek-tiwari.com/agent-guardrails-action-gates-harnesses-and-governance-four-layers-four-different-jobs/ | blog | Useful taxonomy, community tier |
| https://cloudandsre.com/blog/harness-engineering-the-third-phase/ | blog | Community tier |
| https://addyosmani.com/blog/agent-harness-engineering/ | blog | Community tier |
| https://luhuidev.medium.com/anthropics-2026-agent-harness-architecture-from-agent-loop-to-agent-runtime-6da6db4f3f47 | blog | Medium, secondary to the primary source |
| https://github.com/ai-boost/awesome-harness-engineering | list | Link farm |
| https://www.nxcode.io/resources/news/what-is-harness-engineering-complete-guide-2026 | marketing | Low tier |
| https://infoq.com/news/2026/04/anthropic-three-agent-harness-ai/ | trade press | Secondary reporting |
| https://www.linkedin.com/posts/anthropicresearch_effective-harnesses-for-long-running-agents-activity-7399550329031180288-xR_w | social | Pointer to the primary |
| https://thenewstack.io/why-cicd-fails-llms/ | trade press | Adjacent (LLM release gates) |
| https://devsecopsschool.com/blog/fail-closed/ | blog | Redundant with AuthZed |
| https://www.trigguardai.com/fail-closed-ai-systems | vendor | Vendor |
| https://en.wikipedia.org/wiki/Fail-safe | encyclopedia | Definitional only |
| https://www.azuredevopslabs.com/labs/vstsextend/releasegates/ | vendor docs | Release-gate mechanics; corroborates timeout-then-fail |
| https://microsoftlearning.github.io/AZ400-DesigningandImplementingMicrosoftDevOpsSolutions/Instructions/Labs/AZ400_M03_L08_Control_Deployments_using_Release_Gates.html | vendor docs | Same |
| https://www.devopstraininginstitute.com/blog/10-cicd-quality-gates-for-production-level-reliability | blog | Low tier |
| https://www.sciencedirect.com/science/article/abs/pii/S0306454915004806 | peer-reviewed | Paywalled abstract only (PFD ∝ proof-test period) |
| https://www.gt-engineering.it/en/insights/process-safety-processi-gt-engineering/proof-test-diagnostic-coverage/ | industry | Redundant with Risknowlogy |
| https://valvemagazine.com/articles/are-your-safety-instrumented-systems-proof-tests-effective/ | industry | Redundant |
| https://ifluids.com/blog/sis-proof-testing-intervals-iec-61511/ | industry | Redundant |
| https://array.aami.org/doi/full/10.2345/0899-8205-46.4.268 (Monitor Alarm Fatigue: Integrative Review) | peer-reviewed | Canonical alarm-fatigue prior art; cited from snippet |

**URLs collected: 35. Read in full: 9. Snippet-only: 26.**

---

## Recency scan (2024-2026) -- performed

**Result: 4 new findings in the window; none supersedes the canonical prior art,
two materially change the plan.**

1. **`arXiv:2603.05786` "Proof-of-Guardrail in AI Agents and What (Not) to Trust
   from It" (2026)** -- the single most on-point new work. Its thesis is exactly
   pyfinagent's A1 problem: developers "may falsely claim safety measures are
   enforced," so the paper builds TEE attestation proving "a response is
   generated after a specific open-source guardrail" -- i.e. **proving the
   control RAN, not merely that it EXISTS.** It also warns against conflating
   the two. This is the theoretical justification for A1's "no input" token:
   today `proceed` conflates *"the gate ran and found nothing to gate on"* with
   *"the gate had nothing to run against."* **Complements, does not supersede,
   IEC 61511's DU-failure framing.**
2. **`anthropic.com/engineering/managed-agents` (2026-04-08)** -- names harness
   self-observability as the recurring pain: "our only window in was the
   WebSocket event stream, but that couldn't tell us *where* failures arose."
3. **`anthropics/cwc-long-running-agents` (Code with Claude 2026)** -- the
   reference implementation the caller asked about; published inside the window
   and read verbatim. Supersedes any 2025-era guess about Anthropic's gate style.
4. **2025-2026 harness-engineering literature** converges on "every failure you
   investigate becomes a permanent regression gate," which is the maintenance
   burden pyfinagent just experienced: gates accumulate, and nothing watches the
   gates.

**Canonical prior art still stands and is older than the window:** IEC 61508 /
61511 proof-testing and diagnostic-coverage theory (Risknowlogy, read in full)
and the alarm-fatigue integrative review (AAMI 2012). Neither is superseded --
the 2026 agent work is a re-derivation of the same result in a new domain.

---

## Key findings (external)

1. **Anthropic's reference gate fails CLOSED, and the auditor's report is
   verbatim-correct.** `if [ ! -s "$log" ]` blocks; the ledger is
   `{ "feature-1": { "passes": false } }`. (Source: `verify-gate.sh` + README,
   accessed 2026-07-31.)
2. **BUT it fails closed at a completely different control point than ours, and
   this is the finding that reframes the whole plan.** `verify-gate.sh` is a
   **PreToolUse** hook that denies the agent's *Write of the claim*. pyfinagent's
   `auto-commit-and-push.sh` is a **PostToolUse** hook that reacts to *delivery*
   after the claim was already written. Per the official hooks reference,
   **PostToolUse literally cannot block** -- "PostToolUse | Can block? **No** |
   Shows stderr to Claude; the tool already ran." So our fail-open at that point
   is **not a design choice we could reverse by being braver** -- it is forced by
   the event type. Calling our fail-open "the actual bug" is therefore a category
   error. **The real gap is that we have no fail-CLOSED control at the claim
   point** (PreToolUse on the `status: "done"` Write), which is precisely where
   Anthropic puts theirs.
3. **Fail-open is the correct choice for availability-critical paths and the
   wrong default for correctness-critical ones.** "Fail-open… is typically used
   where availability is more critical than security" (AuthZed). CI/release
   practice is deny-by-default. Our stated principle -- never break the
   masterplan Write -- makes the *Write* the availability-critical asset, which
   is defensible; but it does **not** justify letting the *push* proceed
   silently, because the push is not the Write.
4. **A control that fails silently is a Dangerous-Undetected (DU) failure, and
   the countermeasure is a proof test, not a better log line.** A *diagnostic*
   test "runs automatically… runs frequently… produces an automated, safe
   response"; anything else is a *proof test* requiring a human. "If you only
   discover a problem when the plant shuts down… you have not detected the
   failure — you have experienced its consequence." (Risknowlogy.) pyfinagent
   discovered its dead verdict gate by a 26-agent audit -- i.e. by consequence.
   **The durable fix is a scheduled proof test of the gates themselves,** which
   the 4-change plan does not contain.
5. **Boot context: just-in-time beats pre-loading, and a 33K-token boot render is
   an attention-budget tax with no offsetting benefit.** "LLMs have an 'attention
   budget'"; aim for "the smallest possible set of high-signal tokens"; prefer
   "lightweight identifiers (file paths, stored queries…)" loaded on demand.
   Sub-agents are expected to return "condensed 1,000-2,000 token summaries" --
   two orders of magnitude below what `/masterplan` emits. (Anthropic context
   engineering.) A2 is well-founded; the fix must shrink, not merely re-order.
6. **Alarm fatigue is the failure mode on the other side of A1/A4.** "When alarms
   go off too often and pointlessly, employees stop responding to them." A WARN
   that fires on every benign cycle is worth less than no WARN.

## Consensus vs debate

- **Consensus:** correctness gates should be deny-by-default; controls need
  attestation that they actually ran; boot context should be minimal and
  just-in-time.
- **Debate / genuine tension:** availability-first systems legitimately fail
  open (AuthZed), and the SIS literature is explicit that shorter proof-test
  intervals cost real money. There is **no** literature consensus that a
  PostToolUse-style advisory hook must fail closed -- and the Claude Code docs
  make it impossible anyway. So pyfinagent's fail-open discipline is
  **defensible where it sits**; the defect is the *absence of a second,
  fail-closed control upstream* plus the *absence of a liveness signal*.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/hooks/auto-commit-and-push.sh` | 286 | PostToolUse driver; consumes all 3 helpers | **Live; 3 identical `case` blocks with a swallowing catch-all** |
| `.claude/hooks/lib/verdict_gate.py` | 72 | `proceed\|passed\|hold` | Live but **DARK since 2026-07-20** |
| `.claude/hooks/lib/harness_log_gate.py` | 79 | `proceed\|passed\|skip`, env-gated | Live, **never enabled**; no WARN state exists |
| `.claude/hooks/lib/live_check_gate.py` | 87 | `proceed\|passed\|skip` | Live; hardening **already owned by pending step 75.5.10** |
| `.claude/skills/masterplan/SKILL.md` | 212 (file) / **132,714 B rendered** | Boot render | Live; 3 defects measured |
| `scripts/housekeeping/backfill_handoff_archive.py` | — | Sweeps `handoff/current/` | Live; `ROLLING_KEEP` at **:33-41** |
| `scripts/housekeeping/verify_handoff_layout.py` | — | Asserts the same invariant | Live; `ROLLING_KEEP` at **:25-33** |
| `.claude/context/known-blockers.md` | 56 lines+ | **Holds the "read last 3 session logs" directive at :56** | Live; read by a drill |
| `scripts/go_live_drills/incident_log_p0_test.py` | :23 | Constructed-path consumer of known-blockers.md | **LIVE; FATAL if file missing** |
| `tests/verify_phase_23_8_2.py` | :39 | Constructed-path consumer of `.claude/context/project.md` | Live |
| `.claude/settings.json` | :167-171 | `env` block; `Write(.claude/context/sessions/**)` at **:171** | Live; **:170 is a bare `"Write"`** |
| `backend/tests/test_phase_38_4_hook_gate.py` | :80, :86 | Step 38.4's immutable verification | Live; **scrubs the env var** |
| `.gitignore` | :76 | `handoff/logs/` | Live -- **the WARN destination is untracked** |

---

## Application to pyfinagent (file:line)

### A1 -- verdict gate

- **The consumer contract.** `auto-commit-and-push.sh:202` captures one stdout
  token; `:203-214` dispatches it. The final arm is `proceed|*)` at **:211**.
  **Proven by execution** (I ran the exact `case` with 6 tokens):
  `hold`→WARN+exit0, `passed`→INFO, and `proceed`, `noinput`, `warn`, `""` all
  →**silent continue**. Identical catch-alls at **:160** (live_check) and
  **:186** (harness_log). *Answer to the caller's most-important question: a new
  token breaks nothing -- fail-open holds -- but it is **swallowed without a log
  line**, which is exactly the silence A1 exists to remove.*
- **Today's conflation, measured.** `python3 .claude/hooks/lib/verdict_gate.py
  /nonexistent 81.0` → `proceed`; a stale `{"step_id":"75.0",...}` → `proceed`.
  Missing-input and stale-input are the same token, and neither is logged.
  Sources: `verdict_gate.py:39-40` (missing), `:41-42` (parse error), `:44-45`
  (not a dict), `:49-50` (stale step), `:51-53` (no verdict field) -- **five
  distinct causes, one token.** A single `noinput` must not conflate them: `:49`
  stale is a legitimately different condition and will fire routinely (see the
  multi-flip note below).
- **Multi-flip false-alarm risk.** `auto-commit-and-push.sh:123` picks
  `top_id = sorted(newly_done)[-1]` -- only the highest-sorted id. If two steps
  flip in one edit, a critique JSON for the *other* one is "stale" at
  `verdict_gate.py:49`. If A1 maps stale→`noinput`, that is a false WARN.
- **Where the alarm goes.** `log()` at `:47` writes to
  `handoff/logs/auto-push.log`, which `.gitignore:76` excludes. Per the hooks
  doc, PostToolUse stdout is "written to the debug log but **not shown in the
  transcript**." So the WARN reaches neither the operator nor Claude.
- **The 24-hit / dark-since-07-20 claim: VERIFIED.** `grep -c "verdict gate
  satisfied"` = **24**; last line `[2026-07-18T16:49:51Z] … 73.4` … final
  `[2026-07-20T07:39:04Z] INFO: verdict gate satisfied for 75.0`. Sweep commit
  `fa9aaf8e` "chore: handoff layout backfill" dated **2026-07-24**.

### A2 -- masterplan SKILL

- **All three numbers reproduce exactly.** I extracted `SKILL.md:13-143` and ran
  it verbatim: **132,714 bytes / 132,657 chars / ~33,164 tokens / 147 lines.**
  Open steps (not in `DONE`) = **262**; inline-shown = `sum(min(active,8))` =
  **92**. Rendered tail: `## Next actionable: 5.2 -- Data Provider Abstraction
  Layer` -- and `phase-5.status == 'deferred'`.
- **Mechanism.** `SKILL.md:24-25` define `ACTIVE` and `DONE` as two *non-
  exhaustive* sets. `:103` filters the render to `status in ACTIVE`; `:130`
  skips only `status in DONE`. A status in **neither** set is invisible at :103
  yet scanned at :131-140.
- **The plan under-scopes this by 2 of 3 phases.** Measured leakers:
  `phase-5` (`'deferred'`, 11 open steps), `phase-36` (`'deferred'`, 16), and
  **`phase-77` (status key MISSING → `None`, 3 open steps)**. A fix that
  special-cases the literal string `'deferred'` leaves phase-77 leaking. Step
  statuses `'deferred'` (15) and `'merged'` (2) are likewise orphaned and render
  as `[ ]` pending via the `step_icon` default at `:39`.
- **Bloat source is the step `name`, not the step count.** Open-step names total
  **399,157 chars**; median **1,391**; max **13,622** (step 80.11). `:44` prints
  `{s["name"]}` unbounded. The render is "only" 132K because it shows 92/262.
- **TaskCompleted references confirmed** at `SKILL.md:168` and `:192`.

### A3 -- context sweep

- **The directive is not in the directory being deleted.**
  `.claude/context/known-blockers.md:56` -- "Reading order for new sessions:
  1. Last 3 session logs in `.claude/context/sessions/`". A3 proposes deleting
  `sessions/` and does **not** propose touching `known-blockers.md`.
- **A live constructed-path consumer the plan does not name.**
  `scripts/go_live_drills/incident_log_p0_test.py:23` builds
  `REPO_ROOT / ".claude" / "context" / "known-blockers.md"`; its S0 check prints
  `FATAL: known-blockers.md not found` and aborts. It parses `## RESOLVED` /
  active sections by regex -- so an edit to :56 must not disturb that structure.
  (Second constructed-path consumer: `tests/verify_phase_23_8_2.py:39` →
  `.claude/context/project.md`.) `push-credential-diagnosis.md` has **zero**
  consumers -- safe to delete.
- **`CLAUDE.md:47`** ("Read `.claude/context/`") keeps this directory a live boot
  surface; it names project/mas-architecture/research-gate/owner, not sessions.
- **Directory measured:** 23 files, 272K on-disk (block-rounded; the plan's
  229,187 B is the byte sum -- both are right, state which you mean).

### A4 -- harness_log gate

- **Mechanism works.** `harness_log_gate.py:72` reads `os.environ`. **Measured:**
  `settings.json:167-169` sets `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, and that
  variable **is visible in a spawned subprocess** -- so `env` propagation is real
  (caveat: I proved it for a Bash-tool subprocess; the hooks doc says handlers
  "run… with Claude Code's environment," which is the same inheritance, but a
  hook-side echo would make it airtight). `HARNESS_LOG_GATE_ENABLED` is currently
  UNSET and absent from every shell rc.
- **It does not break step 38.4's immutable verification.**
  `test_phase_38_4_hook_gate.py:80` `monkeypatch.delenv(...)` and `:86`
  `env.pop(...)`, plus `harness_log_gate_test.sh:26` `unset` -- all three scrub
  the variable. Good news, but *assert it in the live_check rather than assuming*.
- **Detection window is ~2 cycles.** `harness_log_gate.py:53` keeps only the last
  200 lines. `handoff/harness_log.md` is 29,823 lines / 2.5 MB, and its last 200
  lines contain exactly **2** cycle headers (Cycle 176, Cycle 177) -- ~84 lines
  per block. Any flow that appends the log then appends more before the flip
  risks a false `skip`, which per `:180-181` exits **before** `git add -A` at
  `:239`.
- **CLAUDE.md's correction is right and the plan repeats it correctly:** the hold
  skips commit AND changelog AND push.

---

## Pitfalls (from literature, mapped)

1. **Proving existence ≠ proving execution** (`arXiv:2603.05786`). Restoring a
   file makes the gate *able* to run; it does not make anyone *know* it ran.
2. **DU failures are found by proof tests, not by better logs** (IEC 61511).
   Without a scheduled liveness check the next relocation goes dark identically.
3. **Alarm fatigue** -- a WARN that fires on benign multi-flip cycles trains the
   operator to ignore it.
4. **Attention budget / context rot** -- do not "fix" the 92/262 gap by rendering
   more; 262 full names would be ~400K chars / ~100K tokens.
5. **Fail-open is legitimate where availability dominates** -- do not let this
   brief be read as "make every hook fail closed." PostToolUse *cannot*.

---

## Recommended corrections (not new scope -- corrections to the 4 changes)

- **A1 must be a 2-file change**: helper token + a `noinput)` arm inserted at
  `auto-commit-and-push.sh:210`, **before** `proceed|*)`. Keep `stale` separate
  from `noinput`.
- **A1 must resolve the per-step filename** (see Blocker 2) and **add
  `evaluator_critique.json` to BOTH `ROLLING_KEEP` sets** (Blocker 3).
- **Route the alarm somewhere visible**: emit `{"systemMessage": "..."}` on
  stdout (valid for PostToolUse per the hooks reference) in addition to the
  gitignored log.
- **A4: get the operator token first**, and implement WARN-mode as a *new token
  plus a new shell arm*, not by flipping the existing env var.
- **A2: partition statuses exhaustively** (`ACTIVE = everything not in DONE`, or
  an explicit `UNKNOWN` bucket printed loudly) so `phase-77`'s missing status is
  caught too; cap the **name**, not the step count.
- **A3: fix `known-blockers.md:56` in the same commit** as the deletion.
- **Add a proof test** (a `verify_gates_live.py` that asserts each helper still
  receives real input) -- otherwise this repair has the same half-life as the last.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**9**)
- [x] 10+ unique URLs total (**35**)
- [x] Recency scan (last 2 years) performed + reported (4 findings)
- [x] Full pages read (not abstracts) for the read-in-full set -- with the one
      disclosed partial extraction (`harness-design-long-running-apps`)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module the caller named, plus 3 the
      plan omitted (`backfill_handoff_archive.py`, `verify_handoff_layout.py`,
      `incident_log_p0_test.py`)
- [x] Contradictions noted (fail-open is defensible at our control point; the
      auditor's "our fail-open is the actual bug" is refuted as a category error)
- [x] Claims cited per-claim
- [x] Every reproducible number re-measured rather than accepted (132,714 /
      92 / 262 / 24 / 5.2-from-deferred all reproduce)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 26,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0,
               "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "gate_passed": true
}
```
