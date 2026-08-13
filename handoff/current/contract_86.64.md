# Contract — step 86.64

**Step:** 86.64 — the qa-write-guard cannot see the write channel that would actually be used to evade it
**Priority:** P2  |  **Status at contract time:** pending
**Date:** 2026-08-14

---

## Research gate — PASSED

Run `wf_bb618099-661`, tier `moderate`. `gate_passed: true` recomputed by the script,
**no rail drop**. 9 sources read in full (floor 5), 39 URLs (floor 10), corroborated
39 <= 39 distinct in the brief, all 9 claimed sources present, recency section present,
`brief_status: COMPLETE`. Brief: `handoff/current/research_brief_86.64.md` (34,612 chars).

---

## THE GATE REFUTED THE STEP'S PREMISE. Read this before designing anything.

The step's basis — and the hook's own comment at `:18` — says *"Write/Edit hooks do not
intercept Bash."* **That is true as written and false as read.**

Anthropic's permissions documentation states PreToolUse hooks run **"for every tool
except EndConversation"**, and exit 2 **"stops the tool call before permission rules are
evaluated."** I verified the registration myself:

```
event=PreToolUse   matcher='Write|Edit'
```

**The gap is the MATCHER in `.claude/settings.json`, not a platform limitation.** In
CWE-693 terms that is **"Protection Mechanism Ignored"**, not *missing*.

**Consequence for criterion 4, and it is the whole shape of this step:** its conclusion
may **not** rest on *"the control cannot be made sound because Bash is uninterceptable."*
Bash **is** interceptable. Any honest conclusion must rest on **DECIDABILITY** — can a
hook reliably decide whether a given shell command writes a guarded path?

**Evidence cuts both ways and both belong in the answer:**
- **CARE (2026): 85.64% F1** on shell-command verification — non-trivial, not hopeless.
- **CVE-2025-66032** defeated Claude Code's own validator via **`$IFS` rewriting** — a
  reminder that shell is adversarially hard to parse.

---

## Second, independent hole: the identity is unauthenticated

Phase-86.33 established `agent_type` is **caller-chosen**. So even a perfectly-matched
hook keys on an attribute the caller supplies. **Two independent failures — an ignored
channel and an unauthenticated subject — and fixing only one leaves the control unsound.**

**And the evasion is not hypothetical:** phase-82.39's Q/A already channel-switched, in
its own words — *"blocked my Write, so I moved everything to stdin."*

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. the Bash-channel evasion is DEMONSTRATED, not argued: drive a write to a guarded path through Bash under a qa-shaped agent_type and show the hook does not fire
2. the full set of tool names capable of writing a file is ENUMERATED from the platform's own tool list, not guessed, and each is stated as intercepted or not
3. any change preserves fail-open on internal error -- a hook that blocks its caller when the hook itself breaks is a worse failure than the one being fixed, and that property is proven by making the guard's own helper raise
4. if the conclusion is that the control cannot be made sound (because agent_type is caller-chosen AND Bash is uninterceptable), the statusMessage and documentation are corrected to describe a CONVENTION CHECK rather than a boundary -- accuracy of the claim is itself an acceptable outcome
5. mutation-test any guard added: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

Immutable verification command:
```
bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'
```

---

## Plan

1. **Criterion 1 — DEMONSTRATE the evasion**, do not argue it. Drive a write to a guarded
   path through Bash under a qa-shaped `agent_type` and show the hook does not fire.
   **Note the doubly-blind trap:** even with the matcher fixed, `:124` gates on
   `tool_name in ("Write","Edit")` and would allow Bash anyway. Both must change, and a
   test that only fixes the matcher will still show "no fire" for the wrong reason.
2. **Criterion 2 — ENUMERATE from the platform's tool list**, not by guessing. The gate
   reports **6 of 45 tools write files; the hook covers 2**; the Q/A can call **Bash**
   (load-bearing for its verification work) and `browser_take_screenshot`.
   **`MultiEdit` does not exist** — verified 0 occurrences. **Re-derive all of this.**
   **Also resolve a discrepancy the brief did not flag:** `qa.md`'s `tools:` line lists
   **no Write and no Edit**, yet this session's agent listing showed both on the Q/A's
   runtime surface. Declared frontmatter and effective tools appear to disagree — which
   is exactly what this criterion exists to settle.
3. **Criterion 3 — preserve fail-open on internal error, PROVEN.** The gate supplies the
   design: **CARE fails closed ONLY on parse-failure AND positive high-risk indicators.**
   That **narrow fail-closed** reconciles Saltzer & Schroeder's complete mediation with
   the requirement that a broken hook must never block its caller. A blanket
   fail-closed is forbidden by this criterion.
4. **Criterion 4 — if the control cannot be made sound, correct the docs.** Given the
   premise refutation, the honest answer is now *conditional on decidability*, not on
   capability. Whatever is concluded, **`.claude/settings.json`'s `statusMessage` must
   describe what the control actually does** — a control advertised as protective while
   bypassable buys unearned confidence.
5. **Criterion 5 — mutation-test any guard**, control observed GREEN first, byte-identical
   restore.

---

## Constraints

- **Do NOT weaken any existing guard or gate.** This is a harness-integrity step.
- **Do NOT make the hook blanket fail-closed** — criterion 3 forbids it, and a hook that
  blocks its caller when the hook itself breaks is a worse failure than the one being fixed.
- Paper trading only. No flag promotions, no `.env` writes.
- **F7 from the gate:** all 307 Bash log records are **synthetic** (12-key vs 5-key
  discriminator) — do not treat them as production evidence of the evasion.

---

## References

- `handoff/current/research_brief_86.64.md` — gate PASSED
- `.claude/hooks/qa-write-guard.sh:2,:18,:38,:124,:134` — matcher, self-declared gap, fail-open, the tool_name gate
- `.claude/settings.json` — `PreToolUse` registration, matcher `Write|Edit`, the advertising `statusMessage`
- Saltzer & Schroeder (complete mediation); CWE-693 / CWE-638 / CWE-424;
  CARE (arXiv 2026); CVE-2025-66032 (`$IFS` validator bypass)
- phase-86.33 (`agent_type` is caller-chosen); phase-82.39 (a Q/A that already channel-switched)
