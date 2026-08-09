# live_check — phase-36.27

Required shape (masterplan `36.27.verification.live_check`):

> *"A real researcher spawn through the new rail: the returned envelope
> verbatim, plus the brief file it wrote, plus a deliberate short-of-floor case
> showing rejection."*

All three parts below. Captured 2026-08-09, live tree.

*Created after the pass-1 Q/A pointed out it was missing. Without it the
auto-commit hook holds at `auto-commit-and-push.sh:155/181/206`, which exit-0s
**before** `git add -A` at `:239` — so the status flip would have skipped the
commit, the changelog **and** the push, not merely the push.*

---

## Part 1 — a REAL researcher spawn through the new rail

Run `wf_9880694c-d30`, `.claude/workflows/research-gate.js`. **Not a synthetic
exercise: this is step 86.1's actual research gate**, and its brief is what 86.1
will be built on.

2 agents (stage 1 researcher + stage 2 verifier), 40 tool uses, 191,253 tokens,
686,871 ms.

### The returned envelope, verbatim

> **REGENERATED after EVALUATE pass 2.** The first version of this block was
> hand-transcribed and silently dropped `envelope.summary` (1,193 chars) with
> no elision marker. The Q/A caught it with a leaf-path symmetric difference
> against the journal: zero value mismatches and zero invented fields, so
> nothing gate-bearing was hidden -- but an edited capture inside a block
> labelled *verbatim* is a defect regardless of whether the omission mattered,
> and the masterplan's `live_check` uses the word *verbatim* as its contract.
> This block is now emitted PROGRAMMATICALLY from the run's stored result via
> `json.dumps(..., indent=2)` -- not retyped -- so it cannot drift again.
>
> **Round-trip proven, against the right file.** The block below is byte-equal
> to the workflow's stored return value:
> `EXACT MATCH against workflows/wf_9880694c-d30.json: True`.
> Note the source distinction, which cost me one wrong comparison:
> `subagents/workflows/<run>/journal.jsonl` holds the **per-agent** returns
> (stage 1 and stage 2 separately), while the **script's** return value -- the
> object this block reproduces -- lives in `workflows/<run>.json`. Comparing
> against the journal reports a spurious mismatch because it is a different
> object, not a differing one. (`envelope.summary` measures 1,191 chars here;
> the verdict said 1,193 -- a counting-method difference, not a content one.)

```json
{
  "step_id": "86.1",
  "gate_passed": true,
  "agent_self_reported_gate_passed": true,
  "self_report_disagreed": false,
  "violations": [],
  "checks": [
    "sources_floor_ok: 8 >= 5",
    "urls_floor_ok: 44 >= 10",
    "recency_scan_ok",
    "not_audit_class: coverage.dry informational only",
    "listed_sources_consistent: 8 >= 8",
    "brief_on_disk_ok: handoff/current/research_brief_86.1.md (36790 chars, independently read)",
    "all_8_claimed_sources_present_in_brief"
  ],
  "brief_path": "handoff/current/research_brief_86.1.md",
  "brief_verification": {
    "brief_exists": true,
    "brief_non_empty": true,
    "char_count": 36790,
    "urls_checked": 8,
    "urls_present": 8,
    "urls_missing": []
  },
  "envelope": {
    "tier": "moderate",
    "external_sources_read_in_full": 8,
    "sources_read_in_full": [
      "https://martinfowler.com/articles/feature-toggles.html",
      "https://mir.cs.illinois.edu/marinov/publications/GyoriETAL15PollutionDetection.pdf",
      "https://docs.python.org/3/library/unittest.mock.html",
      "https://docs.pytest.org/en/stable/how-to/monkeypatch.html",
      "https://martinfowler.com/articles/patterns-of-distributed-systems/high-watermark.html",
      "https://cwiki.apache.org/confluence/display/KAFKA/KIP-101+-+Alter+Replication+Protocol+to+use+Leader+Epoch+rather+than+High+Watermark+for+Truncation",
      "https://arxiv.org/html/2509.00466v1",
      "https://arxiv.org/html/2510.26171v1"
    ],
    "snippet_only_sources": 36,
    "urls_collected": 44,
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
    "brief_path": "handoff/current/research_brief_86.1.md",
    "summary": "Premise CONFIRMED and sharpened. RE-DERIVED: reset_peak at kill_switch.py:670 (DARK return :693-694, assign :697, audit :698-700), _AUDIT_PATH :48, _BASELINE_EVENTS :709, _append_audit :432-443, _apply_authoritative_peak :397-430, settings.py:39. MEASURED: the LIVE journal holds ZERO peak rows (62: 44 pause/10 resume/8 sod); all 20 peak_update rows and the 24666.57 max live in handoff/audit/ archives, peak_reset never fired -- so a row written today wins the ts merge-sort outright and destroys 24666.57 permanently (trip point 22199.9 to 11110.5). FOUR non-obvious findings: (1) the isolation asymmetry is INVERTED -- the flag-ON arm (:195-207) IS isolated, the OFF arm is not; (2) a SECOND landmine -- with the flag ON, `assert out is None` at :191 goes RED, so greenness is coupled to operator config; (3) the get_state patch at :188 is vacuous BY IDENTITY (st bound at :187) and module fns read _state directly (:793/:995/:1033/:1047/:1053); (4) redirect-only is a HALF fix -- :697 corrupts the in-memory singleton too. _audit_archive_dir is derived (:89-91). __init__ replays, so redirect BEFORE construction. Verbatim command + 5 criteria + the 86.1/86.6 boundary are in the brief.",
    "gate_passed": true
  }
}
```

**The enforced `gate_passed` is the one that governs.** The agent's own value is
kept separately as `agent_self_reported_gate_passed`; here they agree, so
`self_report_disagreed` is `false`. The check `all_8_claimed_sources_present_in_brief`
comes from stage 2, not from the researcher's own say-so.

---

## Part 2 — the brief file it wrote

```
$ wc -c handoff/current/research_brief_86.1.md
   36998 handoff/current/research_brief_86.1.md
```

`char_count` 36,790 (characters) vs 36,998 (bytes) — the difference is non-ASCII
content, and both figures are correct for what they measure.

**Write-first was observed LIVE, not inferred.** The brief measured **54 lines
on disk while stage 1 was still running**, before any return value existed.
*(The Q/A corroborated this by a route I had not cited: the stage-1 transcript
shows **3 separate `Write` calls** to that path across 38 tool_use blocks.)*

**Main's independent re-check of stage 2** — because an LLM verifier should not
be taken on trust either:

```
claimed=8 listed=8 missing_from_brief=0
MAIN'S INDEPENDENT RE-CHECK AGREES WITH STAGE 2
```

And a spot-check of the brief's own headline measurement:

```
{'pause': 44, 'resume': 10, 'sod_snapshot': 8}     # 44+10+8 = 62 = the whole file
peak rows in LIVE journal: 0
```

Corroborating 86.1's severity: with **zero** peak rows in the live journal, a
`peak_reset` written today wins the `ts` merge-sort outright — there is nothing
later to override it.

---

## Part 3 — a deliberate short-of-floor case, showing rejection

Driven against the **shipped** `enforceGate`, with an envelope that
self-reports `gate_passed: true` while failing three floors and having no brief:

```json
{
 "gate_passed": false,
 "violations": [
  "external_sources_read_in_full=3 < floor 5",
  "urls_collected=4 < floor 10",
  "recency_scan_performed is not true",
  "brief not found on disk at /tmp/does-not-exist.md (write-first not honoured)"
 ],
 "checks": [
  "not_audit_class: coverage.dry informational only",
  "listed_sources_consistent: 3 >= 3"
 ],
 "agent_self_reported_gate_passed": true,
 "self_report_disagreed": true
}
```

**Rejected, not rounded up**, and the agent's `gate_passed: true` was
**overridden** with the disagreement recorded. Every failing floor is named
individually rather than collapsing to one message.

The full battery — including the empty-return cases, the audit-class
`coverage.dry` requirement, the over-claim detector, and the fail-closed
behaviour when stage-2 verification is absent — is
`node scripts/qa/verify_research_gate_workflow.mjs`: **ALL GREEN, 40 passed, 0
failed**, with a 6-mutant matrix, 6/6 killed.

---

## 4. What this live_check does NOT establish

- **No audit-class step was run live.** `coverage.dry` enforcement is covered by
  the checker only.
- **Stage 2 is an LLM agent, not a deterministic file read** — the Workflow
  runtime gives the script no filesystem access, so this is the strongest
  in-rail check available. Main re-verified it by hand *here*; that hand-check
  is **not** automatic on future runs.
- **The URL cross-check is a substring test.** A brief listing a URL it never
  read in full would still pass. It detects *fabricated* sources, not *shallow*
  ones.
- **The floors are enforced after the agent returns**, so a short-of-floor run
  still costs its tokens. This gate catches over-claims; it does not prevent
  wasted work.
- **The workflow is not yet dispatchable by name** — `Workflow({name:'research-gate'})`
  returns "not found" until a session restart; `{scriptPath: …}` works in-session.
  The next session must verify the name resolves.
