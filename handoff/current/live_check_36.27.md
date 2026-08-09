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
    "brief_exists": true, "brief_non_empty": true, "char_count": 36790,
    "urls_checked": 8, "urls_present": 8, "urls_missing": []
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
    "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0,
                 "K_required": 2, "new_findings_last_round": 0, "dry": false},
    "brief_path": "handoff/current/research_brief_86.1.md",
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
