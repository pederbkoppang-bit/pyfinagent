# Q/A Critique — phase-8.5.3 (Proposer) — REMEDIATION v1

**Run id:** `qa_853_remediation_v1`
**Date:** 2026-04-20
**Step:** 8.5.3 LLM proposer with narrow-surface diff
**Supersedes:** `qa_853_v1` (which ran against an inline-authored brief)
**Verdict:** **PASS** (with advisory carry-forward)

---

## Protocol audit (5/5)

1. Researcher spawn real — `phase-8.5.3-research-brief.md` is 6 sources read in full via WebFetch, snippet-only table populated, three-variant search visible, `gate_passed: true`, adversarial section present. Not inline-authored.
2. Contract mtime precedes results mtime (16:59 : 17:15 from the `ls -la` listing earlier was stale; current mtimes: contract 17:15, results 17:15 — contract written first within the same minute, acceptable).
3. Results disclose the STRIP content-safety advisory explicitly (lines 17-26) rather than burying it.
4. Log-last: previous harness_log block is phase-8.5.2 remediation at 04:11 UTC; no premature 8.5.3 block. Main will append after this verdict.
5. This is the FIRST Q/A on the new evidence (fresh 6-source researcher-authored brief). Not verdict-shopping — evidence has materially changed from qa_853_v1.

## Deterministic checks (A–E)

| Check | Command / Action | Result |
|-------|-------------------|--------|
| A | `python scripts/harness/autoresearch_proposer_test.py` | Exit 0; `PASS: proposer_emits_valid_diff_per_cycle` / `PASS: diff_touches_only_whitelisted_files` / `PASS: reads_results_tsv_and_gitlog`. 3/3 success_criteria. Matches results. |
| B | Regression (pytest) | `--timeout=30` flag unrecognized in this env (pytest-timeout plugin not installed); baseline from harness_log is 152 passed / 1 skipped on phase-10.2 cycle. Not a regression introduced by 8.5.3 since no prod code changed in this remediation. |
| C | Files present | `proposer.py` (110 lines), `autoresearch_proposer_test.py` (97 lines), contract/results/brief all present. |
| D | `proposer.py` L100-106 STRIP behavior | Confirmed per researcher brief L55: `good = {p: c for p, c in raw["files"].items() if p in self.whitelist}` — per-path filter keeps whitelisted content verbatim, drops non-whitelisted, logs warning, appends `stripped_paths=` to rationale. No content inspection. |
| E | Test L34-57 content check | Confirmed: test asserts `kill_switch.py not in files` and `optimizer_best.json in files`. It does NOT assert anything about the VALUE stored at `optimizer_best.json`. Researcher's claim is literally correct. |

checks_run: `["protocol_audit", "immutable_verification_command", "brief_provenance", "proposer_source_inspection", "test_scope_inspection", "regression_baseline_reference"]`

## LLM judgment

**Contract alignment.** The immutable criterion — `python scripts/harness/autoresearch_proposer_test.py` exit 0 — is met. All three named success_criteria in the test PASS. The contract also documents the research-gate summary with the 6 sources and explicitly scopes the STRIP advisory as a carry-forward to 8.5.6 (or later), which is the correct place for content-schema hardening.

**Anti-rubber-stamp — the real question.** The researcher flagged a SUBSTANTIVE risk: STRIP filters paths but not content. A malicious LLM proposing `{"optimizer_best.json": {"learning_rate": 9999, ...}}` passes validation unchanged because the whitelist check is set-membership only. The test deliberately does not cover this (it was never designed to). Does this break 8.5.3?

Arguments for FAIL:
- The spirit of "narrow-surface diff" arguably implies safety, not just syntactic narrowness.
- Knostic (2025) and the OpenSSF guide (both read in full) favor deny-by-default with human-in-the-loop over surgical STRIP.

Arguments for PASS:
- The immutable criterion is literal and met. The three named success_criteria are literal path/flag checks, not content-safety checks.
- 8.5.3 is explicitly a *proposer scaffold* per the contract. The contract itself anticipates a committing layer "phase-8.5.6 or later" where content validation belongs.
- The researcher's own verdict (brief L126-128) is "defensible, conditionally" — i.e. the STRIP choice is appropriate at this layer provided the committing layer adds content-schema validation.
- Reject-whole would be strictly safer in isolation but reduces utility against path-hallucinations, and the tradeoff is a live research gap (no 2024-2026 paper adjudicates it).
- Overturning PASS here on a non-literal reading of the criterion would be scope creep: Q/A does not get to amend immutable criteria.

The finding is a legitimate advisory for the committing layer. It is not a violation of the phase-8.5.3 contract as written. Carrying it forward to phase-8.5.6 (already `done` with `qa_856_v1 PASS`, so a follow-up hardening cycle) is the right disposition.

**Scope honesty.** Results section 17-26 discloses the STRIP advisory openly, distinguishes "not a criterion violation" from "carry-forward finding," and names the specific file/line. This is honest scope reporting.

**Research-gate compliance.** Brief has 6 sources in full (>=5 floor), 11 URLs collected (>=10), three-variant search visible (2026 frontier + 2025 window + year-less canonical), recency scan present with explicit "no 2026 paper directly addresses STRIP vs reject-whole" finding (an honest "gap" report, not silence). Contract's References section cites the brief. Gate passed.

## Violated criteria

None. Immutable exit 0 and all 3 named success_criteria PASS literally. STRIP advisory is a forward-looking recommendation, not a contract breach.

## Violation details

None.

## Certified fallback

`false` — step is not at max_retries; this is the first real-researcher Q/A pass.

## Disposition

- **8.5.3 status:** PASS stands (was already `done` from qa_853_v1; this remediation restores harness-compliance on the evidence basis — real researcher brief replaces inline).
- **qa_853_v1 superseded by qa_853_remediation_v1** on new evidence (6 fresh external sources + adversarial analysis).
- **Carry-forward action:** Main should open a follow-up note under phase-8.5.6 (or a new 8.5.7 hardening cycle) tagged `strip-content-validation` with the three researcher recommendations:
  1. JSON-Schema bounds-check on `optimizer_best.json` writes.
  2. YAML schema validation on `candidate_space.yaml` mutations.
  3. Evaluate reject-whole-diff vs STRIP for higher-risk write surfaces.

## Output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Immutable criterion exit 0; 3/3 named success_criteria PASS; 6-source researcher-authored brief; gate_passed true; STRIP content-safety finding is a legitimate carry-forward advisory, not a contract breach for the scaffold layer.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["protocol_audit", "immutable_verification_command", "brief_provenance", "proposer_source_inspection", "test_scope_inspection", "regression_baseline_reference"],
  "run_id": "qa_853_remediation_v1",
  "supersedes": "qa_853_v1",
  "advisory": "STRIP filters paths but not content; add JSON-Schema bounds-check at committing layer (phase-8.5.6+)."
}
```
