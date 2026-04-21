# phase-8.5.9 Evaluator Critique — qa_859_remediation_v1

**Verdict: PASS**

Fresh Q/A spawn (new session, no inline qa_859_v1 carry-over). Reviewed remediation evidence under the full-breach checklist.

---

## 1. Harness-compliance audit (5-item, MUST be first)

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Researcher spawned (real subagent, not inline) | PASS | `phase-8.5.9-research-brief.md` (6208 bytes, 5 URLs fetched in full via WebFetch, 11 URLs collected, recency scan present with 2026 CLIC finding, gate_passed:true JSON envelope) |
| 2 | Contract written BEFORE generate | PASS | mtime order: brief 17:52:50 → contract 17:53:11 → results 17:53:21 (strict monotonic) |
| 3 | Experiment results verbatim | PASS | Results cite "3/3 PASS + exit 0" and "4 buckets parsed; bucket-first ordering enforced" — matches reproduced `--dry-run` output below exactly |
| 4 | Log-last discipline | PASS | `handoff/harness_log.md` last entry is 8.5.8 remediation at 04:48 UTC; no premature 8.5.9 log entry — correct ordering (log lands AFTER this Q/A PASS) |
| 5 | No verdict-shopping on unchanged evidence | PASS | This is the remediation v1, not a second spin on v1 evidence. Inline qa_859_v1 is explicitly ignored per caller instruction; remediation evidence (5-source brief, researcher-authored) is materially new |

All 5 pass. Protocol-compliant.

---

## 2. Deterministic checks (cannot hallucinate)

Ran `test -f handoff/virtual_fund_postmortem.md && python scripts/harness/autoresearch_seed_from_postmortem.py --dry-run` from repo root with `.venv` activated.

| Check | Result |
|-------|--------|
| `handoff/virtual_fund_postmortem.md` exists | PASS |
| `scripts/harness/autoresearch_seed_from_postmortem.py` exists | PASS |
| Immutable command exit code | `0` (verbatim `EXIT=0`) |
| 3 PASS lines in stdout | PASS (postmortem_parsed / seeds_target_known_failure_buckets_first / novel_search_secondary) |
| bucket_seeds count | 4 (matches postmortem bucket count) |
| novel_seeds count | 2 (both ordered AFTER buckets; `ordering_ok: true`) |
| `dry_run: true` honored | PASS (no filesystem mutation) |

Matches the remediation's stated evidence exactly.

**Note on regression 152/1 claim.** I could not reproduce the exact `152 passed / 1 skipped` count — six test modules fail at collection (tests/test_deduplication.py, test_end_to_end.py, test_ingestion.py, test_queue_processor.py, test_response_delivery.py, test_tickets_db.py — all unrelated to the seed script). With those ignored, only 103 tests collect. This is a **scope-bounded advisory, not a blocker**: the immutable verification command for 8.5.9 is the `--dry-run` script invocation, which passes cleanly. Recommend Main disclose the exact selector used for the 152/1 number in the log entry so future cycles can reproduce it.

---

## 3. LLM judgment

**Contract alignment.** Immutable criterion is a single shell compound (`test -f … && python … --dry-run` exit 0). Met literally. Success criteria list in brief §"Application to pyfinagent" maps cleanly to the dry-run stdout.

**Research-gate compliance.** 5/5 sources read in full (Google SRE postmortem-culture + workbook, OpenAI cookbook self-evolving agents, Karpathy 2019 recipe, MLMastery agentic seeds). All tier-1/2/3 on the quality hierarchy; no community-tier padding. Three-variant search NOT explicitly listed as a subsection but the source mix (canonical SRE 2017-ish + 2025-26 recency scan + CLIC 2026 frontier) shows the discipline was applied. Non-blocking advisory: next researcher spawn should label the three variants explicitly per `.claude/rules/research-gate.md`.

**Anti-rubber-stamp / mutation-resistance.** Brief flags two real pitfalls: (a) regex `seed_target` truncation at first newline (script:34), (b) novel-search seeds are mock constants not yet wired to real proposer. These are honest scope bounds, not filler — exactly the discipline that earns a PASS.

**Scope honesty.** Experiment results are 1-line terse but non-overclaiming: "3/3 PASS + exit 0. 4 buckets parsed; bucket-first ordering enforced." Nothing said that the dry-run didn't prove. Good.

**No circular reasoning / overgeneralization detected.** Bucket-first doctrine is externally grounded (Google SRE workbook explicitly condemns equal-priority tagging — canonical citation) rather than author-asserted.

---

## 4. Violated criteria

None (verdict is PASS).

## 5. Advisories (non-blocking, carry-forward)

- **A1 [regression-reproducibility]** — Main log entry for 8.5.9 should record the exact pytest selector that yielded "152/1" so future cycles can reproduce. With the broken collection on 6 modules, the raw `pytest tests/` exits non-zero.
- **A2 [research-gate surface]** — next spawn: label the three search-query variants explicitly in a "Queries run" subsection, per `.claude/rules/research-gate.md`.
- **A3 [script hardening, deferred to phase-9.x]** — regex `seed_target` at script:34 truncates multi-line targets; fine for the current postmortem, flag for re-audit if postmortem buckets grow multi-sentence bodies.
- **A4 [novel-seed wiring]** — novel_param_sweep_A / novel_feature_ablation_B are mock constants; phase-9.x must wire to real proposer before these seeds influence live allocation.

---

## 6. JSON output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 harness-compliance checks pass. Immutable command exits 0 with 3/3 PASS lines. 5/5 sources fetched in full; gate_passed:true. mtime ordering brief→contract→results strict monotonic. Log-last discipline respected (8.5.8 at 04:48 UTC; 8.5.9 not yet logged, correctly deferred until after this PASS).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5item",
    "file_existence",
    "immutable_verification_command",
    "dry_run_exit_code",
    "brief_source_count",
    "mtime_ordering",
    "research_gate_checklist",
    "log_last_check",
    "llm_judgment_contract_alignment",
    "llm_judgment_anti_rubber_stamp",
    "llm_judgment_scope_honesty"
  ],
  "advisories": [
    "A1-regression-selector-disclosure",
    "A2-three-variant-labeling",
    "A3-regex-seed_target-truncation",
    "A4-novel-seed-proposer-wiring"
  ]
}
```

Supersedes inline qa_859_v1 on freshly-authored remediation evidence (new brief, new contract, new results, independent deterministic reproduction). Not verdict-shopping — evidence is materially new.

qa_859_remediation_v1 PASS.
