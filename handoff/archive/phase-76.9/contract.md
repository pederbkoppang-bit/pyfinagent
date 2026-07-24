# Contract — Step 76.9 (P1, operator bug B1: both nightly launchd jobs failing)

Date: 2026-07-24 | Cycle: 154 | Executor tier: DELEGATED sonnet-4.6/high (step tag) | Main: Fable 5 (orchestration + review + mutation matrix) | Gates: opus/max via Workflow

## Research-gate summary (gate PASSED — Workflow wf_f96642e7-247)

`handoff/current/research_brief_76.9.md` — tier=moderate, 7 external sources read in full
per the brief's envelope (the researcher's resumed leg personally fetched an 8th; both legs
reconciled, no clobber), 12 snippet-only, 19 URLs, recency scan performed, 14 internal
files inspected, gate_passed=true. Load-bearing corrections to the step text (re-derived):

- RETRIEVER is set at `run_memo.py:211` (`"arxiv,semantic_scholar,duckduckgo"`), not ~:189.
- Crash path: gpt-researcher 0.14.8 uses retrievers[0] ONLY for planning
  (`skills/researcher.py:62`, upstream issue #1282); the planning call is UNGUARDED while
  the sub-query fan-out is wrapped (try `:465` / except `:569` returns "" on error — exact
  wrap lines approximate, substance verified). arxiv's 429 raises in `ArxivSearch.search`
  (no try/except) → `run_memo.py:114` broad except → ERROR memo → `return 1` →
  `run_nightly.sh:47/:72` rc=1.
- `arxiv==3.0.0` already retries 3×/3.0s and ignores Retry-After (`__init__.py:600`);
  arXiv has 429'd polite 3s clients server-side since ~2026-02-25 (recency finding) →
  backoff-only is UNRELIABLE. Fix must remove arxiv from the fatal planning slot AND
  tolerate network-class failures.
- ABLATION: fix ALREADY CODED (phase-75.11 `scripts/ops/run_ablation.sh`, commit 07182b94,
  verbatim phase-62.6 grep-sanitize at `:21-37`) and ALREADY BOOTSTRAPPED
  (OPS-ROTATE-BOOTSTRAP leg 3, harness_log ops addendum 2026-07-24 ~07:15 UTC,
  operator-attended; live plist verified by Main first-hand, mtime 08:52, runs=0 since).
  Sufficiency PROVEN: backend/.env L81 is the non-KEY orphan `  ON"` (wrapped tail of
  L80's comment), which `grep -E '^[A-Za-z_][A-Za-z0-9_]*='` drops. Remaining work =
  PROVE (fixture + live), not code.
- Step 39.1's verification is unsatisfiable (May-2026 date grep; memos never named
  `-PASS`) → supersede with 76.9, carry its root-cause-documentation intent forward.

## Hypothesis

(1) Moving arxiv out of the planning slot (retrievers[0]) removes the only untolerated
429 path, and a network-class-tolerant exit seam in run_memo.py (WARN memo + rc 0)
converts residual external-API weather into non-paging WARNs while REAL faults still
page through the 75.11 seam (rc 1). (2) The already-bootstrapped run_ablation.sh
sanitize provably survives the malformed .env; a fixture proof + live sourcing-seam
proof close the ~37-night crash class.

## Immutable success criteria (verbatim from .claude/masterplan.json 76.9)

1. "AUTORESEARCH: a run of run_memo.py with the arxiv retriever forced to raise HTTP 429 (stubbed/mocked) completes via the other retrievers and exits 0 with a WARN -- NOT rc=1 (MUTATION: revert the fall-through -> the run exits non-zero); OR, if backoff-only, a real nightly run writes a non-ERROR memo. Recorded verbatim."
2. "ABLATION: the ablation job sources .env via the same sanitize path as run_nightly.sh so a malformed .env line no longer aborts it -- proven by running the ablation entry-point against a fixture .env containing an unbalanced-quote line and observing it does NOT die with 'unexpected EOF' (MUTATION: bypass the sanitize -> the failure returns)"
3. "The exact backend/.env malformed line (~:80-81) is reported verbatim in experiment_results.md for operator repair; NO .env file is edited by this step"
4. "bash -n on the changed shell scripts passes; py_compile on changed Python passes"

Immutable verification command (verbatim):
`bash -n scripts/autoresearch/run_nightly.sh && .venv/bin/python -c "import ast; ast.parse(open('scripts/autoresearch/run_memo.py').read())"`

## Plan

1. **run_memo.py** (executor): (a) RETRIEVER at :211 → `"semantic_scholar,arxiv,duckduckgo"`
   (deprioritize: arxiv leaves the crash-prone PLANNING slot but keeps sub-query coverage,
   where failures are already inside the tolerant fan-out wrap). (b) Network-tolerance
   seam: classify the caught exception at :114 — arxiv HTTPError / HTTP 429/5xx /
   connection-class → write **WARN** memo + `return 0`; everything else keeps ERROR memo +
   `return 1` (75.11 paging seam preserved). Classification by type-name + module prefix +
   message match (`type(e).__name__ == "HTTPError"`, module startswith `arxiv`,
   "429"/"503"/connection tokens) — NO new dependency, NO catch-all widening. WARN memo
   filename must NOT match the `-ERROR-` pattern (downstream counters), keep ASCII-only
   log lines (security.md, brief pitfall P-list).
2. **Tests** (executor): new `backend/tests/test_phase_76_9_launchd_fixes.py`:
   - t-429: run_memo topic-run stubbed to raise `arxiv.HTTPError`(429-shaped) → WARN memo
     written, run returns 0 (criterion 1; mutation-killable: revert fall-through → red).
   - t-real-fault: stub raises ValueError → ERROR memo, returns 1 (seam preserved).
   - t-retriever-order: RETRIEVER string has semantic_scholar first and arxiv NOT at
     position 0 (mutation-killable: revert order → red).
   - t-ablation-fixture: run the REAL `scripts/ops/run_ablation.sh` with `SRE_OPS_REPO`
     pointed at a tmp fixture repo (fixture backend/.env WITH an unbalanced-quote non-KEY
     line mirroring L80/L81's shape, stub .venv/bin/activate, stub
     scripts/ablation/run_ablation.py) → exits 0, no "unexpected EOF", START/END rows in
     fixture log (criterion 2). The test FIRST asserts the fixture .env reproduces the
     EOF failure under raw `. .env` in a bash subshell — the fixture cannot go vacuous
     (feedback_mutation_test_guards_and_fixtures).
3. **Docs/state** (executor): `handoff/autoresearch/root_cause.md` (39.1's carried-forward
   intent: both root causes, dates, citations). Report backend/.env L80-81 VERBATIM in
   experiment_results.md (comment text only, no secret values; NO .env edit —
   operator-gated).
4. **Main after executor** (sequenced AFTER executor completes —
   feedback_executor_sees_mutation_transients): full-diff review; mutation matrix:
   M1 revert the WARN fall-through → t-429 red; M2 revert RETRIEVER order → t-retriever-order
   red; M3 bypass sanitize in run_ablation.sh (raw `. .env`) → t-ablation-fixture red;
   M4 widen classifier to catch-all (network-tolerance swallows real faults) →
   t-real-fault red; M5 STUB/fixture mutation: quote-balance the fixture .env → the
   in-test reproduce assert red.
5. **Main live checks**: live sourcing-seam proof against the REAL backend/.env (sanitized
   stream sources clean in a throwaway shell; raw source still dies — BEFORE/AFTER pair);
   `launchctl kickstart` decision recorded in live_check (ablation full-run feasibility
   gated on what `run_ablation.py --next-untested` touches — historical_macro freeze check
   FIRST; autoresearch kickstart = one memo run on the operator-sanctioned nightly path;
   tonight's 02:00/03:00 crons are the natural full-live evidence either way).
6. **39.1**: status → `superseded` (note pointing here). NOT in the same edit as any
   status:done flip (feedback_masterplan_status_flip_order).
7. Q/A via qa-verdict Workflow (opus/max) on the changed evidence; log Cycle 154 (append
   BEFORE flip); flip 76.9 done (auto-push; manual fallback if the hook stalls).

## Boundaries

- NO edit to backend/.env (operator-gated; report-only).
- NO new backend/requirements.txt deps.
- `run_nightly.sh` expected UNCHANGED (immutable command covers it; its sanitize is already
  correct). Any executor change to it must be disclosed + justified.
- `scripts/ops/run_ablation.sh` expected UNCHANGED (75.11 shipped it; 76.9 proves it).
- No launchctl bootout/bootstrap from this session (62.0 rail); kickstart only per plan §5.
- Existing tests `test_phase_75_sre_ops.py` / `test_phase_39_1_autoresearch_env.py` /
  `test_phase_75_deps.py` must stay green (regression sweep).

## References

- Research brief: handoff/current/research_brief_76.9.md (envelope + per-claim citations:
  arXiv API ToU + user manual, arXiv API Google-group 429 thread, arxiv.py docs +
  installed-source verification, gpt-researcher search-engines docs + issue #1282,
  python-dotenv #487, judy2k dotenv gist)
- Harness-log ops addendum 2026-07-24 (~07:15 UTC) — plist bootstrap provenance
- Baseline BEFORE evidence: scratchpad baseline_76.9.md (launchctl exit 1, verbatim logs,
  .env structural map)
