# Experiment Results -- Step 62.6 (GENERATE)

**Step:** 62.6 -- ops hygiene batch. **Date:** 2026-06-12. **State:** complete pending
Q/A (which also rules on the 39.1 lenient closure).

## Sub-item 1: backend.log rotation -- DONE

385MB (403,648,199 B) rotated live with NO backend restart: cp -> truncate -> gzip.
Archive: handoff/logs/backend.log.20260612T104931Z.gz (18MB; gitignored dir; holds the
pre-redaction FRED key -> compressed, local-only, never deleted). Live file 338B and
STILL receiving writes post-truncate (O_APPEND research validated live). Ongoing
mechanism: size-gated block (>50MB) in healthcheck.sh (30-min watchdog cadence), new
log_rotated field in the health JSON line.

## Sub-item 2a: autoresearch -- DONE at $0

Constrained install (unconstrained pip would have silently upgraded langchain-core
1.2.30 -> 1.4.6): langchain-huggingface==1.2.1, sentence-transformers==5.5.1,
torch==2.12.0, transformers==5.11.0, langchain-core HELD ==1.2.30. SPEND GUARD: new
--preflight-only flag in run_memo.py (exit 0 after deps + embedding preflight, zero
LLM calls); run_nightly.sh pinned to it for the away window; resumption = verbatim
token AUTORESEARCH SPEND: RESUME (pending_tokens.json).

REGRESSION FOUND AND FIXED during the dry run: this morning's operator .env paste left
an unbalanced quote in a comment line -- pydantic-harmless but it KILLED
run_nightly.sh's shell-sourcing (exit 2 before any log line; last night's 02:00 run
likely died the same way). run_nightly.sh now sources a sanitized KEY=value-only
stream; backend_watchdog.sh + healthcheck.sh already used safe greps (audited).
Cosmetic .env cleanup = ENV-LINE-81 operator keystroke (62.7).

Dry invocation through the REAL nightly entrypoint (verbatim):

    [2026-06-12T12:53:08+02:00] START nightly autoresearch
    preflight-only: deps importable, embedding preflight OK, skipping GPTResearcher (zero spend)
    [2026-06-12T12:53:08+02:00] END nightly autoresearch OK
    nightly-exit=0

## Sub-item 2b: ablation -- documented, no fix needed

launchctl: last exit 0, 16 runs. 37/37 numeric features carry TSV verdicts (last
2026-05-24) -> every run since takes the all-tested branch (run_ablation.py:329-331,
exit 0). The original failing night's traceback is unrecoverable (handoff/ablation.log
truncated to 265B by housekeeping). Disposition: fix-not-needed with evidence; job
stays loaded (self-resumes via --next-untested if the feature set grows). No disable.

## Sub-item 3: 39.1 closure -- live_check_39.1.md written (lenient path)

Evidence-by-output per its success_criteria: 11 consecutive ERROR-free exit-0 nights +
today's deps-live exit-0 dry run (criterion a); root_cause.md exists + the NEW
second-failure-mode root cause documented (criterion b); the owner-gated install
executed under the operator-approved plan with the $0 guard + resumption token
(criterion c). The literal verification command is structurally unsatisfiable (only
-ERROR- files exist in its date window; success memos never carry -PASS; exit is
head's) -- Q/A rules. STRICT path (3 deps-live nights, closes 06-15 via PM sessions)
documented as the fallback if Q/A holds.

## Residual: sector-cap log test

test_phase_23_2_6_backend_log_has_skipping_buy_evidence failed post-rotation exactly as
researched -- adapted per its own original comment ("the log was rotated and the test
should adapt"): falls back to the newest gzip archive; 6/6 green.

## Verification (verbatim)

    $ test $(stat -f%z backend.log) -lt 52428800 && python -c "import langchain_huggingface; print('lh OK')"
    lh OK

## File list

scripts/away_ops/healthcheck.sh (rotation block + log_rotated field),
scripts/autoresearch/run_memo.py (--preflight-only), scripts/autoresearch/run_nightly.sh
(sanitized sourcing + preflight pin), backend/tests/test_phase_23_2_6_sector_cap_emit.py
(rotation-aware fallback), handoff/logs/backend.log.20260612T104931Z.gz (archive,
untracked), handoff/current/live_check_39.1.md (NEW), pending_tokens.json (+2 asks),
handoff artifacts.
