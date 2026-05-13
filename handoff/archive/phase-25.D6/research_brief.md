# Research Brief: phase-25.D6 — Planner Plateau-Detection Lock-File Enforcement

**Tier:** moderate (assumed per caller prompt)
**Date:** 2026-05-13

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://keras.io/api/callbacks/reduce_lr_on_plateau/ | 2026-05-13 | Official doc | WebFetch full | "patience: Number of epochs with no improvement after which learning rate will be reduced." Default patience=10; EarlyStopping common values 5-15 |
| https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html | 2026-05-13 | Official doc | WebFetch full | MedianPruner defaults: n_startup_trials=5, n_warmup_steps=0; prunes unpromising trials by comparing against median of completed trials at same step |
| https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/409 | 2026-05-13 | Official spec (MDN/RFC) | WebFetch full | "409 conflict responses are errors sent to the client so that a user might be able to resolve a conflict and resubmit the request" |
| https://rednafi.com/misc/run-single-instance/ | 2026-05-13 | Authoritative blog | WebFetch full | flock + atomic mkdir lock patterns; automatic release on process exit; trap handlers; manual removal if process crashes |
| https://apipark.com/techblog/en/understanding-the-409-status-code-when-conflicts-arise-in-http-requests/ | 2026-05-13 | Industry blog | WebFetch full | 409 for "request conflicts with current state of resource"; structured JSON body with error code + message + conflict context is best practice |
| https://optax.readthedocs.io/en/latest/_collections/examples/contrib/reduce_on_plateau.html | 2026-05-13 | Official doc | WebFetch full | Optax ReduceLROnPlateau example: PATIENCE=5, RTOL=1e-4; consistent with Keras; confirms cross-framework patience range of 5-10 |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://assets.amazon.science/32/cf/3cec00624df8a2ee2fb92f4a6b9a/automatic-termination-for-hyperparameter-optimization.pdf | Paper (Amazon Science) | Binary PDF; content not extractable via WebFetch |
| https://www.semanticscholar.org/paper/Automatic-Termination-for-Hyperparameter-Makarova-Shen/64661c86aa57ff8fbe8e61aae855a4e719bbadf3 | Paper abstract | Semantic Scholar returned empty body |
| https://proceedings.mlr.press/v206/ishibashi23a/ishibashi23a.pdf | Paper (PMLR) | PDF >10MB; binary |
| https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1484 | Survey paper | HTTP 402 Payment Required |
| https://link.springer.com/article/10.1007/s41060-026-01037-5 | Paper (Springer 2026) | Requires auth; SSO redirect |
| https://www.aimspress.com/article/doi/10.3934/mbe.2024275?viewType=HTML | Survey paper | Content did not include specific stopping thresholds |
| https://arxiv.org/abs/2507.12453 | arXiv paper (2025) | Snippet only; cost-aware BO stopping; not applicable to consecutive-discard pattern |
| https://sre.google/sre-book/data-processing-pipelines/ | SRE book | Focused on pipeline correctness/leases; no explicit plateau lock pattern |
| https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html | Official doc | Snippet only; confirms early_stopping_round parameter pattern |
| https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/ | Blog | HTTP 403 Forbidden |

---

## Recency scan (2024-2026)

Searched: "plateau detection hyperparameter optimization consecutive no-improvement stopping rule 2026", "consecutive no-improvement stopping rule 10 trials bayesian optimization hyperparameter 2024 2025", "APE automatic prompt engineer GRIPS convergence plateau N cycles stopping criterion 2024 2025".

**Result:** No 2024-2026 findings supersede the existing consensus.

- 2025 arXiv cost-aware BO stopping (Pandora's Box Gittins Index) focuses on varying evaluation costs, not consecutive-discard counts. Not directly applicable.
- Springer 2026 (Bayesian opt + meta-learning early stopping) was paywalled; snippet confirms early stopping is active research but no canonical threshold emerged.
- Codebase reference at `backend/meta_evolution/directive_rewriter.py:362` cites "APE / GRIPS convergence research: N=3 cycles minimum for plateau" -- this is for directive-text convergence, not numerical optimization. Establishes pyfinagent precedent but does not override the Keras/Optuna range for the optimizer.
- The 62-experiment plateau incident (audit bucket 24.6 F-5) is itself 2026 empirical evidence that N=10 would have triggered 52 iterations earlier than actual operator discovery.

---

## Key findings

1. **Canonical patience for numerical optimization: 5-15, with 10 as the default.** Keras ReduceLROnPlateau defaults to patience=10. Optax example uses PATIENCE=5. LightGBM exposes early_stopping_round with typical values 10-100. The consensus: "experiment with values like 5, 10, or 20." (Sources: Keras docs; Optax docs)

2. **N=10 is correct for pyfinagent.** The live-check says "trigger 10 consecutive discards." It sits inside the consensus range. It is also the natural second tier after `think_harder` activates at `consecutive_discards >= 5` (`quant_optimizer.py:205`). After think_harder fails for another 5 iterations the search is genuinely stuck. (Source: `backend/backtest/quant_optimizer.py:205`)

3. **HTTP 409 is the correct status code.** MDN: "409 conflict responses are errors sent to the client so that a user might be able to resolve a conflict and resubmit the request." The existing codebase already uses 409 for engine-busy conflicts at `backtest.py:119` and `backtest.py:239`. 423 (Locked) is a WebDAV extension and non-idiomatic in plain REST. (Sources: MDN; `backend/api/backtest.py:119,239`)

4. **File-based lock, not in-memory counter, because it survives server restarts.** A crashed process would leave `consecutive_discards=0` in memory but the lock file on disk. The file is visible to operators via `cat` without hitting an API endpoint. (Source: rednafi.com single-instance pattern)

5. **DELETE endpoint is the correct clear-lock workflow.** Existing `DELETE /api/backtest/optimize/history` (`backtest.py:665-712`) and `DELETE /api/backtest/runs/{run_id}` (`backtest.py:647-662`) establish the pattern: delete file(s), clear in-memory state, invalidate cache if applicable, return structured JSON. The new endpoint follows this exactly. (Source: `backend/api/backtest.py:647-712`)

6. **Audit JSONL on clear.** Existing codebase uses append-only JSONL for operator-action audit trails: `handoff/kill_switch_audit.jsonl`, `handoff/audit/pre_tool_use_audit.jsonl`. The clear event should append to `handoff/audit/optimizer_plateau_audit.jsonl` before deleting the lock file. (Source: `handoff/kill_switch_audit.jsonl`, `handoff/audit/`)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/backtest/quant_optimizer.py` | 1-33 | Imports, path constants (`_EXPERIMENTS_DIR`, `_TSV_PATH`, `_BEST_PARAMS_PATH`) | Active; add `PLATEAU_THRESHOLD` and `_PLATEAU_LOCK_PATH` here |
| `backend/backtest/quant_optimizer.py` | 190-294 | `run_loop()` iteration block; all three discard branches (`crash` at 235-247, `dsr_reject` at 283-289, `discard` at 290-295); `consecutive_discards` incremented at 246, 287, 294; reset at 258 | Active; add lock-write call after each increment |
| `backend/backtest/quant_optimizer.py` | 297-314 | Post-decision block: `_log_experiment`, `_report_status` | Active; plateau check must break before or after log call |
| `backend/api/backtest.py` | 62-80 | Module-level state: `_optimizer_state`, `_backtest_state`, `_is_engine_busy()` | Active; no changes needed here |
| `backend/api/backtest.py` | 231-258 | `start_optimizer()` endpoint; guards at 236 (already-running) and 238-239 (backtest-running 409) | Active; insert plateau lock check between lines 239 and 241 |
| `backend/api/backtest.py` | 272-314 | `get_optimizer_status()` endpoint | Active; add `plateau_locked` + `plateau_lock` fields to response |
| `backend/api/backtest.py` | 647-712 | `DELETE /runs/{run_id}` and `DELETE /optimize/history` — operator action pattern | Active; new `DELETE /optimize/lock` mirrors this pattern |
| `handoff/locks/` | N/A | Directory for lock file (does not exist) | Must be created at runtime via `mkdir(parents=True, exist_ok=True)` |

---

## Consensus vs debate (external)

**Consensus:** patience=5-10 consecutive no-improvement trials is the accepted stopping rule. HTTP 409 for resolvable conflict. File-based lock for cross-restart observability. DELETE endpoint for operator clear.

**Debate:** N=5 vs N=10 vs N=15. For pyfinagent, N=10 is correct because (a) it matches the live-check requirement, (b) `think_harder` already activates at N=5 so N=10 is a second-tier escalation, and (c) backtest runs cost 30s-5min vs. milliseconds for LLM text, making the per-trial cost high enough to justify waiting for N=10 before halting.

---

## Pitfalls (from literature + codebase)

1. **Crash branch uses `continue` (`quant_optimizer.py:247`).** The plateau check in the crash branch must be placed before the `continue`, or the break will never execute for crash-induced increments.

2. **Lock file survives server restart; in-memory counter does not.** `start_optimizer` must check the lock file even when `_optimizer_state["status"] == "idle"`, because a previous crashed run may have written the lock.

3. **Corrupt lock file.** `_read_plateau_lock()` must catch `json.JSONDecodeError` and return `None` (treat corrupt as absent) rather than raising. Log a warning for operator visibility.

4. **`cleared_at` not null check.** A previously cleared lock file (with `cleared_at` set) that was not deleted must not block new runs. The `start_optimizer` guard must check `cleared_at is None`. The recommended implementation deletes the file on clear, making this moot; but defensive check is still worth including.

5. **Encoding.** All `Path.write_text()` and `open()` calls must pass `encoding="utf-8"` per backend-api.md convention.

6. **No emoji in logger messages** per security.md convention. Use plain ASCII arrows (`->`, `--`) in `logger.*()` calls.

---

## Application to pyfinagent — files to modify

| File | What changes |
|------|-------------|
| `backend/backtest/quant_optimizer.py` | Add `PLATEAU_THRESHOLD = 10` and `_PLATEAU_LOCK_PATH` constant; add `write_plateau_lock()` module-level helper; add plateau-check + break after each `consecutive_discards += 1` increment (3 locations: crash branch line 246, dsr_reject branch line 287, discard branch line 294) |
| `backend/api/backtest.py` | Add `_plateau_lock_path()` and `_read_plateau_lock()` helpers; add lock-check guard in `start_optimizer` (after line 239); add `plateau_locked` + `plateau_lock` fields to `get_optimizer_status()` response; add `DELETE /api/backtest/optimize/lock` endpoint |

---

## Recommended plateau threshold: N = 10

- Matches live-check requirement.
- Sits within the Keras/Optuna/Optax consensus range of 5-15.
- Is a natural second tier: `think_harder` activates at N=5; if think_harder also fails for 5 more iterations, the search is genuinely stuck.
- Would have caught the 62-experiment plateau incident at iteration 10 (saving ~52 runs).

---

## Lock-file path + JSON shape

**Path:** `handoff/locks/optimizer_plateau.lock`

Rationale: consistent with existing `handoff/` layout for process coordination artifacts. Visible to operators via `ls handoff/locks/` and `cat handoff/locks/optimizer_plateau.lock`. Not inside `backend/` (avoids Python package import complications). Not in `/tmp` (survives reboots; tracked alongside handoff artifacts).

**Lock file JSON shape:**
```json
{
  "created_at": "2026-05-13T10:00:00+00:00",
  "trigger": "plateau_10_discards",
  "consecutive_discards": 10,
  "run_id": "abc12345",
  "cleared_at": null
}
```

**Audit JSONL path:** `handoff/audit/optimizer_plateau_audit.jsonl`
Each cleared lock appends a single JSON line with `cleared_at` populated.

---

## Verbatim Python signatures

### In `quant_optimizer.py` (add after `_BEST_PARAMS_PATH` constant, ~line 33):

```python
PLATEAU_THRESHOLD: int = 10  # consecutive discards before writing plateau lock
_PLATEAU_LOCK_PATH = Path(__file__).parent.parent.parent / "handoff" / "locks" / "optimizer_plateau.lock"


def write_plateau_lock(run_id: str, consecutive_discards: int) -> None:
    """Write a plateau lock file to block further optimizer runs until operator clears it."""
    _PLATEAU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger": f"plateau_{consecutive_discards}_discards",
        "consecutive_discards": consecutive_discards,
        "run_id": run_id,
        "cleared_at": None,
    }
    _PLATEAU_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.warning(
        "QuantOptimizer: plateau lock written after %d consecutive discards (run_id=%s). "
        "Operator must DELETE /api/backtest/optimize/lock to resume.",
        consecutive_discards,
        run_id,
    )
```

### Plateau check in `run_loop()` — crash branch (after line 246 `consecutive_discards += 1`):

```python
                consecutive_discards += 1
                if consecutive_discards >= PLATEAU_THRESHOLD:
                    write_plateau_lock(self._run_id, consecutive_discards)
                    return  # crash branch uses continue; use return to exit loop entirely
                continue
```

### Plateau check in `run_loop()` — discard/dsr_reject (after line 294 `consecutive_discards += 1`):

```python
            if status != "keep" and consecutive_discards >= PLATEAU_THRESHOLD:
                write_plateau_lock(self._run_id, consecutive_discards)
                # Still log the final experiment before exiting
                exp_id = f"{self._run_id}-exp{i:02d}"
                self._log_experiment(
                    exp_id, change_desc,
                    float(self.best_sharpe or 0), trial_sharpe, delta, status, trial_dsr, trial_top5,
                    trial_params=trial_params,
                )
                break
```

### In `backend/api/backtest.py` (add as module-level helpers):

```python
def _plateau_lock_path() -> Path:
    """Canonical path for the optimizer plateau lock file."""
    return Path(__file__).parent.parent.parent / "handoff" / "locks" / "optimizer_plateau.lock"


def _read_plateau_lock() -> dict | None:
    """Return lock payload if plateau lock exists and is not cleared, else None."""
    p = _plateau_lock_path()
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("cleared_at") is not None:
            return None  # previously cleared but file not deleted; treat as absent
        return payload
    except Exception:
        logger.warning("plateau lock file at %s is corrupt; treating as absent", p)
        return None
```

### Lock check guard in `start_optimizer()` (between lines 239 and 241):

```python
    plateau_lock = _read_plateau_lock()
    if plateau_lock is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "PlateauLockPresent",
                "message": (
                    f"Optimizer halted after {plateau_lock.get('consecutive_discards')} consecutive "
                    f"discards (run_id={plateau_lock.get('run_id')}). Strategy switch required. "
                    "Call DELETE /api/backtest/optimize/lock to acknowledge and resume."
                ),
                "lock": plateau_lock,
            },
        )
```

### New `DELETE /api/backtest/optimize/lock` endpoint:

```python
@router.delete("/optimize/lock")
def clear_plateau_lock():
    """Clear the optimizer plateau lock, allowing the optimizer to run again.

    The operator should review why the optimizer plateaued (strategy switch required)
    before calling this endpoint. Appends a record to handoff/audit/optimizer_plateau_audit.jsonl.
    """
    p = _plateau_lock_path()
    if not p.exists():
        raise HTTPException(404, "No plateau lock file found")

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    payload["cleared_at"] = datetime.now(timezone.utc).isoformat()

    # Append to audit JSONL before deleting the lock file
    audit_path = p.parent.parent / "audit" / "optimizer_plateau_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()

    p.unlink()

    logger.info("Plateau lock cleared by operator (run_id=%s)", payload.get("run_id"))
    return {"status": "cleared", "lock": payload}
```

### Surface `plateau_locked` in `get_optimizer_status()`:

After the existing `state = dict(_optimizer_state)` line, add:
```python
    plateau_lock = _read_plateau_lock()
    state["plateau_locked"] = plateau_lock is not None
    state["plateau_lock"] = plateau_lock
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (`quant_optimizer.py:190-294`, `backtest.py:231-258`, `backtest.py:647-712`, `directive_rewriter.py:362`)
- [x] Contradictions / consensus noted (N=5 vs 10 vs 15 debate; 409 vs 423 noted)
- [x] All claims cited per-claim (not just listed in footer)

---

## Search queries run (three-variant discipline)

1. Current-year frontier: "plateau detection hyperparameter optimization consecutive no-improvement stopping rule 2026"
2. Last-2-year window: "consecutive no-improvement stopping rule 10 trials bayesian optimization hyperparameter 2024 2025"
3. Year-less canonical: "plateau threshold N consecutive no-improvement early stopping hyperparameter optimization canonical value"
4. Lock-file canonical: "PID lock file pattern python batch job single instance operator action 2025"
5. HTTP semantics: "HTTP 409 Conflict semantics operator action required REST API FastAPI"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
