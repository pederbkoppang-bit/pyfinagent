"""phase-25.D6 verifier -- planner plateau-detection lock-file enforcement.

Closes phase-24.6 F-5 (62-experiment plateau bypassed planner Rule 1).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_D6.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest  # type: ignore
from fastapi import HTTPException  # type: ignore

REPO = Path(__file__).resolve().parents[1]
QUANT_OPT = REPO / "backend" / "backtest" / "quant_optimizer.py"
API = REPO / "backend" / "api" / "backtest.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    if not QUANT_OPT.exists() or not API.exists():
        print(f"FAIL: required source file missing")
        return 1

    qopt_text = QUANT_OPT.read_text(encoding="utf-8")
    api_text = API.read_text(encoding="utf-8")

    # ---- Claim 1: PLATEAU_THRESHOLD = 10 constant.
    threshold = re.search(r"PLATEAU_THRESHOLD\s*:\s*int\s*=\s*10", qopt_text)
    results.append((
        "PASS" if threshold else "FAIL",
        "plateau_threshold_constant_10",
        "PLATEAU_THRESHOLD: int = 10 must be declared in quant_optimizer.py",
    ))

    # ---- Claim 2: write_plateau_lock signature.
    sig = re.search(
        r"def\s+write_plateau_lock\s*\(\s*run_id:\s*str\s*,\s*consecutive_discards:\s*int\s*\)\s*->\s*None\s*:",
        qopt_text,
    )
    results.append((
        "PASS" if sig else "FAIL",
        "lock_file_strategy_switch_required_written_on_plateau",
        "write_plateau_lock(run_id: str, consecutive_discards: int) -> None must be declared",
    ))

    # ---- Claim 3: api/backtest.py has _plateau_lock_path + _read_plateau_lock.
    path_helper = re.search(r"def\s+_plateau_lock_path\s*\(\s*\)\s*->\s*Path\s*:", api_text)
    read_helper = re.search(
        r"def\s+_read_plateau_lock\s*\(\s*\)\s*->\s*dict\s*\|\s*None\s*:",
        api_text,
    )
    results.append((
        "PASS" if path_helper and read_helper else "FAIL",
        "api_backtest_has_plateau_helpers",
        "backend/api/backtest.py must declare _plateau_lock_path and _read_plateau_lock helpers",
    ))

    # ---- Claim 4: start_optimizer guards on _read_plateau_lock with 409.
    guard = re.search(
        r"plateau_lock\s*=\s*_read_plateau_lock\(\)",
        api_text,
    )
    error_key = '"error": "PlateauLockPresent"' in api_text or "'error': 'PlateauLockPresent'" in api_text
    raises_409 = re.search(r"status_code=409", api_text)
    results.append((
        "PASS" if guard and error_key and raises_409 else "FAIL",
        "optimizer_run_endpoint_returns_409_when_lock_present",
        "start_optimizer must guard via _read_plateau_lock and raise 409 with PlateauLockPresent error key",
    ))

    # ---- Claim 5: DELETE /optimize/lock route registered.
    route = re.search(
        r'@router\.delete\(["\']/optimize/lock["\']\)',
        api_text,
    )
    clear_fn = re.search(r"def\s+clear_plateau_lock\s*\(\s*\)\s*:", api_text)
    results.append((
        "PASS" if route and clear_fn else "FAIL",
        "operator_action_clears_lock",
        "DELETE /optimize/lock route + clear_plateau_lock() function must exist",
    ))

    # ---- Behavioral setup.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.backtest.quant_optimizer", None)
    from backend.backtest import quant_optimizer as qopt  # type: ignore

    # ---- Claim 6: BEHAVIORAL write_plateau_lock writes correct JSON.
    write_ok = False
    write_err = ""
    try:
        td = Path(tempfile.mkdtemp(prefix="phase25d6_"))
        lock_path = td / "optimizer_plateau.lock"
        with patch.object(qopt, "_PLATEAU_LOCK_PATH", lock_path):
            qopt.write_plateau_lock("run_xyz123", 12)
        if not lock_path.exists():
            write_err = "lock file not created"
        else:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            if payload.get("consecutive_discards") != 12:
                write_err = f"consecutive_discards={payload.get('consecutive_discards')}, expected 12"
            elif payload.get("run_id") != "run_xyz123":
                write_err = f"run_id={payload.get('run_id')!r}"
            elif payload.get("trigger") != "plateau_12_discards":
                write_err = f"trigger={payload.get('trigger')!r}"
            elif payload.get("cleared_at") is not None:
                write_err = f"cleared_at must be null initially, got {payload.get('cleared_at')!r}"
            elif not payload.get("created_at"):
                write_err = "created_at missing"
            else:
                write_ok = True
    except Exception as e:
        write_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if write_ok else "FAIL",
        "behavioral_write_plateau_lock_emits_correct_json",
        f"write_plateau_lock must emit JSON with consecutive_discards/run_id/trigger/cleared_at/created_at ({write_err})",
    ))

    # ---- Claim 7: BEHAVIORAL 409 on lock present.
    sys.modules.pop("backend.api.backtest", None)
    from backend.api import backtest as api_mod  # type: ignore

    lock_409_ok = False
    lock_409_err = ""
    try:
        td2 = Path(tempfile.mkdtemp(prefix="phase25d6_409_"))
        lock_path2 = td2 / "optimizer_plateau.lock"
        lock_payload = {
            "created_at": "2026-05-13T00:00:00+00:00",
            "trigger": "plateau_10_discards",
            "consecutive_discards": 10,
            "run_id": "run_abc",
            "cleared_at": None,
        }
        lock_path2.write_text(json.dumps(lock_payload), encoding="utf-8")

        with patch.object(api_mod, "_plateau_lock_path", return_value=lock_path2):
            try:
                import asyncio
                # Reset optimizer state so the guard reaches the lock check.
                api_mod._optimizer_state["status"] = "idle"
                api_mod._backtest_state["status"] = "idle"
                asyncio.run(api_mod.start_optimizer(api_mod.OptimizerStartRequest(max_iterations=0, use_llm=False)))
                lock_409_err = "no exception raised; should have raised 409"
            except HTTPException as exc:
                if exc.status_code != 409:
                    lock_409_err = f"status_code={exc.status_code}, expected 409"
                elif not isinstance(exc.detail, dict):
                    lock_409_err = f"detail is not a dict: {type(exc.detail)}"
                elif exc.detail.get("error") != "PlateauLockPresent":
                    lock_409_err = f"detail.error={exc.detail.get('error')!r}"
                elif (exc.detail.get("lock") or {}).get("run_id") != "run_abc":
                    lock_409_err = f"detail.lock.run_id wrong: {exc.detail.get('lock')}"
                else:
                    lock_409_ok = True
    except Exception as e:
        lock_409_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if lock_409_ok else "FAIL",
        "behavioral_409_when_lock_present_with_detail_payload",
        f"start_optimizer with lock present must raise HTTPException(409) with PlateauLockPresent + lock payload ({lock_409_err})",
    ))

    # ---- Claim 8: BEHAVIORAL clear_plateau_lock removes file + writes audit JSONL.
    clear_ok = False
    clear_err = ""
    try:
        td3 = Path(tempfile.mkdtemp(prefix="phase25d6_clear_"))
        locks_dir = td3 / "locks"
        locks_dir.mkdir()
        lock_path3 = locks_dir / "optimizer_plateau.lock"
        lock_payload3 = {
            "created_at": "2026-05-13T00:00:00+00:00",
            "trigger": "plateau_15_discards",
            "consecutive_discards": 15,
            "run_id": "run_def",
            "cleared_at": None,
        }
        lock_path3.write_text(json.dumps(lock_payload3), encoding="utf-8")

        with patch.object(api_mod, "_plateau_lock_path", return_value=lock_path3):
            response = api_mod.clear_plateau_lock()

        if lock_path3.exists():
            clear_err = "lock file should be removed"
        elif response.get("status") != "cleared":
            clear_err = f"status={response.get('status')!r}"
        elif not response.get("lock", {}).get("cleared_at"):
            clear_err = "lock.cleared_at not set in response"
        else:
            # Check audit JSONL was written.
            audit_path = td3 / "audit" / "optimizer_plateau_audit.jsonl"
            if not audit_path.exists():
                clear_err = "audit JSONL not created"
            else:
                lines = [ln for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) != 1:
                    clear_err = f"audit JSONL has {len(lines)} lines, expected 1"
                else:
                    rec = json.loads(lines[0])
                    if not rec.get("cleared_at"):
                        clear_err = "audit JSONL row missing cleared_at"
                    elif rec.get("run_id") != "run_def":
                        clear_err = f"audit JSONL run_id wrong: {rec.get('run_id')!r}"
                    else:
                        clear_ok = True
    except Exception as e:
        clear_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if clear_ok else "FAIL",
        "behavioral_clear_lock_removes_file_and_audits",
        f"clear_plateau_lock must remove file + append audit JSONL with cleared_at ({clear_err})",
    ))

    # ---- Claim 9: BEHAVIORAL cleared_at != null -> _read_plateau_lock returns None.
    cleared_ok = False
    cleared_err = ""
    try:
        td4 = Path(tempfile.mkdtemp(prefix="phase25d6_cleared_"))
        lock_path4 = td4 / "optimizer_plateau.lock"
        lock_path4.write_text(json.dumps({
            "created_at": "2026-05-13T00:00:00+00:00",
            "trigger": "plateau_10_discards",
            "consecutive_discards": 10,
            "run_id": "run_xyz",
            "cleared_at": "2026-05-13T01:00:00+00:00",
        }), encoding="utf-8")

        with patch.object(api_mod, "_plateau_lock_path", return_value=lock_path4):
            result = api_mod._read_plateau_lock()
        if result is not None:
            cleared_err = f"_read_plateau_lock returned {result!r}, expected None for cleared lock"
        else:
            cleared_ok = True
    except Exception as e:
        cleared_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if cleared_ok else "FAIL",
        "behavioral_cleared_lock_treated_as_absent",
        f"_read_plateau_lock must treat cleared_at != null as absent ({cleared_err})",
    ))

    # ---- Claim 10: BEHAVIORAL corrupt lock -> _read returns None (no crash).
    corrupt_ok = False
    corrupt_err = ""
    try:
        td5 = Path(tempfile.mkdtemp(prefix="phase25d6_corrupt_"))
        lock_path5 = td5 / "optimizer_plateau.lock"
        lock_path5.write_text("not valid json at all", encoding="utf-8")

        with patch.object(api_mod, "_plateau_lock_path", return_value=lock_path5):
            result5 = api_mod._read_plateau_lock()
        if result5 is not None:
            corrupt_err = f"_read_plateau_lock returned {result5!r}, expected None for corrupt JSON"
        else:
            corrupt_ok = True
    except Exception as e:
        corrupt_err = f"raised {type(e).__name__}: {e} (must fail-open to None)"

    results.append((
        "PASS" if corrupt_ok else "FAIL",
        "behavioral_corrupt_lock_fail_open",
        f"_read_plateau_lock must fail-open to None on corrupt JSON ({corrupt_err})",
    ))

    # ---- Claim 11: BEHAVIORAL 404 when no lock + clear called.
    no_lock_ok = False
    no_lock_err = ""
    try:
        td6 = Path(tempfile.mkdtemp(prefix="phase25d6_nolock_"))
        missing_path = td6 / "missing.lock"

        with patch.object(api_mod, "_plateau_lock_path", return_value=missing_path):
            try:
                api_mod.clear_plateau_lock()
                no_lock_err = "no exception raised; should have raised 404"
            except HTTPException as exc:
                if exc.status_code != 404:
                    no_lock_err = f"status_code={exc.status_code}, expected 404"
                else:
                    no_lock_ok = True
    except Exception as e:
        no_lock_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if no_lock_ok else "FAIL",
        "behavioral_clear_raises_404_when_no_lock",
        f"clear_plateau_lock must raise 404 when no lock file ({no_lock_err})",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
