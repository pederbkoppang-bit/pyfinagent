"""phase-82.22: does optimizer_best.json's headline actually belong to the run it names?

`optimizer_best.json` is read by 15 consumers, including the live daily cycle
(`backend/services/autonomous_loop.py:431`) and the incumbent-DSR bar every
rotation challenger must clear (`backend/autoresearch/rotation_runner.py`). If
its `sharpe`/`dsr` were inherited from an earlier run by the warm-start path and
then re-stamped with the current run's id, every one of those readers is quoting
a number to the wrong run.

This checker answers one question: **is there a saved experiment artifact,
belonging to the run the file names, that actually produced these metrics?**

Exit 0 = provenance verified or explicitly declared.
Exit 1 = the file claims metrics no artifact of that run produced.

Usage:
    python scripts/qa/check_optimizer_best_provenance.py [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEST = REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
RESULTS = REPO / "backend" / "backtest" / "experiments" / "results"

TOL = 1e-9


def _artifacts_for(run_id: str) -> list[Path]:
    """Saved experiment artifacts belonging to a run.

    Artifacts are named `<stamp>_<run_id>-expNN.json`, so a substring match on
    the run id is the addressing scheme the optimizer itself uses.
    """
    if not run_id:
        return []
    short = run_id.split("-")[0]
    return sorted(p for p in RESULTS.glob("*.json") if short in p.name)


def check() -> dict:
    if not BEST.exists():
        return {"ok": True, "status": "absent", "detail": f"{BEST} does not exist"}

    data = json.loads(BEST.read_text(encoding="utf-8"))
    run_id = data.get("run_id")
    sharpe = data.get("sharpe")
    dsr = data.get("dsr")
    metrics_run_id = data.get("metrics_run_id")
    schema = data.get("schema_version")

    # ABSENCE MUST NOT READ AS FRESH. A file with no `metrics_run_id` is
    # unattributed -- it is NOT a claim that `run_id` produced the metrics.
    attributed_to = metrics_run_id if metrics_run_id is not None else None

    out = {
        "run_id": run_id,
        "metrics_run_id": metrics_run_id,
        "schema_version": schema,
        "sharpe": sharpe,
        "dsr": dsr,
        "warm_started_from": data.get("warm_started_from"),
        "kept": data.get("kept"),
    }

    if sharpe is None:
        return {**out, "ok": True, "status": "no_metrics",
                "detail": "file records no sharpe; nothing to attribute"}

    # Which run do we check against? The declared producer if present,
    # otherwise the writer -- and if the writer's own artifacts do not
    # contain the metrics, that is precisely the defect.
    target = attributed_to or run_id
    arts = _artifacts_for(target)

    # BOTH statistics must reproduce, not just sharpe.
    #
    # phase-82.22 Q/A cycle-1 [BLOCK]: this loop originally compared `sharpe`
    # alone -- it read `deflated_sharpe` into `observed` and then discarded it.
    # A probe with a MATCHING sharpe and a fabricated dsr (0.99 vs the
    # artifact's 0.05) returned "verified", and so did a file with no `dsr` key
    # at all. DSR is the money-path half: it is the term the promotion gate
    # spends (DSR >= 0.95) and the bar every rotation challenger must clear
    # (rotation_runner). Checking provenance on sharpe while leaving dsr
    # unverified defeats the point of the check.
    matches = []
    dsr_mismatches = []
    observed = []
    for p in arts:
        try:
            a = json.loads(p.read_text(encoding="utf-8")).get("analytics") or {}
        except Exception:
            continue
        s = a.get("sharpe")
        if s is None:
            continue
        art_dsr = a.get("deflated_sharpe")
        observed.append((p.name, s, art_dsr))
        if abs(float(s) - float(sharpe)) < TOL:
            # sharpe reproduces -- now the dsr must too.
            if dsr is None:
                dsr_mismatches.append((p.name, art_dsr, None))
            elif art_dsr is None:
                dsr_mismatches.append((p.name, None, dsr))
            elif abs(float(art_dsr) - float(dsr)) < TOL:
                matches.append(p.name)
            else:
                dsr_mismatches.append((p.name, art_dsr, dsr))

    out.update({
        "checked_against_run": target,
        "artifacts_found": len(arts),
        "artifacts_with_metrics": len(observed),
        "matching_artifacts": matches,
    })

    if matches:
        return {**out, "ok": True, "status": "verified",
                "detail": f"sharpe {sharpe} AND dsr {dsr} both reproduced by {matches[0]}"}

    # sharpe reproduced but dsr did not -- the more insidious shape, because
    # the file looks self-consistent on its headline number.
    if dsr_mismatches:
        name, art_dsr, claimed = dsr_mismatches[0]
        return {**out, "ok": False, "status": "dsr_mis_attributed",
                "dsr_mismatches": dsr_mismatches,
                "detail": (
                    f"sharpe {sharpe} reproduces from {name}, but the recorded "
                    f"dsr={claimed!r} does not match that artifact's "
                    f"deflated_sharpe={art_dsr!r}. DSR is the term the promotion "
                    "gate spends and the bar every rotation challenger must "
                    "clear, so a sharpe-only match is not provenance.")}

    # No artifact of the named run produced this number. Say where it DID
    # come from, if we can find it -- that is the actionable part.
    origin = []
    for p in RESULTS.glob("*.json"):
        try:
            a = json.loads(p.read_text(encoding="utf-8")).get("analytics") or {}
        except Exception:
            continue
        s = a.get("sharpe")
        if s is not None and abs(float(s) - float(sharpe)) < TOL:
            origin.append(p.name)

    detail = (
        f"optimizer_best.json claims sharpe={sharpe} for run {target!r}, but none "
        f"of that run's {len(observed)} saved artifacts produced it."
    )
    if origin:
        detail += f" The value appears in: {', '.join(sorted(origin)[:3])}."
    if attributed_to is None:
        detail += (
            " The file carries no `metrics_run_id`, so its provenance is UNDECLARED"
            " -- it must not be read as self-attributed to `run_id`."
        )
    if observed:
        detail += (
            f" That run's own observed sharpes: "
            f"{sorted({round(s, 6) for _, s, _ in observed})}."
        )
    return {**out, "ok": False, "status": "mis_attributed",
            "detail": detail, "origin_artifacts": sorted(origin)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = check()
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"[optimizer_best provenance] {r['status'].upper()}")
        print(f"  {r['detail']}")
        for k in ("run_id", "metrics_run_id", "schema_version", "kept",
                  "checked_against_run", "artifacts_found", "matching_artifacts"):
            if k in r:
                print(f"  {k}: {r[k]}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
