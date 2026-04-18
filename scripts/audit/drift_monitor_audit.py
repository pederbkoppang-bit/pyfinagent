"""phase-4.8 step 4.8.4 drift-monitor audit.

Three fixtures verify the auto-freeze has teeth:
(a) benign: seeded baseline/current (no feature shift; IC > 0).
    Expect frozen=False.
(b) psi trip: feature-shift fixture (shift + stretch).
    Expect frozen=True + "psi_exceeded" in reasons.
(c) ic trip: predictions and returns anti-correlated across the
    whole window. Expect frozen=True + "ic_sustained_negative"
    in reasons.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.drift_monitor import _seed_model, run  # noqa: E402

OUT = REPO / "handoff" / "drift_monitor_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    # (a) benign
    benign_models = [{"name": "benign", "data": _seed_model("benign-seed")}]
    benign = run(models=benign_models)
    benign_m = benign["models"][0]
    benign_ok = not benign_m["frozen"] and not benign_m["freeze_reasons"]

    # (b) psi trip
    psi_models = [{"name": "psi_trip",
                   "data": _seed_model("psi-trip-seed", psi_anomaly=True)}]
    psi = run(models=psi_models)
    psi_m = psi["models"][0]
    psi_ok = (psi_m["frozen"] and
              psi_m["psi"] > 0.25 and
              any("psi_exceeded" in r for r in psi_m["freeze_reasons"]))

    # (c) ic trip
    ic_models = [{"name": "ic_trip",
                  "data": _seed_model("ic-trip-seed", ic_anomaly=True)}]
    ic = run(models=ic_models)
    ic_m = ic["models"][0]
    ic_ok = (ic_m["frozen"] and
             any("ic_sustained_negative" in r for r in ic_m["freeze_reasons"]))

    all_ok = benign_ok and psi_ok and ic_ok
    verdict = "PASS" if all_ok else "FAIL"

    summary = {
        "step": "4.8.4",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": {
            "benign": {**benign_m, "ok": benign_ok},
            "psi_trip": {**psi_m, "ok": psi_ok},
            "ic_trip": {**ic_m, "ok": ic_ok},
        },
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=float) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "benign_ok": benign_ok,
        "psi_trip_ok": psi_ok,
        "ic_trip_ok": ic_ok,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
