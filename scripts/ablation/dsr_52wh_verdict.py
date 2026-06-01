"""phase-52.3: apply the A-PRIORI ENABLE/REJECT rule to the 52wh-tilt edge.

Reads the PINNED paired returns (handoff/current/_52wh_paired_returns.json, dumped by
sector_neutral_replay.py) and runs the rigorous test stack:
  PRIMARY   -- Ledoit-Wolf (2008) SR-difference, stationary-bootstrap one-sided p (H0: SR_tilt<=SR_base)
  SECONDARY -- DSR on the tilt's ABSOLUTE Sharpe, deflated for the configs tried (weak discriminator)
A-PRIORI RULE (fixed in the 52.3 contract BEFORE running -> no p-hacking):
  ENABLE iff  R1: p_one_sided < 0.05  AND  R2: delta >= +0.05 AND bootstrap-90%-CI lower bound > 0.
$0, deterministic (seeded bootstrap), NO live change.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats as sps

from backend.backtest.analytics import sharpe_diff_test, compute_deflated_sharpe

JSON_PATH = "handoff/current/_52wh_paired_returns.json"


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        J = json.load(f)
    base, tilt = J["baseline"], J["hi52_k0.5"]
    cfg_srs = list(J["config_sharpes"].values())

    # PRIMARY: Ledoit-Wolf SR-difference, stationary bootstrap (5000 resamples, $0)
    r = sharpe_diff_test(tilt, base, periods_per_year=12, n_boot=5000, block=4, seed=42, ci=0.90)

    # SECONDARY: DSR on the tilt's absolute Sharpe, deflated for 5 configs tried
    diffs = np.array([t - b for t, b in zip(tilt, base) if t is not None and b is not None])
    skew = float(sps.skew(diffs)) if len(diffs) > 5 else 0.0
    kurt = float(sps.kurtosis(diffs, fisher=False)) if len(diffs) > 5 else 3.0
    var_srs = float(np.var(cfg_srs)) if len(cfg_srs) > 1 else 0.5
    dsr = compute_deflated_sharpe(observed_sr=r["sr_a"], num_trials=5,
                                  variance_of_srs=var_srs, skewness=skew, kurtosis=kurt, T=r["n"])

    # A-PRIORI RULE
    R1 = r["p_one_sided"] < 0.05
    R2 = (r["delta"] >= 0.05) and (r["ci_low"] > 0)
    enable = R1 and R2

    print("=== phase-52.3: 52wh-tilt robustness verdict (a-priori rule) ===")
    print(f"n_rebalances={r['n']}  n_boot={r['n_boot']}")
    print(f"SR_tilt={r['sr_a']:.3f}  SR_base={r['sr_b']:.3f}  delta={r['delta']:+.3f}")
    print(f"PRIMARY  Ledoit-Wolf stationary-bootstrap p (one-sided, H0 SR_tilt<=SR_base) = {r['p_one_sided']:.4f}")
    print(f"         -> R1 (p < 0.05): {R1}")
    print(f"         bootstrap 90% CI for delta = [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  (se={r['se']:.3f})")
    print(f"         -> R2 (delta >= +0.05 AND CI_low > 0): {R2}")
    print(f"SECONDARY DSR(abs SR={r['sr_a']:.2f}, 5 trials, T={r['n']}) = {dsr:.3f}  (weak here; report only)")
    print()
    if enable:
        print("VERDICT: ENABLE -- the +0.05 52wh edge survives the SR-difference test + magnitude rule.")
        print("         (Live enable is still a SEPARATE operator-gated flag flip, post-Monday.)")
    else:
        print("VERDICT: REJECT -- the +0.05 edge is NOT statistically distinguishable from noise")
        print("         (R1 and/or R2 failed). Keep the 52wh tilt flag OFF (dormant). The measured")
        print("         point estimate is within selection-bias/small-sample noise; per McLean-Pontiff")
        print("         it would decay further live. Pivot: residual momentum (52.4) or accept the engine.")
    return 0 if True else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
