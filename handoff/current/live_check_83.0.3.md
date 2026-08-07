# live_check evidence — step 83.0.3 (PBO false-pass on the MCP surface)

Captured 2026-08-07.

## "BEFORE" — pre-fix behaviour REPRODUCED FROM AN AST-VALIDATED MUTANT (disclosure)

The pre-fix tree state no longer exists (commit `e5bb9f25` / phase-82.27 landed the
checked-wrapper routing), so a historical live capture is IMPOSSIBLE to take
honestly. Per the research brief, the admissible route is an in-memory mutant of
the REAL `risk_server.py` source with the 2-line checked call (anchor count
asserted == 1) reverted to raw `compute_pbo`, `ast.parse`-validated, exec'd into
a fresh module — **never written to disk** (`disk file untouched: True` printed
by the capture script). This is labelled reproduction, NOT a historical capture:

```
MUTANT (pre-fix behaviour reproduced, ast-validated, in-memory only):
{"ok": true, "pbo": 0.0, "threshold": 0.5, "vetoed": false, "n_trials": null, "n_obs": null, "gate_grade": null, "column_corr_mean": null, "columns_diverse": null, "reason": "pbo_within_bounds", "isError": false}
```

The undersized matrix (T=10, S=16) clears every ceiling with the BEST possible
value — the exact manufactured PASS the step describes.

## AFTER — real `mcp__pyfinagent-risk__pbo_check` responses over the live MCP surface

Undersized matrix (T=10, N=4, S=16; rng seed 11, values in
`backend/tests/test_phase_83_0_3_pbo_false_pass.py::_undersized_matrix`) —
verbatim tool response:

```json
{"ok":false,"vetoed":false,"pbo":null,"threshold":0.5,"n_trials":4,"n_obs":10,"reason":"pbo_refused:T=10 < S*2=32; compute_pbo would return a false-good 0.0 that PASSES the ceiling","isError":true}
```

8-column near-identical matrix (T=64, base + eps=0.0005 noise, seed 11; min
pairwise corr 0.99557 measured pre-call) — verbatim tool response:

```json
{"ok":true,"pbo":0.8514374514374514,"threshold":0.5,"vetoed":true,"n_trials":8,"n_obs":64,"gate_grade":false,"column_corr_mean":0.9967753036869336,"columns_diverse":false,"reason":"pbo_exceeds_threshold","isError":true}
```

`columns_diverse: false` as required. Note the HIGH PBO (0.851 → vetoed) on the
near-identical fixture — Bailey/Borwein/Lopez de Prado/Zhu predict exactly this
(similar Sharpes make the in-sample winner noise-selected), so the veto and the
diversity flag agree here by construction, not by accident.

Serialization note (cycle-1 Q/A N3): the matrix was passed to the MCP tool as
JSON rounded to 6 decimal places, so the captured `column_corr_mean`
(0.9967753036869336) differs from the local full-precision float64 value
(0.9967751388680227) at the 1e-7 level; the rank-based `pbo` is bit-identical
under the rounding. Expected, not drift.
