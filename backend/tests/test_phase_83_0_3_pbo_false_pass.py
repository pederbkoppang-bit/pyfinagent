"""phase-83.0.3: the PBO false-pass on the MCP surface -- PROOF step.

The routing fix itself landed under commit e5bb9f25 (phase-82.27): the step
text's anchors (risk_server.py:142-143) are STALE -- they now point at comment
prose about the old defect; the live checked call sits a few lines below.
This file proves the fix and pins it against regression. Criteria map:

 C1  pbo_check routes through compute_pbo_checked -- carried by TWO guards
     (cycle-2 correction: the cycle-1 docstring credited the behavioural
     refusal alone, and the Q/A refuted that with an impostor that refuses
     inline over raw compute_pbo and passes every behavioural test): (a) a
     direct routing SPY asserting compute_pbo_checked is actually invoked
     (test_c1_routing_spy...), and (b) the test_c3 source anchor, which the
     impostor and a kwarg->positional rewording both trip. The behavioural
     refusal below is NECESSARY but NOT SUFFICIENT for routing. The anchor
     is not comment-satisfiable (it requires the 12-space-indented
     assignment, not prose).
 C2  T=10 with S=16 returns an explicit refusal payload, inspected field by
     field via _assert_refusal (shared with the mutation test).
 C3  reverting pbo_check to raw compute_pbo makes the C2 assertions FAIL --
     executed against an ast-validated in-memory mutant of the REAL source
     (never written to disk). The mutant is demonstrably NON-equivalent: it
     reproduces the original false PASS ({ok: True, pbo: 0.0, vetoed:
     False}) on the same input. One killed mutant is a targeted regression
     pin, not a mutation score (equivalent-mutant rates run 4-39% in
     practice, arXiv:2408.01760).
 C4  the computed payload carries gate_grade + columns_diverse; 8 near-
     identical columns (pairwise corr > 0.99, precondition-asserted so seed
     drift fails as a fixture, never as a false verdict) report
     columns_diverse False, with the independent-columns mirror guard
     (else "always False" would satisfy it). gate_grade does NOT
     discriminate diversity at N=8 (< PBO_MIN_TRIALS_GATE_GRADE) -- it is
     asserted present and False on BOTH fixtures, deliberately.
 C5  raw-call-site census: recorded in experiment_results_83.0.3.md as a
     measured before/after DELTA with sites enumerated -- not asserted zero.
 C6  gate.py minimum-trials bypass: disclosure with verbatim lines, in
     experiment_results_83.0.3.md; already pinned live by
     test_phase_82_27_pbo_sweep_producer.py:258-262.

Bailey/Borwein/Lopez de Prado/Zhu (SSRN 2326253) note near-identical columns
produce a HIGH PBO (similar Sharpes => in-sample winner is noise), so no
assertion here expects a low PBO from the degenerate fixture.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import types

import numpy as np
import pytest

from backend.agents.mcp_servers.risk_server import create_risk_server
from backend.backtest.analytics import compute_pbo

_SRC = pathlib.Path(__file__).resolve().parents[2] / (
    "backend/agents/mcp_servers/risk_server.py"
)


def _pbo_check():
    """The undecorated tool function (in-repo idiom: test_phase_82_27:210)."""
    return asyncio.run(create_risk_server().get_tool("pbo_check")).fn


def _undersized_matrix() -> list[list[float]]:
    rng = np.random.default_rng(11)
    return rng.normal(size=(10, 4)).tolist()  # T=10 < S*2=32


def _assert_refusal(r: dict) -> None:
    """The C2 assertion set. Shared with the mutation test (C3) so the mutant
    is graded by the SAME assertions the criterion-2 test uses."""
    assert r["ok"] is False, f"undersized matrix cleared the veto: {r}"
    assert r["pbo"] is None, f"a numeric PBO was returned for T<S*2: {r}"
    assert str(r.get("reason", "")).startswith("pbo_refused:"), r
    assert r["isError"] is True, "a refusal must not read as a clean pass"
    assert r.get("vetoed") is False  # refusal is NO MEASUREMENT, not a veto


# ── C1 + C2: behavioural refusal on the real tool ────────────────────


def test_c1_routing_spy_compute_pbo_checked_is_invoked(monkeypatch):
    """Criterion 1, behaviourally observable (cycle-2, Q/A named fix c):
    the tool's function-scoped import resolves at call time, so spying on
    the analytics module attribute proves the ACTUAL routing -- an inline-
    refusal impostor over raw compute_pbo cannot satisfy this."""
    import backend.backtest.analytics as analytics

    calls = []
    real = analytics.compute_pbo_checked

    def spy(pnl_matrix, S=16):
        calls.append(True)
        return real(pnl_matrix, S=S)

    monkeypatch.setattr(analytics, "compute_pbo_checked", spy)
    r = _pbo_check()(pnl_matrix=_undersized_matrix(), S=16)
    assert calls, "pbo_check did not invoke compute_pbo_checked"
    _assert_refusal(r)


def test_c1_c2_undersized_matrix_refuses_not_passes():
    m = _undersized_matrix()
    r = _pbo_check()(pnl_matrix=m, S=16)
    _assert_refusal(r)
    assert r["n_trials"] == 4 and r["n_obs"] == 10
    # The discriminator this guard protects against: raw compute_pbo returns
    # 0.0 -- the BEST possible value -- on the identical input.
    assert compute_pbo(np.asarray(m), S=16) == 0.0


# ── C3: the raw-compute_pbo revert mutant fails the C2 assertions ────

_ORIG = (
    "            from backend.backtest.analytics import compute_pbo_checked\n"
    "            checked = compute_pbo_checked(pnl_matrix, S=S)\n"
)
_MUT = (
    "            from backend.backtest.analytics import compute_pbo\n"
    "            checked = {\"pbo\": float(compute_pbo(pnl_matrix, S=S)),\n"
    "                       \"n_trials\": None, \"n_obs\": None}\n"
)


def test_c3_reverting_to_raw_compute_pbo_fails_criterion_2():
    src = _SRC.read_text()
    n = src.count(_ORIG)
    # DO NOT DELETE this anchor assertion: with the routing spy it is one of
    # criterion 1's TWO routing guards, and it is the only one that catches a
    # source rewording (kwarg->positional) before the mutant is even built.
    assert n == 1, (
        f"mutation anchor count={n}, want exactly 1 -- a no-match replace "
        "would silently grade the UNMUTATED module"
    )
    mutated = src.replace(_ORIG, _MUT, 1)
    ast.parse(mutated)  # the mutant must be valid python, not a syntax casualty
    mod = types.ModuleType("risk_server_mutant_83_0_3")
    exec(compile(mutated, str(_SRC), "exec"), mod.__dict__)
    mfn = asyncio.run(mod.create_risk_server().get_tool("pbo_check")).fn

    m = _undersized_matrix()
    r_mut = mfn(pnl_matrix=m, S=16)
    # Non-equivalence, pinned: the mutant reproduces the ORIGINAL false PASS.
    assert r_mut["ok"] is True and r_mut["pbo"] == 0.0 and r_mut["vetoed"] is False
    with pytest.raises(AssertionError):
        _assert_refusal(r_mut)


# ── C4: near-identical columns flagged; independent mirror guard ─────


def _near_identical_matrix() -> np.ndarray:
    rng = np.random.default_rng(11)
    t = 600
    base = rng.normal(0.0005, 0.01, size=(t, 1))
    return base + rng.normal(0, 0.0005, size=(t, 8))


def test_c4_near_identical_columns_reported_not_diverse():
    mat = _near_identical_matrix()
    # Fixture precondition FIRST: the eps window is narrow (0.001 noise
    # already drops corr_mean below 0.99); drift must fail HERE.
    corr = np.corrcoef(mat, rowvar=False)
    pair = corr[np.triu_indices_from(corr, k=1)]
    assert pair.min() > 0.99, f"fixture degraded: min pairwise corr {pair.min():.5f}"

    r = _pbo_check()(pnl_matrix=mat.tolist(), S=16)
    assert r["ok"] is True and r["pbo"] is not None
    assert r["n_trials"] == 8
    assert "gate_grade" in r and "columns_diverse" in r
    assert r["columns_diverse"] is False, (
        f"corr_mean={r.get('column_corr_mean')}: near-identical columns "
        "must be flagged"
    )
    assert r["gate_grade"] is False  # N=8 < 10: never gate-grade, diverse or not


def test_c4_mirror_independent_columns_reported_diverse():
    rng = np.random.default_rng(11)
    mat = rng.normal(size=(600, 8)).tolist()
    r = _pbo_check()(pnl_matrix=mat, S=16)
    assert r["ok"] is True and r["pbo"] is not None
    assert r["columns_diverse"] is True, (
        "independent columns flagged non-diverse -- an always-False "
        "diagnostic would vacuously satisfy the near-identical test"
    )
    assert r["gate_grade"] is False  # still N=8 < 10; presence, not diversity


# ── protocol-layer round trip: the refusal survives into structured
#    content over a real in-memory client (the payload-vs-protocol
#    is_error divergence itself is OWNED BY step 83.0.6, not asserted) ─


def test_refusal_survives_client_round_trip():
    from fastmcp import Client

    async def _call():
        async with Client(create_risk_server()) as client:
            return await client.call_tool(
                "pbo_check", {"pnl_matrix": _undersized_matrix(), "S": 16}
            )

    result = asyncio.run(_call())
    payload = result.structured_content
    if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
        payload = payload["result"]  # fastmcp wraps plain-dict returns
    _assert_refusal(payload)
