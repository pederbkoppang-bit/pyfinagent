"""phase-52.4: residual_momentum_replay.resid_mom_signal -- correctness on synthetic data.

Pins the SIGNAL math (not the empirical promote/reject, which is the live_check's job):
- a positive idiosyncratic run in the 12-1 FORMATION window -> iMOM > 0; a negative run -> iMOM < 0;
- a spike ONLY in the recent (skipped) month -> NOT reflected in iMOM (12-1 skip works);
- too-short input -> None. Deterministic (no randomness in the signal).
$0, no network. The replay lives under scripts/ (not a package) -> load by path.
"""
import importlib.util
import pathlib

import numpy as np

_RP = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "residual_momentum_replay.py"
_spec = importlib.util.spec_from_file_location("residmom_uut", _RP)
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)

FORM, SKIP, N = 252, 21, 504


def _build(form_idio=0.0, recent_idio=0.0, seed=0):
    """s = 1.0*market + idiosyncratic; idiosyncratic gets `form_idio` over the 12-1 formation
    window and `recent_idio` over the skipped recent month."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, N)
    idio = rng.normal(0.0, 0.001, N)
    idio[-(FORM + SKIP):-SKIP] += form_idio   # the formation (12-1)
    if recent_idio:
        idio[-SKIP:] += recent_idio           # the skipped recent month
    return 1.0 * m + idio, m


def test_positive_formation_residual_gives_positive_imom():
    s, m = _build(form_idio=+0.004)
    assert rm.resid_mom_signal(s, m) > 0


def test_negative_formation_residual_gives_negative_imom():
    s, m = _build(form_idio=-0.004)
    assert rm.resid_mom_signal(s, m) < 0


def test_skip_recent_month_only_formation_drives_momentum():
    # The 12-1 skip means only the FORMATION (months 12-2) drives the momentum SUM; a recent-only
    # run does NOT create POSITIVE momentum -- it is excluded from the sum, and the OLS alpha (fit
    # over the full window) absorbs it, so a recent positive run pushes iMOM down, not up.
    _, m = _build()
    imom_formation = rm.resid_mom_signal(_build(form_idio=+0.004)[0], m)
    imom_recent = rm.resid_mom_signal(_build(form_idio=0.0, recent_idio=+0.01)[0], m)
    assert imom_formation > 0                  # the FORMATION drives a positive signal
    assert imom_recent <= 0                     # a RECENT-only run does NOT create positive momentum
    assert imom_recent < imom_formation


def test_too_short_returns_none():
    assert rm.resid_mom_signal(np.zeros(50), np.zeros(50)) is None


def test_deterministic():
    s, m = _build(form_idio=+0.002, seed=3)
    assert rm.resid_mom_signal(s, m) == rm.resid_mom_signal(s, m)
