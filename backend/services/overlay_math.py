"""phase-69.3: sign-safe overlay multiplier.

Multiplicative overlays (`base_score * mult`) INVERT when `base_score < 0`: a boost
(mult>1) makes a negative score MORE negative (worse rank) and a penalty (mult<1)
makes it LESS negative (better rank) -- so in broad drawdowns (negative momentum
composites) positive catalysts demote candidates and negative catalysts promote them,
exactly when selection carries the most information.

`sign_safe_mult` applies the tilt so a boost ALWAYS improves rank and a penalty ALWAYS
worsens it, in both sign regimes:

    enabled=True  -> base + abs(base)*(mult-1)   # sign-safe
        base >= 0 : reduces to base*mult          (unchanged intent)
        base <  0 : reduces to base*(2-mult)      (boost moves score UP toward 0)
    enabled=False -> base * mult                  # legacy, BYTE-IDENTICAL

Flag-gated: default-OFF (via `settings.sign_safe_overlays=False`) returns the legacy
`base*mult` so the LIVE engine is byte-identical until the operator flips the flag.
Proof + reference: research_brief_69.0.md §3 / design_audit_burndown_69.md §3.
"""

from __future__ import annotations


def sign_safe_mult(base_score: float, mult: float, *, enabled: bool | None = None) -> float:
    """Apply a multiplicative tilt `mult` to a (possibly signed) `base_score`.

    enabled=None (default) reads `settings.sign_safe_overlays` (default False), so
    every overlay call site is gated by ONE flag with no per-site plumbing.
    """
    if enabled is None:
        try:
            from backend.config.settings import get_settings
            enabled = bool(getattr(get_settings(), "sign_safe_overlays", False))
        except Exception:
            enabled = False
    if not enabled:
        return base_score * mult
    return base_score + abs(base_score) * (mult - 1.0)
