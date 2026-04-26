"""phase-5.6 backend.markets.options -- options-specific machinery.

Exposes Black-Scholes greeks calculation + OCC option-symbol parsing.
The options_ingestion script lives alongside as a module-as-script
(invoked via `python -m backend.markets.options.options_ingestion`).
"""
from __future__ import annotations

from backend.markets.options.greeks import (
    black_scholes_greeks,
    parse_occ_symbol,
)

__all__ = ["black_scholes_greeks", "parse_occ_symbol"]
