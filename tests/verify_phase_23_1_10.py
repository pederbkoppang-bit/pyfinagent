"""phase-23.1.10 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. The ticker-meta route is registered on the paper_trading router
2. _fetch_ticker_meta + _yfinance_ticker_info are importable + callable
3. _yfinance_ticker_info returns the documented fallback shape on a fake ticker
4. ENDPOINT_TTLS has the new paper:ticker_meta entry at 86400s (24h)
"""

from __future__ import annotations

import sys


def main() -> int:
    from backend.api.paper_trading import (
        router,
        _fetch_ticker_meta,
        _yfinance_ticker_info,
    )
    from backend.services.api_cache import ENDPOINT_TTLS

    paths = {getattr(r, "path", None) for r in router.routes}
    assert "/api/paper-trading/ticker-meta" in paths, \
        f"/api/paper-trading/ticker-meta not registered (paths: {sorted(p for p in paths if p)})"

    assert callable(_fetch_ticker_meta), "_fetch_ticker_meta not callable"
    assert callable(_yfinance_ticker_info), "_yfinance_ticker_info not callable"

    # The fallback shape on a never-going-to-exist ticker
    out = _yfinance_ticker_info("ZZZZZ_NOT_A_REAL_TICKER")
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    assert "company_name" in out, f"missing company_name: {out}"
    assert "sector" in out, f"missing sector: {out}"
    assert "source" in out, f"missing source: {out}"
    # Either it gracefully fell back OR it actually got something — both are valid
    assert out["company_name"], "company_name must not be empty"

    assert ENDPOINT_TTLS.get("paper:ticker_meta") == 86400.0, \
        f"ENDPOINT_TTLS['paper:ticker_meta'] must be 86400.0 (24h), got {ENDPOINT_TTLS.get('paper:ticker_meta')}"

    print("ok ticker-meta route registered + helpers callable + 24h TTL configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
