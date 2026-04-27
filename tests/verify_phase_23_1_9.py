"""phase-23.1.9 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. 10 new paper-trading fields are exposed in FullSettings (the read API)
2. 9 of those are writable via SettingsUpdate (paper_starting_capital is read-only)
3. DepositRequest validates a normal amount
4. DepositRequest rejects amount=0 (gt=0 enforced)
5. DepositRequest rejects amount=$2M (le=$1M enforced)
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from backend.api.settings_api import FullSettings, SettingsUpdate
from backend.api.paper_trading import DepositRequest


def main() -> int:
    expected = {
        "paper_max_positions",
        "paper_max_daily_cost_usd",
        "paper_default_stop_loss_pct",
        "paper_screen_top_n",
        "paper_analyze_top_n",
        "paper_transaction_cost_pct",
        "paper_daily_loss_limit_pct",
        "paper_trailing_dd_limit_pct",
        "paper_min_cash_reserve_pct",
        "paper_starting_capital",
    }

    full_fields = set(FullSettings.model_fields.keys())
    missing_full = expected - full_fields
    assert not missing_full, f"FullSettings missing: {missing_full}"

    writable = expected - {"paper_starting_capital"}
    update_fields = set(SettingsUpdate.model_fields.keys())
    missing_update = writable - update_fields
    assert not missing_update, f"SettingsUpdate missing: {missing_update}"

    req = DepositRequest(amount=500.0)
    assert req.amount == 500.0, f"normal amount validation failed: {req.amount}"

    rejected = []
    for val, flag in [(0, "floor"), (2_000_000, "ceiling")]:
        try:
            DepositRequest(amount=val)
        except ValidationError:
            rejected.append(flag)
    assert "floor" in rejected, "DepositRequest should reject amount=0"
    assert "ceiling" in rejected, "DepositRequest should reject amount > $1M"

    print("ok 10 paper fields wired + DepositRequest validates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
