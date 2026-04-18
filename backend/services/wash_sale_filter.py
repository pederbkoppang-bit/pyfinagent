"""phase-4.8 step 4.8.10 wash-sale filter (IRC Sec 1091).

A loss from selling a security is DISALLOWED for tax purposes if
the same (or substantially identical) security is purchased within
a 61-calendar-day window: 30 days BEFORE through 30 days AFTER
the sale.

Pitfall codified here: the window is CALENDAR days, not business
days. `_in_window` uses `(datetime date1 - date2).days` which
includes weekends + holidays.

For paper trading this filter preserves learning-signal accuracy:
a system that ignores wash sales over-reports realized losses and
feeds the evaluator agent misleading P&L.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable

logger = logging.getLogger(__name__)


WINDOW_DAYS = 30                # one-sided; full window is 2 * 30 + 1


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # Accept ISO strings.
    if isinstance(d, str):
        return datetime.fromisoformat(d.split("T")[0]).date()
    raise TypeError(f"unsupported date type: {type(d)}")


@dataclass
class LossEntry:
    symbol: str
    sell_date: date
    disallowed_loss_usd: float
    trade_id: str = ""

    @property
    def window_end(self) -> date:
        return self.sell_date + timedelta(days=WINDOW_DAYS)

    @property
    def window_start(self) -> date:
        return self.sell_date - timedelta(days=WINDOW_DAYS)


@dataclass
class WashSaleLedger:
    """Running ledger of realized losses within the 61-day window.

    `record_loss` adds an entry; `is_wash_sale(symbol, buy_date)`
    scans for any entry whose [window_start, window_end] contains
    buy_date. Entries older than WINDOW_DAYS past the window_end
    are pruned lazily (when `is_wash_sale` is called with a later
    date or explicitly via `prune`).
    """
    entries: list[LossEntry] = field(default_factory=list)

    def record_loss(
        self,
        *,
        symbol: str,
        sell_date: date | datetime | str,
        disallowed_loss_usd: float,
        trade_id: str = "",
    ) -> LossEntry:
        if disallowed_loss_usd <= 0:
            raise ValueError(
                "record_loss: disallowed_loss_usd must be > 0 "
                f"(got {disallowed_loss_usd}); a GAIN is not a "
                "wash-sale candidate"
            )
        entry = LossEntry(
            symbol=symbol.upper(),
            sell_date=_to_date(sell_date),
            disallowed_loss_usd=float(disallowed_loss_usd),
            trade_id=trade_id,
        )
        self.entries.append(entry)
        return entry

    def prune(self, as_of: date | datetime | str | None = None) -> int:
        """Drop entries whose window has fully elapsed. Returns
        number of entries removed."""
        now = _to_date(as_of) if as_of is not None else date.today()
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.window_end >= now]
        return before - len(self.entries)

    def is_wash_sale(
        self,
        symbol: str,
        buy_date: date | datetime | str,
    ) -> tuple[bool, LossEntry | None]:
        """Return (is_wash, matching_entry_or_None)."""
        sym = symbol.upper()
        bd = _to_date(buy_date)
        self.prune(as_of=bd)
        for e in self.entries:
            if e.symbol != sym:
                continue
            if e.window_start <= bd <= e.window_end:
                return True, e
        return False, None


def filter_candidates(
    buys: Iterable[dict],
    ledger: WashSaleLedger,
) -> tuple[list[dict], list[dict]]:
    """Partition candidate buys into (allowed, blocked).

    Each input buy dict must have `symbol` + `trade_date` (date or
    ISO string) + `notional_usd`. Blocked entries are returned with
    added fields `wash_sale: True`, `matched_sell_date`, and
    `disallowed_loss_usd`.
    """
    allowed: list[dict] = []
    blocked: list[dict] = []
    for buy in buys:
        is_ws, entry = ledger.is_wash_sale(
            buy["symbol"], buy["trade_date"],
        )
        if is_ws and entry is not None:
            blocked.append({
                **buy,
                "wash_sale": True,
                "matched_sell_date": entry.sell_date.isoformat(),
                "disallowed_loss_usd": entry.disallowed_loss_usd,
            })
        else:
            allowed.append({**buy, "wash_sale": False})
    return allowed, blocked


__all__ = [
    "LossEntry", "WashSaleLedger", "WINDOW_DAYS",
    "filter_candidates",
]
