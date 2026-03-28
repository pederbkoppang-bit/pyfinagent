"""
Backtest trader — in-memory portfolio simulator with inverse-volatility
position sizing and meta-label probability weighting.
No BQ writes during simulation; all state in-memory.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_entry_price: float
    entry_date: str
    label: int  # +1 BUY, -1 SELL, 0 HOLD


@dataclass
class Trade:
    ticker: str
    action: str  # "BUY" or "SELL"
    quantity: float
    price: float
    date: str
    label: int
    probability: float
    commission: float = 0.0


@dataclass
class DailySnapshot:
    date: str
    nav: float
    cash: float
    positions_value: float
    num_positions: int


class BacktestTrader:
    """
    In-memory portfolio simulator for backtesting.
    Uses inverse-volatility × meta-label probability for position sizing (AQR).
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        max_positions: int = 20,
        transaction_cost_pct: float = 0.1,
        target_vol: float = 0.15,
        max_single_pct: float = 0.10,
        commission_model: str = "flat_pct",
        commission_per_share: float = 0.005,
    ):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.max_positions = max_positions
        self.transaction_cost_pct = transaction_cost_pct
        self.target_vol = target_vol
        self.max_single_pct = max_single_pct
        self.commission_model = commission_model
        self.commission_per_share = commission_per_share

        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[DailySnapshot] = []
        self.total_commission: float = 0.0

    def _compute_commission(self, quantity: float, price: float) -> float:
        """Compute commission based on the active model."""
        if self.commission_model == "per_share":
            return max(quantity * self.commission_per_share, 1.0)
        # Default: flat percentage of notional
        return abs(quantity * price) * self.transaction_cost_pct / 100

    def size_position(
        self, probability: float, stock_vol: float, nav: float,
        turbulence: float = 0.0, turbulence_threshold: float = 1.0,
        amihud_illiquidity: float = 0.0,
        vol_target_scale: float = 1.0,
    ) -> float:
        """
        Inverse-volatility position sizing (AQR / Frazzini & Pedersen 2014):
        dollar_amount = probability × (target_vol / stock_vol) × nav / max_positions
        Capped at max_single_pct × nav.

        Turbulence scaling (FinRL architecture, Mahalanobis distance):
        When turbulence > threshold, positions are scaled down proportionally.
        This reduces exposure during systemic market stress events (e.g., COVID crash,
        2022 bear market). turbulence=0 or threshold=0 disables this feature.

        Amihud illiquidity filter (López de Prado AFML Ch. 18):
        High Amihud = low liquidity = wider bid-ask spreads = higher execution costs.
        Scales position down for illiquid stocks. Typical S&P 500 Amihud values
        are 0.01-1.0 (×1e6 scaled). Values above 5.0 indicate illiquid stocks.
        """
        if stock_vol <= 0 or probability <= 0:
            return 0.0

        vol_scale = min(self.target_vol / stock_vol, 3.0)  # Cap at 3x to prevent extreme sizing
        raw = probability * vol_scale * nav / self.max_positions

        # Turbulence dampening: scale down positions when market is stressed
        if turbulence > 0 and turbulence_threshold > 0 and turbulence > turbulence_threshold:
            # Scale factor: 1.0 at threshold, approaches 0.2 at 5× threshold
            turbulence_ratio = turbulence / turbulence_threshold
            dampening = max(0.2, 1.0 / turbulence_ratio)
            raw *= dampening

        # Amihud liquidity scaling: penalize illiquid stocks
        # S&P 500 median Amihud ≈ 0.1-0.5. Above 2.0 is relatively illiquid.
        # Scale: 1.0 for liquid (amihud ≤ 0.5), down to 0.3 for very illiquid (amihud ≥ 10)
        if amihud_illiquidity and amihud_illiquidity > 0.5:
            liquidity_scale = max(0.3, 1.0 / (1.0 + (amihud_illiquidity - 0.5) / 3.0))
            raw *= liquidity_scale

        # Volatility targeting: scale position to match target annual vol
        # Computed by BacktestEngine._compute_vol_target_scale()
        if vol_target_scale != 1.0:
            raw *= vol_target_scale

        capped = min(raw, nav * self.max_single_pct)
        return max(0.0, capped)

    def execute_trades(
        self,
        signals: list[dict],
        date: str,
        prices: dict[str, float],
    ) -> list[Trade]:
        """
        Execute trades based on ML signals.

        signals: list of {"ticker", "label", "probability", "volatility"}
        prices: dict of ticker → current price
        """
        executed = []
        nav = self._compute_nav(prices)

        # First: close positions where signal flipped to SELL (-1)
        for sig in signals:
            ticker = sig["ticker"]
            label = sig["label"]
            if label == -1 and ticker in self.positions:
                pos = self.positions[ticker]
                price = prices.get(ticker, pos.avg_entry_price)
                proceeds = pos.quantity * price
                cost = self._compute_commission(pos.quantity, price)
                self.cash += proceeds - cost
                self.total_commission += cost

                trade = Trade(
                    ticker=ticker, action="SELL", quantity=pos.quantity,
                    price=price, date=date, label=label,
                    probability=sig.get("probability", 0),
                    commission=cost,
                )
                self.trades.append(trade)
                executed.append(trade)
                del self.positions[ticker]

        # Second: open new BUY positions
        buy_signals = [s for s in signals if s["label"] == 1 and s["ticker"] not in self.positions]
        # Sort by probability (highest confidence first)
        buy_signals.sort(key=lambda x: x.get("probability", 0), reverse=True)

        for sig in buy_signals:
            if len(self.positions) >= self.max_positions:
                break

            ticker = sig["ticker"]
            price = prices.get(ticker)
            if not price or price <= 0:
                continue

            probability = sig.get("probability", 0.5)
            volatility = sig.get("volatility", 0.3)
            amihud = sig.get("amihud_illiquidity", 0.0)
            vt_scale = sig.get("vol_target_scale", 1.0)
            turbulence = sig.get("turbulence", 0.0)  # PHASE 1.3: Market stress indicator
            dollar_amount = self.size_position(
                probability, volatility, nav,
                turbulence=turbulence,
                amihud_illiquidity=amihud,
                vol_target_scale=vt_scale,
            )

            if dollar_amount < 100:  # Minimum trade size
                continue

            # Check cash
            cost_basis = dollar_amount
            quantity = cost_basis / price
            transaction_cost = self._compute_commission(quantity, price)
            total_needed = cost_basis + transaction_cost

            if total_needed > self.cash:
                cost_basis = self.cash * 0.95  # Use 95% of remaining cash
                quantity = cost_basis / price
                transaction_cost = self._compute_commission(quantity, price)
                total_needed = cost_basis + transaction_cost
                if total_needed > self.cash or cost_basis < 100:
                    continue

            self.cash -= total_needed
            self.total_commission += transaction_cost

            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=price,
                entry_date=date,
                label=1,
            )

            trade = Trade(
                ticker=ticker, action="BUY", quantity=quantity,
                price=price, date=date, label=1,
                probability=probability,
                commission=transaction_cost,
            )
            self.trades.append(trade)
            executed.append(trade)

        return executed

    def mark_to_market(self, date: str, prices: dict[str, float]) -> float:
        """Update positions with current prices and return NAV."""
        nav = self._compute_nav(prices)
        positions_value = nav - self.cash

        self.snapshots.append(DailySnapshot(
            date=date,
            nav=nav,
            cash=self.cash,
            positions_value=positions_value,
            num_positions=len(self.positions),
        ))

        return nav

    def close_all_positions(self, date: str, prices: dict[str, float]):
        """Liquidate all positions at end of test window."""
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            price = prices.get(ticker, pos.avg_entry_price)
            proceeds = pos.quantity * price
            cost = self._compute_commission(pos.quantity, price)
            self.cash += proceeds - cost
            self.total_commission += cost

            self.trades.append(Trade(
                ticker=ticker, action="SELL", quantity=pos.quantity,
                price=price, date=date, label=0,
                probability=0,
                commission=cost,
            ))
            del self.positions[ticker]

    def get_returns_series(self) -> list[float]:
        """Get daily return series from snapshots."""
        if len(self.snapshots) < 2:
            return []
        returns = []
        for i in range(1, len(self.snapshots)):
            prev_nav = self.snapshots[i - 1].nav
            curr_nav = self.snapshots[i].nav
            if prev_nav > 0:
                returns.append((curr_nav - prev_nav) / prev_nav)
        return returns

    def _compute_nav(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            pos.quantity * prices.get(pos.ticker, pos.avg_entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def reset(self):
        """Reset for next window while carrying forward capital."""
        # Capital carries forward (from cash after closing all positions)
        self.positions.clear()
        # Don't reset cash — it carries forward from closed positions
        # Don't reset trades/snapshots — they accumulate across windows

    def full_reset(self):
        """Full reset for a new independent backtest run.

        Per Bailey & López de Prado (2014), each optimizer trial must be an
        independent measurement. This resets ALL state so successive
        run_backtest() calls don't contaminate each other's returns.
        """
        self.cash = self.starting_capital
        self.positions.clear()
        self.trades.clear()
        self.snapshots.clear()
        self.total_commission = 0.0
