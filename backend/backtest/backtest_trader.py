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
        
        # PHASE 1.7 IMPROVEMENT: Performance-based position scaling
        self._recent_returns: list[float] = []  # Track last 20 trading days
        self._performance_scale = 1.0  # Scale positions based on recent performance

    def _compute_commission(self, quantity: float, price: float, amihud_illiquidity: float = 0.0) -> float:
        """
        PHASE 1.9: Enhanced transaction cost model (Almgren & Chriss 2000).
        Includes: commission + bid-ask spread + market impact + illiquidity penalty.
        """
        notional = abs(quantity * price)
        
        if self.commission_model == "per_share":
            base_commission = max(quantity * self.commission_per_share, 1.0)
        else:
            # Base commission: flat percentage
            base_commission = notional * self.transaction_cost_pct / 100
        
        # PHASE 1.9 IMPROVEMENT: Market microstructure costs
        
        # 1. Bid-ask spread (depends on liquidity)
        # S&P 500 average spread: ~0.01-0.05%. Illiquid stocks: 0.05-0.20%
        base_spread = 0.02  # 2 bps for liquid stocks
        spread_penalty = amihud_illiquidity * 0.5 if amihud_illiquidity else 0  # Scale with Amihud
        bid_ask_spread = min(base_spread + spread_penalty, 0.20) / 100  # Cap at 20 bps
        spread_cost = notional * bid_ask_spread / 2  # Half spread per side
        
        # 2. Market impact (Almgren-Chriss square-root model)
        # Impact ∝ sqrt(trade_size / avg_volume). Simplified: larger trades cost more.
        # For S&P 500: assume $100k trade has ~1bp impact, scales by sqrt
        trade_size_factor = min(notional / 100_000, 10.0)  # Normalize to $100k, cap at 10x
        market_impact = notional * 0.0001 * (trade_size_factor ** 0.5)  # Square-root scaling
        
        # 3. Additional illiquidity penalty for very illiquid stocks
        illiq_penalty = notional * min(amihud_illiquidity * 0.001, 0.002) if amihud_illiquidity > 5.0 else 0
        
        total_cost = base_commission + spread_cost + market_impact + illiq_penalty
        return total_cost

    def size_position(
        self, probability: float, stock_vol: float, nav: float,
        turbulence: float = 0.0, turbulence_threshold: float = 1.0,
        amihud_illiquidity: float = 0.0,
        vol_target_scale: float = 1.0,
        correlation_penalty: float = 1.0,
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
        
        Correlation penalty (Phase 1.8 - Portfolio Diversification):
        Scales down positions for stocks highly correlated with existing holdings.
        1.0 = no penalty, 0.5 = 50% reduction for highly correlated stocks.
        """
        if stock_vol <= 0 or probability <= 0:
            return 0.0

        # PHASE 1.4 IMPROVEMENT: Fractional Kelly Criterion position sizing
        # Kelly fraction: f = (bp - q) / b
        # For binary outcomes with ML probability estimates:
        # - Assume symmetric TP/SL barriers → expected_win ≈ expected_loss
        # - Kelly becomes: f = 2p - 1 (where p = ML probability)
        # - Use 1/2 Kelly (Thorp recommendation) for reduced volatility
        # - Further scale by inverse volatility for risk-adjusted sizing
        
        kelly_base = max(0.0, 2 * probability - 1)  # Only size when p > 0.5
        kelly_fraction = kelly_base * 0.5  # Half Kelly for stability
        vol_scale = min(self.target_vol / stock_vol, 3.0)  # Cap at 3x to prevent extreme sizing
        raw = kelly_fraction * vol_scale * nav / self.max_positions

        # PHASE 1.7 IMPROVEMENT: Performance-based scaling
        # Scale down after losses, scale up after gains (with limits)
        raw *= self._performance_scale

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

        # PHASE 1.8 IMPROVEMENT: Correlation-based diversification
        # Reduce position size for stocks highly correlated with existing holdings
        raw *= correlation_penalty

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
                cost = self._compute_commission(pos.quantity, price, sig.get("amihud_illiquidity", 0.0))
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
            
            # PHASE 1.8: Compute correlation penalty for diversification
            existing_tickers = list(self.positions.keys())
            corr_penalty = self._compute_correlation_penalty(ticker, existing_tickers)
            
            dollar_amount = self.size_position(
                probability, volatility, nav,
                turbulence=turbulence,
                amihud_illiquidity=amihud,
                vol_target_scale=vt_scale,
                correlation_penalty=corr_penalty,
            )

            if dollar_amount < 100:  # Minimum trade size
                continue

            # Check cash
            cost_basis = dollar_amount
            quantity = cost_basis / price
            transaction_cost = self._compute_commission(quantity, price, amihud)
            total_needed = cost_basis + transaction_cost

            if total_needed > self.cash:
                cost_basis = self.cash * 0.95  # Use 95% of remaining cash
                quantity = cost_basis / price
                transaction_cost = self._compute_commission(quantity, price, amihud)
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

    def _update_performance_scaling(self, nav: float, prev_nav: float):
        """
        PHASE 1.7: Update position sizing based on recent performance.
        Reduces positions after losses, increases after gains (within limits).
        Based on behavioral portfolio theory + risk management best practices.
        """
        if prev_nav > 0:
            daily_return = (nav - prev_nav) / prev_nav
            self._recent_returns.append(daily_return)
            
            # Keep only last 20 trading days
            if len(self._recent_returns) > 20:
                self._recent_returns = self._recent_returns[-20:]
            
            # Compute rolling return over lookback period
            if len(self._recent_returns) >= 5:  # Need at least 5 days
                recent_performance = sum(self._recent_returns)
                
                # Scale factor: 0.5x after -10% drawdown, 1.5x after +10% gain
                # Capped between 0.3 and 1.8 to prevent extreme sizing
                scale = 1.0 + recent_performance * 2.0  # 2x leverage on recent performance
                self._performance_scale = max(0.3, min(1.8, scale))
            else:
                self._performance_scale = 1.0

    def _compute_correlation_penalty(self, candidate_ticker: str, existing_positions: list[str]) -> float:
        """
        PHASE 1.8: Compute correlation penalty for portfolio diversification.
        Returns scaling factor between 0.3 and 1.0 based on correlations with existing positions.
        
        Simple sector-based diversification for now (requires historical data for full correlation).
        Can be enhanced with actual price correlation calculation.
        """
        if not existing_positions:
            return 1.0
            
        # Simplified sector-based correlation penalty
        # In production, would use actual price correlation from historical data
        sector_mapping = {
            # Tech stocks (typically correlated)
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'GOOG': 'tech', 'AMZN': 'tech',
            'TSLA': 'tech', 'NVDA': 'tech', 'META': 'tech', 'NFLX': 'tech', 'CRM': 'tech',
            # Financial (typically correlated)  
            'JPM': 'financial', 'BAC': 'financial', 'WFC': 'financial', 'C': 'financial', 'GS': 'financial',
            # Healthcare
            'JNJ': 'healthcare', 'UNH': 'healthcare', 'PFE': 'healthcare', 'ABBV': 'healthcare',
            # Energy
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy',
        }
        
        candidate_sector = sector_mapping.get(candidate_ticker, 'other')
        same_sector_count = 0
        
        for ticker in existing_positions:
            if sector_mapping.get(ticker, 'other') == candidate_sector and candidate_sector != 'other':
                same_sector_count += 1
        
        # Scale down for same-sector concentration: 1 same-sector = 0.8x, 2+ = 0.5x
        if same_sector_count == 0:
            return 1.0
        elif same_sector_count == 1:
            return 0.8  # 20% reduction
        else:
            return 0.5  # 50% reduction for high sector concentration
            
    def mark_to_market(self, date: str, prices: dict[str, float]) -> float:
        """Update positions with current prices and return NAV."""
        nav = self._compute_nav(prices)
        positions_value = nav - self.cash

        # PHASE 1.7: Update performance scaling based on NAV change
        prev_nav = self.snapshots[-1].nav if self.snapshots else self.starting_capital
        self._update_performance_scaling(nav, prev_nav)

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
