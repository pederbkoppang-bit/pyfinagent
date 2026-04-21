"""phase-4.9 step 4.9.4 Gauntlet regime catalog package."""
from backend.backtest.gauntlet.regimes import REGIMES, RegimeWindow
from backend.backtest.gauntlet.evaluator import evaluate, DRAWDOWN_RATIO_CAP

__all__ = ["REGIMES", "RegimeWindow", "evaluate", "DRAWDOWN_RATIO_CAP"]
