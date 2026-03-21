"""
Walk-forward scheduler — generates expanding-window train/test splits
with embargo periods to prevent autocorrelation leakage (López de Prado Ch. 7).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


@dataclass
class WalkForwardWindow:
    """A single train/test split in the walk-forward schedule."""
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_days: int


class WalkForwardScheduler:
    """
    Generates expanding-window walk-forward splits with embargo.

    Expanding window: train_start is always the same (earliest date),
    but train_end advances each iteration (more training data over time).
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        train_window_months: int = 12,
        test_window_months: int = 3,
        embargo_days: int = 5,
    ):
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.embargo_days = embargo_days

    def generate_windows(self) -> list[WalkForwardWindow]:
        """
        Generate expanding-window walk-forward splits.

        Example (start=2023-01, train=12mo, test=3mo, embargo=5d):
          Window 1: Train [2023-01 → 2023-12] | Embargo 5d | Test [2024-01 → 2024-03]
          Window 2: Train [2023-01 → 2024-03] | Embargo 5d | Test [2024-04 → 2024-06]
          ...
        """
        windows = []
        window_id = 1

        # First train window end
        train_end = self.start_date + relativedelta(months=self.train_window_months) - timedelta(days=1)

        while True:
            # Test window with embargo gap
            test_start = train_end + timedelta(days=self.embargo_days + 1)
            test_end = test_start + relativedelta(months=self.test_window_months) - timedelta(days=1)

            # Don't create a partial test window that extends beyond end_date
            if test_start > self.end_date:
                break

            # Clip test_end to end_date if needed
            if test_end > self.end_date:
                test_end = self.end_date

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=self.start_date,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=self.embargo_days,
            ))

            # Advance: train_end expands to include the previous test period
            train_end = test_end
            window_id += 1

            # Safety: prevent infinite loop
            if window_id > 50:
                break

        return windows
