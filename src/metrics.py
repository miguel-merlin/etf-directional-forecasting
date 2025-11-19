import pandas as pd
import numpy as np
from typing import List
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for all metric calculators."""

    @abstractmethod
    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        """
        Compute features for a single ETF (one price/return series).

        Returns:
        --------
        pd.DataFrame
            DataFrame indexed by dates with one or more feature columns.
        """
        pass


class VolatilityMetric(BaseMetric):
    """Historical volatility metrics (annualized)."""

    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.lookback_periods:
            df[f"vol_{period}d"] = returns.rolling(period).std() * np.sqrt(252)
        return df


class ReturnMetric(BaseMetric):
    """Rolling simple return over different horizons."""

    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.lookback_periods:
            df[f"ret_{period}d"] = prices.pct_change(period)
        return df


class SharpeMetric(BaseMetric):
    """Rolling Sharpe ratio (0% risk-free)."""

    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.lookback_periods:
            mean_ret = returns.rolling(period).mean() * 252
            vol = returns.rolling(period).std() * np.sqrt(252)
            df[f"sharpe_{period}d"] = mean_ret / vol
        return df


class MaxDrawdownMetric(BaseMetric):
    """Rolling maximum drawdown over different horizons."""

    def __init__(self, lookback_periods: List[int]):
        self.lookback_periods = lookback_periods

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.lookback_periods:
            rolling_max = prices.rolling(period, min_periods=1).max()
            drawdown = (prices - rolling_max) / rolling_max
            df[f"max_dd_{period}d"] = drawdown.rolling(period).min()
        return df


class MomentumMetric(BaseMetric):
    """12-1 month momentum."""

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        df["mom_12_1"] = prices.pct_change(252) - prices.pct_change(21)
        return df


class MovingAverageMetric(BaseMetric):
    """Moving averages and price-to-MA features."""

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        df[f"sma_{self.short_window}"] = prices.rolling(self.short_window).mean()
        df[f"sma_{self.long_window}"] = prices.rolling(self.long_window).mean()
        df["price_to_sma50"] = prices / df[f"sma_{self.short_window}"] - 1
        df["price_to_sma200"] = prices / df[f"sma_{self.long_window}"] - 1
        return df


class VolOfVolMetric(BaseMetric):
    """Volatility of volatility over medium-term windows."""

    def __init__(self, vol_windows: List[int] = [60, 120], volofvol_window: int = 20):
        self.vol_windows = vol_windows
        self.volofvol_window = volofvol_window

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.vol_windows:
            vol = returns.rolling(period).std()
            df[f"vol_of_vol_{period}d"] = vol.rolling(self.volofvol_window).std()
        return df


class HigherMomentMetric(BaseMetric):
    """Rolling skewness and kurtosis."""

    def __init__(self, windows: List[int] = [60, 120]):
        self.windows = windows

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.windows:
            df[f"skew_{period}d"] = returns.rolling(period).skew()
            df[f"kurt_{period}d"] = returns.rolling(period).kurt()
        return df


class UpDownCaptureMetric(BaseMetric):
    """Rolling average of up and down days (up/down capture)."""

    def __init__(self, windows: List[int] = [60, 120]):
        self.windows = windows

    def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=prices.index)
        for period in self.windows:
            up_days = returns.where(returns > 0).rolling(period).mean()
            down_days = returns.where(returns < 0).rolling(period).mean()
            df[f"up_capture_{period}d"] = up_days
            df[f"down_capture_{period}d"] = down_days
        return df


def get_metrics(
    lookback_periods: List[int],
    short_window: int,
    long_window: int,
    vol_windows: List[int],
    volofvol_window: int,
) -> List[BaseMetric]:
    """Return a list of all available metric calculator classes."""
    return [
        VolatilityMetric(lookback_periods),
        ReturnMetric(lookback_periods),
        SharpeMetric(lookback_periods),
        MaxDrawdownMetric(lookback_periods),
        MomentumMetric(),
        MovingAverageMetric(short_window=short_window, long_window=long_window),
        VolOfVolMetric(vol_windows=vol_windows, volofvol_window=volofvol_window),
        HigherMomentMetric(windows=vol_windows),
        UpDownCaptureMetric(windows=vol_windows),
    ]
