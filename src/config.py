from dataclasses import dataclass
from typing import Sequence


@dataclass
class FetchConfig:
    """Configuration for fetching ticker data from Yahoo Finance."""

    ticker_pattern: str = "data/*.csv"
    output_dir: str = "data"


@dataclass
class RankingConfig:
    """Configuration for ranking stored ETF CSV files."""

    data_dir: str = "data/etfs"
    output_file: str = "etf_rankings.csv"
    metrics_to_display: Sequence[str] = (
        "Sharpe_Ratio",
        "Annualized_Return_%",
        "Volatility_%",
    )


@dataclass
class ETFReturnModelingConfig:
    """Configuration for modelling ETF returns."""

    data_dir: str = "data/etfs"
    plot_dir: str = "results/plots"
    target_return_period_months: int = 6
