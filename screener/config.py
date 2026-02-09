from dataclasses import dataclass
from typing import Sequence
from enum import Enum


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
    rank_predictive_metrics: bool = False


class ModelType(Enum):
    """Model types for ETF return modeling."""

    ENUMERATION = "enumeration"
    LOGISTIC = "logistic"


@dataclass
class ETFReturnModelingConfig:
    """Configuration for modelling ETF returns."""

    data_dir: str = "data/etfs"
    results_dir: str = "results"
    target_return_period_months: int = 6
    n_bins: int = 5
    model: ModelType = ModelType.ENUMERATION
