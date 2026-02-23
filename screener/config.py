from dataclasses import dataclass
from typing import Sequence
from enum import Enum


@dataclass
class FetchConfig:
    """Configuration for fetching ticker data from Yahoo Finance."""

    ticker_pattern: str = "data/*.csv"
    output_dir: str = "data"


@dataclass
class FetchMacroConfig:
    """Configuration for fetching macro data from FRED."""

    series_ids: Sequence[str] = ("T10Y2Y", "CPIAUCSL", "GS10", "SP500")
    output_dir: str = "data/macro"


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
    exclude_major_event_dates: bool = False


class ModelType(Enum):
    """Model types for ETF return modeling."""

    ENUMERATION = "enumeration"
    LOGISTIC = "logistic"
    STEPWISE = "stepwise"


@dataclass
class ETFReturnModelingConfig:
    """Configuration for modelling ETF returns."""

    data_dir: str = "data/etfs"
    macro_dir: str = "data/macro"
    results_dir: str = "results"
    target_return_period_months: int = 6
    n_bins: int = 5
    model: ModelType = ModelType.ENUMERATION
    model_class_weight: str = "none"
    exclude_major_event_dates: bool = False
