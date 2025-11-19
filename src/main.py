from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

from utils import parse_tickers_from_csvs, fetch_yfinance_data
from metrics import rank_etfs, display_rankings


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
    metrics_to_display: Sequence[str] = ("Sharpe_Ratio", "Annualized_Return_%", "Volatility_%")


def run_fetch_workflow(config: FetchConfig) -> None:
    """Fetch Yahoo Finance data for tickers parsed from CSV sources."""

    tickers = parse_tickers_from_csvs(config.ticker_pattern)
    if not tickers:
        print("No tickers found to process. Fetch skipped.")
        return

    fetch_yfinance_data(tickers, output_dir=config.output_dir)


def run_ranking_workflow(config: RankingConfig) -> None:
    """Rank ETFs stored under the configured directory."""

    rankings = rank_etfs(config.data_dir)

    if rankings.empty:
        return

    for metric in config.metrics_to_display:
        print(f"\n=== Rankings for {metric} ===")
        display_rankings(rankings, metric)

    rankings.to_csv(config.output_file, index_label="Rank")
    print(f"Rankings saved to '{config.output_file}'")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF Screener features")
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Parse tickers from CSVs and fetch their Yahoo Finance histories.",
    )
    parser.add_argument(
        "--ticker-pattern",
        default="data/*.csv",
        help="Glob pattern for CSV files that contain a 'ticker' column.",
    )
    parser.add_argument(
        "--fetch-output-dir",
        default="data",
        help="Directory where fetched info and history files should be saved.",
    )
    parser.add_argument(
        "--rank-etfs",
        action="store_true",
        help="Rank ETFs stored as CSV files.",
    )
    parser.add_argument(
        "--etf-dir",
        default="data/etfs",
        help="Directory that contains the ETF CSV files to rank.",
    )
    parser.add_argument(
        "--rankings-output",
        default="etf_rankings.csv",
        help="Path for the consolidated rankings CSV output.",
    )
    parser.add_argument(
        "--display-metrics",
        nargs="+",
        default=["Sharpe_Ratio", "Annualized_Return_%", "Volatility_%"],
        help="List of metrics to display when showing rankings.",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    fetch_config = FetchConfig(
        ticker_pattern=args.ticker_pattern,
        output_dir=args.fetch_output_dir,
    )
    ranking_config = RankingConfig(
        data_dir=args.etf_dir,
        output_file=args.rankings_output,
        metrics_to_display=tuple(args.display_metrics),
    )

    requested_feature = args.fetch_data or args.rank_etfs

    if args.fetch_data:
        run_fetch_workflow(fetch_config)

    if args.rank_etfs or not requested_feature:
        run_ranking_workflow(ranking_config)


if __name__ == "__main__":
    main()
