from __future__ import annotations

import argparse
from typing import Sequence

from utils import parse_tickers_from_csvs, fetch_yfinance_data, load_etf_data_from_csvs
from ranking import rank_etfs, display_rankings
from modeling import ETFReturnPredictor
from config import FetchConfig, RankingConfig, ETFReturnModelingConfig


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


def run_etf_modeling_workflow(config: ETFReturnModelingConfig) -> None:
    """Model ETF returns based on historical data and features."""
    etf_price_data = load_etf_data_from_csvs(config.data_dir)
    predictor = ETFReturnPredictor(etf_price_data)
    target = predictor.create_target_variable(months=config.target_return_period_months)
    print("Target variable created. Shape:", target.shape)
    print(f"Overall positive rate: {target.mean().mean():.3f}")

    features = predictor.calculate_features()
    print(f"\nFeatures calculated. Shape: {features.shape}")
    print(f"Number of metrics: {len([c for c in features.columns if c != 'etf'])}")

    print("\nAnalyzing all metrics...")
    summary = predictor.analyze_all_metrics(n_bins=config.n_bins)

    print("\n" + "=" * 80)
    print("TOP 10 PREDICTIVE METRICS (by Information Gain)")
    print("=" * 80)
    print(summary.head(10).to_string(index=False))

    print("\n\nGenerating visualizations for all metrics...")
    predictor.plot_top_metrics(summary=summary)

    print("\n\nSaving probability plots for each metric (with bin ranges and KL)")
    predictor.plot_metric_probabilities_for_metrics(summary["metric"].tolist())


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
    parser.add_argument(
        "--model-etf-returns",
        action="store_true",
        help="Model ETF returns based on historical data and features.",
    )
    parser.add_argument(
        "--model-plot-dir",
        default="results/plots",
        help="Directory where modeling plots should be saved.",
    )
    parser.add_argument(
        "--model-target-months",
        type=int,
        default=6,
        help="Number of months ahead for the target return variable.",
    )
    parser.add_argument(
        "--model-bins",
        type=int,
        default=5,
        help="Number of bins to use when discretizing metrics during modeling.",
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
    etf_returns_modeling_config = ETFReturnModelingConfig(
        data_dir=args.etf_dir,
        plot_dir=args.model_plot_dir,
        target_return_period_months=args.model_target_months,
        n_bins=args.model_bins,
    )

    if args.fetch_data:
        run_fetch_workflow(fetch_config)
    if args.rank_etfs:
        run_ranking_workflow(ranking_config)
    if args.model_etf_returns:
        run_etf_modeling_workflow(etf_returns_modeling_config)


if __name__ == "__main__":
    main()
