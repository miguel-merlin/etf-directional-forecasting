import argparse
from typing import Sequence
import pandas as pd
import os

from screener.utils import (
    parse_tickers_from_csvs,
    fetch_yfinance_data,
    load_etf_data_from_csvs,
    fetch_fred_data,
    load_macro_data,
)
from screener.ranking import ETFRanker
from screener.modeling import ETFReturnPredictor
from screener.config import (
    FetchConfig,
    RankingConfig,
    ETFReturnModelingConfig,
    ModelType,
    FetchMacroConfig,
)


def run_fetch_workflow(config: FetchConfig) -> None:
    """Fetch Yahoo Finance data for tickers parsed from CSV sources."""

    tickers = parse_tickers_from_csvs(config.ticker_pattern)
    if not tickers:
        print("No tickers found to process. Fetch skipped.")
        return

    fetch_yfinance_data(tickers, output_dir=config.output_dir)


def run_fetch_macro_workflow(config: FetchMacroConfig) -> None:
    """Fetch macro data from FRED."""
    fetch_fred_data(list(config.series_ids), output_dir=config.output_dir)


def run_ranking_workflow(config: RankingConfig) -> None:
    """Rank ETFs stored under the configured directory."""
    ranker = ETFRanker(config.data_dir)
    rankings = pd.DataFrame()
    if config.rank_predictive_metrics:
        predictor = ETFReturnPredictor(load_etf_data_from_csvs(config.data_dir))
        rankings = ETFRanker.rank_predictive_metrics(predictor)
    else:
        rankings = ranker.rank()

    if rankings.empty:
        return

    for metric in config.metrics_to_display:
        print(f"\n=== Rankings for {metric} ===")
        ranker.display(rankings, metric)

    rankings.to_csv(config.output_file, index_label="Rank")
    print(f"Rankings saved to '{config.output_file}'")


def run_etf_modeling_workflow(config: ETFReturnModelingConfig) -> None:
    """Model ETF returns based on historical data and features."""
    etf_price_data = load_etf_data_from_csvs(config.data_dir)
    predictor = ETFReturnPredictor(
        etf_price_data, results_dir=config.results_dir, model_type=config.model.value
    )
    target = predictor.create_target_variable(months=config.target_return_period_months)
    print("Target variable created. Shape:", target.shape)
    print(f"Overall positive rate: {target.mean().mean():.3f}")

    # Load macro data if available
    macro_data = pd.DataFrame()
    if os.path.exists(config.macro_dir):
        print(f"Loading macro data from {config.macro_dir}...")
        macro_data = load_macro_data(config.macro_dir)
        print(f"Macro data loaded. Columns: {', '.join(macro_data.columns)}")

    features = predictor.calculate_features(macro_data=macro_data)
    print(f"\nFeatures calculated. Shape: {features.shape}")
    print(f"Number of metrics: {len([c for c in features.columns if c != 'etf'])}")

    predictor.model_etf_returns()

    if config.model == ModelType.ENUMERATION:
        summary = predictor.metric_summary
        print("\n" + "=" * 80)
        print("TOP 10 PREDICTIVE METRICS (by Information Gain)")
        print("=" * 80)

        print("\n\nGenerating visualizations for all metrics...")
        predictor.plot_top_metrics(summary=summary)

        print("\n\nSaving probability plots for each metric (with bin ranges and KL)")
        predictor.plot_metric_probabilities_for_metrics(summary["metric"].tolist())
    elif config.model == ModelType.LOGISTIC:
        print(
            "\nLogistic Regression modeling complete. No specific plots generated for individual features."
        )
    elif config.model == ModelType.STEPWISE:
        print(
            "\nStepwise Feature Selection complete. Check the output above for selected features and AUC improvement."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF Screener features")

    ## Fetching arguments
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Parse tickers from CSVs and fetch their Yahoo Finance histories.",
    )
    parser.add_argument(
        "--ticker-pattern",
        default="data/*.csv",
        help="File path, directory, or glob for CSVs containing a 'ticker' column.",
    )
    parser.add_argument(
        "--fetch-output-dir",
        default="data",
        help="Directory where fetched info and history files should be saved.",
    )

    ## Macro arguments
    parser.add_argument(
        "--fetch-macro",
        action="store_true",
        help="Fetch macro data from FRED.",
    )
    parser.add_argument(
        "--fred-series",
        nargs="+",
        default=["T10Y2Y", "CPIAUCSL", "GS10", "SP500"],
        help="List of FRED series IDs to fetch.",
    )
    parser.add_argument(
        "--macro-dir",
        default="data/macro",
        help="Directory where macro data should be saved/loaded.",
    )

    ## Ranking arguments
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
        "--rank-predictive-metrics",
        action="store_true",
        help="Rank ETFs using predictive metrics from the modeling module.",
    )

    ## Modeling arguments
    parser.add_argument(
        "--model-etf-returns",
        action="store_true",
        help="Model ETF returns based on historical data and features.",
    )
    parser.add_argument(
        "--model-results-dir",
        default="results",
        help="Base directory where modeling results (plots and bin details) should be saved.",
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
    parser.add_argument(
        "--model-type",
        choices=[mt.value for mt in ModelType],
        default=ModelType.ENUMERATION.value,
        help="Modeling approach to use when estimating ETF returns.",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    fetch_config = FetchConfig(
        ticker_pattern=args.ticker_pattern,
        output_dir=args.fetch_output_dir,
    )
    fetch_macro_config = FetchMacroConfig(
        series_ids=tuple(args.fred_series),
        output_dir=args.macro_dir,
    )
    ranking_config = RankingConfig(
        data_dir=args.etf_dir,
        output_file=args.rankings_output,
        metrics_to_display=tuple(args.display_metrics),
        rank_predictive_metrics=args.rank_predictive_metrics,
    )
    etf_returns_modeling_config = ETFReturnModelingConfig(
        data_dir=args.etf_dir,
        macro_dir=args.macro_dir,
        results_dir=args.model_results_dir,
        target_return_period_months=args.model_target_months,
        n_bins=args.model_bins,
        model=ModelType(args.model_type),
    )

    if args.fetch_data:
        run_fetch_workflow(fetch_config)
    if args.fetch_macro:
        run_fetch_macro_workflow(fetch_macro_config)
    if args.rank_etfs:
        run_ranking_workflow(ranking_config)
    if args.model_etf_returns:
        run_etf_modeling_workflow(etf_returns_modeling_config)


if __name__ == "__main__":
    main()
