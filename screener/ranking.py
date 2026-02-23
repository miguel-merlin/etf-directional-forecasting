from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from screener.modeling import ETFReturnPredictor
from screener.utils import MAJOR_EVENT_DATE_RANGES, filter_dataframe_by_date_ranges


class ETFRanker:
    """Ranks ETFs based on performance metrics or predictor outputs."""

    def __init__(self, data_dir: str = "data", exclude_major_event_dates: bool = False):
        self.data_dir = Path(data_dir)
        self.exclude_major_event_dates = exclude_major_event_dates

    @staticmethod
    def calculate_metrics(
        df: pd.DataFrame, ticker: str, exclude_major_event_dates: bool = False
    ) -> Dict:
        """Calculate performance metrics for a single ETF."""
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date")
        if exclude_major_event_dates:
            df, _dropped_rows = filter_dataframe_by_date_ranges(
                df, "Date", MAJOR_EVENT_DATE_RANGES
            )

        if len(df) < 2:
            raise ValueError("Insufficient data points after date exclusions.")

        df["Daily_Return"] = df["Close"].pct_change()
        total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
        years = days / 365.25
        annualized_return = (
            ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / years) - 1) * 100
            if years > 0
            else 0
        )

        volatility = df["Daily_Return"].std() * np.sqrt(252) * 100

        risk_free_rate = 4.0
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        )

        cumulative = (1 + df["Daily_Return"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        current_year = df["Date"].iloc[-1].year
        ytd_data = df[df["Date"].dt.year == current_year]  # type: ignore
        if len(ytd_data) > 0:
            ytd_return = (
                ytd_data["Close"].iloc[-1] / ytd_data["Close"].iloc[0] - 1
            ) * 100
        else:
            ytd_return = 0

        return {
            "Ticker": ticker,
            "Total_Return_%": round(total_return, 2),
            "Annualized_Return_%": round(annualized_return, 2),
            "Volatility_%": round(volatility, 2),
            "Sharpe_Ratio": round(sharpe_ratio, 2),
            "Max_Drawdown_%": round(max_drawdown, 2),
            "YTD_Return_%": round(ytd_return, 2),
            "Current_Price": round(df["Close"].iloc[-1], 2),
            "Data_Points": len(df),
        }

    def rank(self) -> pd.DataFrame:
        """Rank all ETFs in the directory based on performance metrics."""
        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return pd.DataFrame()

        results: List[Dict] = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                ticker = csv_file.stem
                metrics = self.calculate_metrics(
                    df,
                    ticker,
                    exclude_major_event_dates=self.exclude_major_event_dates,
                )
                results.append(metrics)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")

        if not results:
            print("No ETFs were successfully processed")
            return pd.DataFrame()

        rankings_df = pd.DataFrame(results)

        rankings_df = rankings_df.sort_values(
            "Sharpe_Ratio", ascending=False
        ).reset_index(drop=True)
        rankings_df.index = rankings_df.index + 1

        return rankings_df

    @staticmethod
    def display(rankings_df: pd.DataFrame, metric: str = "Sharpe_Ratio"):
        """Display rankings sorted by a specific metric."""
        valid_metrics = [
            col for col in rankings_df.columns if col not in {"Ticker", "Data_Points"}
        ]

        if metric not in valid_metrics:
            print(f"Invalid metric. Choose from: {valid_metrics}")
            return

        ascending = "Drawdown" in metric or "Volatility" in metric
        sorted_df = rankings_df.sort_values(metric, ascending=ascending).reset_index(
            drop=True
        )
        sorted_df.index = sorted_df.index + 1

        print(f"\n{'='*80}")
        print(f"ETF Rankings by {metric}")
        print(f"{'='*80}\n")
        print(sorted_df.to_string())
        print(f"\n{'='*80}\n")

    @staticmethod
    def rank_predictive_metrics(
        predictor: ETFReturnPredictor, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank predictive metrics using the predictor outputs.

        Parameters
        ----------
        predictor : ETFReturnPredictor
            Predictor instance that exposes get_top_metrics.
        top_n : Optional[int]
            Limit the number of ranked metrics.
        """
        ranked_metrics = predictor.get_top_metrics(top_n=top_n)

        if not ranked_metrics:
            return pd.DataFrame()

        rows = []
        for rank, metric in enumerate(ranked_metrics, start=1):
            metric_dict = asdict(metric)
            metric_dict["metric"] = metric_dict.pop("name")
            rows.append({"Rank": rank, **metric_dict})

        rankings_df = pd.DataFrame(rows)
        return rankings_df


def calculate_metrics(df: pd.DataFrame, ticker: str) -> Dict:
    return ETFRanker.calculate_metrics(df, ticker)


def rank_etfs(
    data_dir: str = "data", exclude_major_event_dates: bool = False
) -> pd.DataFrame:
    return ETFRanker(
        data_dir, exclude_major_event_dates=exclude_major_event_dates
    ).rank()


def display_rankings(rankings_df: pd.DataFrame, metric: str = "Sharpe_Ratio"):
    ETFRanker.display(rankings_df, metric)
