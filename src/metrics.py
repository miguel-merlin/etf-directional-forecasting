import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def calculate_metrics(df: pd.DataFrame, ticker: str) -> Dict:
    """Calculate performance metrics for a single ETF."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["Daily_Return"] = df["Close"].pct_change()
    total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    years = days / 365.25
    annualized_return = (
        ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / years) - 1) * 100
        if years > 0
        else 0
    )

    # Volatility (annualized standard deviation)
    volatility = df["Daily_Return"].std() * np.sqrt(252) * 100

    # Sharpe ratio (assuming risk-free rate of 4%)
    risk_free_rate = 4.0
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    )

    # Maximum drawdown
    cumulative = (1 + df["Daily_Return"]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # YTD return
    current_year = df["Date"].iloc[-1].year
    ytd_data = df[df["Date"].dt.year == current_year]
    if len(ytd_data) > 0:
        ytd_return = (ytd_data["Close"].iloc[-1] / ytd_data["Close"].iloc[0] - 1) * 100
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


def rank_etfs(data_dir: str = "data") -> pd.DataFrame:
    """Rank all ETFs in the directory based on performance metrics."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    results = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            ticker = csv_file.stem
            metrics = calculate_metrics(df, ticker)
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    if not results:
        print("No ETFs were successfully processed")
        return pd.DataFrame()

    rankings_df = pd.DataFrame(results)

    rankings_df = rankings_df.sort_values("Sharpe_Ratio", ascending=False).reset_index(
        drop=True
    )
    rankings_df.index = rankings_df.index + 1

    return rankings_df


def display_rankings(rankings_df: pd.DataFrame, metric: str = "Sharpe_Ratio"):
    """Display rankings sorted by a specific metric."""
    valid_metrics = [
        col for col in rankings_df.columns if col != "Ticker" and col != "Data_Points"
    ]

    if metric not in valid_metrics:
        print(f"Invalid metric. Choose from: {valid_metrics}")
        return

    ascending = True if "Drawdown" in metric or "Volatility" in metric else False
    sorted_df = rankings_df.sort_values(metric, ascending=ascending).reset_index(
        drop=True
    )
    sorted_df.index = sorted_df.index + 1

    print(f"\n{'='*80}")
    print(f"ETF Rankings by {metric}")
    print(f"{'='*80}\n")
    print(sorted_df.to_string())
    print(f"\n{'='*80}\n")
