import pandas as pd
import glob
import os
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import pandas_datareader.data as web

MAJOR_EVENT_DATE_RANGES: Sequence[tuple[str, str, str]] = (
    ("2001-09-11", "2001-09-21", "9/11 market disruption"),
    ("2007-07-01", "2009-06-30", "subprime crisis"),
    ("2020-03-01", "2020-05-31", "COVID peak"),
)


def filter_dataframe_by_date_ranges(
    df: pd.DataFrame,
    date_col: str,
    date_ranges: Sequence[tuple[str, str, str]],
) -> tuple[pd.DataFrame, int]:
    """Drop rows where date_col falls within any inclusive date range."""
    if df.empty:
        return df.copy(), 0

    dates = pd.to_datetime(df[date_col], utc=True)
    keep_mask = pd.Series(True, index=df.index)

    for start, end, _label in date_ranges:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        keep_mask &= ~((dates >= start_ts) & (dates <= end_ts))

    dropped = int((~keep_mask).sum())
    return df.loc[keep_mask].copy(), dropped


def filter_index_by_date_ranges(
    df: pd.DataFrame,
    date_ranges: Sequence[tuple[str, str, str]],
) -> tuple[pd.DataFrame, int]:
    """Drop rows where index falls within any inclusive date range."""
    if df.empty:
        return df.copy(), 0

    dates = pd.to_datetime(df.index, utc=True)
    keep_mask = pd.Series(True, index=df.index)

    for start, end, _label in date_ranges:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        keep_mask &= ~((dates >= start_ts) & (dates <= end_ts))

    dropped = int((~keep_mask).sum())
    return df.loc[keep_mask].copy(), dropped


def _collect_csv_files(source: str) -> list[str]:
    """Return CSV paths from a file, directory, or glob pattern."""
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(p) for p in sorted(path.glob("*.csv"))]
    return sorted(glob.glob(source))


def parse_tickers_from_csvs(file_pattern="*.csv"):
    """
    Parse ticker names from CSV files.

    Parameters:
    file_pattern (str): File path, directory, or glob pattern for CSV files.

    Returns:
    list: List of unique ticker names
    """
    all_tickers = []
    csv_files = _collect_csv_files(file_pattern)

    if not csv_files:
        print(f"No CSV files found matching: {file_pattern}")
        return []

    print(f"Found {len(csv_files)} CSV file(s)")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            ticker_col = None
            for col in df.columns:
                if "ticker" in col.lower():
                    ticker_col = col
                    break

            if ticker_col:
                tickers = df[ticker_col].dropna().str.strip().tolist()
                all_tickers.extend(tickers)
                print(f"  {file}: {len(tickers)} ticker(s) found")
            else:
                print(f"  {file}: No 'Ticker' column found")

        except Exception as e:
            print(f"  Error processing {file}: {e}")

    unique_tickers = sorted(list(set(all_tickers)))

    print(f"\nTotal unique tickers: {len(unique_tickers)}")

    return unique_tickers


def fetch_yfinance_data(tickers, output_dir="data"):
    """
    Fetch all available data from yfinance for given tickers.

    Parameters:
    tickers (list): List of ticker symbols
    output_dir (str): Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tickers = sorted(list(set([t.split()[0].strip().upper() for t in tickers])))
    existing_files = {p.stem.upper() for p in Path(output_dir).glob("*.csv")}
    tickers_to_fetch = [t for t in clean_tickers if t not in existing_files]

    skipped_count = len(clean_tickers) - len(tickers_to_fetch)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} tickers already present in '{output_dir}'")

    if not tickers_to_fetch:
        print("No new tickers to fetch.")
        return pd.DataFrame(), {}

    all_info = []
    all_history = {}

    print(f"\nFetching data for {len(tickers_to_fetch)} tickers...")

    for i, ticker in enumerate(tickers_to_fetch, 1):
        print(f"[{i}/{len(tickers_to_fetch)}] Fetching {ticker}...", end=" ")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            info["ticker"] = ticker
            all_info.append(info)
            history = stock.history(period="max")
            if not history.empty:
                history["ticker"] = ticker
                all_history[ticker] = history

            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            all_info.append({"ticker": ticker, "error": str(e)})

    if all_info:
        info_df = pd.DataFrame(all_info)
        info_file = os.path.join(output_dir, f"ticker_info_{timestamp}.csv")
        info_df.to_csv(info_file, index=False)
        print(f"\n✓ Ticker info saved to: {info_file}")

    if all_history:
        for ticker, hist in all_history.items():
            ticker_file = os.path.join(output_dir, f"{ticker}.csv")
            hist.to_csv(ticker_file)

        print(f"✓ Individual histories saved to: {output_dir}/")

    print("\n✓ All data fetched successfully!")

    return info_df, all_history


def load_etf_data_from_csvs(
    data_dir: str, exclude_major_event_dates: bool = False
) -> pd.DataFrame:
    """
    Load ETF data from multiple CSV files and create price DataFrame.

    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files with ETF data

    Returns:
    --------
    pd.DataFrame with DateTimeIndex and ETF names as columns
    """

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    price_data = {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date")
            df.set_index("Date", inplace=True)
            etf_name = csv_file.stem
            if "Close" in df.columns:
                price_data[etf_name] = df["Close"]
            else:
                print(f"Warning: 'Close' column not found in {csv_file}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    if not price_data:
        print("No valid price data loaded")
        return pd.DataFrame()

    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.ffill()
    prices_df = prices_df.dropna(how="all")

    if exclude_major_event_dates:
        prices_df, dropped_rows = filter_index_by_date_ranges(
            prices_df, MAJOR_EVENT_DATE_RANGES
        )
        print(
            f"Excluded {dropped_rows} date row(s) across major event windows: "
            f"{', '.join(f'{label} ({start} to {end})' for start, end, label in MAJOR_EVENT_DATE_RANGES)}"
        )

    print(f"Loaded {len(prices_df.columns)} ETFs with {len(prices_df)} dates")
    print(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    print(f"ETFs: {', '.join(prices_df.columns.tolist())}")

    return prices_df


def fetch_fred_data(series_ids: List[str], output_dir: str = "data/macro") -> None:
    """
    Fetch macro data from FRED and save to CSV.

    Parameters:
    -----------
    series_ids : List[str]
        List of FRED series IDs (e.g., 'T10Y2Y', 'CPIAUCSL')
    output_dir : str
        Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFetching {len(series_ids)} series from FRED...")

    for series_id in series_ids:
        print(f"  Fetching {series_id}...", end=" ")
        try:
            # We fetch max period available
            df = web.DataReader(series_id, "fred", start="1970-01-01")
            if not df.empty:
                output_file = Path(output_dir) / f"{series_id}.csv"
                df.to_csv(output_file)
                print("✓")
            else:
                print("✗ (Empty response)")
        except Exception as e:
            print(f"✗ Error: {e}")


def load_macro_data(macro_dir: str = "data/macro") -> pd.DataFrame:
    """
    Load all macro CSVs from a directory and combine them.

    Returns:
    --------
    pd.DataFrame with DateTimeIndex
    """
    macro_path = Path(macro_dir)
    csv_files = list(macro_path.glob("*.csv"))

    if not csv_files:
        return pd.DataFrame()

    macro_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["DATE"] = pd.to_datetime(df["DATE"])
            df.set_index("DATE", inplace=True)
            macro_dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not macro_dfs:
        return pd.DataFrame()

    combined = pd.concat(macro_dfs, axis=1)
    combined = combined.sort_index()
    # ffill macro data as it usually has lower frequency than daily
    combined = combined.ffill()

    return combined
