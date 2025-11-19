import pandas as pd
import glob
import os
import yfinance as yf
from datetime import datetime


def parse_tickers_from_csvs(file_pattern="*.csv"):
    """
    Parse ticker names from CSV files.

    Parameters:
    file_pattern (str): Glob pattern to match CSV files (default: '*.csv')

    Returns:
    list: List of unique ticker names
    """
    all_tickers = []
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {file_pattern}")
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

    all_info = []
    all_history = {}

    print(f"\nFetching data for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Fetching {ticker}...", end=" ")
        ticker = ticker.split()[0]

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
        history_dir = os.path.join(output_dir, f"individual_histories_{timestamp}")
        os.makedirs(history_dir, exist_ok=True)

        for ticker, hist in all_history.items():
            ticker_file = os.path.join(history_dir, f"{ticker}.csv")
            hist.to_csv(ticker_file)

        print(f"✓ Individual histories saved to: {history_dir}/")

    print("\n✓ All data fetched successfully!")

    return info_df, all_history
