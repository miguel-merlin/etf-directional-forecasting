from utils import parse_tickers_from_csvs, fetch_yfinance_data
import pandas as pd

if __name__ == "__main__":
    tickers = parse_tickers_from_csvs("data/*.csv")

    if tickers:
        print("\nTicker symbols extracted:")
        for ticker in tickers:
            print(f"  {ticker}")
        info_df, history_data = fetch_yfinance_data(tickers, output_dir="data")

        print(f"\n✓ Process complete! Check the 'data' directory for output files.")
    else:
        print("No tickers found to process.")
