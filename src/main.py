from utils import parse_tickers_from_csvs, fetch_yfinance_data
from metrics import rank_etfs, display_rankings

if __name__ == "__main__":
    # tickers = parse_tickers_from_csvs("data/*.csv")

    # if tickers:
    #     print("\nTicker symbols extracted:")
    #     for ticker in tickers:
    #         print(f"  {ticker}")
    #     info_df, history_data = fetch_yfinance_data(tickers, output_dir="data")

    #     print(f"\n✓ Process complete! Check the 'data' directory for output files.")
    # else:
    #     print("No tickers found to process.")

    rankings = rank_etfs("data/etfs")

    if not rankings.empty:
        print("\n=== Top ETFs by Sharpe Ratio ===")
        display_rankings(rankings, "Sharpe_Ratio")

        print("\n=== Top ETFs by Annualized Return ===")
        display_rankings(rankings, "Annualized_Return_%")

        print("\n=== Lowest Volatility ETFs ===")
        display_rankings(rankings, "Volatility_%")

        rankings.to_csv("etf_rankings.csv", index_label="Rank")
        print("Rankings saved to 'etf_rankings.csv'")
