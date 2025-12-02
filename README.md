# ETF Screener

ETF screener that consumes locally stored price histories, computes common risk/return ratios, and produces easy-to-scan rankings. Utilities are also available to parse ticker symbols from watchlist CSVs and download their full Yahoo Finance histories so you can keep the inputs fresh.

## Features
- Batch import of ETF histories from `data/etfs/*.csv`
- Daily-return derived metrics such as annualized return, volatility, Sharpe ratio, max drawdown, and YTD return
- Console rankings plus an `etf_rankings.csv` export for further analysis
- Helper utilities to (a) extract symbols from multiple CSV sources and (b) pull their entire trading history via `yfinance`
- Model 3- to 12-month ETF return probabilities, rank the most predictive metrics, and auto-generate probability plots

## Repository layout
| Path | Purpose |
| --- | --- |
| `src/main.py` | Entry point that ranks ETFs stored under `data/etfs`. |
| `src/metrics.py` | Calculates portfolio statistics and prints the rankings table. |
| `src/modeling.py` | Builds signal features, models forward returns, and plots conditional probabilities. |
| `src/config.py` | Typed configuration containers shared across the CLI workflows. |
| `src/utils.py` | Helper functions for parsing tickers and downloading new price histories. |
| `data/` | Sample output plus your downloadable price histories. Each ETF should be its own CSV inside `data/etfs/`. |

## Requirements
Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Preparing the data
The screener expects a CSV per ETF stored inside `data/etfs/`. Each file should contain, at minimum, the Yahoo Finance OHLCV columns (`Date, Open, High, Low, Close, Adj Close, Volume`). You can obtain those files in two ways:

1. **Provide your own histories.** Drop the CSVs under `data/etfs/` with filenames that match the ticker (for example `SPY.csv`).
2. **Let the helper utilities fetch them.**
   ```python
   from src.utils import parse_tickers_from_csvs, fetch_yfinance_data

   tickers = parse_tickers_from_csvs("data/screener/*.csv")  # glob for the CSVs you care about
   info_df, histories = fetch_yfinance_data(tickers, output_dir="data")
   ```
   - `parse_tickers_from_csvs` can scan a single CSV (like `data/watchlist.csv`), every CSV in a directory, or any glob pattern and looks for a column whose name contains "ticker".
   - `fetch_yfinance_data` saves a `ticker_info_<timestamp>.csv` file and an `individual_histories_<timestamp>/` folder containing one CSV per symbol. Copy or move the ETFs you want ranked into `data/etfs/`.

## Running the screener
The CLI exposes one flag per feature so you can fetch data, rank ETFs, or do both in a single command.

- `python -m src.main --rank-etfs` – default behavior; omitting the flag does the same.
- `python -m src.main --fetch-data` – parse tickers via `--ticker-pattern` (default `data/*.csv`) and save downloads under `--fetch-output-dir` (default `data`). The pattern now accepts direct CSV paths (e.g., `data/watchlist.csv`) in addition to directories or globs, so you can run the fetcher on a single watchlist file.
- `python -m src.main --rank-etfs --display-metrics Sharpe_Ratio Annualized_Return_%` – pass custom metrics to print, and `--rankings-output`/`--etf-dir` to change file locations.
- `python -m src.main --model-etf-returns` – compute predictive features, evaluate their power against a forward return target, and save plots to `--model-plot-dir` (default `results/plots`).
- Combine both flags to fetch data and immediately rank the refreshed histories.

When ranking runs it will:
1. Read every CSV in `data/etfs/` (or the directory you supplied).
2. Compute the metrics defined in `src/metrics.py`.
3. Print ranked tables for each metric you requested.
4. Export the consolidated results to `etf_rankings.csv` by default.

If no files are found you will see an error message such as `No CSV files found in data/etfs`.

## Modeling ETF returns
The modeling workflow estimates the probability that each ETF posts a positive forward return (6 months ahead by default) based on the technical metrics produced in `src/modeling.py`.

```bash
python -m src.main --model-etf-returns \
  --etf-dir data/etfs \
  --model-target-months 6 \
  --model-plot-dir results/plots
```

What you get:
- Console summary of the top predictive metrics ranked by information gain, chi-square statistics, and sample counts.
- A grid of plots for the top metrics plus a detailed chart for the best one, stored in `results/plots/`.
- Saved probability tables (per metric) inside `ETFReturnPredictor.results` while the process is running, so you can reuse them inside notebooks if needed.

Adjust `--model-target-months` to change the look-ahead window (e.g., 3, 6, or 12 months). Set `--model-plot-dir` if you want the PNGs to live elsewhere.

## Metrics explained
All returns assume trading days and use percentages unless noted otherwise.

- **Total_Return_%** – Cumulative return from the first to the last available price.
- **Annualized_Return_%** – Compound annual growth rate computed from the same period.
- **Volatility_%** – Annualized standard deviation of daily returns (252 trading days).
- **Sharpe_Ratio** – `(annualized_return - 4% risk free) / volatility`.
- **Max_Drawdown_%** – Maximum peak-to-trough decline based on cumulative returns.
- **YTD_Return_%** – Return between the first and last trading day within the current calendar year.
- **Current_Price** – Last close in the CSV.
- **Data_Points** – Total number of records processed for that ETF.

## Customizing
- Change `data/etfs` inside `src/main.py` if your histories live elsewhere.
- Adjust the risk-free rate in `src/metrics.py` to match your assumptions.
- Extend `display_rankings` if you need additional printouts or prefer different sort orders.
- Use `--model-target-months` and `--model-plot-dir` to tune the return-modeling workflow without touching the source.

## Troubleshooting
- **`No CSV files found`** – Ensure the directory exists and the files carry the `.csv` extension.
- **`Error processing <file>`** – Open the file and verify it contains numeric price columns with ISO-formatted dates.
- **Inconsistent metrics** – Make sure every CSV has the same currency and no duplicate dates; consider re-downloading via the helper utilities.
