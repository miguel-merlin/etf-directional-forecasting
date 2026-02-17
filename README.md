# ETF Screener

ETF screener that consumes locally stored price histories, computes common risk/return ratios, and produces easy-to-scan rankings. Utilities are also available to parse ticker symbols from watchlist CSVs and download their full Yahoo Finance histories so you can keep the inputs fresh.

## Features
- Batch import of ETF histories from `data/etfs/*.csv`
- Macro data integration from FRED (Federal Reserve Economic Data) stored in `data/macro/*.csv`
- Daily-return derived metrics such as annualized return, volatility, Sharpe ratio, max drawdown, and YTD return
- Console rankings plus an `etf_rankings.csv` export for further analysis
- Helper utilities to (a) extract symbols from multiple CSV sources and (b) pull their entire trading history via `yfinance` or FRED
- Model 3- to 12-month ETF return probabilities using Enumeration (binning), Logistic Regression, or Stepwise Forward Selection
- Rank the most predictive metrics (Technical and Macro) based on Information Gain and Chi-Square significance

## Repository layout
| Path | Purpose |
| --- | --- |
| `screener/main.py` | Entry point that orchestrates rankings, fetching, and modeling workflows. |
| `screener/metrics.py` | Contains `BaseMetric` strategy classes for Technical and Macro indicators. |
| `screener/modeling.py` | `ETFReturnPredictor` logic for binning, logistic modeling, and stepwise selection. |
| `screener/ranking.py` | `ETFRanker` for descriptive stats or predictive power rankings. |
| `screener/config.py` | Typed configuration containers for all CLI workflows. |
| `screener/utils.py` | I/O helpers for Yahoo Finance, FRED, and local CSV parsing. |
| `data/` | Data store for `etfs/` price histories and `macro/` indicators. |
| `results/` | Output directory for modeling artifacts (plots, summaries, bin details, logistic diagnostics). |

## Documentation
- [Architecture & Pipeline](docs/architecture.md): Detailed overview of the system design and data flow.
- [Theoretical Framework](docs/overview.md): Explanation of the financial theory and statistical methods used.

## Requirements
Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip3 install -r requirements.txt
```

## Preparing the data
The screener expects a CSV per ETF stored inside `data/etfs/`. Each file should contain, at minimum, the Yahoo Finance OHLCV columns (`Date, Open, High, Low, Close, Adj Close, Volume`). You can obtain those files in two ways:

1. **Provide your own histories.** Drop the CSVs under `data/etfs/` with filenames that match the ticker (for example `SPY.csv`).
2. **Let the helper utilities fetch them.**
   ```python
   from screener.utils import parse_tickers_from_csvs, fetch_yfinance_data

   tickers = parse_tickers_from_csvs("data/screener/*.csv")  # glob for the CSVs you care about
   info_df, histories = fetch_yfinance_data(tickers, output_dir="data")
   ```
   - `parse_tickers_from_csvs` can scan a single CSV (like `data/watchlist.csv`), every CSV in a directory, or any glob pattern and looks for a column whose name contains "ticker".
   - `fetch_yfinance_data` saves a `ticker_info_<timestamp>.csv` file and an `individual_histories_<timestamp>/` folder containing one CSV per symbol. Copy or move the ETFs you want ranked into `data/etfs/`.

## Running the screener
The CLI exposes flags for fetching data, ranking ETFs, and modeling returns.

- `python -m screener.main --fetch-data` – parse tickers via `--ticker-pattern` (default `data/*.csv`) and save downloads under `--fetch-output-dir` (default `data`).
- `python -m screener.main --fetch-macro --fred-series T10Y2Y CPIAUCSL` – fetch specified series from FRED and save to `data/macro`.
- `python -m screener.main --rank-etfs` – calculate performance metrics for ETFs in `data/etfs`.
- `python -m screener.main --rank-predictive-metrics` – use the modeling engine to rank all available metrics by their Information Gain.
- `python -m screener.main --rank-etfs --display-metrics Sharpe_Ratio Annualized_Return_%` – pass custom metrics to display.

## Modeling ETF returns
The modeling workflow estimates the probability of positive forward returns based on technical and macro indicators.

```bash
python3 -m screener.main --model-etf-returns \
  --etf-dir data/etfs \
  --macro-dir data/macro \
  --model-target-months 6 \
  --model-type stepwise
```

**Model Types:**
- `enumeration` (default) – Quantile-based binning to estimate conditional probabilities $P(Return > 0 | Bin)$.
- `logistic` – Trains a Logistic Regression classifier on all available features and saves diagnostics/plots.
- `stepwise` – Performs forward feature selection to find the most predictive subset of indicators using ROC-AUC.

**What you get (`enumeration`):**
- **Probability Plots:** Visualizations in `results/plots/` showing the relationship between metric bins and return probabilities.
- **Experiment Summary:** A `results/experiment_summary.txt` file containing metadata, analyzed variables, and top performing factors.
- **Bin Details:** Individual `.txt` files in `results/` for each metric detailing the probability distribution across bins.
- **Information Gain (IG):** A ranking of metrics based on how much they reduce uncertainty about future returns.

**What you get (`logistic`):**
- **Logistic Summary:** `results/logistic_experiment_summary.txt` with train/test/full-fit metrics (ROC-AUC, AP, Brier, accuracy, precision, recall, F1) and top features.
- **Row-Level Predictions:** `results/logistic_predictions.csv` with `date`, `etf`, `target`, `split`, `evaluation_probability`, and `full_model_probability`.
- **Feature Importance:** `results/logistic_feature_importance.csv` with coefficients, standardized coefficients, absolute standardized effect, and odds ratios.
- **Diagnostic Plots:** `results/plots/logistic_roc_curve.png`, `results/plots/logistic_probability_distribution.png`, and `results/plots/logistic_top_feature_importance.png`.

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
- Change `data/etfs` inside `screener/main.py` if your histories live elsewhere.
- Adjust the risk-free rate in `screener/metrics.py` to match your assumptions.
- Use `--model-target-months` and `--model-results-dir` to tune the return-modeling workflow.
- Extend `ETFRanker.display` if you need additional printouts or prefer different sort orders.

## Troubleshooting
- **`No CSV files found`** – Ensure the directory exists and the files carry the `.csv` extension.
- **`Error processing <file>`** – Open the file and verify it contains numeric price columns with ISO-formatted dates.
- **Inconsistent metrics** – Make sure every CSV has the same currency and no duplicate dates; consider re-downloading via the helper utilities.
