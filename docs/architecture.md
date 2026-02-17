# System Architecture & Pipeline

This document outlines the software architecture and end-to-end data pipeline of the `screener` codebase. It connects the theoretical concepts defined in [Overview](overview.md) to the actual Python implementation.

## 1. High-Level Overview

The `screener` is a modular CLI application designed to:
1.  **Ingest** historical price data for ETFs.
2.  **Transform** raw prices into technical indicators and statistical metrics.
3.  **Model** the predictive power of these metrics against future returns.
4.  **Rank** ETFs or Metrics based on performance or predictive signal strength.

## 2. Module Structure

The codebase is organized into functional modules with clear separation of concerns:

| Module | Role | Key Components |
| :--- | :--- | :--- |
| **`main.py`** | **Orchestrator** | Entry point, CLI argument parsing (`argparse`), workflow dispatch. |
| **`config.py`** | **Configuration** | Typed DataClasses (`FetchConfig`, `RankingConfig`, `ETFReturnModelingConfig`) defining parameters. |
| **`utils.py`** | **I/O & Ingestion** | `fetch_yfinance_data`, `fetch_fred_data`, `load_etf_data_from_csvs`. Handles Yahoo Finance and FRED APIs. |
| **`metrics.py`** | **Feature Engineering** | `BaseMetric` strategy and concrete implementations (Technical and `MacroMetric`). |
| **`modeling.py`** | **Core Logic** | `ETFReturnPredictor`. Binning, probability estimation, Logistic Regression, and Stepwise selection. |
| **`ranking.py`** | **Reporting** | `ETFRanker`. Generates sorted lists of ETFs or predictive Metrics. |

## 3. End-to-End Pipeline

The system operates via three primary workflows which can be executed independently or sequentially.

### Phase 1: Data Ingestion (`--fetch-data`, `--fetch-macro`)
*   **Goal:** Build a local database of historical ETF prices and macroeconomic indicators.
*   **Process:**
    1.  **ETF Discovery:** Scans input CSVs to extract ticker symbols and queries Yahoo Finance.
    2.  **Macro Fetch:** Queries FRED (Federal Reserve Economic Data) via `pandas_datareader` for series like Treasury yields (T10Y2Y), CPI, or SP500.
    3.  **Storage:** Saves price histories in `data/etfs/` and macro series in `data/macro/`.

### Phase 2: Feature Engineering & Modeling (`--model-etf-returns`)
*   **Goal:** Determine which metrics (Technical and Macro) actually predict future returns.
*   **Process:**
    1.  **Load:** Aggregates ETF CSVs and combines them with macro data (aligned via forward-filling).
    2.  **Target Generation:** Calculates forward returns (default: 6 months) and binarizes them.
    3.  **Feature Calculation:**
        *   Instantiates `BaseMetric` subclasses including `Volatility`, `Momentum`, `Sharpe`, and `MacroMetric`.
        *   `MacroMetric` uses `align_macro` to map low-frequency macro data (monthly/quarterly) to daily ETF price indexes.
    4.  **Statistical Analysis & Modeling:**
        *   **Enumeration Mode (Default):** Discretizes metrics into quantile bins and estimates $P(Return > 0 | Bin)$ using Wilson Score Intervals.
        *   **Logistic Regression Mode:** Trains a binary classifier on the full feature set, evaluates train/test/full-fit diagnostics, and saves coefficient-based interpretability artifacts.
        *   **Stepwise Mode (Forward Selection):** Greedily adds features to a logistic model to maximize ROC-AUC.
    5.  **Output:** 
        *   **Enumeration artifacts:** `results/experiment_summary.txt`, `results/plots/*probability*.png`, and `results/*_bin_details.txt`.
        *   **Logistic artifacts:** `results/logistic_experiment_summary.txt`, `results/logistic_predictions.csv`, `results/logistic_feature_importance.csv`.
        *   **Logistic diagnostic plots:** `results/plots/logistic_roc_curve.png`, `results/plots/logistic_probability_distribution.png`, `results/plots/logistic_top_feature_importance.png`.

### Phase 3: Ranking (`--rank-etfs`, `--rank-predictive-metrics`)
*   **Goal:** Provide actionable lists of ETFs or Metrics.
*   **Modes:**
    *   **Descriptive Ranking:** Sorts ETFs by realized performance (Sharpe, YTD Return, etc.).
    *   **Predictive Ranking:** Ranks *metrics* by their Information Gain (KL Divergence) as calculated by the modeling engine.

## 4. Key Architectural Patterns

### The Metric Strategy Pattern (`metrics.py`)
To allow easy extension of the feature set, the system uses a Strategy pattern. All metrics inherit from `BaseMetric` and must implement:

```python
def compute(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
    ...
```

To add a new indicator (e.g., RSI), one simply subclasses `BaseMetric` and adds it to the `get_metrics` factory list.

### The Predictor Engine (`modeling.py`)
The `ETFReturnPredictor` class encapsulates the theoretical framework. It decouples the *calculation* of features from the *evaluation* of their quality.
*   **Input:** Raw Price Data.
*   **Operations:** Feature Engineering -> Target Alignment -> Statistical Testing.
*   **Output:** `RankedMetric` objects containing Information Gain, p-values, and probability distributions.

## 5. Directory Structure (Runtime)

```
project_root/
├── data/
│   ├── etfs/            # Individual ticker CSVs (fetched data)
│   └── ...
├── results/
│   ├── plots/           # Enumeration + logistic diagnostic plots (PNGs)
│   ├── experiment_summary.txt
│   ├── logistic_experiment_summary.txt
│   ├── logistic_predictions.csv
│   └── logistic_feature_importance.csv
├── screener/            # Source code
├── docs/                # Documentation
└── etf_rankings.csv     # Output of the ranking workflow
```
