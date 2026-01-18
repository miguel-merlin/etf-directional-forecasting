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
| **`utils.py`** | **I/O & Ingestion** | `fetch_yfinance_data`, `load_etf_data_from_csvs`. Handles interaction with the filesystem and Yahoo Finance API. |
| **`metrics.py`** | **Feature Engineering** | `BaseMetric` (Abstract Class) and concrete implementations (`VolatilityMetric`, `SharpeMetric`, etc.). |
| **`modeling.py`** | **Core Logic** | `ETFReturnPredictor`. Implements the binning, conditional probability estimation, and statistical testing (Information Gain). |
| **`ranking.py`** | **Reporting** | `ETFRanker`. Generates sorted lists of ETFs based on descriptive stats or metrics based on predictive power. |

## 3. End-to-End Pipeline

The system operates via three primary workflows which can be executed independently or sequentially.

### Phase 1: Data Ingestion (`--fetch-data`)
*   **Goal:** Build a local database of historical ETF prices.
*   **Process:**
    1.  **Discovery:** Scans input CSVs (defined by `--ticker-pattern`) to extract unique ticker symbols.
    2.  **Fetch:** Queries the Yahoo Finance API (via `yfinance`) for max historical data.
    3.  **Storage:** Saves individual CSV files (e.g., `data/SPY.csv`, `data/QQQ.csv`) containing Open, High, Low, Close, Volume data.

### Phase 2: Feature Engineering & Modeling (`--model-etf-returns`)
*   **Goal:** Determine which metrics actually predict future returns.
*   **Process:**
    1.  **Load:** `utils.load_etf_data_from_csvs` aggregates all individual CSVs into a single Price DataFrame (Dates x ETFs).
    2.  **Target Generation:**
        *   Calculates the forward return (default: 6 months).
        *   Binarizes the target: `1` if Return > 0, `0` otherwise.
    3.  **Feature Calculation:**
        *   `ETFReturnPredictor` instantiates metric calculators from `metrics.py`.
        *   Computes rolling windows (e.g., 20d, 60d, 120d) for Volatility, Sharpe, Momentum, Skewness, etc.
    4.  **Statistical Analysis (Enumeration Mode):**
        *   **Discretization:** Continuous metrics are binned (default: 5 quantile bins).
        *   **Probability Estimation:** Calculates $P(Return > 0 | Bin)$ for each bin.
        *   **Validation:** Computes Wilson Score Intervals (uncertainty), Information Gain (predictive power), and Chi-Square statistics (significance).
    5.  **Output:** Generates visualization plots in `results/plots` showing the probability curves for top metrics.

### Phase 3: Ranking (`--rank-etfs`)
*   **Goal:** Provide actionable lists of ETFs or Metrics.
*   **Modes:**
    *   **Descriptive Ranking:** Calculates standard stats (YTD Return, Sharpe Ratio, Max Drawdown) for each ETF and sorts them. Useful for finding "best performing" funds currently.
    *   **Predictive Ranking (`--rank-predictive-metrics`):** Uses the modeling engine to rank *metrics* by Information Gain. Useful for answering "Which technical indicator should I trust?"

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
│   └── plots/           # Generated probability curves (PNGs)
├── screener/            # Source code
├── docs/                # Documentation
└── etf_rankings.csv     # Output of the ranking workflow
```
