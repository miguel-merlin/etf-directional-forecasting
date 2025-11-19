import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import stats
import os
from metrics import (
    BaseMetric,
    VolatilityMetric,
    ReturnMetric,
    SharpeMetric,
    MaxDrawdownMetric,
    MomentumMetric,
    MovingAverageMetric,
    VolOfVolMetric,
    HigherMomentMetric,
    UpDownCaptureMetric,
)


class ETFReturnPredictor:
    """
    Analyze predictive metrics for 6-month forward returns across ETFs.
    Models P(I=1 | metric) where I indicates positive 6-month return.
    """

    def __init__(self, price_data: pd.DataFrame, plot_dir: str = "results/plots"):
        """
        Initialize with price data.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with DateTimeIndex and columns for each ETF's prices
        """
        self.prices = price_data
        self.returns = price_data.pct_change()
        self.features = pd.DataFrame()
        self.target = pd.DataFrame()
        self.results = {}

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self.plot_dir = plot_dir

    def calculate_forward_return(self, months: int = 6) -> pd.DataFrame:
        """Calculate forward returns for specified horizon."""
        forward_return = (
            self.prices.shift(-months * 21) / self.prices - 1
        )  # Approx 21 trading days/month
        return forward_return

    def create_target_variable(self, months: int = 6) -> pd.DataFrame:
        """Create binary indicator: 1 if positive return, 0 otherwise."""
        forward_return = self.calculate_forward_return(months)
        self.target = (forward_return > 0).astype(int)
        return self.target

    def calculate_features(
        self, lookback_periods: List[int] = [20, 60, 120, 252]
    ) -> pd.DataFrame:
        """
        Calculate comprehensive set of predictive features using metric classes.

        Features include:
        - Historical volatility (multiple windows)
        - Returns (multiple windows)
        - Momentum indicators
        - Risk-adjusted returns (Sharpe ratios)
        - Drawdown metrics
        - Trend indicators
        - Volatility of volatility
        - Skewness and Kurtosis
        - Up/Down capture
        """
        metric_calculators: List[BaseMetric] = [
            VolatilityMetric(lookback_periods),
            ReturnMetric(lookback_periods),
            SharpeMetric(lookback_periods),
            MaxDrawdownMetric(lookback_periods),
            MomentumMetric(),
            MovingAverageMetric(short_window=50, long_window=200),
            VolOfVolMetric(vol_windows=[60, 120], volofvol_window=20),
            HigherMomentMetric(windows=[60, 120]),
            UpDownCaptureMetric(windows=[60, 120]),
        ]

        features_dict = {}

        for etf in self.prices.columns:
            etf_features = pd.DataFrame(index=self.prices.index)
            prices = self.prices[etf]
            returns = self.returns[etf]
            for metric_calc in metric_calculators:
                metric_df = metric_calc.compute(prices, returns)
                etf_features = pd.concat([etf_features, metric_df], axis=1)

            etf_features["etf"] = etf
            features_dict[etf] = etf_features

        # Combine all ETFs
        self.features = pd.concat(features_dict.values(), axis=0)
        return self.features

    def discretize_metric(
        self, metric_values: pd.Series, n_bins: int = 5, method: str = "quantile"
    ) -> pd.Series:
        """
        Discretize continuous metric into bins for probability estimation.

        Parameters:
        -----------
        metric_values : pd.Series
            Continuous metric values
        n_bins : int
            Number of bins to create
        method : str
            'quantile' for equal-frequency bins or 'uniform' for equal-width bins
        """
        if method == "quantile":
            return pd.qcut(metric_values, q=n_bins, labels=False, duplicates="drop")
        else:
            return pd.cut(metric_values, bins=n_bins, labels=False)

    def _prepare_flat_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Flatten feature/target frames and ensure a date column is present."""
        target_flat = self.target.stack().reset_index()
        target_flat.columns = ["date", "etf", "target"]

        features_flat = self.features.reset_index()

        if "date" not in features_flat.columns:
            original_index_name = self.prices.index.name
            if original_index_name and original_index_name in features_flat.columns:
                features_flat.rename(
                    columns={original_index_name: "date"}, inplace=True
                )
            elif "index" in features_flat.columns:
                features_flat.rename(columns={"index": "date"}, inplace=True)
            else:
                alt_date_cols = [
                    col for col in features_flat.columns if col.lower() == "date"
                ]
                if alt_date_cols:
                    features_flat.rename(
                        columns={alt_date_cols[0]: "date"}, inplace=True
                    )
                else:
                    raise KeyError(
                        "No date column found after resetting indexes; ensure price data has a named index."
                    )

        return features_flat, target_flat

    def estimate_conditional_probability(
        self,
        metric_name: str,
        n_bins: int = 5,
        method: str = "quantile",
        features_flat: pd.DataFrame = pd.DataFrame(),
        target_flat: pd.DataFrame = pd.DataFrame(),
    ) -> Dict:
        """
        Estimate P(I=1 | metric) for a given metric.

        Returns:
        --------
        dict with:
            - probabilities: P(I=1) for each bin
            - bin_edges: boundaries of bins
            - counts: number of observations per bin
            - metric_name: name of the metric
        """
        if features_flat.empty or target_flat.empty:
            features_flat, target_flat = self._prepare_flat_data()

        merged = pd.merge(
            features_flat[["date", "etf", metric_name]],
            target_flat,
            on=["date", "etf"],
            how="inner",
        )
        merged = merged.dropna(subset=[metric_name, "target"])

        if len(merged) == 0:
            return {
                "probabilities": pd.DataFrame(),
                "overall_positive_rate": 0,
                "n_observations": 0,
            }

        try:
            merged["bin"] = self.discretize_metric(merged[metric_name], n_bins, method)
        except Exception:
            try:
                merged["bin"] = self.discretize_metric(
                    merged[metric_name],
                    min(n_bins, len(merged[metric_name].unique())),
                    method,
                )
            except Exception:
                return {
                    "probabilities": pd.DataFrame(),
                    "overall_positive_rate": 0,
                    "n_observations": 0,
                }

        prob_by_bin = merged.groupby("bin").agg(
            {"target": ["mean", "count", "sum"], metric_name: ["min", "max", "mean"]}
        )

        prob_by_bin.columns = [
            "prob_positive",
            "count",
            "positive_count",
            "metric_min",
            "metric_max",
            "metric_mean",
        ]

        # Calculate confidence intervals (Wilson score interval)
        z = 1.96  # 95% confidence
        n = prob_by_bin["count"]
        p = prob_by_bin["prob_positive"]

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

        prob_by_bin["ci_lower"] = center - margin
        prob_by_bin["ci_upper"] = center + margin

        return {
            "metric_name": metric_name,
            "probabilities": prob_by_bin,
            "overall_positive_rate": merged["target"].mean(),
            "n_observations": len(merged),
        }

    def analyze_all_metrics(self, n_bins: int = 5) -> pd.DataFrame:
        """
        Analyze all calculated metrics and rank by predictive power.

        Returns summary DataFrame with metrics ranked by information gain.
        """
        metric_cols = [col for col in self.features.columns if col != "etf"]
        results_list = []
        features_flat, target_flat = self._prepare_flat_data()

        for metric in metric_cols:
            result = self.estimate_conditional_probability(
                metric,
                n_bins,
                features_flat=features_flat,
                target_flat=target_flat,
            )
            if result is None:
                continue

            probs = result["probabilities"]
            if probs.empty:
                continue

            overall_rate = result["overall_positive_rate"]

            # Calculate information gain (KL divergence from base rate)
            prob_dist = probs["prob_positive"].values
            weights = probs["count"].values / probs["count"].sum()

            # Avoid log(0)
            prob_dist = np.clip(prob_dist, 1e-10, 1 - 1e-10)

            ig = np.sum(
                weights
                * (
                    prob_dist * np.log(prob_dist / overall_rate)
                    + (1 - prob_dist) * np.log((1 - prob_dist) / (1 - overall_rate))
                )
            )

            # Range of probabilities
            prob_range = prob_dist.max() - prob_dist.min()

            # Chi-square test for independence
            observed = np.column_stack(
                [
                    probs["positive_count"].values,
                    probs["count"].values - probs["positive_count"].values,
                ]
            )
            chi2, p_value = stats.chi2_contingency(observed)[:2]

            results_list.append(
                {
                    "metric": metric,
                    "information_gain": ig,
                    "prob_range": prob_range,
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "min_prob": prob_dist.min(),
                    "max_prob": prob_dist.max(),
                    "n_observations": result["n_observations"],
                }
            )

            # Store detailed results
            self.results[metric] = result

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values("information_gain", ascending=False)

        return results_df

    def plot_metric_probabilities(
        self, metric_name: str, figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Visualize P(I=1 | metric) with confidence intervals.
        """
        if metric_name not in self.results:
            print(
                f"Metric {metric_name} not found in results. Run analyze_all_metrics first."
            )
            return

        result = self.results[metric_name]
        probs = result["probabilities"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        x = range(len(probs))
        ax1.plot(
            x,
            probs["prob_positive"],
            "o-",
            linewidth=2,
            markersize=8,
            label="P(I=1|metric)",
        )
        ax1.fill_between(x, probs["ci_lower"], probs["ci_upper"], alpha=0.3)
        ax1.axhline(
            result["overall_positive_rate"],
            color="r",
            linestyle="--",
            label=f'Overall rate: {result["overall_positive_rate"]:.3f}',
        )
        ax1.set_xlabel("Metric Bin (Low to High)")
        ax1.set_ylabel("P(Positive 6m Return)")
        ax1.set_title(f"Conditional Probability: {metric_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])

        # Plot 2: Sample sizes
        ax2.bar(x, probs["count"], alpha=0.6, color="steelblue")
        ax2.set_xlabel("Metric Bin (Low to High)")
        ax2.set_ylabel("Number of Observations")
        ax2.set_title("Sample Size per Bin")
        ax2.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.plot_dir + f"/{metric_name}_probability_plot.png")

        # Print bin details
        print(f"\n{metric_name} - Bin Details:")
        print("=" * 80)
        print(
            probs[
                [
                    "metric_min",
                    "metric_max",
                    "metric_mean",
                    "prob_positive",
                    "count",
                    "ci_lower",
                    "ci_upper",
                ]
            ].to_string()
        )

    def plot_top_metrics(self, n_metrics: int = 6, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot the top N most predictive metrics in a grid.
        """
        summary = self.analyze_all_metrics()
        top_metrics = summary.head(n_metrics)["metric"].tolist()

        n_rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, metric in enumerate(top_metrics):
            result = self.results[metric]
            probs = result["probabilities"]

            x = range(len(probs))
            axes[idx].plot(x, probs["prob_positive"], "o-", linewidth=2, markersize=6)
            axes[idx].fill_between(x, probs["ci_lower"], probs["ci_upper"], alpha=0.3)
            axes[idx].axhline(
                result["overall_positive_rate"],
                color="r",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            axes[idx].set_xlabel("Bin")
            axes[idx].set_ylabel("P(I=1)")
            axes[idx].set_title(
                f'{metric}\n(IG: {summary.iloc[idx]["information_gain"]:.4f})'
            )
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim([0, 1])

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(self.plot_dir + "/top_metrics_probability_plots.png")
