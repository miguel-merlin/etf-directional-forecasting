import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from metrics import BaseMetric, get_metrics


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
        metric_calculators: List[BaseMetric] = get_metrics(
            lookback_periods=lookback_periods,
            short_window=50,
            long_window=200,
            vol_windows=[60, 120],
            volofvol_window=20,
        )

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

    def _prepare_probability_plot_data(
        self, metric_name: str
    ) -> Tuple[pd.DataFrame, float]:
        """Return probability frame augmented with bin labels and KL divergence."""
        result = self.results[metric_name]
        probs = result["probabilities"].copy()

        probs["bin_label"] = [
            f"{metric_min:.4g} – {metric_max:.4g}"
            for metric_min, metric_max in zip(
                probs["metric_min"].values, probs["metric_max"].values
            )
        ]

        overall_rate = np.clip(result["overall_positive_rate"], 1e-6, 1 - 1e-6)
        probs["prob_positive"] = np.clip(probs["prob_positive"], 1e-6, 1 - 1e-6)
        probs["kl_divergence"] = probs["prob_positive"] * np.log(
            probs["prob_positive"] / overall_rate
        ) + (1 - probs["prob_positive"]) * np.log(
            (1 - probs["prob_positive"]) / (1 - overall_rate)
        )

        return probs, overall_rate

    def plot_metric_probabilities(
        self, metric_name: str, figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Visualize P(I=1 | metric) with confidence intervals.
        Single-axis version (no subplots).
        """
        if metric_name not in self.results:
            print(
                f"Metric {metric_name} not found in results. Run analyze_all_metrics first."
            )
            return

        probs, overall_rate = self._prepare_probability_plot_data(metric_name)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(probs))

        # --- Probability line ---
        ax.plot(
            x,
            probs["prob_positive"],
            "o-",
            linewidth=2,
            markersize=8,
            label="P(I=1|metric)",
        )

        # Confidence intervals
        ax.fill_between(x, probs["ci_lower"], probs["ci_upper"], alpha=0.3)

        # Overall rate horizontal line
        ax.axhline(
            overall_rate,
            color="r",
            linestyle="--",
            label=f"Overall rate: {overall_rate:.3f}",
        )

        ax.set_xlabel("Metric Bin (Low → High)")
        ax.set_ylabel("P(Positive 6m Return)")
        ax.set_title(f"Conditional Probability: {metric_name}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        # X tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(probs["bin_label"], rotation=45, ha="right")

        # --- KL Divergence on twin axis ---
        ax_kl = ax.twinx()
        ax_kl.bar(
            x,
            probs["kl_divergence"],
            alpha=0.25,
            color="darkorange",
            width=0.5,
            label="KL divergence",
        )
        ax_kl.set_ylabel("KL Divergence")

        # Merge legends
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax_kl.get_legend_handles_labels()
        ax.legend(lines + bars, labels + bar_labels, loc="upper left")

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
                    "bin_label",
                    "metric_mean",
                    "prob_positive",
                    "count",
                    "ci_lower",
                    "ci_upper",
                    "kl_divergence",
                ]
            ].to_string()
        )

    def plot_metric_probabilities_for_metrics(self, metrics: Sequence[str]) -> None:
        """Generate and save probability plots for the requested metrics."""

        for metric in metrics:
            print(f"  Plotting detailed probabilities for {metric}...")
            self.plot_metric_probabilities(metric)

    def plot_top_metrics(
        self,
        n_metrics: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 10),
        summary: Optional[pd.DataFrame] = None,
    ) -> None:
        """Plot probability curves for either all or the top-N metrics in a grid."""

        summary_df = summary if summary is not None else self.analyze_all_metrics()
        metrics = summary_df["metric"].tolist()
        if n_metrics is not None:
            metrics = metrics[:n_metrics]

        if not metrics:
            print("No metrics available to plot.")
            return

        n_cols = 2
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        fig_height = max(figsize[1], n_rows * 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height))
        axes = axes.flatten()
        info_gain_map = (
            summary_df.set_index("metric")["information_gain"].to_dict()
            if "information_gain" in summary_df.columns
            else {}
        )

        for idx, metric in enumerate(metrics):
            probs, overall_rate = self._prepare_probability_plot_data(metric)
            x = np.arange(len(probs))
            axes[idx].plot(x, probs["prob_positive"], "o-", linewidth=2, markersize=6)
            axes[idx].fill_between(x, probs["ci_lower"], probs["ci_upper"], alpha=0.3)
            axes[idx].axhline(
                overall_rate,
                color="r",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            axes[idx].set_ylabel("P(I=1)")
            axes[idx].set_title(
                f"{metric}\n(IG: {info_gain_map.get(metric, float('nan')):.4f})"
            )
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim([0, 1])
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(
                probs["bin_label"], rotation=45, ha="right", fontsize=8
            )

        for idx in range(len(metrics), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        filename = (
            "all_metrics_probability_grid.png"
            if n_metrics is None or n_metrics >= len(summary_df)
            else "top_metrics_probability_plots.png"
        )
        plt.savefig(os.path.join(self.plot_dir, filename))
