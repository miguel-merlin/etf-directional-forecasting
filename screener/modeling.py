import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from metrics import BaseMetric, get_metrics


@dataclass
class RankedMetric:
    """Container for ranked predictive metric information."""

    name: str
    information_gain: float
    prob_range: float
    chi2_statistic: float
    p_value: float
    min_prob: float
    max_prob: float
    n_observations: int


class ETFReturnPredictor:
    """
    Analyze predictive metrics for 6-month forward returns across ETFs.
    Models P(I=1 | metric) where I indicates positive 6-month return.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        results_dir: str = "results",
        model_type: str = "enumeration",
    ):
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
        self.metric_summary = pd.DataFrame()
        self.logistic_model = None  # Initialize logistic model
        self.model_type = model_type  # Store the model type

        self.results_dir = results_dir
        self.plot_dir = os.path.join(results_dir, "plots")

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

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
        self,
        lookback_periods: List[int] = [20, 60, 120, 252],
        macro_data: pd.DataFrame = pd.DataFrame(),
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
        - Macro-economic indicators (if provided)
        """
        metric_calculators: List[BaseMetric] = get_metrics(
            lookback_periods=lookback_periods,
            short_window=50,
            long_window=200,
            vol_windows=[60, 120],
            volofvol_window=20,
            macro_data=macro_data,
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

        # Robust preprocessing: Replace infinite values with NaN to avoid downstream errors
        self.features.replace([np.inf, -np.inf], np.nan, inplace=True)

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
        # Ensure no infinite values before discretization
        metric_values = metric_values.replace([np.inf, -np.inf], np.nan).dropna()

        if metric_values.empty:
            return pd.Series(dtype=float)

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

    def train_logistic_regression_model(
        self, features: pd.DataFrame, target: pd.Series
    ):
        """
        Trains a logistic regression model.

        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame of features.
        target : pd.Series
            Series of target variable (0 or 1).
        """
        self.logistic_model = LogisticRegression(solver="liblinear", random_state=42)

        # Robust preprocessing: Replace infinite values with NaN
        combined_data = pd.concat([features, target], axis=1).replace(
            [np.inf, -np.inf], np.nan
        )

        # Drop rows where target or features are NaN
        combined_data = combined_data.dropna()

        X = combined_data[features.columns]
        y = combined_data[target.name]

        if not X.empty and not y.empty:
            self.logistic_model.fit(X, y)
            print(
                f"Logistic Regression model trained successfully on {len(X)} samples."
            )
        else:
            print("No valid data to train the Logistic Regression model.")
            self.logistic_model = None

    def predict_logistic_regression(self, features: pd.DataFrame) -> pd.Series:
        """
        Predicts probabilities using the trained logistic regression model.

        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame of features for prediction.

        Returns:
        --------
        pd.Series
            Predicted probabilities of the positive class.
        """
        if self.logistic_model:
            # Handle potential NaNs/Infs in prediction data
            X = features.replace([np.inf, -np.inf], np.nan).fillna(0)

            predictions = self.logistic_model.predict_proba(X)[:, 1]
            return pd.Series(predictions, index=features.index)
        else:
            print("Logistic Regression model not trained.")
            return pd.Series([], dtype=float)

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

        # Robust preprocessing: Replace infinite values with NaN before dropping
        merged[metric_name] = merged[metric_name].replace([np.inf, -np.inf], np.nan)
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
        if not results_list:
            # Return an empty frame with the expected schema so downstream code can handle it
            results_df = results_df.reindex(
                columns=[
                    "metric",
                    "information_gain",
                    "prob_range",
                    "chi2_statistic",
                    "p_value",
                    "min_prob",
                    "max_prob",
                    "n_observations",
                ]
            )
        else:
            results_df = results_df.sort_values("information_gain", ascending=False)
        self.metric_summary = results_df

        # Save metadata summary
        self.save_experiment_metadata()

        return results_df

    def save_experiment_metadata(self) -> None:
        """Save experiment metadata including analyzed variables and top performing metrics."""
        if self.metric_summary.empty:
            return

        metadata_file = os.path.join(self.results_dir, "experiment_summary.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        all_metrics = self.metric_summary["metric"].tolist()
        macro_metrics = [m for m in all_metrics if m.startswith("macro_")]
        tech_metrics = [m for m in all_metrics if not m.startswith("macro_")]

        top_10 = self.metric_summary.head(10)

        with open(metadata_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ETF SCREENER EXPERIMENT SUMMARY\n")
            f.write(f"Date: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"DATASET OVERVIEW:\n")
            f.write(f"- ETFs Analyzed: {', '.join(self.prices.columns)}\n")
            f.write(f"- Date Range: {self.prices.index.min()} to {self.prices.index.max()}\n")
            f.write(f"- Total Observations (ETF-Dates): {len(self.features)}\n\n")

            f.write(f"VARIABLES ANALYZED ({len(all_metrics)} total):\n")
            f.write(f"- Technical Metrics ({len(tech_metrics)}):\n")
            f.write(f"  {', '.join(tech_metrics[:10])} ...\n")
            f.write(f"- Macro Variables ({len(macro_metrics)}):\n")
            if macro_metrics:
                f.write(f"  {', '.join(macro_metrics)}\n")
            else:
                f.write(f"  None\n")
            f.write("\n")

            f.write(f"TOP 10 PREDICTIVE FACTORS (by Information Gain):\n")
            f.write("-" * 80 + "\n")
            f.write(top_10[["metric", "information_gain", "prob_range", "n_observations"]].to_string(index=False))
            f.write("\n" + "-" * 80 + "\n")

        print(f"\n✓ Experiment metadata saved to: {metadata_file}")

    def get_top_metrics(self, top_n: Optional[int] = None) -> List[RankedMetric]:
        """
        Return the ranked metric summaries as dataclass instances.

        Parameters
        ----------
        top_n : Optional[int]
            Limit to the top-N metrics. If None, return all ranked metrics.
        """
        if self.metric_summary.empty:
            summary_df = self.analyze_all_metrics()
        else:
            summary_df = self.metric_summary

        if summary_df.empty:
            return []

        top_df = summary_df if top_n is None else summary_df.head(top_n)

        ranked_metrics = [
            RankedMetric(
                name=row["metric"],
                information_gain=row["information_gain"],
                prob_range=row["prob_range"],
                chi2_statistic=row["chi2_statistic"],
                p_value=row["p_value"],
                min_prob=row["min_prob"],
                max_prob=row["max_prob"],
                n_observations=int(row["n_observations"]),
            )
            for _, row in top_df.iterrows()
        ]

        return ranked_metrics

    def model_etf_returns(self):
        """
        Orchestrates the modeling process based on the configured model_type.
        """
        if self.model_type == "enumeration":
            print("Running enumeration-based analysis...")
            self.analyze_all_metrics()
        elif self.model_type == "logistic":
            print("Running logistic regression modeling...")
            if self.features.empty or self.target.empty:
                print(
                    "Features or target not calculated. Please run calculate_features and create_target_variable first."
                )
                return

            # Prepare data for logistic regression
            features_flat = self.features.copy()
            target_flat = self.target.stack().reset_index(
                level=0, drop=True
            )  # Align target to features index

            # Drop rows with NaN values in features or target
            combined_data = pd.concat([features_flat, target_flat.rename("target")], axis=1).dropna()  # type: ignore
            X = combined_data.drop(
                columns=["target", "etf"]
            )  # 'etf' is an identifier, not a feature
            y = combined_data["target"]

            if X.empty or y.empty:
                print("No valid data after cleaning for logistic regression.")
                return

            # Train the model
            self.train_logistic_regression_model(X, y)

            # If model trained, make predictions
            if self.logistic_model:
                predictions = self.predict_logistic_regression(X)
                self.results["logistic_regression_predictions"] = predictions
                print("Logistic regression predictions generated.")
            else:
                print(
                    "Logistic regression model could not be trained, no predictions generated."
                )
        else:
            print(f"Unknown model type: {self.model_type}")

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
        plt.savefig(os.path.join(self.plot_dir, f"{metric_name}_probability_plot.png"))
        plt.close()

        # Prepare bin details text
        details_txt = probs[
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

        # Print bin details
        print(f"\n{metric_name} - Bin Details:")
        print("=" * 80)
        print(details_txt)

        # Save bin details to txt file in results directory
        txt_filename = os.path.join(self.results_dir, f"{metric_name}_bin_details.txt")
        with open(txt_filename, "w") as f:
            f.write(f"{metric_name} - Bin Details:\n")
            f.write("=" * 80 + "\n")
            f.write(details_txt)
            f.write("\n")

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
        plt.close()
