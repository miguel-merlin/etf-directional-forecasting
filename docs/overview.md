# Theoretical Framework: ETF Returns Modeling

## 1. The Financial Intuition
The core objective of this framework is to determine if specific asset characteristics (metrics) today contain information about future performance. In quantitative finance, predicting exact returns is notoriously difficult due to low signal-to-noise ratios.

This model shifts the goalpost: instead of asking *"What will the return be?"* (regression), we ask *"Does this metric shift the odds of a positive return?"* (probabilistic classification).

### Why Binning?
We model the relationship between a metric $M_t$ and future performance non-parametrically using **binning**. This approach is preferred over simple linear regression for three key reasons:
1.  **Non-Linearity:** Financial relationships are often non-linear (e.g., a "U-shaped" curve where both extremely low and high volatility might signal risk). Binning captures these shapes without forcing a linear model.
2.  **Data Efficiency:** By aggregating observations into bins, we reduce idiosyncratic noise, allowing the "signal" of the regime to emerge.
3.  **Interpretability:** It allows us to view the market in distinct regimes (e.g., "When volatility is in the top decile, the probability of a positive return drops to 40%").

---

## 2. Mathematical Formulation

### Defining the Target
We analyze the **6-month forward return** over an approximate 126-trading-day horizon. Let $P_t$ be the price at time $t$:

$$R_{t}^{(6m)} = \frac{P_{t+126}}{P_{t}} - 1$$

To reduce noise, we convert this continuous variable into a **binary indicator** $I_t$, which equals 1 if the return is positive and 0 otherwise:

$$I_{t} = \begin{cases} 1, & \text{if } R_{t}^{(6m)} > 0 \\ 0, & \text{otherwise} \end{cases}$$

### Conditional Probability
Our goal is to estimate the probability function:
$$m \mapsto P(I_{t}=1 | M_{t}=m)$$

Directly estimating this for every unique value $m$ is impossible in finite samples. Therefore, we discretize the metric domain into $B$ bins, $\mathcal{B}_0, \dots, \mathcal{B}_{B-1}$. We typically use **quantile bins** (e.g., quintiles) to ensure each bin has a balanced sample size $n_b$.

The empirical probability estimate for a specific bin is simply the fraction of positive outcomes observed in that regime:

$$\hat{p}_{b} = \frac{s_{b}}{n_{b}} = \frac{\text{Count of Positive Returns in Bin } b}{\text{Total Observations in Bin } b}$$

---

## 3. Quantifying Uncertainty: Wilson Score Intervals
Financial data is sparse. Some bins may have few observations, making the estimate $\hat{p}_b$ unreliable. To account for this, we calculate **Wilson Score Intervals** instead of standard normal approximation intervals.

The Wilson interval is asymmetric and robust for small sample sizes or probabilities near 0 or 1. For a confidence level $z$ (default 1.96 for 95%), the center and margin of the interval are calculated as:

$$Center = \frac{\hat{p}_{b} + \frac{z^2}{2n_{b}}}{1 + \frac{z^2}{n_{b}}}$$

$$Margin = \frac{z}{1 + \frac{z^2}{n_{b}}} \sqrt{\frac{\hat{p}_{b}(1-\hat{p}_{b})}{n_{b}} + \frac{z^2}{4n_{b}^2}}$$

**Intuition:** If the error bars (Wilson intervals) for two bins overlap significantly, we cannot confidently claim the metric distinguishes between those two regimes.

---

## 4. Ranking Methodology
To systematically select the "best" metrics from a large universe, we rely on three summary statistics.

### A. Information Gain (IG)
This measures how much knowing the metric value "surprises" us relative to the baseline probability. It is the expected Kullback-Leibler (KL) divergence between the bin probabilities and the global base rate $\bar{p}$.

$$IG = \sum_{b} w_{b} \left[ \hat{p}_{b} \log\left(\frac{\hat{p}_{b}}{\bar{p}}\right) + (1-\hat{p}_{b}) \log\left(\frac{1-\hat{p}_{b}}{1-\bar{p}}\right) \right]$$

* **High IG:** The metric creates distinct regimes where the probability of success is significantly different from the average.
* **Zero IG:** The metric provides no new information.

### B. Probability Range
A simpler metric for "Economic Significance." It measures the spread between the best and worst regimes: $Range = \max_{b}\hat{p}_{b} - \min_{b}\hat{p}_{b}$.

### C. Chi-Square Test
A hypothesis test to check if the distribution of positive/negative outcomes depends on the bin index. A p-value $< 0.05$ suggests the metric effectively sorts returns into different probability classes.

---

## 5. Parametric Extension: Logistic Regression
While binning is non-parametric, the framework supports **Logistic Regression** for continuous modeling: $P(I_{t}=1 | M_{t}) = \sigma(\beta_{0} + \beta_{1}M_{t})$.

### Logistic Diagnostics and Outputs
The logistic pipeline now produces explicit diagnostics and artifacts that mirror the enumeration workflow's emphasis on interpretability:

1.  **Performance summary (`logistic_experiment_summary.txt`)**
    - Includes train, test, and full-fit metrics:
    - ROC-AUC, Average Precision, Brier score, accuracy, precision, recall, and F1.
    - Reports class prevalence and predicted-positive prevalence to detect threshold bias.

2.  **Prediction-level output (`logistic_predictions.csv`)**
    - Contains `date`, `etf`, `target`, and split labels (`train`/`test`).
    - Stores both evaluation probabilities (from split-specific model evaluation) and full-fit probabilities.
    - Enables calibration checks and error analysis by ETF and time period.

3.  **Coefficient interpretability (`logistic_feature_importance.csv`)**
    - Raw coefficient (effect on log-odds),
    - Standardized coefficient (scale-adjusted effect),
    - Absolute standardized coefficient (ranking by effect size),
    - Odds ratio (multiplicative odds impact).

4.  **Diagnostic plots (`results/plots/`)**
    - ROC curve (`logistic_roc_curve.png`) for ranking quality.
    - Predicted probability distributions by class (`logistic_probability_distribution.png`) for class separation.
    - Top standardized feature effects (`logistic_top_feature_importance.png`) for interpretability.

### Stepwise Forward Feature Selection
In high-dimensional feature spaces (technical + macro indicators), we use **Forward Selection** to identify the most predictive subset of variables. 

**The Algorithm:**
1.  **Baseline:** Start with an empty model ($AUC = 0.5$).
2.  **Greedy Search:** Test every available feature not already in the model.
3.  **Evaluation:** Select the feature that maximizes the Area Under the ROC Curve (ROC-AUC) on a test set.
4.  **Termination:** Stop when no remaining feature improves the AUC by more than a threshold (default 0.001) or a maximum number of features is reached.
