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

The Wilson interval is asymmetric and robust for small sample sizes or probabilities near 0 or 1. For a confidence level $z$, the center and margin of the interval are calculated as:

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
* **Zero IG:** The metric provides no new information; the probability of success is the same as the base rate regardless of the metric's value.

### B. Probability Range
A simpler metric for "Economic Significance." It measures the spread between the best and worst regimes:

$$Range = \max_{b}\hat{p}_{b} - \min_{b}\hat{p}_{b}$$

**Intuition:** A metric might be statistically significant but have a tiny range (e.g., moving probability from 51% to 52%). We prefer metrics with a large range (e.g., 40% vs 70%), as they offer more actionable trade signals.

### C. Chi-Square Test
A hypothesis test to check if the distribution of positive/negative outcomes depends on the bin index.
* **$H_0$:** Independence (The metric predicts nothing).
* **$H_1$:** Dependence (The metric predicts something).
* A p-value $< 0.05$ suggests the metric effectively sorts returns into different probability classes.

---

## 5. Parametric Extension: Logistic Regression
While binning is the primary focus, the framework is compatible with **Logistic Regression** for continuous modeling. This assumes a monotonic relationship between the metric and the log-odds of a positive return:

$$P(I_{t}=1 | M_{t}) = \sigma(\beta_{0} + \beta_{1}M_{t})$$

The binned plots serve as a diagnostic tool for these models. If the binned probability curve is non-monotonic (e.g., U-shaped), a simple logistic model would be misspecified and fail to capture the signal.