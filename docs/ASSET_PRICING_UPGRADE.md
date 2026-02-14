# Asset Pricing & Robust Portfolio Construction Upgrade

## Overview

This upgrade adds institutional-grade factor risk premia estimation and covariance shrinkage to QuantLab, elevating the platform from "student backtester" to "systematic researcher."

**Key Additions:**
1. **Fama-MacBeth cross-sectional regression** with Newey-West HAC standard errors
2. **Covariance shrinkage** (Ledoit-Wolf + factor model) for robust portfolio construction
3. **Out-of-sample validation** comparing sample vs shrinkage estimators

---

## 1. Why Rankings Are Not Enough

Factor backtesting typically relies on:
- Cross-sectional rankings (quintile portfolios)
- Information Coefficient (IC) tests

**Limitations:**
- Rankings obscure **magnitude** of factor effects
- No inference on risk premia (compensation per unit of factor exposure)
- Cannot answer: *"What is the expected return for 1σ increase in momentum?"*

**Solution:** Fama-MacBeth regression estimates factor **risk premia** (λ) with statistical significance tests.

---

## 2. Fama-MacBeth Estimator

### Model

Two-stage cross-sectional regression:

**Stage 1 (Cross-Sectional):** For each date $t$:

$$
r_{i,t+1} = a_t + \sum_{k=1}^K \beta_{k,t} \cdot x_{k,i,t} + \epsilon_{i,t+1}
$$

Where:
- $r_{i,t+1}$ = forward return of asset $i$ (strictly future)
- $x_{k,i,t}$ = factor $k$ exposure of asset $i$ at time $t$ (lagged, no lookahead)
- $\beta_{k,t}$ = cross-sectional factor loading at time $t$

**Stage 2 (Time-Series Average):** Risk premium estimate:

$$
\hat{\lambda}_k = \frac{1}{T} \sum_{t=1}^T \beta_{k,t}
$$

### Newey-West HAC Standard Errors

Cross-sectional betas $\beta_{k,t}$ are **not IID**:
- Serial correlation (momentum crashes, regime persistence)
- Heteroskedasticity (volatility clustering)

**Newey-West correction:**

$$
\text{Var}(\hat{\lambda}_k) = \frac{1}{T} \left( \Gamma_0 + \sum_{j=1}^L w_j (\Gamma_j + \Gamma_j') \right)
$$

Where:
- $\Gamma_j$ = autocovariance at lag $j$
- $w_j = 1 - \frac{j}{L+1}$ = Bartlett kernel weights
- $L$ = lag truncation (default: 3 for monthly, 5 for daily)

**T-statistic:**

$$
t_k = \frac{\hat{\lambda}_k}{\text{SE}_{\text{HAC}}(\hat{\lambda}_k)}
$$

### Interpretation

- $\hat{\lambda}_k > 0$: Positive risk premium (compensated factor)
- $t_k > 2$: Statistically significant at ~5% level
- $\hat{\lambda}_k \times 12$ (monthly) = annualized premium

**Example:**
- Momentum $\hat{\lambda} = 0.015$ per month, $t = 2.8$
- Interpretation: 1σ increase in momentum → +18% annual return (significant)

---

## 3. Estimation Error in Covariance

Sample covariance matrix $\hat{\Sigma}$ is **noisy** when $T \ll N$:
- Eigenvalues are biased (too spread out)
- Small sample → large estimation error → unstable weights

**Minimum variance portfolio:**

$$
w_{\text{MV}} \propto \Sigma^{-1} \mathbf{1}
$$

With noisy $\hat{\Sigma}$, $\hat{w}_{\text{MV}}$ has:
- Extreme positions (corner solutions)
- High turnover
- Poor out-of-sample performance

**Shrinkage idea:** Regularize toward a **stable target**:

$$
\Sigma_{\text{shrunk}} = (1 - \delta) \hat{\Sigma}_{\text{sample}} + \delta \cdot \Sigma_{\text{target}}
$$

Where $\delta \in [0, 1]$ is shrinkage intensity.

---

## 4. Ledoit-Wolf + Factor Model Covariance

### 4.1 Ledoit-Wolf Shrinkage

**Target:** Constant correlation matrix

$$
\Sigma_{\text{target}} = \text{diag}(\sigma_1^2, \dots, \sigma_N^2) + \bar{\rho} \cdot \mathbf{11}^T
$$

**Optimal $\delta$:** Estimated via minimizing expected loss (Ledoit & Wolf, 2004)

**Properties:**
- Reduces condition number of $\Sigma$
- Stabilizes inverse $\Sigma^{-1}$
- Oracle optimal under certain asymptotics

### 4.2 Factor Model Covariance Target

**Model:**

$$
R = B F + \epsilon
$$

Where:
- $F$ = $K$ factor returns (e.g., market, size, value)
- $B$ = factor exposures (estimated via time-series regression)
- $\epsilon$ = idiosyncratic noise

**Target covariance:**

$$
\Sigma_{\text{factor}} = B \Sigma_F B^T + \text{diag}(\sigma_{\epsilon_1}^2, \dots, \sigma_{\epsilon_N}^2)
$$

**Advantages:**
- Incorporates economic structure (systematic vs idiosyncratic risk)
- Lower dimensional ($K \ll N$) → less estimation error
- Used in institutional risk models (Barra, Axioma)

**Combined shrinkage:**

$$
\Sigma_{\text{shrunk}} = (1 - \delta) \hat{\Sigma}_{\text{sample}} + \delta \cdot \Sigma_{\text{factor}}
$$

---

## 5. Expected Results

### 5.1 Fama-MacBeth

**Output:**
- Risk premia table with $\hat{\lambda}_k$, $\text{SE}_{\text{HAC}}$, $t$-stat, $p$-value
- Time-series plots of $\beta_{k,t}$ (premia stability)
- Cross-sectional $R^2$ distribution

**Typical findings:**
- Momentum: $\hat{\lambda} \approx 0.01$ monthly, $t > 2$ (significant)
- Volatility: $\hat{\lambda} < 0$ (low-vol anomaly)
- Quality: $\hat{\lambda} > 0$ but marginal significance

### 5.2 Shrinkage

**Improvements vs sample covariance:**
- **Lower turnover:** 20-40% reduction (fewer extreme trades)
- **More stable weights:** Max|w| decreases, diversification improves
- **OOS vol accuracy:** Realized vol closer to predicted (1-3% error reduction)

**Trade-off:**
- In-sample fit slightly worse (by construction)
- Out-of-sample robustness **much better**

---

## 6. Usage

### 6.1 Fama-MacBeth Analysis

```bash
# Monthly regression (default)
python scripts/run_fama_macbeth.py --start 2018-01-01 --end 2024-12-31

# Daily regression (more granular)
python scripts/run_fama_macbeth.py --freq D --ret-col ret_fwd_1d --min-obs 50

# Custom Newey-West lags
python scripts/run_fama_macbeth.py --nw-lags 5
```

**Outputs:**
- `data/processed/fmb_premia.csv` - Summary table
- `data/processed/fmb_betas_by_date.csv` - Time series of $\beta_t$
- `reports/figures/fmb_premia_rolling.png` - Rolling premia
- `reports/figures/fmb_r2_hist.png` - R² distribution

### 6.2 Shrinkage Study

```bash
# Compare sample vs Ledoit-Wolf
python scripts/run_shrinkage_study.py --start 2018-01-01 --end 2024-12-31

# Adjust training window
python scripts/run_shrinkage_study.py --train-window 504  # 2 years
```

**Outputs:**
- `data/processed/shrinkage_study.csv` - Performance summary
- `reports/figures/shrinkage_oos_vol.png` - OOS volatility
- `reports/figures/shrinkage_turnover.png` - Cumulative turnover
- `reports/figures/risk_contrib_sample_vs_shrinkage.png` - Risk decomposition

---

## 7. Research Discipline

### No Lookahead Bias

**Critical:**
- Forward returns $r_{i,t+1}$ computed using *strictly future prices*
- Factor exposures $x_{i,t}$ use *only information available at $t$*

**Implementation:**
```python
# CORRECT: lag factors, shift returns forward
ret_fwd = prices.pct_change(1).shift(-1)
factor = prices.shift(1) / prices.shift(252) - 1  # lagged

# WRONG: using contemporaneous factors
factor = prices / prices.shift(251) - 1  # includes t+1 info!
```

### HAC Standard Errors

Do **not** assume IID errors:
- Use Newey-West with appropriate lags
- Report $t$-stats from HAC SEs, not naive SEs

### Out-of-Sample Validation

Always evaluate on **held-out data**:
- Train covariance on window $[t - W, t]$
- Compute weights at $t$
- Evaluate performance on $[t+1, t+H]$

---

## 8. CV-Ready Bullets

Add to your CV/LinkedIn under QuantLab:

> - **Estimated factor risk premia via Fama-MacBeth with HAC errors; stabilized covariance via shrinkage for OOS robustness**
> - Implemented Fama-MacBeth cross-sectional regression with Newey-West HAC t-statistics to estimate factor risk premia and validate alpha significance
> - Built covariance shrinkage framework (Ledoit-Wolf + factor-model targets) to reduce estimation error and improve out-of-sample portfolio stability and turnover

---

## 9. References

- Fama, E. F., & MacBeth, J. D. (1973). *Risk, Return, and Equilibrium: Empirical Tests*. Journal of Political Economy.
- Newey, W. K., & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. Econometrica.
- Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. Journal of Portfolio Management.

---

## 10. Next Steps

**Immediate:**
1. Run `python scripts/run_fama_macbeth.py` to generate premia estimates
2. Run `python scripts/run_shrinkage_study.py` to compare methods
3. Review outputs in `data/processed/` and `reports/figures/`

**Extensions:**
- Add more factors (value, carry, sentiment)
- Implement factor model shrinkage with real factor returns (FF3, FF5)
- Add quadratic programming for long-only min-var
- Extend to mean-variance optimization (not just min-var)

---

*This upgrade transforms QuantLab from a backtesting tool into an institutional-grade research platform.*
