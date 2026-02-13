"""
Statistical Testing Module for Factor Research.

Implements research-grade statistical inference:
- Bootstrap confidence intervals
- Multiple hypothesis testing corrections
- T-statistics with proper standard errors
- Distribution testing

References:
- Newey-West (1987) - HAC standard errors
- White (2000) - Reality Check for Data Snooping
- Harvey et al. (2016) - Multiple testing in cross-section
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.config import get_logger

logger = get_logger(__name__)


def bootstrap_confidence_interval(
    data: Union[pd.Series, np.ndarray],
    statistic: str = 'mean',
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : array-like
        Sample data
    statistic : str
        'mean', 'sharpe', 'median', 'std'
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    n_bootstrap : int
        Number of bootstrap samples
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (point_estimate, lower_ci, upper_ci)
    
    Example
    -------
    >>> returns = strategy_returns
    >>> est, lower, upper = bootstrap_confidence_interval(returns, 'sharpe')
    >>> print(f"Sharpe: {est:.2f} [{lower:.2f}, {upper:.2f}]")
    """
    np.random.seed(seed)

    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 10:
        return np.nan, np.nan, np.nan

    # Define statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'sharpe':
        def stat_func(x):
            return np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'std':
        stat_func = np.std
    else:
        stat_func = np.mean

    # Point estimate
    point_est = stat_func(data)

    # Bootstrap
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_est, lower, upper


def bootstrap_hypothesis_test(
    data: Union[pd.Series, np.ndarray],
    null_value: float = 0.0,
    alternative: str = 'two-sided',
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap hypothesis test for the mean.
    
    H0: μ = null_value
    
    Parameters
    ----------
    data : array-like
        Sample data
    null_value : float
        Hypothesized mean under null
    alternative : str
        'two-sided', 'greater', or 'less'
    n_bootstrap : int
        Number of bootstrap samples
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Test results with p-value, t-stat, etc.
    """
    np.random.seed(seed)

    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 10:
        return {'t_stat': np.nan, 'p_value': np.nan, 'reject_h0': False}

    # Observed statistic
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    t_stat = (sample_mean - null_value) / (sample_std / np.sqrt(n))

    # Center data under null hypothesis
    centered_data = data - sample_mean + null_value

    # Bootstrap under null
    bootstrap_t_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(centered_data, size=n, replace=True)
        boot_mean = np.mean(sample)
        boot_std = np.std(sample, ddof=1)

        if boot_std > 0:
            bootstrap_t_stats[i] = (boot_mean - null_value) / (boot_std / np.sqrt(n))
        else:
            bootstrap_t_stats[i] = 0

    # P-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(bootstrap_t_stats) >= np.abs(t_stat))
    elif alternative == 'greater':
        p_value = np.mean(bootstrap_t_stats >= t_stat)
    else:  # 'less'
        p_value = np.mean(bootstrap_t_stats <= t_stat)

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'n': n,
        'reject_h0_5pct': p_value < 0.05,
        'reject_h0_1pct': p_value < 0.01
    }


def newey_west_se(
    data: Union[pd.Series, np.ndarray],
    lags: int = None
) -> Tuple[float, float, float]:
    """
    Compute Newey-West HAC standard errors for the mean.
    
    Accounts for autocorrelation in time series data.
    
    Parameters
    ----------
    data : array-like
        Time series data
    lags : int, optional
        Number of lags (default: floor(4*(T/100)^(2/9)))
    
    Returns
    -------
    tuple
        (mean, newey_west_se, t_stat)
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 10:
        return np.nan, np.nan, np.nan

    # Optimal lag selection (Newey-West 1994)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    mean = np.mean(data)
    demeaned = data - mean

    # Variance term (lag 0)
    gamma_0 = np.sum(demeaned ** 2) / n

    # Add autocovariance terms with Bartlett weights
    nw_var = gamma_0

    for j in range(1, lags + 1):
        gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / n
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        nw_var += 2 * weight * gamma_j

    # Standard error
    nw_se = np.sqrt(nw_var / n)

    # T-statistic
    t_stat = mean / nw_se if nw_se > 0 else 0

    return mean, nw_se, t_stat


def sharpe_ratio_test(
    returns: Union[pd.Series, np.ndarray],
    null_sharpe: float = 0.0,
    annualize: bool = True
) -> Dict[str, float]:
    """
    Statistical test for Sharpe ratio using Lo (2002) adjustment.
    
    Parameters
    ----------
    returns : array-like
        Return series
    null_sharpe : float
        Sharpe ratio under null hypothesis
    annualize : bool
        Whether to annualize
    
    Returns
    -------
    dict
        Test results
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n < 30:
        return {'sharpe': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan}

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)

    if std == 0:
        return {'sharpe': np.nan, 'se': np.nan, 't_stat': np.nan, 'p_value': np.nan}

    sharpe = mean / std

    # Annualize
    ann_factor = np.sqrt(252) if annualize else 1
    sharpe_ann = sharpe * ann_factor

    # Standard error (Lo 2002)
    # SE(SR) ≈ sqrt((1 + 0.5*SR^2) / n)
    se = np.sqrt((1 + 0.5 * sharpe ** 2) / n) * ann_factor

    # T-statistic
    t_stat = (sharpe_ann - null_sharpe) / se

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    return {
        'sharpe': sharpe_ann,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01,
        'n': n
    }


def ic_significance_test(
    ic_series: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Test if Information Coefficient is significantly different from zero.
    
    Parameters
    ----------
    ic_series : array-like
        Time series of IC values
    
    Returns
    -------
    dict
        Test statistics
    """
    ic = np.asarray(ic_series)
    ic = ic[~np.isnan(ic)]
    n = len(ic)

    if n < 10:
        return {'mean_ic': np.nan, 't_stat': np.nan, 'p_value': np.nan}

    mean_ic = np.mean(ic)
    std_ic = np.std(ic, ddof=1)

    # T-statistic
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0

    # P-value (two-sided)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))

    # Information Ratio (mean IC / std IC)
    ir = mean_ic / std_ic if std_ic > 0 else 0

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        't_stat': t_stat,
        'p_value': p_value,
        'ir': ir,
        'pct_positive': (ic > 0).mean(),
        'n': n,
        'significant_5pct': p_value < 0.05
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Correct p-values for multiple hypothesis testing.
    
    Parameters
    ----------
    p_values : list
        List of p-values from multiple tests
    method : str
        'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
    
    Returns
    -------
    pd.DataFrame
        Original and adjusted p-values
    """
    n = len(p_values)
    p_arr = np.array(p_values)

    if method == 'bonferroni':
        adjusted = np.minimum(p_arr * n, 1.0)

    elif method == 'holm':
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_arr)
        adjusted = np.zeros(n)

        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_arr[idx] * (n - i), 1.0)

        # Enforce monotonicity
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i - 1]
            adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])

    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR control
        sorted_idx = np.argsort(p_arr)
        sorted_p = p_arr[sorted_idx]

        adjusted_sorted = np.zeros(n)

        for i in range(n - 1, -1, -1):
            if i == n - 1:
                adjusted_sorted[i] = sorted_p[i]
            else:
                adjusted_sorted[i] = min(
                    sorted_p[i] * n / (i + 1),
                    adjusted_sorted[i + 1]
                )

        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = adjusted_sorted[i]

    else:
        adjusted = p_arr

    return pd.DataFrame({
        'original_p': p_values,
        'adjusted_p': adjusted,
        'reject_5pct': adjusted < 0.05,
        'reject_1pct': adjusted < 0.01
    })


def jarque_bera_test(
    data: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Jarque-Bera test for normality.
    
    Parameters
    ----------
    data : array-like
        Sample data
    
    Returns
    -------
    dict
        Test results
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 20:
        return {'jb_stat': np.nan, 'p_value': np.nan, 'is_normal': None}

    # Skewness and kurtosis
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    skew = np.mean(((data - mean) / std) ** 3)
    kurt = np.mean(((data - mean) / std) ** 4)
    excess_kurt = kurt - 3

    # JB statistic
    jb = n / 6 * (skew ** 2 + 0.25 * excess_kurt ** 2)

    # P-value (chi-squared with 2 df)
    p_value = 1 - stats.chi2.cdf(jb, df=2)

    return {
        'jb_stat': jb,
        'p_value': p_value,
        'skewness': skew,
        'excess_kurtosis': excess_kurt,
        'is_normal': p_value > 0.05,
        'n': n
    }


def factor_decay_test(
    ic_in_sample: np.ndarray,
    ic_out_of_sample: np.ndarray
) -> Dict[str, float]:
    """
    Test for factor decay (McLean & Pontiff style).
    
    Tests if out-of-sample IC is significantly lower than in-sample.
    
    Parameters
    ----------
    ic_in_sample : array
        In-sample IC values
    ic_out_of_sample : array
        Out-of-sample IC values
    
    Returns
    -------
    dict
        Test results
    """
    ic_in = np.asarray(ic_in_sample)
    ic_out = np.asarray(ic_out_of_sample)

    ic_in = ic_in[~np.isnan(ic_in)]
    ic_out = ic_out[~np.isnan(ic_out)]

    if len(ic_in) < 10 or len(ic_out) < 10:
        return {'decay_pct': np.nan, 'p_value': np.nan}

    mean_in = np.mean(ic_in)
    mean_out = np.mean(ic_out)

    # Decay percentage
    decay_pct = (mean_in - mean_out) / mean_in * 100 if mean_in != 0 else 0

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(ic_in, ic_out)

    return {
        'mean_ic_in_sample': mean_in,
        'mean_ic_out_of_sample': mean_out,
        'decay_pct': decay_pct,
        't_stat': t_stat,
        'p_value': p_value,
        'significant_decay': p_value < 0.05 and mean_out < mean_in
    }


def comprehensive_factor_statistics(
    returns: pd.Series,
    ic_series: pd.Series = None,
    benchmark_returns: pd.Series = None
) -> pd.DataFrame:
    """
    Compute comprehensive statistical summary for a factor strategy.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    ic_series : pd.Series, optional
        Information coefficient series
    benchmark_returns : pd.Series, optional
        Benchmark returns for alpha calculation
    
    Returns
    -------
    pd.DataFrame
        Comprehensive statistics
    """
    results = []

    returns = returns.dropna()
    n = len(returns)

    if n < 30:
        return pd.DataFrame()

    # Basic statistics
    mean_ret = returns.mean()
    std_ret = returns.std()

    # Sharpe with proper test
    sharpe_test = sharpe_ratio_test(returns)
    results.append({
        'metric': 'Sharpe Ratio',
        'value': sharpe_test['sharpe'],
        'std_error': sharpe_test['se'],
        't_stat': sharpe_test['t_stat'],
        'p_value': sharpe_test['p_value']
    })

    # Mean return with Newey-West SE
    mean, nw_se, nw_t = newey_west_se(returns)
    results.append({
        'metric': 'Mean Return (Annual)',
        'value': mean * 252,
        'std_error': nw_se * np.sqrt(252),
        't_stat': nw_t,
        'p_value': 2 * (1 - stats.norm.cdf(np.abs(nw_t)))
    })

    # Bootstrap CI for Sharpe
    sharpe_est, sharpe_lo, sharpe_hi = bootstrap_confidence_interval(
        returns, 'sharpe', confidence=0.95
    )
    results.append({
        'metric': 'Sharpe 95% CI Lower',
        'value': sharpe_lo,
        'std_error': np.nan,
        't_stat': np.nan,
        'p_value': np.nan
    })
    results.append({
        'metric': 'Sharpe 95% CI Upper',
        'value': sharpe_hi,
        'std_error': np.nan,
        't_stat': np.nan,
        'p_value': np.nan
    })

    # IC statistics if provided
    if ic_series is not None:
        ic_test = ic_significance_test(ic_series)
        results.append({
            'metric': 'Mean IC',
            'value': ic_test['mean_ic'],
            'std_error': ic_test['std_ic'] / np.sqrt(ic_test['n']),
            't_stat': ic_test['t_stat'],
            'p_value': ic_test['p_value']
        })
        results.append({
            'metric': 'IC Information Ratio',
            'value': ic_test['ir'],
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan
        })

    # Alpha if benchmark provided
    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)

        if len(common) > 30:
            y = returns.loc[common].values
            X = np.column_stack([np.ones(len(common)), benchmark_returns.loc[common].values])

            betas = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = betas[0]

            # Alpha t-stat
            y_pred = X @ betas
            residuals = y - y_pred
            mse = np.sum(residuals ** 2) / (len(y) - 2)
            var_alpha = mse * np.linalg.inv(X.T @ X)[0, 0]
            se_alpha = np.sqrt(var_alpha)
            t_alpha = alpha / se_alpha

            results.append({
                'metric': 'Alpha (Annual)',
                'value': alpha * 252,
                'std_error': se_alpha * np.sqrt(252),
                't_stat': t_alpha,
                'p_value': 2 * (1 - stats.t.cdf(np.abs(t_alpha), df=len(y) - 2))
            })

    # Normality test
    jb_test = jarque_bera_test(returns)
    results.append({
        'metric': 'Jarque-Bera (Normality)',
        'value': jb_test['jb_stat'],
        'std_error': np.nan,
        't_stat': np.nan,
        'p_value': jb_test['p_value']
    })

    return pd.DataFrame(results)
