"""
VaR Backtesting Module.

Implements research-grade VaR validation:
- Kupiec unconditional coverage test
- Christoffersen independence test
- Combined conditional coverage test
- Traffic light approach (Basel)

References:
- Kupiec (1995) - "Techniques for Verifying the Accuracy of Risk Measurement Models"
- Christoffersen (1998) - "Evaluating Interval Forecasts"
- Basel Committee (1996) - "Supervisory Framework for Backtesting"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from dataclasses import dataclass

from quantlab.config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """VaR backtest result container."""
    n_observations: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_lr: float
    christoffersen_pvalue: float
    conditional_lr: float
    conditional_pvalue: float
    basel_zone: str
    violations_series: pd.Series


def count_violations(
    returns: pd.Series,
    var_series: pd.Series,
    var_sign: str = 'negative'
) -> Tuple[int, pd.Series]:
    """
    Count VaR violations.
    
    Parameters
    ----------
    returns : pd.Series
        Actual returns
    var_series : pd.Series
        VaR forecast series (same index)
    var_sign : str
        'negative' if VaR is reported as positive loss, 'positive' otherwise
    
    Returns
    -------
    tuple
        (number of violations, violation indicator series)
    """
    # Align series
    common_idx = returns.index.intersection(var_series.index)
    ret = returns.loc[common_idx]
    var = var_series.loc[common_idx]
    
    # VaR violation: actual loss exceeds VaR
    if var_sign == 'negative':
        violations = ret < -var  # Return more negative than -VaR
    else:
        violations = ret < var
    
    return violations.sum(), violations.astype(int)


def kupiec_test(
    n_violations: int,
    n_observations: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Kupiec (1995) unconditional coverage test.
    
    Tests H0: True violation rate = expected rate (1 - confidence)
    
    LR = -2 * ln[(1-p)^(T-x) * p^x / (1-p̂)^(T-x) * p̂^x]
    
    where:
        p = expected violation rate
        p̂ = actual violation rate
        T = number of observations
        x = number of violations
    
    Parameters
    ----------
    n_violations : int
        Number of VaR violations
    n_observations : int
        Total number of observations
    confidence : float
        VaR confidence level
    
    Returns
    -------
    tuple
        (likelihood ratio statistic, p-value)
    """
    p = 1 - confidence  # Expected violation rate
    T = n_observations
    x = n_violations
    
    if x == 0:
        # No violations - compute limit
        lr = -2 * (T * np.log(1 - p) - T * np.log(1))
        return lr, 1 - stats.chi2.cdf(lr, df=1)
    
    if x == T:
        # All violations
        lr = -2 * (T * np.log(p) - T * np.log(1))
        return lr, 1 - stats.chi2.cdf(lr, df=1)
    
    p_hat = x / T  # Actual violation rate
    
    # Log-likelihood ratio
    lr = -2 * (
        (T - x) * np.log(1 - p) + x * np.log(p)
        - (T - x) * np.log(1 - p_hat) - x * np.log(p_hat)
    )
    
    # P-value (chi-squared with 1 df)
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    
    return lr, pvalue


def christoffersen_independence_test(
    violations: pd.Series
) -> Tuple[float, float]:
    """
    Christoffersen (1998) independence test.
    
    Tests H0: Violations are independently distributed
    
    Parameters
    ----------
    violations : pd.Series
        Binary violation series (0/1)
    
    Returns
    -------
    tuple
        (likelihood ratio statistic, p-value)
    """
    v = violations.values
    
    # Count transitions
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))  # No viol -> No viol
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))  # No viol -> Viol
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))  # Viol -> No viol
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))  # Viol -> Viol
    
    # Transition probabilities
    n0 = n00 + n01
    n1 = n10 + n11
    
    if n0 == 0 or n1 == 0:
        return 0.0, 1.0  # Can't compute, assume independence
    
    pi01 = n01 / n0 if n0 > 0 else 0  # P(viol | no viol yesterday)
    pi11 = n11 / n1 if n1 > 0 else 0  # P(viol | viol yesterday)
    
    # Under independence, both should equal overall rate
    pi = (n01 + n11) / (n0 + n1)
    
    # Handle edge cases
    eps = 1e-10
    pi01 = max(min(pi01, 1 - eps), eps)
    pi11 = max(min(pi11, 1 - eps), eps)
    pi = max(min(pi, 1 - eps), eps)
    
    # Log-likelihood under independence
    ll_indep = n00 * np.log(1 - pi) + n01 * np.log(pi) + n10 * np.log(1 - pi) + n11 * np.log(pi)
    
    # Log-likelihood under dependence
    ll_dep = (n00 * np.log(1 - pi01) + n01 * np.log(pi01) 
              + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    
    # LR statistic
    lr = -2 * (ll_indep - ll_dep)
    
    # P-value (chi-squared with 1 df)
    pvalue = 1 - stats.chi2.cdf(lr, df=1)
    
    return lr, pvalue


def conditional_coverage_test(
    violations: pd.Series,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Christoffersen (1998) conditional coverage test.
    
    Combined test: LR_CC = LR_UC + LR_IND
    
    Parameters
    ----------
    violations : pd.Series
        Binary violation series
    confidence : float
        VaR confidence level
    
    Returns
    -------
    tuple
        (combined LR statistic, p-value)
    """
    n = len(violations)
    x = violations.sum()
    
    lr_uc, _ = kupiec_test(x, n, confidence)
    lr_ind, _ = christoffersen_independence_test(violations)
    
    lr_cc = lr_uc + lr_ind
    pvalue = 1 - stats.chi2.cdf(lr_cc, df=2)
    
    return lr_cc, pvalue


def basel_traffic_light(
    n_violations: int,
    n_observations: int = 250,
    confidence: float = 0.99
) -> str:
    """
    Basel traffic light approach for VaR model validation.
    
    Based on 250 trading days at 99% confidence:
    - Green: 0-4 violations
    - Yellow: 5-9 violations
    - Red: 10+ violations
    
    Parameters
    ----------
    n_violations : int
        Number of VaR violations
    n_observations : int
        Number of observations (typically 250)
    confidence : float
        VaR confidence level
    
    Returns
    -------
    str
        'GREEN', 'YELLOW', or 'RED'
    """
    # Scale for different observation counts
    expected = n_observations * (1 - confidence)
    
    # Basel zones (scaled)
    scale = n_observations / 250
    green_upper = int(4 * scale)
    yellow_upper = int(9 * scale)
    
    if n_violations <= green_upper:
        return 'GREEN'
    elif n_violations <= yellow_upper:
        return 'YELLOW'
    else:
        return 'RED'


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
    var_sign: str = 'negative'
) -> BacktestResult:
    """
    Comprehensive VaR backtest.
    
    Performs:
    - Kupiec unconditional coverage test
    - Christoffersen independence test
    - Conditional coverage test
    - Basel traffic light assessment
    
    Parameters
    ----------
    returns : pd.Series
        Actual return series
    var_series : pd.Series
        VaR forecast series
    confidence : float
        VaR confidence level
    var_sign : str
        'negative' if VaR reported as positive number
    
    Returns
    -------
    BacktestResult
        Comprehensive backtest results
    """
    n_violations, violations = count_violations(returns, var_series, var_sign)
    n_obs = len(violations)
    
    expected_rate = 1 - confidence
    actual_rate = n_violations / n_obs if n_obs > 0 else 0
    
    # Kupiec test
    kupiec_lr, kupiec_pval = kupiec_test(n_violations, n_obs, confidence)
    
    # Christoffersen independence test
    christ_lr, christ_pval = christoffersen_independence_test(violations)
    
    # Conditional coverage test
    cc_lr, cc_pval = conditional_coverage_test(violations, confidence)
    
    # Basel zone
    zone = basel_traffic_light(n_violations, n_obs, confidence)
    
    return BacktestResult(
        n_observations=n_obs,
        n_violations=n_violations,
        violation_rate=actual_rate,
        expected_rate=expected_rate,
        kupiec_lr=kupiec_lr,
        kupiec_pvalue=kupiec_pval,
        christoffersen_lr=christ_lr,
        christoffersen_pvalue=christ_pval,
        conditional_lr=cc_lr,
        conditional_pvalue=cc_pval,
        basel_zone=zone,
        violations_series=violations
    )


def rolling_var_backtest(
    returns: pd.Series,
    var_func,
    window: int = 250,
    confidence: float = 0.95,
    **var_kwargs
) -> pd.DataFrame:
    """
    Rolling VaR backtest with dynamic forecasts.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    var_func : callable
        VaR function that takes returns and confidence
    window : int
        Rolling estimation window
    confidence : float
        VaR confidence level
    **var_kwargs
        Additional arguments for var_func
    
    Returns
    -------
    pd.DataFrame
        Rolling backtest results
    """
    results = []
    
    for i in range(window, len(returns)):
        # Training data
        train_ret = returns.iloc[i-window:i]
        
        # Forecast VaR for next day
        var_forecast = var_func(train_ret, confidence, **var_kwargs)
        
        # Actual return
        actual_ret = returns.iloc[i]
        
        # Violation
        violation = 1 if actual_ret < -var_forecast else 0
        
        results.append({
            'date': returns.index[i],
            'var_forecast': var_forecast,
            'actual_return': actual_ret,
            'violation': violation
        })
    
    return pd.DataFrame(results).set_index('date')


def var_model_comparison(
    returns: pd.Series,
    var_methods: Dict[str, callable],
    window: int = 250,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compare multiple VaR models using backtesting.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    var_methods : dict
        Dictionary of VaR functions {'Historical': hist_var, 'GARCH': garch_var}
    window : int
        Rolling window
    confidence : float
        Confidence level
    
    Returns
    -------
    pd.DataFrame
        Comparison of model performance
    """
    results = []
    
    for name, var_func in var_methods.items():
        try:
            # Run rolling backtest
            backtest_df = rolling_var_backtest(
                returns, var_func, window, confidence
            )
            
            # Get violations
            violations = backtest_df['violation']
            var_forecasts = backtest_df['var_forecast']
            actual_returns = backtest_df['actual_return']
            
            n_obs = len(violations)
            n_viol = violations.sum()
            
            # Tests
            kupiec_lr, kupiec_pval = kupiec_test(n_viol, n_obs, confidence)
            christ_lr, christ_pval = christoffersen_independence_test(violations)
            
            results.append({
                'model': name,
                'n_violations': n_viol,
                'violation_rate': n_viol / n_obs,
                'expected_rate': 1 - confidence,
                'kupiec_pvalue': kupiec_pval,
                'independence_pvalue': christ_pval,
                'mean_var': var_forecasts.mean(),
                'basel_zone': basel_traffic_light(n_viol, n_obs, confidence)
            })
            
        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")
            continue
    
    return pd.DataFrame(results)


def violation_clustering_analysis(
    violations: pd.Series
) -> Dict[str, float]:
    """
    Analyze clustering of VaR violations.
    
    Parameters
    ----------
    violations : pd.Series
        Binary violation series
    
    Returns
    -------
    dict
        Clustering statistics
    """
    v = violations.values
    
    # Count consecutive violations
    consecutive = []
    count = 0
    
    for i in range(len(v)):
        if v[i] == 1:
            count += 1
        else:
            if count > 0:
                consecutive.append(count)
            count = 0
    
    if count > 0:
        consecutive.append(count)
    
    if not consecutive:
        return {
            'max_consecutive': 0,
            'mean_consecutive': 0,
            'n_clusters': 0,
            'cluster_rate': 0
        }
    
    return {
        'max_consecutive': max(consecutive),
        'mean_consecutive': np.mean(consecutive),
        'n_clusters': len(consecutive),
        'cluster_rate': sum(c for c in consecutive if c > 1) / sum(consecutive)
    }
