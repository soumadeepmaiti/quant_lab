"""
Market Impact Regression Module.

Implements research-grade impact analysis:
- Kyle's lambda estimation (ΔP = λ × OFI)
- Price impact regression
- Permanent vs temporary impact decomposition
- Adverse selection measurement

References:
- Kyle (1985) - "Continuous Auctions and Insider Trading"
- Hasbrouck (1991) - "Measuring the Information Content of Stock Trades"
- Glosten & Harris (1988) - "Estimating the Components of the Bid-Ask Spread"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from dataclasses import dataclass

from quantlab.config import get_logger

logger = get_logger(__name__)


@dataclass
class ImpactRegressionResult:
    """Container for impact regression results."""
    lambda_coef: float      # Kyle's lambda
    lambda_se: float        # Standard error
    lambda_t_stat: float    # T-statistic
    lambda_pvalue: float    # P-value
    r_squared: float        # R-squared
    n_observations: int     # Sample size
    intercept: float        # Regression intercept
    
    
def estimate_kyle_lambda(
    price_changes: np.ndarray,
    order_flow: np.ndarray,
    method: str = 'ols'
) -> ImpactRegressionResult:
    """
    Estimate Kyle's lambda: the price impact coefficient.
    
    Model: ΔP_t = α + λ × OFI_t + ε_t
    
    where:
        ΔP_t = price change
        OFI_t = order flow imbalance (signed volume)
        λ = Kyle's lambda (price impact)
    
    Parameters
    ----------
    price_changes : np.ndarray
        Price changes (mid-price differences)
    order_flow : np.ndarray
        Signed order flow (positive = buy pressure)
    method : str
        'ols' or 'wls' (weighted by inverse variance)
    
    Returns
    -------
    ImpactRegressionResult
        Regression results with lambda estimate
    
    Interpretation
    --------------
    λ represents how much the price moves per unit of order flow:
        - Higher λ → less liquid market
        - Lower λ → more liquid market
        
    Example
    -------
    >>> ofi = cumulative_buy_volume - cumulative_sell_volume
    >>> result = estimate_kyle_lambda(price_changes, ofi)
    >>> print(f"Kyle's lambda: {result.lambda_coef:.6f}")
    """
    # Clean data
    valid = ~(np.isnan(price_changes) | np.isnan(order_flow))
    dp = price_changes[valid]
    ofi = order_flow[valid]
    n = len(dp)
    
    if n < 30:
        return ImpactRegressionResult(
            lambda_coef=np.nan, lambda_se=np.nan, lambda_t_stat=np.nan,
            lambda_pvalue=np.nan, r_squared=np.nan, n_observations=n,
            intercept=np.nan
        )
    
    # OLS regression: ΔP = α + λ × OFI
    X = np.column_stack([np.ones(n), ofi])
    y = dp
    
    if method == 'ols':
        # Standard OLS
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        # WLS with inverse variance weights
        # Weight by inverse of squared order flow (give less weight to large trades)
        weights = 1 / (np.abs(ofi) + 1)
        W = np.diag(weights)
        betas = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    
    intercept, lambda_coef = betas
    
    # Residuals and R-squared
    y_pred = X @ betas
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Standard errors (homoskedastic)
    mse = ss_res / (n - 2)
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    se_lambda = np.sqrt(var_beta[1])
    
    # T-statistic and p-value
    t_stat = lambda_coef / se_lambda if se_lambda > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
    
    return ImpactRegressionResult(
        lambda_coef=lambda_coef,
        lambda_se=se_lambda,
        lambda_t_stat=t_stat,
        lambda_pvalue=p_value,
        r_squared=r_squared,
        n_observations=n,
        intercept=intercept
    )


def estimate_power_law_impact(
    price_impacts: np.ndarray,
    order_sizes: np.ndarray
) -> Dict[str, float]:
    """
    Estimate power law impact scaling: ΔP = k × Q^α
    
    Takes log: log(ΔP) = log(k) + α × log(Q)
    
    Literature suggests α ≈ 0.5 (square root law)
    
    Parameters
    ----------
    price_impacts : np.ndarray
        Absolute price impacts (positive values)
    order_sizes : np.ndarray
        Order sizes
    
    Returns
    -------
    dict
        Power law parameters and statistics
    
    Interpretation
    --------------
    α ≈ 0.5: Square root law (theoretical prediction)
    α < 0.5: Impact grows slower than sqrt
    α > 0.5: Impact grows faster than sqrt
    """
    # Filter positive values for log transform
    valid = (price_impacts > 0) & (order_sizes > 0)
    impacts = price_impacts[valid]
    sizes = order_sizes[valid]
    
    n = len(impacts)
    
    if n < 20:
        return {
            'alpha': np.nan, 'k': np.nan, 'alpha_se': np.nan,
            't_stat': np.nan, 'r_squared': np.nan, 'n': n
        }
    
    # Log transform
    log_impact = np.log(impacts)
    log_size = np.log(sizes)
    
    # OLS on logs
    X = np.column_stack([np.ones(n), log_size])
    y = log_impact
    
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    log_k, alpha = betas
    k = np.exp(log_k)
    
    # R-squared
    y_pred = X @ betas
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Standard error for alpha
    mse = ss_res / (n - 2)
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    se_alpha = np.sqrt(var_beta[1])
    
    t_stat = (alpha - 0.5) / se_alpha  # Test H0: alpha = 0.5
    
    return {
        'alpha': alpha,
        'alpha_se': se_alpha,
        'k': k,
        't_stat_vs_half': t_stat,
        'p_value_vs_half': 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2)),
        'consistent_with_sqrt': abs(alpha - 0.5) < 2 * se_alpha,
        'r_squared': r_squared,
        'n': n
    }


def permanent_transitory_decomposition(
    price_changes: np.ndarray,
    order_signs: np.ndarray,
    lags: int = 5
) -> Dict[str, float]:
    """
    Decompose price impact into permanent and transitory components.
    
    Uses Hasbrouck (1991) VAR approach:
    - Permanent component: information-driven price change
    - Transitory component: temporary price pressure (reverses)
    
    Parameters
    ----------
    price_changes : np.ndarray
        Price changes
    order_signs : np.ndarray
        Order direction (+1 buy, -1 sell)
    lags : int
        Number of lags for VAR
    
    Returns
    -------
    dict
        Decomposition results
    """
    n = len(price_changes)
    
    if n < 50:
        return {'permanent': np.nan, 'transitory': np.nan, 'info_share': np.nan}
    
    # Simple approach: look at price reversion after trades
    post_trade_returns = []
    
    for i in range(len(order_signs) - lags):
        if order_signs[i] != 0:
            # Cumulative return over next 'lags' periods
            cum_return = np.sum(price_changes[i+1:i+1+lags])
            initial_impact = price_changes[i] * order_signs[i]  # Signed impact
            post_trade_returns.append((initial_impact, cum_return, order_signs[i]))
    
    if not post_trade_returns:
        return {'permanent': np.nan, 'transitory': np.nan, 'info_share': np.nan}
    
    df = pd.DataFrame(post_trade_returns, columns=['initial_impact', 'subsequent', 'direction'])
    
    # Buy trades
    buys = df[df['direction'] == 1]
    sells = df[df['direction'] == -1]
    
    # Average impact and reversion
    if len(buys) > 5:
        buy_impact = buys['initial_impact'].mean()
        buy_subsequent = buys['subsequent'].mean()
    else:
        buy_impact = buy_subsequent = 0
    
    if len(sells) > 5:
        sell_impact = sells['initial_impact'].mean()
        sell_subsequent = sells['subsequent'].mean()
    else:
        sell_impact = sell_subsequent = 0
    
    avg_impact = (abs(buy_impact) + abs(sell_impact)) / 2
    avg_reversion = (buy_subsequent + sell_subsequent) / 2  # If negative = reversal for buys
    
    # Permanent = impact that remains
    # Transitory = impact that reverses
    total_impact = avg_impact
    permanent = total_impact + avg_reversion  # What remains after reversion
    transitory = -avg_reversion  # What reversed
    
    # Information share = permanent / total
    info_share = permanent / total_impact if total_impact > 0 else 0
    
    return {
        'permanent': permanent,
        'transitory': transitory,
        'total_impact': total_impact,
        'info_share': info_share,
        'reversal_fraction': transitory / total_impact if total_impact > 0 else 0,
        'n_trades': len(df)
    }


def adverse_selection_measure(
    trade_prices: np.ndarray,
    trade_directions: np.ndarray,
    mid_prices_after: np.ndarray,
    horizons: List[int] = [1, 5, 10, 30]
) -> pd.DataFrame:
    """
    Measure adverse selection (post-trade price drift).
    
    If price continues in trade direction → adverse selection (informed trader)
    If price reverts → no adverse selection (liquidity trader)
    
    Parameters
    ----------
    trade_prices : np.ndarray
        Execution prices
    trade_directions : np.ndarray
        +1 for buy, -1 for sell
    mid_prices_after : np.ndarray
        Mid prices at various horizons after trade
    horizons : list
        Time horizons to measure
    
    Returns
    -------
    pd.DataFrame
        Adverse selection metrics by horizon
    """
    results = []
    
    for h in horizons:
        if h >= len(mid_prices_after):
            continue
        
        # Price change from trade to horizon h
        price_change = mid_prices_after[h:] - trade_prices[:-h]
        direction = trade_directions[:-h]
        
        # Signed return (positive if price moved in trade direction)
        signed_return = price_change * direction
        
        valid = ~np.isnan(signed_return)
        signed_return = signed_return[valid]
        
        if len(signed_return) < 10:
            continue
        
        mean_drift = np.mean(signed_return)
        std_drift = np.std(signed_return)
        t_stat = mean_drift / (std_drift / np.sqrt(len(signed_return))) if std_drift > 0 else 0
        
        results.append({
            'horizon': h,
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            't_stat': t_stat,
            'p_value': 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(signed_return) - 1)),
            'pct_favorable': (signed_return > 0).mean(),
            'adverse_selection': mean_drift > 0 and t_stat > 1.96,
            'n_trades': len(signed_return)
        })
    
    return pd.DataFrame(results)


def toxicity_index(
    order_flow: np.ndarray,
    price_changes: np.ndarray,
    window: int = 50
) -> pd.Series:
    """
    Calculate VPIN-style toxicity index.
    
    High toxicity = informed trading activity
    Low toxicity = normal market making
    
    Parameters
    ----------
    order_flow : np.ndarray
        Signed order flow
    price_changes : np.ndarray
        Price changes
    window : int
        Rolling window
    
    Returns
    -------
    pd.Series
        Toxicity index over time
    """
    n = len(order_flow)
    toxicity = np.zeros(n)
    
    for i in range(window, n):
        window_ofi = order_flow[i-window:i]
        window_dp = price_changes[i-window:i]
        
        # Volume imbalance
        buy_vol = np.sum(window_ofi[window_ofi > 0])
        sell_vol = np.sum(np.abs(window_ofi[window_ofi < 0]))
        total_vol = buy_vol + sell_vol
        
        if total_vol > 0:
            imbalance = abs(buy_vol - sell_vol) / total_vol
        else:
            imbalance = 0
        
        # Price variance
        price_var = np.var(window_dp) if len(window_dp) > 1 else 0
        
        # Toxicity = imbalance × price variance
        toxicity[i] = imbalance * np.sqrt(price_var) * 100
    
    return pd.Series(toxicity)


def rolling_impact_estimation(
    price_changes: pd.Series,
    order_flow: pd.Series,
    window: int = 100
) -> pd.DataFrame:
    """
    Rolling estimation of Kyle's lambda.
    
    Parameters
    ----------
    price_changes : pd.Series
        Price changes
    order_flow : pd.Series
        Order flow
    window : int
        Rolling window
    
    Returns
    -------
    pd.DataFrame
        Rolling lambda estimates with confidence intervals
    """
    results = []
    
    for i in range(window, len(price_changes)):
        dp = price_changes.iloc[i-window:i].values
        ofi = order_flow.iloc[i-window:i].values
        
        result = estimate_kyle_lambda(dp, ofi)
        
        results.append({
            'date': price_changes.index[i],
            'lambda': result.lambda_coef,
            'lambda_se': result.lambda_se,
            'lambda_lower': result.lambda_coef - 1.96 * result.lambda_se,
            'lambda_upper': result.lambda_coef + 1.96 * result.lambda_se,
            'r_squared': result.r_squared,
            'significant': result.lambda_pvalue < 0.05
        })
    
    return pd.DataFrame(results).set_index('date')
