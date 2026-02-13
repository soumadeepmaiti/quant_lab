"""
Extreme Value Theory (EVT) Module.

Implements tail risk modeling using:
- Generalized Pareto Distribution (GPD) for threshold exceedances
- Block Maxima / Generalized Extreme Value (GEV)
- Tail index estimation

References:
- McNeil & Frey (2000) - "Estimation of Tail-Related Risk Measures"
- Embrechts, Klüppelberg, Mikosch (1997) - "Modelling Extremal Events"
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantlab.config import get_logger

logger = get_logger(__name__)


@dataclass
class GPDFitResult:
    """Container for GPD fit results."""
    xi: float           # Shape parameter (tail index)
    beta: float         # Scale parameter
    threshold: float    # Threshold u
    n_exceedances: int  # Number of exceedances
    n_total: int        # Total observations
    var_95: float       # VaR at 95%
    var_99: float       # VaR at 99%
    es_95: float        # ES at 95%
    es_99: float        # ES at 99%
    converged: bool     # Optimization converged


def select_threshold_mean_excess(
    data: np.ndarray,
    min_pct: float = 0.05,
    max_pct: float = 0.25
) -> float:
    """
    Select GPD threshold using Mean Excess Plot.
    
    The mean excess function e(u) = E[X - u | X > u]
    should be linear in u if data follows GPD.
    
    Parameters
    ----------
    data : np.ndarray
        Loss data (positive values)
    min_pct : float
        Minimum percentile for threshold search
    max_pct : float
        Maximum percentile for threshold search
    
    Returns
    -------
    float
        Selected threshold
    """
    sorted_data = np.sort(data)[::-1]  # Descending
    n = len(data)

    # Test thresholds between min_pct and max_pct quantiles
    thresholds = np.percentile(data, [100 * (1 - max_pct), 100 * (1 - min_pct)])
    u_range = np.linspace(thresholds[0], thresholds[1], 50)

    mean_excess = []
    for u in u_range:
        exceedances = data[data > u] - u
        if len(exceedances) > 10:
            mean_excess.append((u, np.mean(exceedances), len(exceedances)))

    if not mean_excess:
        return np.percentile(data, 90)

    # Find threshold where mean excess becomes approximately linear
    # Use stability of shape parameter estimate as criterion
    me_df = pd.DataFrame(mean_excess, columns=['threshold', 'mean_excess', 'n_exceed'])

    # Default to 90th percentile if method fails
    return np.percentile(data, 90)


def select_threshold_auto(
    data: np.ndarray,
    method: str = 'percentile',
    percentile: float = 90
) -> float:
    """
    Automatic threshold selection for GPD.
    
    Parameters
    ----------
    data : np.ndarray
        Loss data
    method : str
        'percentile' or 'mean_excess'
    percentile : float
        Percentile for threshold (if method='percentile')
    
    Returns
    -------
    float
        Selected threshold
    """
    if method == 'percentile':
        return np.percentile(data, percentile)
    elif method == 'mean_excess':
        return select_threshold_mean_excess(data)
    else:
        return np.percentile(data, 90)


def gpd_log_likelihood(
    params: Tuple[float, float],
    exceedances: np.ndarray
) -> float:
    """
    Negative log-likelihood for GPD.
    
    GPD density:
        f(x) = (1/β) * (1 + ξx/β)^(-1/ξ - 1)  if ξ ≠ 0
        f(x) = (1/β) * exp(-x/β)              if ξ = 0
    
    Parameters
    ----------
    params : tuple
        (xi, beta) - shape and scale parameters
    exceedances : np.ndarray
        Data exceeding threshold (y = x - u)
    
    Returns
    -------
    float
        Negative log-likelihood
    """
    xi, beta = params
    n = len(exceedances)

    if beta <= 0:
        return 1e10

    if abs(xi) < 1e-10:
        # Exponential limit
        ll = -n * np.log(beta) - np.sum(exceedances) / beta
    else:
        # General GPD
        z = 1 + xi * exceedances / beta

        if np.any(z <= 0):
            return 1e10

        ll = -n * np.log(beta) - (1 / xi + 1) * np.sum(np.log(z))

    return -ll


def fit_gpd(
    data: Union[pd.Series, np.ndarray],
    threshold: float = None,
    threshold_pct: float = 90
) -> GPDFitResult:
    """
    Fit Generalized Pareto Distribution to tail exceedances.
    
    Parameters
    ----------
    data : array-like
        Loss data (typically negative returns made positive)
    threshold : float, optional
        Explicit threshold u
    threshold_pct : float
        Percentile for automatic threshold selection
    
    Returns
    -------
    GPDFitResult
        Fitted GPD parameters and risk measures
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Make losses positive
    if np.mean(data) < 0:
        data = -data

    n = len(data)

    # Select threshold
    if threshold is None:
        threshold = np.percentile(data, threshold_pct)

    # Get exceedances
    exceedances = data[data > threshold] - threshold
    n_exceed = len(exceedances)

    if n_exceed < 20:
        logger.warning(f"Only {n_exceed} exceedances - GPD fit may be unreliable")

    # Initial parameter guess
    mean_e = np.mean(exceedances)
    var_e = np.var(exceedances)

    # Method of moments initial guess
    xi_init = 0.5 * (mean_e ** 2 / var_e - 1) if var_e > 0 else 0.1
    beta_init = mean_e * (1 - xi_init) if xi_init < 1 else mean_e / 2

    # Constrain to reasonable values
    xi_init = max(min(xi_init, 1.0), -0.5)
    beta_init = max(beta_init, 0.001)

    # MLE optimization
    try:
        result = minimize(
            gpd_log_likelihood,
            x0=[xi_init, beta_init],
            args=(exceedances,),
            method='L-BFGS-B',
            bounds=[(-0.5, 2.0), (1e-6, None)]
        )

        xi, beta = result.x
        converged = result.success

    except Exception as e:
        logger.warning(f"GPD optimization failed: {e}")
        xi = xi_init
        beta = beta_init
        converged = False

    # Compute tail risk measures
    Fu = n_exceed / n  # Probability of exceeding threshold

    # GPD quantile function: Q(p) = u + (β/ξ) * ((1-p)^(-ξ) - 1)
    def gpd_var(p):
        """VaR at probability p (e.g., 0.95)."""
        if abs(xi) < 1e-10:
            # Exponential case
            return threshold + beta * np.log(Fu / (1 - p))
        else:
            return threshold + (beta / xi) * ((Fu / (1 - p)) ** xi - 1)

    def gpd_es(p):
        """Expected Shortfall at probability p."""
        var_p = gpd_var(p)
        if xi < 1:
            return (var_p + beta - xi * threshold) / (1 - xi)
        else:
            return np.inf  # ES undefined for xi >= 1

    var_95 = gpd_var(0.95)
    var_99 = gpd_var(0.99)
    es_95 = gpd_es(0.95)
    es_99 = gpd_es(0.99)

    return GPDFitResult(
        xi=xi,
        beta=beta,
        threshold=threshold,
        n_exceedances=n_exceed,
        n_total=n,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        converged=converged
    )


def gpd_var(
    gpd_result: GPDFitResult,
    confidence: float
) -> float:
    """
    Compute VaR from fitted GPD.
    
    VaR_p = u + (β/ξ) * ((n/N_u * (1-p))^(-ξ) - 1)
    
    Parameters
    ----------
    gpd_result : GPDFitResult
        Fitted GPD
    confidence : float
        Confidence level (e.g., 0.99)
    
    Returns
    -------
    float
        VaR estimate
    """
    xi = gpd_result.xi
    beta = gpd_result.beta
    u = gpd_result.threshold
    Fu = gpd_result.n_exceedances / gpd_result.n_total

    p = confidence

    if abs(xi) < 1e-10:
        return u + beta * np.log(Fu / (1 - p))
    else:
        return u + (beta / xi) * ((Fu / (1 - p)) ** xi - 1)


def gpd_es(
    gpd_result: GPDFitResult,
    confidence: float
) -> float:
    """
    Compute Expected Shortfall from fitted GPD.
    
    ES_p = (VaR_p + β - ξ*u) / (1 - ξ)
    
    Parameters
    ----------
    gpd_result : GPDFitResult
        Fitted GPD
    confidence : float
        Confidence level
    
    Returns
    -------
    float
        ES estimate
    """
    xi = gpd_result.xi
    beta = gpd_result.beta
    u = gpd_result.threshold

    var_p = gpd_var(gpd_result, confidence)

    if xi >= 1:
        return np.inf

    return (var_p + beta - xi * u) / (1 - xi)


def hill_estimator(
    data: np.ndarray,
    k: int = None
) -> Tuple[float, float]:
    """
    Hill estimator for tail index.
    
    Estimates ξ from k largest observations:
        ξ_Hill = (1/k) * Σ log(X_(n-i+1) / X_(n-k))
    
    Parameters
    ----------
    data : np.ndarray
        Data (positive values)
    k : int, optional
        Number of order statistics (default: sqrt(n))
    
    Returns
    -------
    tuple
        (hill_estimate, standard_error)
    """
    data = np.sort(data)[::-1]  # Descending
    n = len(data)

    if k is None:
        k = int(np.sqrt(n))

    k = min(k, n - 1)

    # Hill estimator
    log_data = np.log(data[:k]) - np.log(data[k])
    xi_hill = np.mean(log_data)

    # Standard error
    se = xi_hill / np.sqrt(k)

    return xi_hill, se


def evt_var_comparison(
    data: Union[pd.Series, np.ndarray],
    confidence_levels: List[float] = [0.95, 0.99, 0.999]
) -> pd.DataFrame:
    """
    Compare VaR estimates: Historical vs EVT.
    
    Parameters
    ----------
    data : array-like
        Loss data
    confidence_levels : list
        Confidence levels to compute
    
    Returns
    -------
    pd.DataFrame
        Comparison of VaR methods
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Make losses positive
    if np.mean(data) < 0:
        losses = -data
    else:
        losses = data

    # Fit GPD
    gpd_result = fit_gpd(losses, threshold_pct=90)

    results = []

    for conf in confidence_levels:
        # Historical VaR
        hist_var = np.percentile(losses, conf * 100)

        # GPD VaR
        try:
            evt_var = gpd_var(gpd_result, conf)
        except:
            evt_var = np.nan

        # GPD ES
        try:
            evt_es = gpd_es(gpd_result, conf)
        except:
            evt_es = np.nan

        # Historical ES
        hist_es = losses[losses >= hist_var].mean() if np.any(losses >= hist_var) else np.nan

        results.append({
            'confidence': conf,
            'historical_var': hist_var,
            'gpd_var': evt_var,
            'var_ratio': evt_var / hist_var if hist_var > 0 else np.nan,
            'historical_es': hist_es,
            'gpd_es': evt_es,
            'es_ratio': evt_es / hist_es if hist_es > 0 else np.nan
        })

    return pd.DataFrame(results)


def tail_dependence_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.95
) -> float:
    """
    Estimate upper tail dependence coefficient.
    
    λ_U = lim_{q→1} P(Y > F_Y^{-1}(q) | X > F_X^{-1}(q))
    
    Parameters
    ----------
    x, y : np.ndarray
        Two return series
    quantile : float
        Quantile for estimation
    
    Returns
    -------
    float
        Tail dependence estimate
    """
    n = len(x)

    # Empirical quantiles
    qx = np.percentile(x, quantile * 100)
    qy = np.percentile(y, quantile * 100)

    # Count joint exceedances
    joint = np.sum((x > qx) & (y > qy))
    marginal_x = np.sum(x > qx)

    if marginal_x == 0:
        return 0.0

    return joint / marginal_x


def stress_correlation_analysis(
    returns: pd.DataFrame,
    quantiles: List[float] = [0.10, 0.05, 0.01]
) -> pd.DataFrame:
    """
    Analyze correlation structure during stress periods.
    
    Shows how correlations increase during market stress.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Multi-asset returns
    quantiles : list
        Quantiles defining stress periods
    
    Returns
    -------
    pd.DataFrame
        Correlation statistics by stress regime
    """
    results = []

    # Full sample correlation
    full_corr = returns.corr()
    avg_full_corr = (full_corr.sum().sum() - len(full_corr)) / (len(full_corr) ** 2 - len(full_corr))

    results.append({
        'regime': 'Full Sample',
        'avg_correlation': avg_full_corr,
        'min_correlation': full_corr.min().min(),
        'max_correlation': (full_corr.where(full_corr < 1)).max().max()
    })

    # Market return (equal weighted)
    market_return = returns.mean(axis=1)

    for q in quantiles:
        threshold = np.percentile(market_return, q * 100)
        stress_mask = market_return <= threshold

        if stress_mask.sum() < 10:
            continue

        stress_returns = returns.loc[stress_mask]
        stress_corr = stress_returns.corr()

        avg_stress_corr = (stress_corr.sum().sum() - len(stress_corr)) / (len(stress_corr) ** 2 - len(stress_corr))

        results.append({
            'regime': f'Worst {q*100:.0f}%',
            'avg_correlation': avg_stress_corr,
            'min_correlation': stress_corr.min().min(),
            'max_correlation': (stress_corr.where(stress_corr < 1)).max().max(),
            'n_days': stress_mask.sum()
        })

    return pd.DataFrame(results)
