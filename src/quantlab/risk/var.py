"""
Value at Risk (VaR) calculations.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Historical VaR.
    
    Uses actual historical returns to estimate VaR.
    VaR = negative of the (1-confidence)th percentile.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns (Series for portfolio, DataFrame for assets)
    confidence : float
        Confidence level (0.95 = 95%)
    weights : np.ndarray, optional
        Portfolio weights if returns is DataFrame
    
    Returns
    -------
    float
        VaR as positive number (loss)
    """
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = returns

    alpha = 1 - confidence
    var = -np.percentile(portfolio_returns.dropna(), alpha * 100)

    return var


def parametric(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Parametric (Variance-Covariance) VaR.
    
    Assumes returns are normally distributed.
    VaR = μ - z × σ
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns
    confidence : float
        Confidence level
    weights : np.ndarray, optional
        Portfolio weights
    
    Returns
    -------
    float
        VaR as positive number
    """
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Portfolio mean and variance
        mu = (returns.mean() * weights).sum()
        cov_matrix = returns.cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        sigma = np.sqrt(portfolio_var)
    else:
        mu = returns.mean()
        sigma = returns.std()

    z = norm.ppf(1 - confidence)
    var = -(mu + z * sigma)

    return var


def monte_carlo(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Calculate Monte Carlo VaR.
    
    Simulates future returns using historical parameters.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns
    confidence : float
        Confidence level
    weights : np.ndarray, optional
        Portfolio weights
    n_simulations : int
        Number of simulations
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    float
        VaR as positive number
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Simulate from multivariate normal
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        simulated = np.random.multivariate_normal(
            mean=mean_returns,
            cov=cov_matrix,
            size=n_simulations
        )

        portfolio_returns = simulated @ weights
    else:
        mu = returns.mean()
        sigma = returns.std()
        portfolio_returns = np.random.normal(mu, sigma, n_simulations)

    alpha = 1 - confidence
    var = -np.percentile(portfolio_returns, alpha * 100)

    return var


def calculate_all(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
    weights: Optional[np.ndarray] = None,
    n_simulations: int = 10000
) -> pd.DataFrame:
    """
    Calculate VaR using all methods.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns
    confidence_levels : List[float]
        Confidence levels to calculate
    weights : np.ndarray, optional
        Portfolio weights
    n_simulations : int
        Monte Carlo simulations
    
    Returns
    -------
    pd.DataFrame
        VaR by method and confidence level
    """
    results = []

    for conf in confidence_levels:
        results.append({
            'confidence': f'{int(conf*100)}%',
            'historical': historical(returns, conf, weights),
            'parametric': parametric(returns, conf, weights),
            'monte_carlo': monte_carlo(returns, conf, weights, n_simulations)
        })

    return pd.DataFrame(results).set_index('confidence')


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate rolling VaR.
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    window : int
        Rolling window
    confidence : float
        Confidence level
    method : str
        'historical' or 'parametric'
    
    Returns
    -------
    pd.Series
        Rolling VaR
    """
    var_series = []
    dates = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]

        if method == 'parametric':
            var = parametric(window_returns, confidence)
        else:
            var = historical(window_returns, confidence)

        var_series.append(var)
        dates.append(returns.index[i])

    return pd.Series(var_series, index=dates, name=f'VaR_{int(confidence*100)}')


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Backtest VaR predictions.
    
    Parameters
    ----------
    returns : pd.Series
        Actual returns
    var_series : pd.Series
        VaR predictions (positive values)
    confidence : float
        VaR confidence level
    
    Returns
    -------
    Dict[str, float]
        Backtest statistics
    """
    common_idx = returns.index.intersection(var_series.index)
    actual = returns.loc[common_idx]
    var = var_series.loc[common_idx]

    # Count breaches (actual loss > VaR)
    breaches = (-actual > var).sum()
    n_periods = len(actual)
    breach_rate = breaches / n_periods
    expected_rate = 1 - confidence

    return {
        'n_breaches': int(breaches),
        'n_periods': n_periods,
        'breach_rate': breach_rate,
        'expected_rate': expected_rate,
        'breach_ratio': breach_rate / expected_rate if expected_rate > 0 else 0
    }
