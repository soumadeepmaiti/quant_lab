"""
Expected Shortfall (Conditional VaR) calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Union


def historical(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Historical Expected Shortfall.
    
    ES is the average of all losses that exceed the VaR threshold.
    ES = E[Loss | Loss > VaR]
    
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
        ES as positive number (loss)
    """
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = returns
    
    portfolio_returns = portfolio_returns.dropna()
    
    alpha = 1 - confidence
    var_threshold = np.percentile(portfolio_returns, alpha * 100)
    
    # Get tail returns (worse than VaR)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    
    # ES is the average of tail losses
    es = -tail_returns.mean()
    
    return es


def parametric(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Parametric Expected Shortfall.
    
    For normal distribution:
    ES = μ - σ × φ(z) / (1-α)
    
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
        ES as positive number
    """
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        mu = (returns.mean() * weights).sum()
        cov_matrix = returns.cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        sigma = np.sqrt(portfolio_var)
    else:
        mu = returns.mean()
        sigma = returns.std()
    
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    phi_z = norm.pdf(z)
    
    # ES formula for normal distribution
    es = -(mu - sigma * phi_z / alpha)
    
    return es


def monte_carlo(
    returns: Union[pd.Series, pd.DataFrame],
    confidence: float = 0.95,
    weights: Optional[np.ndarray] = None,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Calculate Monte Carlo Expected Shortfall.
    
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
        Random seed
    
    Returns
    -------
    float
        ES as positive number
    """
    if seed is not None:
        np.random.seed(seed)
    
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
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
    var_threshold = np.percentile(portfolio_returns, alpha * 100)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    
    es = -tail_returns.mean()
    
    return es


def calculate_all(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
    weights: Optional[np.ndarray] = None,
    n_simulations: int = 10000
) -> pd.DataFrame:
    """
    Calculate ES using all methods.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns
    confidence_levels : List[float]
        Confidence levels
    weights : np.ndarray, optional
        Portfolio weights
    n_simulations : int
        Monte Carlo simulations
    
    Returns
    -------
    pd.DataFrame
        ES by method and confidence level
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


def var_es_comparison(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_levels: List[float] = [0.90, 0.95, 0.99],
    weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare VaR and ES at different confidence levels.
    
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Historical returns
    confidence_levels : List[float]
        Confidence levels
    weights : np.ndarray, optional
        Portfolio weights
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    from quantlab.risk.var import historical as var_hist
    
    results = []
    
    for conf in confidence_levels:
        var = var_hist(returns, conf, weights)
        es = historical(returns, conf, weights)
        
        results.append({
            'confidence': f'{int(conf*100)}%',
            'var': var,
            'es': es,
            'es_var_ratio': es / var if var > 0 else 0
        })
    
    return pd.DataFrame(results).set_index('confidence')
