"""
Performance evaluation metrics.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Trading periods per year (252 for daily)
    
    Returns
    -------
    Dict[str, float]
        Performance metrics
    """
    returns = returns.dropna()

    if len(returns) < 20:
        return {}

    # Cumulative returns
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # Annualized metrics
    n_years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(periods_per_year)

    # Risk-free adjusted
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_daily

    # Sharpe ratio
    sharpe = (excess_returns.mean() / daily_vol) * np.sqrt(periods_per_year) if daily_vol > 0 else 0

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0
    sortino = annual_return / downside_std if downside_std > 0 else 0

    # Maximum drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns > 0).mean()

    # Other statistics
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'n_periods': len(returns)
    }


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown time series.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    
    Returns
    -------
    pd.Series
        Drawdown series (negative values)
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    window : int
        Rolling window
    
    Returns
    -------
    pd.Series
        Rolling Sharpe ratio
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)


def calculate_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate alpha and beta relative to benchmark.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    
    Returns
    -------
    Dict[str, float]
        alpha, beta, r_squared
    """
    # Align data
    common_idx = returns.index.intersection(benchmark_returns.index)
    r = returns.loc[common_idx]
    b = benchmark_returns.loc[common_idx]

    # Linear regression
    cov = r.cov(b)
    var_b = b.var()
    beta = cov / var_b if var_b > 0 else 0

    alpha = r.mean() - beta * b.mean()
    alpha_annual = alpha * 252

    # R-squared
    predicted = alpha + beta * b
    ss_res = ((r - predicted) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'alpha': alpha_annual,
        'beta': beta,
        'r_squared': r_squared
    }


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate Information Ratio.
    
    IR = (Portfolio Return - Benchmark Return) / Tracking Error
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    
    Returns
    -------
    float
        Information ratio
    """
    common_idx = returns.index.intersection(benchmark_returns.index)
    excess = returns.loc[common_idx] - benchmark_returns.loc[common_idx]

    tracking_error = excess.std() * np.sqrt(252)
    excess_return = excess.mean() * 252

    return excess_return / tracking_error if tracking_error > 0 else 0


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns matrix.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    
    Returns
    -------
    pd.DataFrame
        Year x Month matrix of returns
    """
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })

    pivot = df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    return pivot
