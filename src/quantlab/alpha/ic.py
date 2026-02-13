"""
Information Coefficient (IC) analysis for factor evaluation.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def calculate_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'spearman'
) -> pd.Series:
    """
    Calculate Information Coefficient time series.
    
    IC = Cross-sectional correlation between factor values
         and subsequent returns for each time period.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x tickers)
    forward_returns : pd.DataFrame
        Forward returns (dates x tickers)
    method : str
        'spearman' (rank) or 'pearson' (linear)
    
    Returns
    -------
    pd.Series
        IC values indexed by date
    """
    common_dates = factor.index.intersection(forward_returns.index)

    ic_values = []
    dates = []

    for date in common_dates:
        factor_row = factor.loc[date]
        return_row = forward_returns.loc[date]

        # Remove NaN
        valid = ~(factor_row.isna() | return_row.isna())

        if valid.sum() >= 5:
            f = factor_row[valid]
            r = return_row[valid]

            if method == 'spearman':
                ic, _ = spearmanr(f, r)
            else:
                ic, _ = pearsonr(f, r)

            if not np.isnan(ic):
                ic_values.append(ic)
                dates.append(date)

    return pd.Series(ic_values, index=dates, name='IC')


def calculate_forward_returns(
    prices: pd.DataFrame,
    periods: List[int] = [5, 10, 21]
) -> Dict[str, pd.DataFrame]:
    """
    Calculate forward returns for multiple horizons.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    periods : List[int]
        Forward periods in trading days
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        {period_name: forward_returns_df}
    """
    forward_returns = {}

    for period in periods:
        fwd_ret = prices.shift(-period) / prices - 1
        forward_returns[f'{period}D'] = fwd_ret

    return forward_returns


def ic_analysis(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = 'spearman'
) -> Dict[str, float]:
    """
    Comprehensive IC analysis with statistics.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    forward_returns : pd.DataFrame
        Forward returns
    method : str
        Correlation method
    
    Returns
    -------
    Dict[str, float]
        IC statistics: mean, std, ir, t_stat, pct_positive
    """
    ic_series = calculate_ic(factor, forward_returns, method)

    if len(ic_series) == 0:
        return {}

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0
    pct_positive = (ic_series > 0).mean()

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ir': ir,
        't_stat': t_stat,
        'pct_positive': pct_positive,
        'n_periods': len(ic_series)
    }


def ic_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    periods: List[int] = [1, 5, 10, 21, 42, 63]
) -> pd.DataFrame:
    """
    Calculate IC decay over different forward horizons.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    prices : pd.DataFrame
        Price data
    periods : List[int]
        Forward periods to test
    
    Returns
    -------
    pd.DataFrame
        IC statistics for each period
    """
    results = []

    for period in periods:
        fwd_ret = prices.shift(-period) / prices - 1
        stats = ic_analysis(factor, fwd_ret)
        stats['period'] = period
        results.append(stats)

    return pd.DataFrame(results).set_index('period')


def quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Calculate average forward returns by factor quantile.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    forward_returns : pd.DataFrame
        Forward returns
    n_quantiles : int
        Number of quantiles (5 = quintiles, 10 = deciles)
    
    Returns
    -------
    pd.DataFrame
        Mean returns for each quantile
    """
    common_dates = factor.index.intersection(forward_returns.index)

    quantile_rets = {q: [] for q in range(1, n_quantiles + 1)}

    for date in common_dates:
        factor_row = factor.loc[date].dropna()
        return_row = forward_returns.loc[date]

        if len(factor_row) >= n_quantiles:
            # Assign quantiles
            quantiles = pd.qcut(factor_row, n_quantiles, labels=range(1, n_quantiles + 1))

            for q in range(1, n_quantiles + 1):
                q_tickers = quantiles[quantiles == q].index
                q_returns = return_row[q_tickers].dropna()
                if len(q_returns) > 0:
                    quantile_rets[q].append(q_returns.mean())

    # Calculate mean for each quantile
    mean_returns = {q: np.mean(rets) if rets else np.nan
                   for q, rets in quantile_rets.items()}

    return pd.DataFrame({
        'quantile': list(mean_returns.keys()),
        'mean_return': list(mean_returns.values())
    }).set_index('quantile')


def factor_turnover(
    factor: pd.DataFrame,
    n_quantiles: int = 5,
    top_quantile: int = None
) -> pd.Series:
    """
    Calculate factor portfolio turnover.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    n_quantiles : int
        Number of quantiles
    top_quantile : int, optional
        Specific quantile to track (default: top)
    
    Returns
    -------
    pd.Series
        Turnover percentage per period
    """
    if top_quantile is None:
        top_quantile = n_quantiles

    turnovers = []
    prev_holdings = None

    for date in factor.index:
        factor_row = factor.loc[date].dropna()

        if len(factor_row) >= n_quantiles:
            quantiles = pd.qcut(factor_row, n_quantiles, labels=range(1, n_quantiles + 1))
            current_holdings = set(quantiles[quantiles == top_quantile].index)

            if prev_holdings is not None:
                # Calculate turnover
                common = len(current_holdings & prev_holdings)
                turnover = 1 - common / len(current_holdings) if current_holdings else 0
                turnovers.append({'date': date, 'turnover': turnover})

            prev_holdings = current_holdings

    return pd.DataFrame(turnovers).set_index('date')['turnover']
