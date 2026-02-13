"""
Portfolio construction for factor strategies.
"""

from typing import Tuple

import pandas as pd


def construct_long_short(
    factor: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    weighting: str = 'equal'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct long-short portfolio weights from factor values.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x tickers)
    top_pct : float
        Percentage of stocks to go long (0.2 = top 20%)
    bottom_pct : float
        Percentage of stocks to short (0.2 = bottom 20%)
    weighting : str
        'equal' or 'factor' (weight by factor value)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (long_weights, short_weights) DataFrames
    """
    long_weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    short_weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date in factor.index:
        factor_row = factor.loc[date].dropna()

        if len(factor_row) < 5:
            continue

        n_long = max(1, int(len(factor_row) * top_pct))
        n_short = max(1, int(len(factor_row) * bottom_pct))

        sorted_tickers = factor_row.sort_values(ascending=False)
        long_tickers = sorted_tickers.head(n_long).index
        short_tickers = sorted_tickers.tail(n_short).index

        if weighting == 'equal':
            long_weights.loc[date, long_tickers] = 1.0 / n_long
            short_weights.loc[date, short_tickers] = 1.0 / n_short
        else:
            # Weight by factor value
            long_vals = factor_row[long_tickers]
            short_vals = -factor_row[short_tickers]  # Invert for shorts

            long_weights.loc[date, long_tickers] = long_vals / long_vals.sum()
            short_weights.loc[date, short_tickers] = short_vals / short_vals.sum()

    return long_weights, short_weights


def construct_long_only(
    factor: pd.DataFrame,
    top_pct: float = 0.2,
    weighting: str = 'equal',
    max_weight: float = 0.1
) -> pd.DataFrame:
    """
    Construct long-only portfolio weights.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    top_pct : float
        Percentage of stocks to hold
    weighting : str
        'equal', 'factor', or 'market_cap'
    max_weight : float
        Maximum weight per stock
    
    Returns
    -------
    pd.DataFrame
        Portfolio weights
    """
    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)

    for date in factor.index:
        factor_row = factor.loc[date].dropna()

        if len(factor_row) < 2:
            continue

        n_stocks = max(1, int(len(factor_row) * top_pct))
        sorted_tickers = factor_row.sort_values(ascending=False)
        selected = sorted_tickers.head(n_stocks).index

        if weighting == 'equal':
            w = 1.0 / n_stocks
            weights.loc[date, selected] = min(w, max_weight)
        else:
            vals = factor_row[selected]
            raw_weights = vals / vals.sum()
            weights.loc[date, selected] = raw_weights.clip(upper=max_weight)

        # Renormalize to sum to 1
        row_sum = weights.loc[date].sum()
        if row_sum > 0:
            weights.loc[date] = weights.loc[date] / row_sum

    return weights


def calculate_portfolio_returns(
    long_weights: pd.DataFrame,
    short_weights: pd.DataFrame,
    returns: pd.DataFrame,
    leverage: float = 1.0
) -> pd.Series:
    """
    Calculate portfolio returns from weights.
    
    Parameters
    ----------
    long_weights : pd.DataFrame
        Long position weights
    short_weights : pd.DataFrame
        Short position weights
    returns : pd.DataFrame
        Asset returns
    leverage : float
        Gross leverage (1.0 = dollar neutral)
    
    Returns
    -------
    pd.Series
        Portfolio returns
    """
    # Align data
    common_dates = long_weights.index.intersection(returns.index)
    common_tickers = long_weights.columns.intersection(returns.columns)

    long_w = long_weights.loc[common_dates, common_tickers]
    short_w = short_weights.loc[common_dates, common_tickers]
    ret = returns.loc[common_dates, common_tickers]

    # Calculate returns
    long_ret = (long_w.shift(1) * ret).sum(axis=1)
    short_ret = -(short_w.shift(1) * ret).sum(axis=1)  # Negative for short

    portfolio_ret = (long_ret + short_ret) * leverage / 2

    return portfolio_ret.dropna()


def rebalance_weights(
    weights: pd.DataFrame,
    rebalance_freq: str = 'M'
) -> pd.DataFrame:
    """
    Apply rebalancing frequency to weights.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Daily weights
    rebalance_freq : str
        'D' (daily), 'W' (weekly), 'M' (monthly)
    
    Returns
    -------
    pd.DataFrame
        Rebalanced weights
    """
    if rebalance_freq == 'D':
        return weights

    # Get rebalance dates
    if rebalance_freq == 'W':
        rebal_dates = weights.resample('W').last().index
    elif rebalance_freq == 'M':
        rebal_dates = weights.resample('ME').last().index
    else:
        rebal_dates = weights.resample(rebalance_freq).last().index

    # Create new weights that hold positions until next rebalance
    new_weights = pd.DataFrame(index=weights.index, columns=weights.columns, dtype=float)

    current_weights = None
    for date in weights.index:
        if date in rebal_dates or current_weights is None:
            current_weights = weights.loc[date]
        new_weights.loc[date] = current_weights

    return new_weights.fillna(0.0)
