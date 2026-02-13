"""
Data cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional

from quantlab.config import get_logger

logger = get_logger(__name__)


def clean_prices(
    prices: pd.DataFrame,
    fill_method: str = 'ffill',
    max_missing_pct: float = 0.2
) -> pd.DataFrame:
    """
    Clean price data by handling missing values.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Raw price data
    fill_method : str
        Method to fill NaN: 'ffill', 'bfill', 'interpolate'
    max_missing_pct : float
        Maximum allowed missing percentage per column
    
    Returns
    -------
    pd.DataFrame
        Cleaned price data
    """
    # Calculate missing percentage per column
    missing_pct = prices.isna().sum() / len(prices)
    
    # Remove columns with too much missing data
    valid_cols = missing_pct[missing_pct <= max_missing_pct].index
    removed_cols = set(prices.columns) - set(valid_cols)
    
    if removed_cols:
        logger.warning(f"Removing {len(removed_cols)} columns with >{max_missing_pct:.0%} missing: {removed_cols}")
    
    prices = prices[valid_cols].copy()
    
    # Fill missing values
    if fill_method == 'ffill':
        prices = prices.ffill().bfill()
    elif fill_method == 'bfill':
        prices = prices.bfill().ffill()
    elif fill_method == 'interpolate':
        prices = prices.interpolate(method='linear').ffill().bfill()
    
    return prices


def detect_outliers(
    returns: pd.DataFrame,
    method: str = 'zscore',
    threshold: float = 5.0
) -> pd.DataFrame:
    """
    Detect outliers in return data.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    method : str
        Detection method: 'zscore', 'iqr', 'mad'
    threshold : float
        Threshold for outlier detection
    
    Returns
    -------
    pd.DataFrame
        Boolean mask of outliers
    """
    if method == 'zscore':
        z_scores = (returns - returns.mean()) / returns.std()
        return np.abs(z_scores) > threshold
    
    elif method == 'iqr':
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (returns < lower) | (returns > upper)
    
    elif method == 'mad':
        median = returns.median()
        mad = np.abs(returns - median).median()
        modified_z = 0.6745 * (returns - median) / (mad + 1e-10)
        return np.abs(modified_z) > threshold
    
    return pd.DataFrame(False, index=returns.index, columns=returns.columns)


def winsorize_returns(
    returns: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.DataFrame:
    """
    Winsorize returns to limit extreme values.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    lower_pct : float
        Lower percentile for clipping
    upper_pct : float
        Upper percentile for clipping
    
    Returns
    -------
    pd.DataFrame
        Winsorized returns
    """
    lower_bound = returns.quantile(lower_pct)
    upper_bound = returns.quantile(upper_pct)
    
    return returns.clip(lower=lower_bound, upper=upper_bound, axis=1)


def adjust_for_splits(
    prices: pd.DataFrame,
    splits: pd.DataFrame
) -> pd.DataFrame:
    """
    Adjust prices for stock splits.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Unadjusted prices
    splits : pd.DataFrame
        Split data with columns: date, ticker, ratio
    
    Returns
    -------
    pd.DataFrame
        Split-adjusted prices
    """
    adjusted = prices.copy()
    
    for _, split in splits.iterrows():
        ticker = split['ticker']
        date = split['date']
        ratio = split['ratio']
        
        if ticker in adjusted.columns:
            # Adjust prices before split date
            mask = adjusted.index < pd.Timestamp(date)
            adjusted.loc[mask, ticker] = adjusted.loc[mask, ticker] / ratio
    
    return adjusted


def align_data(
    *dataframes: pd.DataFrame,
    how: str = 'inner'
) -> tuple:
    """
    Align multiple DataFrames by index.
    
    Parameters
    ----------
    *dataframes : pd.DataFrame
        DataFrames to align
    how : str
        Join method: 'inner', 'outer', 'left', 'right'
    
    Returns
    -------
    tuple
        Aligned DataFrames
    """
    if len(dataframes) < 2:
        return dataframes
    
    # Get common index
    if how == 'inner':
        common_idx = dataframes[0].index
        for df in dataframes[1:]:
            common_idx = common_idx.intersection(df.index)
        return tuple(df.loc[common_idx] for df in dataframes)
    
    elif how == 'outer':
        all_idx = dataframes[0].index
        for df in dataframes[1:]:
            all_idx = all_idx.union(df.index)
        return tuple(df.reindex(all_idx) for df in dataframes)
    
    return dataframes
