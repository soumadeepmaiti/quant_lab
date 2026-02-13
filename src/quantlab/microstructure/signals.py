"""
Microstructure signals: OFI, spread, depth, VPIN.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def order_flow_imbalance(
    events: List[Dict],
    window: int = 1
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI).
    
    OFI = Δ(bid_size) - Δ(ask_size)
    
    Positive OFI = buying pressure
    Negative OFI = selling pressure
    
    Parameters
    ----------
    events : List[Dict]
        List of order book events with bid/ask states
    window : int
        Smoothing window
    
    Returns
    -------
    pd.Series
        OFI values
    """
    ofi_values = []

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]

        # Bid side contribution
        bid_delta = 0
        if curr.get('best_bid') and prev.get('best_bid'):
            if curr['best_bid'] >= prev['best_bid']:
                bid_delta = curr.get('bid_size', 0) - prev.get('bid_size', 0)
            else:
                bid_delta = -prev.get('bid_size', 0)

        # Ask side contribution
        ask_delta = 0
        if curr.get('best_ask') and prev.get('best_ask'):
            if curr['best_ask'] <= prev['best_ask']:
                ask_delta = curr.get('ask_size', 0) - prev.get('ask_size', 0)
            else:
                ask_delta = -prev.get('ask_size', 0)

        ofi = bid_delta - ask_delta
        ofi_values.append(ofi)

    ofi_series = pd.Series(ofi_values)

    if window > 1:
        ofi_series = ofi_series.rolling(window, min_periods=1).mean()

    return ofi_series


def calculate_ofi_from_book(
    bid_before: Tuple[float, int],
    ask_before: Tuple[float, int],
    bid_after: Tuple[float, int],
    ask_after: Tuple[float, int]
) -> float:
    """
    Calculate OFI from before/after book states.
    
    Parameters
    ----------
    bid_before, ask_before : Tuple[float, int]
        (price, size) before event
    bid_after, ask_after : Tuple[float, int]
        (price, size) after event
    
    Returns
    -------
    float
        OFI value
    """
    bid_p_before, bid_s_before = bid_before
    ask_p_before, ask_s_before = ask_before
    bid_p_after, bid_s_after = bid_after
    ask_p_after, ask_s_after = ask_after

    # Bid contribution
    bid_delta = 0
    if bid_p_before and bid_p_after:
        if bid_p_after >= bid_p_before:
            bid_delta = (bid_s_after or 0) - (bid_s_before or 0)
        else:
            bid_delta = -(bid_s_before or 0)

    # Ask contribution
    ask_delta = 0
    if ask_p_before and ask_p_after:
        if ask_p_after <= ask_p_before:
            ask_delta = (ask_s_after or 0) - (ask_s_before or 0)
        else:
            ask_delta = -(ask_s_before or 0)

    return bid_delta - ask_delta


def relative_spread(
    bid_prices: pd.Series,
    ask_prices: pd.Series
) -> pd.Series:
    """
    Calculate relative bid-ask spread.
    
    Relative Spread = (Ask - Bid) / Mid
    
    Parameters
    ----------
    bid_prices : pd.Series
        Best bid prices
    ask_prices : pd.Series
        Best ask prices
    
    Returns
    -------
    pd.Series
        Relative spread in basis points
    """
    mid = (bid_prices + ask_prices) / 2
    spread = ask_prices - bid_prices
    relative = (spread / mid) * 10000  # Basis points
    return relative


def depth_imbalance(
    bid_sizes: pd.Series,
    ask_sizes: pd.Series
) -> pd.Series:
    """
    Calculate depth imbalance.
    
    Imbalance = (Bid_Size - Ask_Size) / (Bid_Size + Ask_Size)
    
    Parameters
    ----------
    bid_sizes : pd.Series
        Total bid depth
    ask_sizes : pd.Series
        Total ask depth
    
    Returns
    -------
    pd.Series
        Imbalance values (-1 to 1)
    """
    total = bid_sizes + ask_sizes
    imbalance = (bid_sizes - ask_sizes) / total.replace(0, np.nan)
    return imbalance


def vpin(
    trades: pd.DataFrame,
    bucket_size: int = 50,
    n_buckets: int = 50
) -> pd.Series:
    """
    Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
    
    VPIN measures order flow toxicity.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with columns: price, size, side
    bucket_size : int
        Volume per bucket
    n_buckets : int
        Number of buckets for moving average
    
    Returns
    -------
    pd.Series
        VPIN values
    """
    if len(trades) == 0:
        return pd.Series()

    # Classify trades as buy/sell (if not already classified)
    if 'side' not in trades.columns:
        # Use tick rule: price up = buy, price down = sell
        trades = trades.copy()
        price_diff = trades['price'].diff()
        trades['side'] = np.where(price_diff > 0, 'BUY', 'SELL')

    # Create volume buckets
    cumulative_volume = trades['size'].cumsum()
    bucket_ids = (cumulative_volume // bucket_size).astype(int)

    # Calculate buy/sell volume per bucket
    trades['bucket'] = bucket_ids

    buy_vol = trades[trades['side'] == 'BUY'].groupby('bucket')['size'].sum()
    sell_vol = trades[trades['side'] == 'SELL'].groupby('bucket')['size'].sum()

    buy_vol = buy_vol.reindex(range(bucket_ids.max() + 1), fill_value=0)
    sell_vol = sell_vol.reindex(range(bucket_ids.max() + 1), fill_value=0)

    # Calculate order imbalance per bucket
    imbalance = np.abs(buy_vol - sell_vol)
    total_vol = buy_vol + sell_vol

    # VPIN is rolling average of |OI| / V
    vpin_values = imbalance.rolling(n_buckets).sum() / total_vol.rolling(n_buckets).sum()

    return vpin_values


def kyle_lambda(
    price_changes: pd.Series,
    signed_volume: pd.Series,
    window: int = 100
) -> pd.Series:
    """
    Estimate Kyle's lambda (price impact coefficient).
    
    ΔP = λ × (Buy_Volume - Sell_Volume)
    
    Parameters
    ----------
    price_changes : pd.Series
        Price changes
    signed_volume : pd.Series
        Signed order flow
    window : int
        Rolling window for estimation
    
    Returns
    -------
    pd.Series
        Kyle lambda estimates
    """
    lambdas = []

    for i in range(window, len(price_changes)):
        dp = price_changes.iloc[i - window:i]
        sv = signed_volume.iloc[i - window:i]

        # Simple linear regression
        cov = dp.cov(sv)
        var_sv = sv.var()

        lambda_est = cov / var_sv if var_sv > 0 else np.nan
        lambdas.append(lambda_est)

    return pd.Series(lambdas, index=price_changes.index[window:])


def trade_flow_toxicity(
    trades: pd.DataFrame,
    window: int = 100
) -> pd.Series:
    """
    Calculate trade flow toxicity metric.
    
    Toxicity = |Net_Buyer_Volume| / Total_Volume
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade data
    window : int
        Rolling window
    
    Returns
    -------
    pd.Series
        Toxicity values (0 to 1)
    """
    if 'side' not in trades.columns:
        return pd.Series()

    buy_vol = trades[trades['side'] == 'BUY']['size']
    sell_vol = trades[trades['side'] == 'SELL']['size']

    net_flow = buy_vol.cumsum() - sell_vol.cumsum()
    total_vol = buy_vol.cumsum() + sell_vol.cumsum()

    toxicity = np.abs(net_flow).rolling(window).sum() / total_vol.rolling(window).sum()

    return toxicity
