"""
Market impact models and slippage metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from quantlab.microstructure.lob import LimitOrderBook


def analyze_market_order(
    book: LimitOrderBook,
    size: int,
    side: str = 'BUY'
) -> Dict:
    """
    Analyze market impact of executing a large order.
    
    Parameters
    ----------
    book : LimitOrderBook
        Order book to execute against
    size : int
        Order size
    side : str
        'BUY' or 'SELL'
    
    Returns
    -------
    Dict
        Impact metrics
    """
    # Pre-trade state
    pre_mid = book.get_mid_price()
    pre_spread = book.get_spread()
    pre_bid, _ = book.get_best_bid()
    pre_ask, _ = book.get_best_ask()
    
    # Execute
    trades, vwap = book.execute_market_order(side, size)
    
    # Post-trade state
    post_mid = book.get_mid_price()
    post_spread = book.get_spread()
    
    total_executed = sum(t.quantity for t in trades)
    
    # Calculate slippage
    if side == 'BUY':
        slippage_vs_mid = vwap - pre_mid if pre_mid else 0
        slippage_vs_best = vwap - pre_ask if pre_ask else 0
    else:
        slippage_vs_mid = pre_mid - vwap if pre_mid else 0
        slippage_vs_best = pre_bid - vwap if pre_bid else 0
    
    price_impact = abs(post_mid - pre_mid) if (post_mid and pre_mid) else 0
    
    return {
        'order_size': size,
        'executed_size': total_executed,
        'num_trades': len(trades),
        'vwap': vwap,
        'pre_mid': pre_mid,
        'post_mid': post_mid,
        'pre_spread': pre_spread,
        'post_spread': post_spread,
        'slippage_vs_mid': slippage_vs_mid,
        'slippage_vs_best': slippage_vs_best,
        'slippage_bps': (slippage_vs_mid / pre_mid * 10000) if pre_mid else 0,
        'impact_bps': (price_impact / pre_mid * 10000) if pre_mid else 0,
        'trades': trades
    }


def square_root_impact(
    size: float,
    daily_volume: float,
    volatility: float,
    eta: float = 0.1
) -> float:
    """
    Calculate market impact using square root model.
    
    Impact ≈ η × σ × √(Q / V)
    
    Parameters
    ----------
    size : float
        Order size
    daily_volume : float
        Average daily volume
    volatility : float
        Daily volatility
    eta : float
        Impact coefficient (typically 0.05-0.2)
    
    Returns
    -------
    float
        Expected price impact (percentage)
    """
    participation = size / daily_volume
    impact = eta * volatility * np.sqrt(participation)
    return impact


def linear_impact(
    size: float,
    daily_volume: float,
    lambda_coef: float = 0.1
) -> float:
    """
    Calculate market impact using linear model.
    
    Impact = λ × (Q / V)
    
    Parameters
    ----------
    size : float
        Order size
    daily_volume : float
        Average daily volume
    lambda_coef : float
        Impact coefficient
    
    Returns
    -------
    float
        Expected price impact (percentage)
    """
    participation = size / daily_volume
    return lambda_coef * participation


def almgren_chriss_impact(
    size: float,
    daily_volume: float,
    volatility: float,
    time_horizon: float,
    gamma: float = 0.1,
    eta: float = 0.1
) -> Tuple[float, float]:
    """
    Calculate temporary and permanent impact (Almgren-Chriss).
    
    Parameters
    ----------
    size : float
        Total order size
    daily_volume : float
        Average daily volume
    volatility : float
        Daily volatility
    time_horizon : float
        Execution time horizon (fraction of day)
    gamma : float
        Permanent impact coefficient
    eta : float
        Temporary impact coefficient
    
    Returns
    -------
    Tuple[float, float]
        (temporary_impact, permanent_impact)
    """
    participation = size / daily_volume
    
    # Permanent impact (linear)
    permanent = gamma * participation
    
    # Temporary impact (depends on execution speed)
    execution_rate = size / (time_horizon * daily_volume)
    temporary = eta * volatility * np.sqrt(execution_rate)
    
    return temporary, permanent


def implementation_shortfall(
    decision_price: float,
    execution_prices: pd.Series,
    execution_sizes: pd.Series,
    side: str = 'BUY'
) -> Dict:
    """
    Calculate implementation shortfall.
    
    IS = (VWAP - Decision_Price) × Total_Size
    
    Parameters
    ----------
    decision_price : float
        Price when decision was made
    execution_prices : pd.Series
        Execution prices
    execution_sizes : pd.Series
        Execution sizes
    side : str
        'BUY' or 'SELL'
    
    Returns
    -------
    Dict
        Implementation shortfall metrics
    """
    total_size = execution_sizes.sum()
    total_value = (execution_prices * execution_sizes).sum()
    vwap = total_value / total_size if total_size > 0 else 0
    
    if side == 'BUY':
        shortfall = vwap - decision_price
    else:
        shortfall = decision_price - vwap
    
    shortfall_pct = shortfall / decision_price if decision_price > 0 else 0
    shortfall_bps = shortfall_pct * 10000
    
    return {
        'decision_price': decision_price,
        'vwap': vwap,
        'total_size': total_size,
        'shortfall': shortfall,
        'shortfall_pct': shortfall_pct,
        'shortfall_bps': shortfall_bps,
        'shortfall_dollar': shortfall * total_size
    }


def impact_by_size(
    book_factory,
    sizes: list,
    side: str = 'BUY'
) -> pd.DataFrame:
    """
    Analyze impact across different order sizes.
    
    Parameters
    ----------
    book_factory : callable
        Function that creates a fresh order book
    sizes : list
        Order sizes to test
    side : str
        Order side
    
    Returns
    -------
    pd.DataFrame
        Impact metrics by size
    """
    results = []
    
    for size in sizes:
        book = book_factory()
        metrics = analyze_market_order(book, size, side)
        results.append({
            'size': size,
            'slippage_bps': metrics['slippage_bps'],
            'impact_bps': metrics['impact_bps'],
            'vwap': metrics['vwap']
        })
    
    return pd.DataFrame(results)
