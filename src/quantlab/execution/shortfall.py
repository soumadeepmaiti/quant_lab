"""
Implementation Shortfall Decomposition Module.

Implements institutional-grade execution analysis:
- IS decomposition (spread, impact, timing, opportunity)
- Execution quality metrics
- Benchmark comparison (VWAP, arrival price)
- Cost attribution

References:
- Perold (1988) - "The Implementation Shortfall"
- Almgren & Chriss (2000) - "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2003) - "Optimal Trading Strategies"
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from quantlab.config import get_logger

logger = get_logger(__name__)


@dataclass
class ISDecomposition:
    """Implementation Shortfall decomposition result."""
    total_is: float           # Total implementation shortfall
    spread_cost: float        # Bid-ask spread cost
    impact_cost: float        # Market impact cost
    timing_cost: float        # Timing/delay cost
    opportunity_cost: float   # Unfilled portion cost
    execution_rate: float     # Percentage of order filled

    # Per-share costs (in bps)
    spread_bps: float
    impact_bps: float
    timing_bps: float
    total_bps: float


def implementation_shortfall(
    decision_price: float,
    execution_prices: np.ndarray,
    execution_sizes: np.ndarray,
    total_order_size: int,
    final_price: float,
    bid_ask_spread: float,
    side: str = 'BUY'
) -> ISDecomposition:
    """
    Calculate Implementation Shortfall and decompose into components.
    
    IS = Paper Return - Actual Return
       = (Final Price - Decision Price) × Shares Intended
         - Σ(Final Price - Execution Price) × Shares Executed
    
    Components:
    1. Spread Cost: Half spread × Executed Shares
    2. Impact Cost: Price moved due to our trading
    3. Timing Cost: Price moved while we were trading
    4. Opportunity Cost: Missed profit on unfilled shares
    
    Parameters
    ----------
    decision_price : float
        Price at decision time (arrival price)
    execution_prices : np.ndarray
        Execution prices for each fill
    execution_sizes : np.ndarray
        Sizes for each fill
    total_order_size : int
        Total intended order size
    final_price : float
        Price at end of execution period
    bid_ask_spread : float
        Average bid-ask spread
    side : str
        'BUY' or 'SELL'
    
    Returns
    -------
    ISDecomposition
        Detailed cost breakdown
    
    Example
    -------
    >>> result = implementation_shortfall(
    ...     decision_price=100.0,
    ...     execution_prices=np.array([100.05, 100.10, 100.15]),
    ...     execution_sizes=np.array([1000, 1000, 1000]),
    ...     total_order_size=5000,
    ...     final_price=100.20,
    ...     bid_ask_spread=0.02,
    ...     side='BUY'
    ... )
    """
    executed_size = np.sum(execution_sizes)
    unfilled_size = total_order_size - executed_size
    execution_rate = executed_size / total_order_size if total_order_size > 0 else 0

    # Average execution price
    if executed_size > 0:
        vwap = np.sum(execution_prices * execution_sizes) / executed_size
    else:
        vwap = decision_price

    # Direction multiplier
    sign = 1 if side.upper() == 'BUY' else -1

    # Paper return (what we would have made if executed at decision price)
    paper_return = sign * (final_price - decision_price) * total_order_size

    # Actual return (what we actually made)
    actual_return = sign * (final_price - vwap) * executed_size

    # Total Implementation Shortfall
    total_is = paper_return - actual_return

    # --- Cost Decomposition ---

    # 1. Spread Cost: immediate cost of crossing the spread
    half_spread = bid_ask_spread / 2
    spread_cost = half_spread * executed_size

    # 2. Impact Cost: price moved due to our trading
    # Estimated as execution price vs mid-price at execution time
    # Simplified: difference between VWAP and (decision_price + half_spread)
    expected_price = decision_price + sign * half_spread
    impact_cost = sign * (vwap - expected_price) * executed_size
    impact_cost = max(0, impact_cost)  # Impact should be non-negative

    # 3. Timing Cost: price drift during execution
    # Price change from decision to end, minus what we captured
    price_drift = sign * (final_price - decision_price)
    timing_cost = price_drift * executed_size - (final_price - vwap) * executed_size * sign
    timing_cost = max(0, timing_cost)

    # 4. Opportunity Cost: missed profit on unfilled shares
    opportunity_cost = sign * (final_price - decision_price) * unfilled_size
    opportunity_cost = max(0, opportunity_cost)  # Only if favorable price move

    # Convert to basis points
    spread_bps = (spread_cost / (decision_price * executed_size)) * 10000 if executed_size > 0 else 0
    impact_bps = (impact_cost / (decision_price * executed_size)) * 10000 if executed_size > 0 else 0
    timing_bps = (timing_cost / (decision_price * executed_size)) * 10000 if executed_size > 0 else 0
    total_bps = spread_bps + impact_bps + timing_bps

    return ISDecomposition(
        total_is=total_is,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        timing_cost=timing_cost,
        opportunity_cost=opportunity_cost,
        execution_rate=execution_rate,
        spread_bps=spread_bps,
        impact_bps=impact_bps,
        timing_bps=timing_bps,
        total_bps=total_bps
    )


def vwap_benchmark(
    execution_prices: np.ndarray,
    execution_sizes: np.ndarray,
    market_prices: np.ndarray,
    market_volumes: np.ndarray
) -> Dict[str, float]:
    """
    Calculate VWAP benchmark performance.
    
    Parameters
    ----------
    execution_prices : np.ndarray
        Our execution prices
    execution_sizes : np.ndarray
        Our execution sizes
    market_prices : np.ndarray
        Market VWAP prices during execution
    market_volumes : np.ndarray
        Market volumes during execution
    
    Returns
    -------
    dict
        VWAP comparison metrics
    """
    # Our VWAP
    our_size = np.sum(execution_sizes)
    our_vwap = np.sum(execution_prices * execution_sizes) / our_size if our_size > 0 else 0

    # Market VWAP
    total_volume = np.sum(market_volumes)
    market_vwap = np.sum(market_prices * market_volumes) / total_volume if total_volume > 0 else 0

    # VWAP slippage
    slippage = our_vwap - market_vwap
    slippage_bps = (slippage / market_vwap) * 10000 if market_vwap > 0 else 0

    return {
        'our_vwap': our_vwap,
        'market_vwap': market_vwap,
        'slippage': slippage,
        'slippage_bps': slippage_bps,
        'outperformed': slippage < 0  # Negative = we got better price
    }


def arrival_price_benchmark(
    execution_prices: np.ndarray,
    execution_sizes: np.ndarray,
    arrival_price: float,
    side: str = 'BUY'
) -> Dict[str, float]:
    """
    Calculate arrival price benchmark performance.
    
    Parameters
    ----------
    execution_prices : np.ndarray
        Execution prices
    execution_sizes : np.ndarray
        Execution sizes
    arrival_price : float
        Price at order arrival
    side : str
        'BUY' or 'SELL'
    
    Returns
    -------
    dict
        Arrival price comparison
    """
    total_size = np.sum(execution_sizes)
    vwap = np.sum(execution_prices * execution_sizes) / total_size if total_size > 0 else 0

    sign = 1 if side.upper() == 'BUY' else -1

    slippage = sign * (vwap - arrival_price)
    slippage_bps = (slippage / arrival_price) * 10000

    return {
        'vwap': vwap,
        'arrival_price': arrival_price,
        'slippage': slippage,
        'slippage_bps': slippage_bps,
        'total_cost': slippage * total_size
    }


def execution_quality_metrics(
    execution_prices: np.ndarray,
    execution_sizes: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    side: str = 'BUY'
) -> Dict[str, float]:
    """
    Calculate execution quality metrics.
    
    Parameters
    ----------
    execution_prices : np.ndarray
        Execution prices
    execution_sizes : np.ndarray
        Execution sizes
    best_bid : np.ndarray
        Best bid at execution time
    best_ask : np.ndarray
        Best ask at execution time
    side : str
        'BUY' or 'SELL'
    
    Returns
    -------
    dict
        Quality metrics
    """
    n_executions = len(execution_prices)

    if n_executions == 0:
        return {}

    # Mid price at each execution
    mid_prices = (best_bid + best_ask) / 2
    spreads = best_ask - best_bid

    # Price improvement analysis
    if side.upper() == 'BUY':
        # Buy: better if execution < ask
        reference = best_ask
        improvement = reference - execution_prices
    else:
        # Sell: better if execution > bid
        reference = best_bid
        improvement = execution_prices - reference

    # Effective spread = 2 × |execution - mid|
    effective_spread = 2 * np.abs(execution_prices - mid_prices)

    # Realized spread (with price 5 periods later - simplified)
    # This would need future prices, so we estimate

    # Fill rate at each price level
    size_weighted_improvement = np.sum(improvement * execution_sizes) / np.sum(execution_sizes)

    return {
        'n_executions': n_executions,
        'avg_improvement': np.mean(improvement),
        'size_weighted_improvement': size_weighted_improvement,
        'pct_at_midpoint': (np.abs(improvement) < spreads / 4).mean(),
        'avg_effective_spread': np.mean(effective_spread),
        'avg_quoted_spread': np.mean(spreads),
        'effective_vs_quoted': np.mean(effective_spread) / np.mean(spreads) if np.mean(spreads) > 0 else 1
    }


def cost_curve_analysis(
    order_sizes: List[int],
    execution_func,
    base_price: float = 100.0
) -> pd.DataFrame:
    """
    Generate execution cost curve for different order sizes.
    
    Parameters
    ----------
    order_sizes : list
        Order sizes to test
    execution_func : callable
        Function that takes size and returns (execution_prices, execution_sizes)
    base_price : float
        Reference price for bps calculation
    
    Returns
    -------
    pd.DataFrame
        Cost curve data
    """
    results = []

    for size in order_sizes:
        exec_prices, exec_sizes = execution_func(size)

        if len(exec_prices) == 0:
            continue

        vwap = np.sum(exec_prices * exec_sizes) / np.sum(exec_sizes)
        slippage = vwap - base_price
        slippage_bps = slippage / base_price * 10000

        # Impact per share
        impact_per_share = slippage / size if size > 0 else 0

        results.append({
            'order_size': size,
            'vwap': vwap,
            'slippage': slippage,
            'slippage_bps': slippage_bps,
            'impact_per_share': impact_per_share,
            'total_cost': slippage * size
        })

    return pd.DataFrame(results)


def compare_execution_strategies(
    strategies: Dict[str, callable],
    order_sizes: List[int],
    base_price: float = 100.0
) -> pd.DataFrame:
    """
    Compare multiple execution strategies across order sizes.
    
    Parameters
    ----------
    strategies : dict
        Dictionary of strategy name to execution function
    order_sizes : list
        Order sizes to test
    base_price : float
        Reference price
    
    Returns
    -------
    pd.DataFrame
        Comparison across strategies and sizes
    """
    all_results = []

    for strategy_name, exec_func in strategies.items():
        for size in order_sizes:
            try:
                exec_prices, exec_sizes = exec_func(size)

                if len(exec_prices) == 0:
                    continue

                vwap = np.sum(exec_prices * exec_sizes) / np.sum(exec_sizes)
                slippage_bps = (vwap - base_price) / base_price * 10000

                all_results.append({
                    'strategy': strategy_name,
                    'order_size': size,
                    'slippage_bps': slippage_bps,
                    'n_fills': len(exec_prices)
                })
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed for size {size}: {e}")
                continue

    return pd.DataFrame(all_results)


def is_attribution_report(
    is_decomp: ISDecomposition,
    order_info: Dict[str, any]
) -> str:
    """
    Generate human-readable IS attribution report.
    
    Parameters
    ----------
    is_decomp : ISDecomposition
        Decomposition result
    order_info : dict
        Order details
    
    Returns
    -------
    str
        Formatted report
    """
    report = f"""
================================================================================
                    IMPLEMENTATION SHORTFALL REPORT
================================================================================

Order Details:
  Symbol:        {order_info.get('symbol', 'N/A')}
  Side:          {order_info.get('side', 'N/A')}
  Target Size:   {order_info.get('target_size', 'N/A'):,} shares
  Executed:      {order_info.get('executed_size', 'N/A'):,} shares ({is_decomp.execution_rate*100:.1f}%)

Cost Breakdown (basis points):
--------------------------------------------------------------------------------
  Spread Cost:      {is_decomp.spread_bps:8.2f} bps
  Impact Cost:      {is_decomp.impact_bps:8.2f} bps
  Timing Cost:      {is_decomp.timing_bps:8.2f} bps
  ─────────────────────────────────────
  Total:            {is_decomp.total_bps:8.2f} bps

Dollar Costs:
--------------------------------------------------------------------------------
  Spread:          ${is_decomp.spread_cost:12,.2f}
  Impact:          ${is_decomp.impact_cost:12,.2f}
  Timing:          ${is_decomp.timing_cost:12,.2f}
  Opportunity:     ${is_decomp.opportunity_cost:12,.2f}
  ─────────────────────────────────────
  Total IS:        ${is_decomp.total_is:12,.2f}

================================================================================
"""
    return report


def liquidity_regime_analysis(
    execution_results: List[ISDecomposition],
    spreads: np.ndarray,
    volumes: np.ndarray
) -> pd.DataFrame:
    """
    Analyze execution costs across liquidity regimes.
    
    Parameters
    ----------
    execution_results : list
        IS results for different executions
    spreads : np.ndarray
        Spreads at execution time
    volumes : np.ndarray
        Volumes at execution time
    
    Returns
    -------
    pd.DataFrame
        Cost analysis by liquidity regime
    """
    # Define regimes based on spread percentiles
    spread_pct = np.percentile(spreads, [33, 67])

    results = []

    for i, is_result in enumerate(execution_results):
        if i >= len(spreads):
            continue

        spread = spreads[i]
        volume = volumes[i]

        # Classify regime
        if spread <= spread_pct[0]:
            regime = 'Tight Spread'
        elif spread <= spread_pct[1]:
            regime = 'Normal Spread'
        else:
            regime = 'Wide Spread'

        results.append({
            'regime': regime,
            'spread': spread,
            'volume': volume,
            'total_bps': is_result.total_bps,
            'impact_bps': is_result.impact_bps,
            'spread_bps': is_result.spread_bps
        })

    df = pd.DataFrame(results)

    # Aggregate by regime
    summary = df.groupby('regime').agg({
        'total_bps': ['mean', 'std', 'count'],
        'impact_bps': 'mean',
        'spread_bps': 'mean'
    }).round(2)

    return summary
