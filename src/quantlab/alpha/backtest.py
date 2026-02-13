"""
Backtesting engine for factor strategies.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from quantlab.alpha.evaluation import calculate_metrics
from quantlab.alpha.portfolio import (
    calculate_portfolio_returns,
    construct_long_short,
    rebalance_weights,
)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio_returns: pd.Series
    long_returns: pd.Series
    short_returns: pd.Series
    long_weights: pd.DataFrame
    short_weights: pd.DataFrame
    metrics: Dict[str, float]

    def __repr__(self):
        return (
            f"BacktestResult(\n"
            f"  sharpe={self.metrics.get('sharpe', 0):.2f},\n"
            f"  annual_return={self.metrics.get('annual_return', 0):.1%},\n"
            f"  max_drawdown={self.metrics.get('max_drawdown', 0):.1%}\n"
            f")"
        )


def run_long_short(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    rebalance_freq: str = 'M',
    weighting: str = 'equal',
    transaction_cost_bps: float = 0.0
) -> BacktestResult:
    """
    Run a long-short factor backtest.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x tickers)
    returns : pd.DataFrame
        Asset returns (dates x tickers)
    top_pct : float
        Percentage to go long
    bottom_pct : float
        Percentage to go short
    rebalance_freq : str
        'D', 'W', 'M' for daily, weekly, monthly
    weighting : str
        'equal' or 'factor'
    transaction_cost_bps : float
        Transaction cost in basis points (one-way)
    
    Returns
    -------
    BacktestResult
        Backtest results container
    """
    # Construct portfolio weights
    long_w, short_w = construct_long_short(
        factor, top_pct, bottom_pct, weighting
    )

    # Apply rebalancing
    long_w = rebalance_weights(long_w, rebalance_freq)
    short_w = rebalance_weights(short_w, rebalance_freq)

    # Calculate returns
    portfolio_ret = calculate_portfolio_returns(long_w, short_w, returns)

    # Apply transaction costs
    if transaction_cost_bps > 0:
        turnover = _calculate_turnover(long_w, short_w)
        cost = turnover * transaction_cost_bps / 10000
        portfolio_ret = portfolio_ret - cost

    # Calculate long-only and short-only returns for analysis
    long_ret = (long_w.shift(1) * returns).sum(axis=1).loc[portfolio_ret.index]
    short_ret = -(short_w.shift(1) * returns).sum(axis=1).loc[portfolio_ret.index]

    # Calculate metrics
    metrics = calculate_metrics(portfolio_ret)

    return BacktestResult(
        portfolio_returns=portfolio_ret,
        long_returns=long_ret,
        short_returns=short_ret,
        long_weights=long_w,
        short_weights=short_w,
        metrics=metrics
    )


def run_long_only(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    top_pct: float = 0.2,
    rebalance_freq: str = 'M',
    weighting: str = 'equal',
    transaction_cost_bps: float = 0.0
) -> BacktestResult:
    """
    Run a long-only factor backtest.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    returns : pd.DataFrame
        Asset returns
    benchmark_returns : pd.Series, optional
        Benchmark returns for alpha calculation
    top_pct : float
        Percentage of stocks to hold
    rebalance_freq : str
        Rebalancing frequency
    weighting : str
        Weight scheme
    transaction_cost_bps : float
        Transaction costs
    
    Returns
    -------
    BacktestResult
        Backtest results
    """
    from quantlab.alpha.portfolio import construct_long_only

    # Construct weights
    weights = construct_long_only(factor, top_pct, weighting)
    weights = rebalance_weights(weights, rebalance_freq)

    # Calculate returns
    portfolio_ret = (weights.shift(1) * returns).sum(axis=1).dropna()

    # Apply costs
    if transaction_cost_bps > 0:
        turnover = weights.diff().abs().sum(axis=1) / 2
        cost = turnover * transaction_cost_bps / 10000
        portfolio_ret = portfolio_ret - cost.loc[portfolio_ret.index]

    # Metrics
    metrics = calculate_metrics(portfolio_ret)

    # Add alpha vs benchmark if provided
    if benchmark_returns is not None:
        common_idx = portfolio_ret.index.intersection(benchmark_returns.index)
        excess_ret = portfolio_ret.loc[common_idx] - benchmark_returns.loc[common_idx]
        metrics['alpha'] = excess_ret.mean() * 252

    return BacktestResult(
        portfolio_returns=portfolio_ret,
        long_returns=portfolio_ret,
        short_returns=pd.Series(0, index=portfolio_ret.index),
        long_weights=weights,
        short_weights=pd.DataFrame(0, index=weights.index, columns=weights.columns),
        metrics=metrics
    )


def _calculate_turnover(
    long_w: pd.DataFrame,
    short_w: pd.DataFrame
) -> pd.Series:
    """Calculate portfolio turnover."""
    long_turnover = long_w.diff().abs().sum(axis=1) / 2
    short_turnover = short_w.diff().abs().sum(axis=1) / 2
    return long_turnover + short_turnover


def compare_strategies(
    strategies: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Compare multiple strategy backtests.
    
    Parameters
    ----------
    strategies : Dict[str, BacktestResult]
        {strategy_name: backtest_result}
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    rows = []
    for name, result in strategies.items():
        row = {'strategy': name}
        row.update(result.metrics)
        rows.append(row)

    return pd.DataFrame(rows).set_index('strategy')


def rolling_backtest(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 252,
    **kwargs
) -> pd.DataFrame:
    """
    Run rolling window backtest for robustness analysis.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    returns : pd.DataFrame
        Asset returns
    window : int
        Rolling window in trading days
    **kwargs
        Additional arguments for run_long_short
    
    Returns
    -------
    pd.DataFrame
        Rolling metrics
    """
    results = []
    dates = factor.index[window:]

    for i, end_date in enumerate(dates):
        start_idx = i
        end_idx = i + window

        factor_window = factor.iloc[start_idx:end_idx]
        returns_window = returns.iloc[start_idx:end_idx]

        try:
            result = run_long_short(factor_window, returns_window, **kwargs)
            results.append({
                'date': end_date,
                **result.metrics
            })
        except Exception:
            continue

    return pd.DataFrame(results).set_index('date')
