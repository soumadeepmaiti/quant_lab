"""
Regime Analysis & Parameter Sensitivity Module.

Implements research-grade robustness testing:
- Multi-regime performance analysis
- Parameter sensitivity heatmaps
- Out-of-sample stability testing

References:
- Harvey et al. (2016) - "...and the Cross-Section of Expected Returns"
- McLean & Pontiff (2016) - "Does Academic Research Destroy Stock Return Predictability?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from itertools import product
import warnings

from quantlab.config import get_logger

logger = get_logger(__name__)


# Define market regimes
REGIMES = {
    'pre_2018': ('2016-01-01', '2017-12-31'),
    'pre_covid': ('2018-01-01', '2020-02-28'),
    'covid_crash': ('2020-03-01', '2020-06-30'),
    'post_covid_rally': ('2020-07-01', '2021-12-31'),
    'rate_hike_2022': ('2022-01-01', '2022-12-31'),
    'post_2022': ('2023-01-01', '2026-12-31'),
}


def split_by_regime(
    data: pd.DataFrame,
    regimes: Dict[str, Tuple[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split data into regime-specific subsets.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with DatetimeIndex
    regimes : dict, optional
        Dictionary of regime names to (start, end) date tuples
    
    Returns
    -------
    dict
        Dictionary of regime name to data subset
    """
    if regimes is None:
        regimes = REGIMES
    
    result = {}
    
    for name, (start, end) in regimes.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        
        mask = (data.index >= start_dt) & (data.index <= end_dt)
        subset = data.loc[mask]
        
        if len(subset) > 0:
            result[name] = subset
    
    return result


def regime_performance(
    returns: pd.Series,
    regimes: Dict[str, Tuple[str, str]] = None,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate performance metrics for each regime.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    regimes : dict, optional
        Regime definitions
    annualize : bool
        Whether to annualize returns and volatility
    
    Returns
    -------
    pd.DataFrame
        Performance metrics by regime
    """
    if regimes is None:
        regimes = REGIMES
    
    results = []
    ann_factor = np.sqrt(252) if annualize else 1
    
    for name, (start, end) in regimes.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        
        mask = (returns.index >= start_dt) & (returns.index <= end_dt)
        regime_ret = returns.loc[mask].dropna()
        
        if len(regime_ret) < 5:
            continue
        
        # Calculate metrics
        mean_ret = regime_ret.mean()
        std_ret = regime_ret.std()
        sharpe = mean_ret / std_ret * ann_factor if std_ret > 0 else 0
        
        # Cumulative return
        cum_ret = (1 + regime_ret).prod() - 1
        
        # Max drawdown
        cum_wealth = (1 + regime_ret).cumprod()
        running_max = cum_wealth.expanding().max()
        drawdown = (cum_wealth - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (regime_ret > 0).mean()
        
        # Skewness and kurtosis
        skew = regime_ret.skew()
        kurt = regime_ret.kurtosis()
        
        results.append({
            'regime': name,
            'start': start,
            'end': end,
            'n_days': len(regime_ret),
            'annual_return': mean_ret * 252 if annualize else mean_ret,
            'annual_vol': std_ret * ann_factor,
            'sharpe': sharpe,
            'cum_return': cum_ret,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'skewness': skew,
            'kurtosis': kurt
        })
    
    return pd.DataFrame(results)


def rolling_ic_analysis(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60
) -> pd.DataFrame:
    """
    Compute rolling IC (Information Coefficient) time series.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates × stocks)
    forward_returns : pd.DataFrame
        Forward returns (dates × stocks)
    window : int
        Rolling window in days
    min_periods : int
        Minimum periods required
    
    Returns
    -------
    pd.DataFrame
        Rolling IC statistics
    """
    # Compute daily cross-sectional IC
    ic_series = []
    
    common_dates = factor.index.intersection(forward_returns.index)
    
    for date in common_dates:
        f = factor.loc[date].dropna()
        r = forward_returns.loc[date].reindex(f.index).dropna()
        
        common = f.index.intersection(r.index)
        
        if len(common) >= 10:
            # Spearman rank correlation
            ic = f[common].corr(r[common], method='spearman')
            ic_series.append({'date': date, 'ic': ic})
    
    if not ic_series:
        return pd.DataFrame()
    
    ic_df = pd.DataFrame(ic_series).set_index('date')
    
    # Rolling statistics
    rolling = ic_df['ic'].rolling(window=window, min_periods=min_periods)
    
    result = pd.DataFrame({
        'ic': ic_df['ic'],
        'ic_rolling_mean': rolling.mean(),
        'ic_rolling_std': rolling.std(),
        'ic_rolling_t': rolling.mean() / (rolling.std() / np.sqrt(window)),
        'ic_rolling_ir': rolling.mean() / rolling.std()  # Information Ratio
    })
    
    return result


def ic_decay_analysis(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: List[int] = [1, 5, 10, 21, 63, 126, 252]
) -> pd.DataFrame:
    """
    Analyze IC decay over different forward return horizons.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    prices : pd.DataFrame
        Price data for computing forward returns
    horizons : list
        Forward return horizons in days
    
    Returns
    -------
    pd.DataFrame
        IC statistics by horizon
    """
    results = []
    
    for horizon in horizons:
        # Compute forward returns
        fwd_ret = prices.shift(-horizon) / prices - 1
        
        # Compute IC for each date
        ic_list = []
        common_dates = factor.index.intersection(fwd_ret.index)
        
        for date in common_dates[:-horizon]:
            f = factor.loc[date].dropna()
            r = fwd_ret.loc[date].reindex(f.index).dropna()
            
            common = f.index.intersection(r.index)
            
            if len(common) >= 10:
                ic = f[common].corr(r[common], method='spearman')
                ic_list.append(ic)
        
        if ic_list:
            ic_arr = np.array(ic_list)
            
            results.append({
                'horizon_days': horizon,
                'mean_ic': np.mean(ic_arr),
                'std_ic': np.std(ic_arr),
                't_stat': np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(len(ic_arr))),
                'ir': np.mean(ic_arr) / np.std(ic_arr),
                'pct_positive': (ic_arr > 0).mean(),
                'n_periods': len(ic_arr)
            })
    
    return pd.DataFrame(results)


def parameter_sensitivity(
    factor_func: Callable,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    param_grid: Dict[str, List],
    metric: str = 'sharpe',
    top_pct: float = 0.2,
    bottom_pct: float = 0.2
) -> pd.DataFrame:
    """
    Test factor performance sensitivity to parameters.
    
    Parameters
    ----------
    factor_func : callable
        Factor function that takes prices and **kwargs
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Return data
    param_grid : dict
        Parameter grid {'lookback': [20, 60, 120], 'skip': [0, 5]}
    metric : str
        Performance metric to compute
    top_pct : float
        Long leg percentile
    bottom_pct : float
        Short leg percentile
    
    Returns
    -------
    pd.DataFrame
        Performance for each parameter combination
    """
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        
        try:
            # Compute factor
            factor = factor_func(prices, **params)
            
            # Simple long-short backtest
            portfolio_returns = []
            
            for i, date in enumerate(factor.index[:-1]):
                factor_vals = factor.loc[date].dropna()
                
                if len(factor_vals) < 10:
                    continue
                
                ranks = factor_vals.rank(pct=True)
                long_stocks = ranks[ranks >= (1 - top_pct)].index
                short_stocks = ranks[ranks <= bottom_pct].index
                
                next_date = factor.index[i + 1]
                if next_date not in returns.index:
                    continue
                
                next_ret = returns.loc[next_date]
                long_ret = next_ret.reindex(long_stocks).mean()
                short_ret = next_ret.reindex(short_stocks).mean()
                
                if pd.notna(long_ret) and pd.notna(short_ret):
                    portfolio_returns.append(long_ret - short_ret)
            
            if len(portfolio_returns) < 20:
                continue
            
            ret_arr = np.array(portfolio_returns)
            
            # Compute metric
            if metric == 'sharpe':
                value = np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252)
            elif metric == 'return':
                value = np.mean(ret_arr) * 252
            elif metric == 'ir':
                value = np.mean(ret_arr) / np.std(ret_arr)
            elif metric == 'win_rate':
                value = (ret_arr > 0).mean()
            else:
                value = np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252)
            
            result = params.copy()
            result[metric] = value
            result['n_periods'] = len(portfolio_returns)
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Parameter combo {params} failed: {e}")
            continue
    
    return pd.DataFrame(results)


def create_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = 'sharpe'
) -> pd.DataFrame:
    """
    Create 2D heatmap from sensitivity results.
    
    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Output from parameter_sensitivity
    param1 : str
        Row parameter
    param2 : str
        Column parameter
    metric : str
        Metric to display
    
    Returns
    -------
    pd.DataFrame
        Pivoted heatmap data
    """
    if sensitivity_df.empty:
        return pd.DataFrame()
    
    return sensitivity_df.pivot_table(
        index=param1,
        columns=param2,
        values=metric,
        aggfunc='mean'
    )


def walk_forward_analysis(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    train_period: int = 252,
    test_period: int = 63,
    step: int = 21,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2
) -> pd.DataFrame:
    """
    Walk-forward out-of-sample testing.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    returns : pd.DataFrame
        Stock returns
    train_period : int
        Training window in days
    test_period : int
        Test window in days
    step : int
        Step size in days
    top_pct : float
        Long leg percentile
    bottom_pct : float
        Short leg percentile
    
    Returns
    -------
    pd.DataFrame
        Walk-forward results by period
    """
    results = []
    
    dates = factor.index
    n_dates = len(dates)
    
    i = train_period
    
    while i + test_period <= n_dates:
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[min(i + test_period - 1, n_dates - 1)]
        
        # Compute factor rank thresholds on training data
        train_factor = factor.iloc[:i]
        
        # Test on out-of-sample period
        test_returns = []
        
        for j in range(i, min(i + test_period, n_dates - 1)):
            date = dates[j]
            factor_vals = factor.loc[date].dropna()
            
            if len(factor_vals) < 10:
                continue
            
            ranks = factor_vals.rank(pct=True)
            long_stocks = ranks[ranks >= (1 - top_pct)].index
            short_stocks = ranks[ranks <= bottom_pct].index
            
            next_date = dates[j + 1]
            if next_date not in returns.index:
                continue
            
            next_ret = returns.loc[next_date]
            long_ret = next_ret.reindex(long_stocks).mean()
            short_ret = next_ret.reindex(short_stocks).mean()
            
            if pd.notna(long_ret) and pd.notna(short_ret):
                test_returns.append(long_ret - short_ret)
        
        if len(test_returns) >= 5:
            ret_arr = np.array(test_returns)
            
            results.append({
                'test_start': test_start,
                'test_end': test_end,
                'n_days': len(test_returns),
                'mean_return': np.mean(ret_arr),
                'std_return': np.std(ret_arr),
                'sharpe': np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252) if np.std(ret_arr) > 0 else 0,
                'cum_return': (1 + pd.Series(ret_arr)).prod() - 1,
                'win_rate': (ret_arr > 0).mean()
            })
        
        i += step
    
    return pd.DataFrame(results)


def turnover_analysis(
    factor: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    rebalance_freq: str = 'M'
) -> pd.DataFrame:
    """
    Analyze portfolio turnover from factor-based rebalancing.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values
    top_pct : float
        Long leg percentile
    bottom_pct : float
        Short leg percentile
    rebalance_freq : str
        'D', 'W', or 'M'
    
    Returns
    -------
    pd.DataFrame
        Turnover statistics by rebalance date
    """
    # Get rebalance dates
    if rebalance_freq == 'D':
        rebal_dates = factor.index
    elif rebalance_freq == 'W':
        rebal_dates = factor.resample('W').last().index
    else:  # 'M'
        rebal_dates = factor.resample('ME').last().index
    
    rebal_dates = [d for d in rebal_dates if d in factor.index]
    
    results = []
    prev_long = set()
    prev_short = set()
    
    for date in rebal_dates:
        factor_vals = factor.loc[date].dropna()
        
        if len(factor_vals) < 10:
            continue
        
        ranks = factor_vals.rank(pct=True)
        long_stocks = set(ranks[ranks >= (1 - top_pct)].index)
        short_stocks = set(ranks[ranks <= bottom_pct].index)
        
        if prev_long:
            # Turnover = stocks entering + stocks exiting
            long_turnover = len(long_stocks - prev_long) + len(prev_long - long_stocks)
            short_turnover = len(short_stocks - prev_short) + len(prev_short - short_stocks)
            
            # Normalize by portfolio size
            long_turnover_pct = long_turnover / (2 * len(long_stocks)) if long_stocks else 0
            short_turnover_pct = short_turnover / (2 * len(short_stocks)) if short_stocks else 0
            
            results.append({
                'date': date,
                'n_long': len(long_stocks),
                'n_short': len(short_stocks),
                'long_turnover': long_turnover_pct,
                'short_turnover': short_turnover_pct,
                'total_turnover': (long_turnover_pct + short_turnover_pct) / 2
            })
        
        prev_long = long_stocks
        prev_short = short_stocks
    
    return pd.DataFrame(results)


def transaction_cost_impact(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 15.0
) -> pd.DataFrame:
    """
    Calculate net returns after transaction costs.
    
    Parameters
    ----------
    gross_returns : pd.Series
        Gross strategy returns
    turnover : pd.Series
        Portfolio turnover per period
    cost_bps : float
        Transaction cost in basis points
    
    Returns
    -------
    pd.DataFrame
        Gross vs net performance comparison
    """
    # Align series
    common_dates = gross_returns.index.intersection(turnover.index)
    
    gross = gross_returns.loc[common_dates]
    turn = turnover.loc[common_dates]
    
    # Transaction costs
    costs = turn * cost_bps / 10000
    
    net_returns = gross - costs
    
    # Performance comparison
    gross_sharpe = gross.mean() / gross.std() * np.sqrt(252) if gross.std() > 0 else 0
    net_sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    return pd.DataFrame({
        'metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Total Cost'],
        'gross': [
            gross.mean() * 252,
            gross.std() * np.sqrt(252),
            gross_sharpe,
            0
        ],
        'net': [
            net_returns.mean() * 252,
            net_returns.std() * np.sqrt(252),
            net_sharpe,
            costs.sum()
        ]
    })
