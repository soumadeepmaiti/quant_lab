"""
Factor calculation functions for alpha research.
"""

from typing import Optional

import numpy as np
import pandas as pd


def momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip_recent: int = 0
) -> pd.DataFrame:
    """
    Calculate price momentum factor.
    
    Momentum = (Price_today / Price_N_days_ago) - 1
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data (dates x tickers)
    lookback : int
        Lookback period in trading days (252 = 1 year)
    skip_recent : int
        Days to skip (avoid short-term reversal)
    
    Returns
    -------
    pd.DataFrame
        Momentum factor values
    """
    if skip_recent > 0:
        return prices.shift(skip_recent) / prices.shift(lookback + skip_recent) - 1
    return prices / prices.shift(lookback) - 1


def rsi(
    prices: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    window : int
        RSI window (standard is 14)
    
    Returns
    -------
    pd.DataFrame
        RSI values (0-100)
    """
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def volatility(
    returns: pd.DataFrame,
    window: int = 60,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    window : int
        Rolling window in days
    annualize : bool
        Whether to annualize (multiply by sqrt(252))
    
    Returns
    -------
    pd.DataFrame
        Volatility values
    """
    vol = returns.rolling(window=window, min_periods=window).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def mean_reversion(
    prices: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate mean reversion signal.
    
    Signal = -(Price - MA) / MA
    
    Negative so oversold stocks rank higher.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    window : int
        Moving average window
    
    Returns
    -------
    pd.DataFrame
        Mean reversion signal
    """
    ma = prices.rolling(window=window).mean()
    deviation = (prices - ma) / ma
    return -deviation  # Negative so oversold ranks high


def beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling beta relative to market.
    
    Beta = Cov(stock, market) / Var(market)
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns
    market_returns : pd.Series
        Market index returns
    window : int
        Rolling window
    
    Returns
    -------
    pd.DataFrame
        Beta values
    """
    betas = pd.DataFrame(index=returns.index, columns=returns.columns)

    market_var = market_returns.rolling(window=window).var()

    for col in returns.columns:
        cov = returns[col].rolling(window=window).cov(market_returns)
        betas[col] = cov / market_var

    return betas.astype(float)


def value_proxy(
    prices: pd.DataFrame,
    earnings: Optional[pd.DataFrame] = None,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Calculate a value proxy factor.
    
    If earnings provided: Earnings Yield = E/P
    Otherwise: Use inverse of price momentum as proxy
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    earnings : pd.DataFrame, optional
        Earnings per share data
    lookback : int
        Lookback for price-based proxy
    
    Returns
    -------
    pd.DataFrame
        Value factor (higher = cheaper)
    """
    if earnings is not None:
        return earnings / prices

    # Use inverse momentum as rough proxy for value
    mom = momentum(prices, lookback=lookback)
    return -mom  # Negative momentum = "value" stocks


def size(
    prices: pd.DataFrame,
    shares_outstanding: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate size factor (market cap proxy).
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    shares_outstanding : pd.DataFrame, optional
        Shares outstanding
    
    Returns
    -------
    pd.DataFrame
        Size factor (negative log market cap for SMB)
    """
    if shares_outstanding is not None:
        market_cap = prices * shares_outstanding
    else:
        # Use price as proxy (assuming similar share counts)
        market_cap = prices

    # Negative log so small stocks rank higher (SMB)
    return -np.log(market_cap)


def quality_proxy(
    returns: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate quality proxy using return stability.
    
    Quality = -Volatility (more stable = higher quality)
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    window : int
        Window for volatility calculation
    
    Returns
    -------
    pd.DataFrame
        Quality factor
    """
    vol = volatility(returns, window=window, annualize=True)
    return -vol  # Lower volatility = higher quality


def skewness(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Calculate rolling skewness of returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    window : int
        Rolling window
    
    Returns
    -------
    pd.DataFrame
        Skewness values
    """
    return returns.rolling(window=window).skew()


def max_drawdown_factor(
    prices: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling maximum drawdown.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    window : int
        Rolling window
    
    Returns
    -------
    pd.DataFrame
        Max drawdown values (negative)
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.rolling(window=window).min()


def composite_factor(
    factors: dict,
    weights: Optional[dict] = None,
    method: str = 'rank_average'
) -> pd.DataFrame:
    """
    Create a composite factor from multiple factors.
    
    Parameters
    ----------
    factors : dict
        {factor_name: factor_df}
    weights : dict, optional
        {factor_name: weight} (defaults to equal weight)
    method : str
        'rank_average', 'z_score', or 'weighted_sum'
    
    Returns
    -------
    pd.DataFrame
        Composite factor values
    """
    if weights is None:
        weights = {name: 1.0 / len(factors) for name in factors}

    if method == 'rank_average':
        # Convert to percentile ranks and average
        ranked = {name: df.rank(axis=1, pct=True) for name, df in factors.items()}
        composite = sum(ranked[name] * weights[name] for name in factors)

    elif method == 'z_score':
        # Z-score each factor and average
        z_scored = {}
        for name, df in factors.items():
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            z_scored[name] = df.sub(mean, axis=0).div(std + 1e-10, axis=0)
        composite = sum(z_scored[name] * weights[name] for name in factors)

    else:  # weighted_sum
        composite = sum(factors[name] * weights[name] for name in factors)

    return composite
