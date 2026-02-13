"""
Universe builders for stock selection.
"""

from typing import List, Optional

import pandas as pd
import yfinance as yf

from quantlab.config import get_logger

logger = get_logger(__name__)


# Pre-defined universes
DOW30 = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

SP500_SAMPLE = [
    # Technology (15)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE',
    'CSCO', 'ACN', 'INTC', 'IBM', 'QCOM',
    # Healthcare (15)
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ISRG',
    # Financials (10)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
    # Consumer (10)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'WMT',
    # Energy & Industrial (10)
    'XOM', 'CVX', 'COP', 'CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT',
    # Communications (5)
    'DIS', 'NFLX', 'CMCSA', 'VZ', 'T'
]

SECTOR_MAP = {
    'technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE'],
    'healthcare': ['UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY'],
    'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB'],
    'consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'WMT'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'industrials': ['CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM']
}


def get_universe(name: str) -> List[str]:
    """
    Get a predefined universe of tickers.
    
    Parameters
    ----------
    name : str
        Universe name: 'dow30', 'sp500', or sector name
    
    Returns
    -------
    List[str]
        List of ticker symbols
    """
    name_lower = name.lower()

    if name_lower == 'dow30':
        return DOW30.copy()
    elif name_lower in ('sp500', 'sp500_sample'):
        return SP500_SAMPLE.copy()
    elif name_lower in SECTOR_MAP:
        return SECTOR_MAP[name_lower].copy()
    else:
        logger.warning(f"Unknown universe: {name}, returning SP500")
        return SP500_SAMPLE.copy()


def get_universe_with_metadata(name: str) -> pd.DataFrame:
    """
    Get universe tickers with metadata from Yahoo Finance.
    
    Parameters
    ----------
    name : str
        Universe name
    
    Returns
    -------
    pd.DataFrame
        Tickers with columns: symbol, name, sector, industry, market_cap
    """
    tickers = get_universe(name)

    metadata = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            metadata.append({
                'symbol': ticker,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0)
            })
        except Exception as e:
            logger.warning(f"Failed to get metadata for {ticker}: {e}")
            metadata.append({'symbol': ticker})

    return pd.DataFrame(metadata)


def filter_by_market_cap(
    tickers: List[str],
    min_cap: Optional[float] = None,
    max_cap: Optional[float] = None
) -> List[str]:
    """
    Filter tickers by market capitalization.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    min_cap : float, optional
        Minimum market cap in dollars
    max_cap : float, optional
        Maximum market cap in dollars
    
    Returns
    -------
    List[str]
        Filtered list of tickers
    """
    filtered = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get('marketCap', 0)

            if min_cap and market_cap < min_cap:
                continue
            if max_cap and market_cap > max_cap:
                continue

            filtered.append(ticker)
        except Exception:
            continue

    return filtered


def get_custom_universe(
    tickers: List[str],
    validate: bool = True
) -> List[str]:
    """
    Create a custom universe with optional validation.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    validate : bool
        Whether to validate tickers exist
    
    Returns
    -------
    List[str]
        Valid ticker list
    """
    if not validate:
        return tickers

    valid = []
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            # Check if we can get basic info
            if yf_ticker.info.get('regularMarketPrice'):
                valid.append(ticker)
            else:
                logger.warning(f"Invalid ticker: {ticker}")
        except Exception:
            logger.warning(f"Could not validate: {ticker}")

    return valid
