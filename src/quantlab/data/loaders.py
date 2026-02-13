"""
Data loaders for various data formats.
"""

from typing import List, Optional, Union
from pathlib import Path
import pandas as pd
import yfinance as yf

from quantlab.config import settings, get_logger
from quantlab.data.polygon_client import PolygonClient

logger = get_logger(__name__)


def load_daily_bars_yf(
    tickers: List[str],
    start_date: str,
    end_date: str,
    adjusted: bool = True
) -> pd.DataFrame:
    """
    Load daily OHLCV bars using Yahoo Finance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    adjusted : bool
        Use adjusted prices
    
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (date, ticker) or columns per ticker
    """
    logger.info(f"Downloading data for {len(tickers)} tickers via Yahoo Finance")
    
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=adjusted,
        progress=False
    )
    
    return data


def load_daily_bars_polygon(
    tickers: List[str],
    start_date: str,
    end_date: str,
    adjusted: bool = True
) -> pd.DataFrame:
    """
    Load daily OHLCV bars using Polygon.io API.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    adjusted : bool
        Use adjusted prices
    
    Returns
    -------
    pd.DataFrame
        Price data
    """
    client = PolygonClient()
    
    all_data = {}
    
    for ticker in tickers:
        logger.info(f"Fetching {ticker} from Polygon")
        try:
            df = client.get_aggregates(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                timespan='day',
                adjusted=adjusted
            )
            if not df.empty:
                all_data[ticker] = df['close']
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.DataFrame(all_data)


def load_prices(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = 'yfinance'
) -> pd.DataFrame:
    """
    Load closing prices for tickers.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str, optional
        Start date (defaults to settings)
    end_date : str, optional
        End date (defaults to settings)
    source : str
        Data source: 'yfinance' or 'polygon'
    
    Returns
    -------
    pd.DataFrame
        Closing prices with date index, ticker columns
    """
    start_date = start_date or settings.default_start_date
    end_date = end_date or settings.default_end_date
    
    if source == 'polygon':
        return load_daily_bars_polygon(tickers, start_date, end_date)
    else:
        data = load_daily_bars_yf(tickers, start_date, end_date)
        if 'Close' in data.columns or ('Close' in str(data.columns)):
            return data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
        return data


def load_returns(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = 'yfinance'
) -> pd.DataFrame:
    """
    Load daily returns for tickers.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str, optional
        Start date
    end_date : str, optional
        End date
    source : str
        Data source
    
    Returns
    -------
    pd.DataFrame
        Daily returns
    """
    prices = load_prices(tickers, start_date, end_date, source)
    returns = prices.pct_change().dropna()
    return returns


def load_from_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load data from parquet file."""
    return pd.read_parquet(path)


def save_to_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save data to parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Saved to {path}")


def load_from_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(path, **kwargs)


def save_to_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save data to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    logger.info(f"Saved to {path}")
