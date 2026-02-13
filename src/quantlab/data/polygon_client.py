"""
Polygon.io / Massive.com API client wrapper for market data.

Note: Polygon.io rebranded to Massive.com in October 2025.
Both API endpoints are supported and interchangeable.
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import time
import os

from quantlab.config import settings, get_logger

logger = get_logger(__name__)


class PolygonClient:
    """
    REST client wrapper for Polygon.io / Massive.com API.
    
    Note: Polygon.io rebranded to Massive.com on October 30, 2025.
    All existing API keys, accounts, and integrations remain compatible.
    
    Provides methods for fetching:
    - Stock aggregates (OHLCV bars)
    - Trades and quotes
    - Reference data (tickers, exchanges)
    
    Usage:
        client = PolygonClient()
        bars = client.get_aggregates('AAPL', '2024-01-01', '2024-12-31')
    """
    
    # Polygon.io URL (also works for Massive.com keys)
    BASE_URL = "https://api.polygon.io"
    # Alternative: MASSIVE_URL = "https://api.massive.com"  # Same API
    
    def __init__(self, api_key: Optional[str] = None, use_massive: bool = False):
        """
        Initialize Polygon/Massive client.
        
        Parameters
        ----------
        api_key : str, optional
            API key. Defaults to POLYGON_API_KEY or MASSIVE_API_KEY env var
        use_massive : bool
            If True, use Massive.com URL (currently same as Polygon)
        """
        self.api_key = api_key or settings.polygon_api_key
        if not self.api_key:
            raise ValueError("Polygon/Massive API key not provided. "
                           "Set POLYGON_API_KEY or MASSIVE_API_KEY environment variable.")
        
        self.session = requests.Session()
        self.rate_limit_remaining = 5  # Conservative default
        self.last_request_time = 0.0
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make a request to the Polygon API.
        
        Parameters
        ----------
        endpoint : str
            API endpoint (without base URL)
        params : dict, optional
            Query parameters
        retries : int
            Number of retries on failure
        
        Returns
        -------
        dict
            API response JSON
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.2:  # 5 requests per second max
            time.sleep(0.2 - elapsed)
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    response.raise_for_status()
                    
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return {}
    
    def get_aggregates(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1,
        adjusted: bool = True,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get aggregate bars (OHLCV) for a ticker.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        timespan : str
            Bar size: 'minute', 'hour', 'day', 'week', 'month'
        multiplier : int
            Bar multiplier (e.g., 5 for 5-minute bars)
        adjusted : bool
            Whether to adjust for splits/dividends
        limit : int
            Maximum number of bars
        
        Returns
        -------
        pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume, vwap
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit
        }
        
        data = self._request(endpoint, params)
        
        if not data.get("results"):
            logger.warning(f"No data for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["results"])
        
        # Rename columns
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions"
        })
        
        # Convert timestamp (ms) to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        
        return df
    
    def get_trades(
        self,
        ticker: str,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get trades for a ticker on a specific date.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        date : str
            Date (YYYY-MM-DD)
        limit : int
            Maximum number of trades
        
        Returns
        -------
        pd.DataFrame
            Trade data with columns: price, size, timestamp, conditions
        """
        endpoint = f"/v3/trades/{ticker}"
        
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": limit,
            "sort": "timestamp"
        }
        
        all_trades = []
        next_url = None
        
        while True:
            if next_url:
                response = self.session.get(next_url + f"&apiKey={self.api_key}")
                data = response.json()
            else:
                data = self._request(endpoint, params)
            
            if data.get("results"):
                all_trades.extend(data["results"])
            
            next_url = data.get("next_url")
            if not next_url or len(all_trades) >= limit:
                break
        
        if not all_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_trades)
        
        # Parse timestamp
        if "sip_timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns")
        
        return df[["timestamp", "price", "size", "conditions"]].set_index("timestamp") if len(df) > 0 else df
    
    def get_quotes(
        self,
        ticker: str,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get NBBO quotes for a ticker on a specific date.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        date : str
            Date (YYYY-MM-DD)
        limit : int
            Maximum number of quotes
        
        Returns
        -------
        pd.DataFrame
            Quote data with columns: bid_price, bid_size, ask_price, ask_size
        """
        endpoint = f"/v3/quotes/{ticker}"
        
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": limit,
            "sort": "timestamp"
        }
        
        data = self._request(endpoint, params)
        
        if not data.get("results"):
            return pd.DataFrame()
        
        df = pd.DataFrame(data["results"])
        
        if "sip_timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns")
        
        columns = ["timestamp", "bid_price", "bid_size", "ask_price", "ask_size"]
        available = [c for c in columns if c in df.columns]
        
        return df[available].set_index("timestamp") if "timestamp" in available else df
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """
        Get details for a specific ticker.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        
        Returns
        -------
        dict
            Ticker details including name, market cap, etc.
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        data = self._request(endpoint)
        return data.get("results", {})
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns
        -------
        dict
            Market status information
        """
        endpoint = "/v1/marketstatus/now"
        return self._request(endpoint)
    
    def search_tickers(
        self,
        search: str = "",
        ticker_type: str = "CS",
        market: str = "stocks",
        active: bool = True,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Search for tickers.
        
        Parameters
        ----------
        search : str
            Search query
        ticker_type : str
            Ticker type (CS=common stock, ETF, etc.)
        market : str
            Market (stocks, crypto, fx)
        active : bool
            Only active tickers
        limit : int
            Maximum results
        
        Returns
        -------
        pd.DataFrame
            Matching tickers
        """
        endpoint = "/v3/reference/tickers"
        
        params = {
            "type": ticker_type,
            "market": market,
            "active": str(active).lower(),
            "limit": limit
        }
        
        if search:
            params["search"] = search
        
        data = self._request(endpoint, params)
        
        if not data.get("results"):
            return pd.DataFrame()
        
        return pd.DataFrame(data["results"])


# Convenience function
def get_polygon_client() -> PolygonClient:
    """Get a configured Polygon client."""
    return PolygonClient()
