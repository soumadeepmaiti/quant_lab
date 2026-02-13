"""Data module - loaders, cleaners, and API clients."""

from quantlab.data.polygon_client import PolygonClient, get_polygon_client
from quantlab.data.universe import (
    get_universe,
    get_universe_with_metadata,
    DOW30,
    SP500_SAMPLE,
    SECTOR_MAP
)
from quantlab.data.loaders import (
    load_prices,
    load_returns,
    load_daily_bars_yf,
    load_daily_bars_polygon,
    load_from_parquet,
    save_to_parquet
)
from quantlab.data.cleaners import (
    clean_prices,
    detect_outliers,
    winsorize_returns,
    align_data
)

__all__ = [
    # Clients
    "PolygonClient",
    "get_polygon_client",
    # Universe
    "get_universe",
    "get_universe_with_metadata",
    "DOW30",
    "SP500_SAMPLE",
    "SECTOR_MAP",
    # Loaders
    "load_prices",
    "load_returns",
    "load_daily_bars_yf",
    "load_daily_bars_polygon",
    "load_from_parquet",
    "save_to_parquet",
    # Cleaners
    "clean_prices",
    "detect_outliers",
    "winsorize_returns",
    "align_data",
]
