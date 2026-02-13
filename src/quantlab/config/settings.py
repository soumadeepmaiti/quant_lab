"""
Configuration settings loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class Settings:
    """
    Application settings loaded from environment variables.
    
    Usage:
        from quantlab.config import settings
        print(settings.polygon_api_key)
    """

    # API Keys (Polygon.io rebranded to Massive.com in Oct 2025)
    polygon_api_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", "") or os.getenv("MASSIVE_API_KEY", ""))
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", ""))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))

    # Cache
    cache_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    cache_expiry_hours: int = field(default_factory=lambda: int(os.getenv("CACHE_EXPIRY_HOURS", "24")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Analysis defaults
    default_start_date: str = field(default_factory=lambda: os.getenv("DEFAULT_START_DATE", "2016-01-01"))
    default_end_date: str = field(default_factory=lambda: os.getenv("DEFAULT_END_DATE", "2026-01-01"))
    default_universe: str = field(default_factory=lambda: os.getenv("DEFAULT_UNIVERSE", "sp500"))
    default_portfolio_value: float = field(default_factory=lambda: float(os.getenv("DEFAULT_PORTFOLIO_VALUE", "1000000")))

    # VaR settings
    var_confidence_levels: List[float] = field(default_factory=lambda: [
        float(x) for x in os.getenv("VAR_CONFIDENCE_LEVELS", "0.90,0.95,0.99").split(",")
    ])
    mc_simulations: int = field(default_factory=lambda: int(os.getenv("MC_SIMULATIONS", "10000")))

    # Execution settings
    vwap_default_slices: int = field(default_factory=lambda: int(os.getenv("VWAP_DEFAULT_SLICES", "10")))
    market_sim_orders: int = field(default_factory=lambda: int(os.getenv("MARKET_SIM_ORDERS", "50")))

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for dir_path in [self.raw_data_dir, self.interim_data_dir,
                         self.processed_data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate required settings."""
        if not self.polygon_api_key:
            print("Warning: POLYGON_API_KEY not set")
            return False
        return True


# Global settings instance
settings = Settings()
