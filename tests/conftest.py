"""Pytest configuration and fixtures."""
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    import numpy as np
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices
