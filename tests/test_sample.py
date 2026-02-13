"""Sample tests for quantlab package."""
import numpy as np
import pytest


class TestSampleFixture:
    """Test that fixtures work properly."""

    def test_sample_returns_shape(self, sample_returns):
        """Test sample returns have correct shape."""
        assert len(sample_returns) == 252

    def test_sample_returns_mean(self, sample_returns):
        """Test sample returns statistics."""
        assert np.isclose(np.mean(sample_returns), 0.001, atol=0.005)

    def test_sample_prices_shape(self, sample_prices):
        """Test sample prices have correct shape."""
        assert len(sample_prices) == 252

    def test_sample_prices_positive(self, sample_prices):
        """Test sample prices are all positive."""
        assert np.all(sample_prices > 0)
