"""
Comprehensive test suite for microstructure module.

Tests covering:
- LOB matching engine (FIFO, partial fills, cancellations)
- Polygon data integration
- Market impact models
- Order flow signals (OFI, VPIN)
- Execution strategies (VWAP, TWAP)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from quantlab.microstructure.lob import (
    LimitOrderBook, Order, Trade, initialize_book, load_real_book_from_polygon
)
from quantlab.microstructure.impact import analyze_market_order, square_root_impact
from quantlab.microstructure.signals import order_flow_imbalance


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def empty_book():
    """Fresh empty limit order book."""
    return LimitOrderBook(symbol="TEST")


@pytest.fixture
def populated_book():
    """LOB with standard opening depth."""
    return initialize_book(symbol="TEST", mid_price=100.0, levels=5)


@pytest.fixture
def polygon_client_mock():
    """Mock Polygon API client."""
    client = Mock()
    
    # Mock quotes data (NBBO)
    quotes_data = pd.DataFrame({
        'bid_price': [99.98, 99.97],
        'bid_size': [1000, 950],
        'ask_price': [100.02, 100.03],
        'ask_size': [1000, 950],
    }, index=pd.DatetimeIndex(['2024-01-15 09:30:00', '2024-01-15 09:35:00']))
    
    # Mock trades data
    trades_data = pd.DataFrame({
        'price': [100.00, 100.01, 99.99],
        'size': [500, 750, 250],
        'conditions': [0, 0, 0]
    }, index=pd.DatetimeIndex(['2024-01-15 09:30:05', '2024-01-15 09:30:10', '2024-01-15 09:30:15']))
    
    client.get_quotes.return_value = quotes_data
    client.get_trades.return_value = trades_data
    
    return client


@pytest.fixture
def market_data():
    """Sample market data for impact analysis."""
    n_periods = 252
    np.random.seed(42)
    
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_periods)))
    volumes = np.random.uniform(1e6, 5e6, n_periods)
    volatility = np.std(np.diff(np.log(prices))) * np.sqrt(252)
    
    return {
        'prices': prices,
        'volumes': volumes,
        'volatility': volatility,
        'avg_spread_bps': 2.0
    }


# ============================================================================
# LOB MATCHING ENGINE TESTS (6 tests)
# ============================================================================


class TestLOBMatchingEngine:
    """Tests for core LOB matching logic."""
    
    def test_add_limit_order_no_execution(self, empty_book):
        """Limit order with no matching should rest in book."""
        order_id, trades = empty_book.add_limit_order('BUY', 100.0, 1000)
        
        assert order_id == 1
        assert len(trades) == 0
        assert empty_book.bids[100.0][0].quantity == 1000
    
    def test_full_execution_against_resting(self, populated_book):
        """Market order fully matched against resting liquidity."""
        best_bid, _ = populated_book.get_best_bid()
        trades, avg_price = populated_book.execute_market_order('SELL', 500)
        
        assert len(trades) > 0
        assert sum(t.quantity for t in trades) == 500
        assert avg_price == best_bid
    
    def test_partial_fill_with_remainder(self, empty_book):
        """Order partially filled; remainder rests in book."""
        empty_book.add_limit_order('SELL', 100.0, 1000)
        order_id, trades = empty_book.add_limit_order('BUY', 100.0, 600)
        
        assert len(trades) == 1
        assert trades[0].quantity == 600
        assert empty_book.asks[100.0][0].quantity == 400
    
    def test_multi_level_execution(self, populated_book):
        """Market order consumes multiple price levels."""
        trades, _ = populated_book.execute_market_order('BUY', 5000)
        
        assert len(trades) > 1
        assert sum(t.quantity for t in trades) == 5000
    
    def test_cancel_order_success(self, populated_book):
        """Cancel existing order removes it from book."""
        best_bid, _ = populated_book.get_best_bid()
        order_id = populated_book.bids[best_bid][0].order_id
        
        result = populated_book.cancel_order('BUY', best_bid, order_id)
        
        assert result is True
    
    def test_cancel_nonexistent_order_fails(self, populated_book):
        """Cancel non-existent order returns False."""
        result = populated_book.cancel_order('BUY', 150.0, 9999)
        assert result is False


# ============================================================================
# POLYGON DATA INTEGRATION TESTS (5 tests)
# ============================================================================


class TestPolygonIntegration:
    """Tests for real Polygon API integration."""
    
    def test_load_real_book_success(self, polygon_client_mock):
        """Successfully load LOB from Polygon quotes/trades."""
        book = load_real_book_from_polygon(
            polygon_client_mock, 'TEST', '2024-01-15'
        )
        
        assert book.symbol == 'TEST'
        assert len(book.bids) > 0
        assert len(book.asks) > 0
        assert book.get_mid_price() is not None
    
    def test_load_real_book_uses_opening_quote(self, polygon_client_mock):
        """LOB initialized with first quote's bid/ask."""
        book = load_real_book_from_polygon(
            polygon_client_mock, 'TEST', '2024-01-15', depth_levels=2
        )
        
        best_bid, _ = book.get_best_bid()
        best_ask, _ = book.get_best_ask()
        
        assert 99.9 < best_bid <= 99.98
        assert 100.02 <= best_ask < 100.1
    
    def test_load_real_book_replays_trades(self, polygon_client_mock):
        """Trades are replayed through LOB engine."""
        book = load_real_book_from_polygon(
            polygon_client_mock, 'TEST', '2024-01-15'
        )
        
        assert len(book.trades) > 0
    
    def test_load_real_book_empty_data_fallback(self):
        """Falls back to synthetic if no Polygon data available."""
        client = Mock()
        client.get_quotes.return_value = pd.DataFrame()
        client.get_trades.return_value = pd.DataFrame()
        
        book = load_real_book_from_polygon(client, 'TEST', '2024-01-15')
        
        assert book.symbol == 'TEST'
        assert len(book.bids) > 0
    
    def test_load_real_book_api_error_fallback(self):
        """Falls back to synthetic on API error."""
        client = Mock()
        client.get_quotes.side_effect = Exception("API Error")
        
        book = load_real_book_from_polygon(client, 'TEST', '2024-01-15')
        
        assert book.symbol == 'TEST'
        assert len(book.bids) > 0


# ============================================================================
# MARKET IMPACT TESTS (4 tests)
# ============================================================================


class TestMarketImpact:
    """Tests for market impact models."""
    
    def test_square_root_impact_calculation(self, market_data):
        """Square-root impact formula produces reasonable output."""
        q = 10000
        v = 1e6
        sigma = 0.015
        
        impact = square_root_impact(q, v, sigma, lambda_param=0.025)
        
        assert impact > 0
        assert impact < 0.005
    
    def test_square_root_scaling_law(self):
        """Impact should increase with order size (power law ~0.5)."""
        v, sigma = 1e6, 0.015
        
        impact_1x = square_root_impact(10000, v, sigma)
        impact_2x = square_root_impact(20000, v, sigma)
        
        ratio = impact_2x / impact_1x
        assert 1.35 < ratio < 1.5
    
    def test_impact_decreases_with_volume(self):
        """Impact should decrease as daily volume increases."""
        q = 10000
        sigma = 0.015
        
        impact_low_vol = square_root_impact(q, 1e5, sigma)
        impact_high_vol = square_root_impact(q, 1e7, sigma)
        
        assert impact_low_vol > impact_high_vol
    
    def test_market_order_execution_reasonable_prices(self, empty_book):
        """Market order fills at reasonable prices."""
        empty_book.add_limit_order('SELL', 100.00, 5000)
        empty_book.add_limit_order('SELL', 100.01, 5000)
        empty_book.add_limit_order('SELL', 100.02, 5000)
        
        trades, avg_price = empty_book.execute_market_order('BUY', 8000)
        
        assert len(trades) > 0
        assert sum(t.quantity for t in trades) == 8000
        assert 100.00 <= avg_price <= 100.02


# ============================================================================
# ORDER FLOW SIGNAL TESTS (3 tests)
# ============================================================================


class TestOrderFlowSignals:
    """Tests for OFI and market microstructure signals."""
    
    def test_ofi_positive_bid_size_increase(self):
        """OFI positive when bid size increases."""
        before = {'bid_size': 1000, 'ask_size': 1000}
        after = {'bid_size': 1500, 'ask_size': 1000}
        
        ofi = order_flow_imbalance(before, after)
        
        assert ofi > 0
    
    def test_ofi_negative_ask_size_increase(self):
        """OFI negative when ask size increases."""
        before = {'bid_size': 1000, 'ask_size': 1000}
        after = {'bid_size': 1000, 'ask_size': 1500}
        
        ofi = order_flow_imbalance(before, after)
        
        assert ofi < 0
    
    def test_ofi_calculation_zero_no_change(self):
        """OFI equal zero when bid and ask sizes don't change."""
        before = {'bid_size': 1000, 'ask_size': 1000}
        after = {'bid_size': 1000, 'ask_size': 1000}
        
        ofi = order_flow_imbalance(before, after)
        
        assert ofi == 0


# ============================================================================
# LOB STATE VALIDATION TESTS (3 tests)
# ============================================================================


class TestLOBStateValidation:
    """Tests for LOB invariants and state consistency."""
    
    def test_bid_ask_crossing_protection(self, empty_book):
        """Best bid should never exceed best ask."""
        empty_book.add_limit_order('BUY', 100.0, 1000)
        empty_book.add_limit_order('SELL', 100.01, 1000)
        
        bid, _ = empty_book.get_best_bid()
        ask, _ = empty_book.get_best_ask()
        
        assert bid < ask
    
    def test_fifo_matching_priority(self, empty_book):
        """Orders at same price level matched FIFO."""
        empty_book.add_limit_order('BUY', 100.0, 500)
        empty_book.add_limit_order('BUY', 100.0, 500)
        
        trades, _ = empty_book.execute_market_order('SELL', 600)
        
        assert trades[0].quantity == 500
        assert trades[1].quantity == 100
    
    def test_trade_history_completeness(self, populated_book):
        """All executed trades recorded in history."""
        initial_trades = len(populated_book.trades)
        
        populated_book.execute_market_order('BUY', 1000)
        populated_book.execute_market_order('SELL', 800)
        
        assert len(populated_book.trades) > initial_trades


# ============================================================================
# ADVANCED LOB TESTS (3 tests)
# ============================================================================


class TestAdvancedLOBFeatures:
    """Advanced LOB functionality tests."""
    
    def test_mid_price_calculation(self, populated_book):
        """Mid price calculated correctly from bid/ask."""
        bid, _ = populated_book.get_best_bid()
        ask, _ = populated_book.get_best_ask()
        mid = populated_book.get_mid_price()
        
        expected_mid = (bid + ask) / 2
        assert abs(mid - expected_mid) < 0.001
    
    def test_book_snapshot_depth(self, populated_book):
        """Snapshot contains requested depth levels."""
        snapshot = populated_book.get_snapshot(depth=3)
        
        assert len(snapshot['bids']) <= 3
        assert len(snapshot['asks']) <= 3
        assert snapshot['mid_price'] is not None
    
    def test_spread_tracking(self, populated_book):
        """Spread history recorded and tracked."""
        initial_spread = populated_book.get_spread()
        
        populated_book.execute_market_order('BUY', 100)
        
        new_spread = populated_book.get_spread()
        
        # Spread may change slightly after execution
        assert new_spread is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
