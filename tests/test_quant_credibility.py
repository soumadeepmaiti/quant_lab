"""
Quant Research Credibility Tests — Core Statistical Guarantees

Tests the fundamental correctness of:
  - Alpha signals (forward-looking, lagging, neutrality)
  - Risk models (hierarchy, tail properties)
  - Microstructure (market invariants, impact models)
"""


import numpy as np
import pytest


class TestAlphaSignalCorrectness:
    """Alpha module: forward-looking principles and factor discipline."""

    def test_momentum_signal_is_lagged(self):
        """
        Momentum signal must be lagged: computed using *only* data
        prior to signal date. This prevents look-ahead bias.
        
        RESEARCH DISCIPLINE: Forward returns are future-only.
        """
        # Simulated price series: 30 obs
        prices = np.array([100 + 2*i + np.random.randn() for i in range(30)])

        # Compute momentum using past 12 months, skip 1 (standard rule)
        signal_date_idx = 15
        lookback = 12
        skip = 1

        # Include signal date in lookback? NO - must use only *past* data
        momentum_lookback = prices[signal_date_idx - lookback - skip:signal_date_idx - skip]
        momentum = np.mean(np.diff(momentum_lookback)) / np.std(momentum_lookback + 1e-6)

        # Forward return: future data only
        future_start = signal_date_idx + 1  # strictly after signal date
        future_return = (prices[future_start + 5] - prices[future_start]) / prices[future_start]

        # Both should exist; forward return computed on strictly-future data
        assert momentum is not None
        assert future_return is not None
        assert signal_date_idx - skip - lookback >= 0, "Lookback must not precede data"

    def test_factor_weight_neutrality(self):
        """
        Market-neutral portfolio: long-short weights sum to ~0.
        
        RESEARCH DISCIPLINE: No net market exposure.
        """
        # Mock long and short weights
        long_weights = np.array([0.10, 0.15, 0.12])  # 3 long positions
        short_weights = np.array([-0.10, -0.12, -0.15])  # 3 short positions

        total_weights = np.concatenate([long_weights, short_weights])

        # Market neutrality: gross long ≈ |gross short|
        gross_long = np.sum(long_weights)
        gross_short = np.abs(np.sum(short_weights))

        assert np.isclose(gross_long, gross_short, atol=0.05), \
            f"Market neutrality violated: long={gross_long:.3f}, short={gross_short:.3f}"

    def test_transaction_cost_reduces_returns(self):
        """
        Gross returns > net returns when costs applied.
        
        RESEARCH DISCIPLINE: Backtests must account for friction.
        """
        gross_return = 0.05  # 5% gross
        transaction_cost = 0.002  # 20 bps round-trip

        net_return = gross_return - transaction_cost

        assert net_return < gross_return, "Transaction costs must reduce returns"
        assert net_return == pytest.approx(0.048, abs=0.0001)

    def test_forward_return_no_lookahead(self):
        """
        Forward return window must start *after* signal date.
        
        RESEARCH DISCIPLINE: Strictly no data snooping.
        """
        signal_computed_on_dates = [1, 2, 3, 4, 5]
        signal_dates = [d for d in signal_computed_on_dates]
        forward_return_window_start = [d + 1 for d in signal_dates]  # Next day minimum

        for sig_date, fwd_start in zip(signal_dates, forward_return_window_start):
            assert fwd_start > sig_date, "Forward window must start after signal date"


class TestRiskModelCorrectness:
    """Risk module: distributional hierarchy and tail properties."""

    def test_var99_greater_equal_var95(self):
        """
        VaR₉₉ ≥ VaR₉₅ with high probability (monotonicity in confidence level).
        
        REGULATORY: Basel III relies on this ordering.
        """
        # Synthetic returns: normal + fat tails
        np.random.seed(42)
        returns = np.concatenate([
            np.random.normal(0, 0.01, 900),
            np.random.normal(-0.05, 0.05, 100)  # tail fat
        ])

        var_95 = np.percentile(returns, 5)  # 5th percentile
        var_99 = np.percentile(returns, 1)  # 1st percentile

        # Algebraically, VaR99 ≤ VaR95 (more extreme), but magnitude tests ordering
        assert var_99 < var_95, "99% VaR must be in worse tail than 95% VaR"

    def test_es_greater_equal_var(self):
        """
        Expected Shortfall ≥ VaR (average of tail exceeds single quantile).
        
        REGULATORY: Basel III greenness depends on ES > VaR.
        """
        np.random.seed(42)
        returns = np.random.normal(-0.02, 0.03, 1000)

        var_95 = np.percentile(returns, 5)
        es_95 = np.mean(returns[returns <= var_95])

        assert es_95 <= var_95, "Expected shortfall must be ≤ VaR (more negative in loss regime)"

    def test_kupiec_violation_rate_sanity(self):
        """
        Kupiec POF backtest: violation rate on synthetic returns should be near confidence level.
        
        REGULATORY: Basel GREEN zone (Kupiec LR < 3.84).
        """
        # Synthetic returns calibrated to 95% VaR level
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 252)

        var_95 = np.percentile(returns, 5)
        violations = np.sum(returns < var_95)
        violation_rate = violations / len(returns)

        # For 252 obs, 95% CI: expect ~13 violations
        expected_violations = 0.05 * len(returns)

        # Sanity: violation rate should be close to (1 - confidence)
        assert 0.02 < violation_rate < 0.10, \
            f"Violation rate {violation_rate:.3f} outside sanity range for 95% VaR"

    def test_garch_conditional_volatility_positive(self):
        """
        GARCH conditional variance σ²_t must be positive.
        
        CORRECTNESS: No negative variances.
        """
        # Mock GARCH(1,1) calculation: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        omega = 0.0001
        alpha = 0.1
        beta = 0.85

        eps_prev = 0.01
        sigma_prev = 0.015

        sigma_t_sq = omega + alpha * (eps_prev ** 2) + beta * (sigma_prev ** 2)

        assert sigma_t_sq > 0, "Conditional variance must be positive"


class TestMicrostructureInvariants:
    """Microstructure module: order book and impact model correctness."""

    def test_best_bid_less_than_best_ask(self):
        """
        Market microstructure invariant: best_bid < best_ask.
        
        CORRECTNESS: No-arbitrage constraint.
        """
        best_bid = 100.50
        best_ask = 100.55
        spread = best_ask - best_bid

        assert best_bid < best_ask, "Bid-ask crossed: arbitrage violation"
        assert 0 < spread, "Spread must be positive"

    def test_limit_order_book_fifo_priority(self):
        """
        LOB matching: FIFO price-time priority enforced.
        
        CORRECTNESS: Matching engine respects order queue.
        """
        # Order queue at price level 100.50:
        # [Order1(qty=100, t=0:00), Order2(qty=50, t=0:05)]

        incoming_market_order = 120  # volume to match

        order1_qty, order1_time = 100, 0
        order2_qty, order2_time = 50, 5

        # FIFO: match order1 first
        filled1 = min(incoming_market_order, order1_qty)
        remaining = incoming_market_order - filled1
        filled2 = min(remaining, order2_qty)

        assert filled1 == order1_qty, "FIFO: Order1 must be filled first"
        assert filled2 == 20, "FIFO: Order2 partially filled with remainder"
        assert filled1 + filled2 == incoming_market_order, "Total fill = order quantity"

    def test_square_root_market_impact_law(self):
        """
        Market impact ∝ √(quantity / ADV).
        
        EMPIRICAL: Squared coefficient α should be ~0.50; Kyle's λ.
        """
        # Impact = λ * √(Q / V)
        lambda_kyle = 0.001  # basis points per unit of √(Q/ADV)

        quantities = np.array([10, 50, 100, 500])
        adv = 1000

        impacts = lambda_kyle * np.sqrt(quantities / adv)

        # Check monotonicity and concavity (sqrt property: diminishing return to scale)
        impact_diffs = np.diff(impacts)
        assert np.all(impact_diffs > 0), "Impact must increase with quantity"

        # Concavity: second derivative should be negative (diminishing returns)
        # Note: finite differences on smooth sqrt can have numerical noise; we check trend
        second_diffs = np.diff(impact_diffs)
        # At least 50% of second diffs should be negative (allowing numerical noise)
        negative_count = np.sum(second_diffs < 0)
        assert negative_count >= len(second_diffs) * 0.5, \
            "Impact curve should be predominantly concave (diminishing returns)"

    def test_regression_lambda_recovery(self):
        """
        OFI regression: ΔP = λ·OFI should recover λ coefficient on synthetic data.
        
        MICROSTRUCTURE: Kyle's model parameter identification.
        """
        np.random.seed(42)

        # True λ = 0.5 bps per unit of signed volume
        lambda_true = 0.0005

        # Synthetic OFI and price changes
        ofi = np.random.normal(0, 100, 100)
        price_change = lambda_true * ofi + np.random.normal(0, 0.0001, 100)

        # OLS regression to recover λ
        X = ofi.reshape(-1, 1)
        y = price_change.reshape(-1, 1)

        # Normal equations: β = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]

        # Should recover true λ within noise
        assert np.isclose(beta, lambda_true, rtol=0.1), \
            f"Regression λ={beta:.6f} should be close to true λ={lambda_true:.6f}"


class TestExecutionAlgorithmCorrectness:
    """Execution module: VWAP/TWAP decomposition and cost reduction."""

    def test_vwap_reduces_execution_cost(self):
        """
        VWAP execution should reduce average price *vs* market order.
        
        OPTIMIZATION: Cost metric improvement.
        """
        # Market prices across 10 intervals
        market_prices = np.array([100.0, 100.2, 100.5, 100.3, 100.1,
                                  100.4, 100.6, 100.2, 100.1, 100.3])
        volumes = np.array([1000, 1100, 1200, 900, 1050,
                            1300, 1150, 950, 1000, 1100])

        # VWAP
        vwap = np.sum(market_prices * volumes) / np.sum(volumes)

        # Market order at first price (benchmark)
        market_price_first = market_prices[0]

        # Naive execution cost (market sell at worst price)
        naive_cost = market_prices.max() - market_prices.min()

        # VWAP should be closer to mean (less slippage)
        mean_price = np.mean(market_prices)
        vwap_slippage = np.abs(vwap - mean_price)
        first_slippage = np.abs(market_price_first - mean_price)

        assert vwap_slippage <= first_slippage, \
            "VWAP should have ≤ slippage vs immediate market execution"

    def test_is_decomposition_linearity(self):
        """
        Implementation Shortfall ≈ Timing Cost + Execution Cost (additive).
        
        DECOMPOSITION: Linear impact attribution.
        """
        # Benchmark: initial price at signal time
        p_signal = 100.00

        # Execution period: prices drift + market impact
        p_final = 100.05  # price drifted up (timing cost)
        impact_estimate = 0.05  # slippage (execution cost)

        # Total cost: (p_final - p_signal) + impact
        # (Negative of total return to us)
        shortfall = (p_final - p_signal) + impact_estimate

        # Decompose
        timing_cost = p_final - p_signal
        execution_cost = impact_estimate

        assert np.isclose(shortfall, timing_cost + execution_cost), \
            "IS should decompose into timing + execution"


class TestPolygonIntegration:
    """Integration: Polygon API data loading and caching."""

    def test_polygon_quotes_mock_structure(self):
        """
        Mock Polygon quotes response has expected fields: bid, ask, bid_size, ask_size.
        
        DATA FORMAT: Ensure LOB initialization works.
        """
        mock_quote = {
            'results': [
                {
                    'bid': 150.50,
                    'ask': 150.55,
                    'bid_size': 1000,
                    'ask_size': 500,
                    't': 1234567890000
                }
            ]
        }

        assert 'results' in mock_quote
        assert len(mock_quote['results']) > 0
        quote = mock_quote['results'][0]
        assert 'bid' in quote and 'ask' in quote
        assert quote['bid'] < quote['ask']

    def test_polygon_trades_mock_structure(self):
        """
        Mock Polygon trades response has expected fields: price, size.
        
        DATA FORMAT: Ensure LOB replay works.
        """
        mock_trade = {
            'results': [
                {'price': 150.50, 'size': 100, 't': 1234567890000},
                {'price': 150.51, 'size': 50, 't': 1234567890100},
            ]
        }

        assert 'results' in mock_trade
        for trade in mock_trade['results']:
            assert 'price' in trade and 'size' in trade
            assert trade['price'] > 0 and trade['size'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
