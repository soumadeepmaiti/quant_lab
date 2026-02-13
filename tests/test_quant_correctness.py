import numpy as np
import pytest


def var(returns, alpha=0.95):
    # VaR as the lower alpha percentile (losses are negative returns)
    q = 100 * (1 - alpha)
    return np.percentile(returns, q)


def expected_shortfall(returns, alpha=0.95):
    v = var(returns, alpha)
    tail = returns[returns <= v]
    if len(tail) == 0:
        return v
    return tail.mean()


def test_var_monotonicity():
    rng = np.random.default_rng(42)
    returns = rng.normal(-0.001, 0.02, size=10000)
    v95 = var(returns, 0.95)
    v99 = var(returns, 0.99)
    # In loss terms, the 99% VaR should be at least as severe as the 95% VaR
    assert abs(v99) >= abs(v95)


def test_es_ge_var():
    rng = np.random.default_rng(1)
    returns = rng.normal(-0.0005, 0.01, size=5000)
    v95 = var(returns, 0.95)
    es95 = expected_shortfall(returns, 0.95)
    assert abs(es95) >= abs(v95)


def test_transaction_costs_reduce_returns():
    gross = np.array([0.01, 0.02, -0.005, 0.0])
    tc = 0.001  # 10bps per trade example
    net = gross - tc
    assert np.all(net <= gross + 1e-12)


def test_weights_sum_to_zero_after_neutralization():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=50)
    # simple long-short neutralization: subtract mean
    weights = raw - raw.mean()
    assert abs(weights.sum()) < 1e-12


def test_lob_best_bid_less_than_best_ask():
    bids = np.array([100.0, 99.5, 99.0])
    asks = np.array([100.5, 101.0, 102.0])
    assert bids.max() < asks.min()


def test_regression_recovers_lambda():
    rng = np.random.default_rng(123)
    n = 500
    ofi = rng.normal(scale=10.0, size=n)
    lam_true = 0.0025
    noise = rng.normal(scale=0.01, size=n)
    delta_p = lam_true * ofi + noise
    est, *_ = np.linalg.lstsq(ofi.reshape(-1, 1), delta_p, rcond=None)
    lam_hat = est[0]
    assert pytest.approx(lam_true, rel=0.2) == lam_hat


def test_no_lookahead_in_lagging():
    # Construct a simple signal that is shifted by 1 day to avoid lookahead
    prices = np.arange(100.0, 200.0)
    returns = np.diff(prices) / prices[:-1]
    signal = np.roll(returns, 1)  # lagged signal
    # ensure signal at time t uses information from <= t-1
    # i.e., signal[0] is contaminated indicator (here rolled) and should be ignored for first return
    assert np.all(signal[1:] == returns[:-1])


def test_var_coverage_synthetic():
    rng = np.random.default_rng(21)
    returns = rng.normal(loc=0.0, scale=0.01, size=2000)
    v95 = var(returns, 0.95)
    violations = np.mean(returns <= v95)
    # Expect roughly 5% violations within a reasonable tolerance
    assert abs(violations - 0.05) < 0.02
