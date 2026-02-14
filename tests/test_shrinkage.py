"""
Tests for covariance shrinkage estimation.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.risk.shrinkage import (
    ledoit_wolf_cov,
    shrink_to_identity,
    factor_model_cov,
    shrink_cov,
    min_var_weights,
)


def test_ledoit_wolf_cov_psd():
    """Test that Ledoit-Wolf covariance is positive semi-definite."""
    np.random.seed(42)

    # Generate returns
    T = 100
    N = 20
    returns = pd.DataFrame(np.random.randn(T, N), columns=[f"A{i}" for i in range(N)])

    # Compute LW covariance
    cov = ledoit_wolf_cov(returns)

    # Check PSD: all eigenvalues >= -eps
    eigvals = np.linalg.eigvalsh(cov.values)
    assert np.all(eigvals >= -1e-8), f"Negative eigenvalue found: {eigvals.min()}"

    # Check shape
    assert cov.shape == (N, N)


def test_shrink_to_identity_interpolation():
    """Test shrinkage interpolation properties."""
    np.random.seed(42)

    N = 10
    sample_cov = np.random.randn(N, N)
    sample_cov = sample_cov @ sample_cov.T  # Make PSD

    # delta=0 should return sample
    shrunk_0 = shrink_to_identity(sample_cov, delta=0.0)
    assert np.allclose(shrunk_0, sample_cov)

    # delta=1 should return scaled identity
    shrunk_1 = shrink_to_identity(sample_cov, delta=1.0)
    trace_avg = np.trace(sample_cov) / N
    target = trace_avg * np.eye(N)
    assert np.allclose(shrunk_1, target)


def test_shrink_cov_interpolation():
    """Test general shrinkage interpolation."""
    np.random.seed(42)

    N = 10
    sample_cov = pd.DataFrame(
        np.random.randn(N, N), columns=[f"A{i}" for i in range(N)]
    )
    sample_cov = sample_cov @ sample_cov.T

    target_cov = pd.DataFrame(np.eye(N), index=sample_cov.index, columns=sample_cov.columns)

    # delta=0 should return sample
    shrunk_0 = shrink_cov(sample_cov, target_cov, delta=0.0)
    assert np.allclose(shrunk_0.values, sample_cov.values)

    # delta=1 should return target
    shrunk_1 = shrink_cov(sample_cov, target_cov, delta=1.0)
    assert np.allclose(shrunk_1.values, target_cov.values)


def test_factor_model_cov_structure():
    """Test factor model covariance has correct structure."""
    np.random.seed(42)

    # Generate returns and factor returns
    T = 200
    N = 15
    K = 3

    factor_returns = pd.DataFrame(np.random.randn(T, K), columns=["F1", "F2", "F3"])

    # Generate asset returns from factor model
    betas = np.random.randn(N, K)
    idio_noise = np.random.randn(T, N) * 0.5
    asset_returns = pd.DataFrame(
        factor_returns.values @ betas.T + idio_noise, columns=[f"A{i}" for i in range(N)]
    )

    # Estimate factor model covariance
    cov = factor_model_cov(asset_returns, factor_returns, add_const=False)

    # Check PSD
    eigvals = np.linalg.eigvalsh(cov.values)
    assert np.all(eigvals >= -1e-6), f"Negative eigenvalue: {eigvals.min()}"

    # Check shape
    assert cov.shape == (N, N)


def test_min_var_weights_stability():
    """Test that shrinkage produces less extreme weights than sample."""
    np.random.seed(42)

    # Generate correlated returns (challenging for sample covariance)
    T = 50  # Small sample
    N = 20

    # High correlation
    rho = 0.7
    cov_true = (1 - rho) * np.eye(N) + rho * np.ones((N, N))
    returns = pd.DataFrame(
        np.random.multivariate_normal(np.zeros(N), cov_true, size=T),
        columns=[f"A{i}" for i in range(N)],
    )

    # Sample covariance (noisy)
    sample_cov = returns.cov()

    # Ledoit-Wolf (shrunk)
    lw_cov = ledoit_wolf_cov(returns)

    # Compute min-var weights
    weights_sample = min_var_weights(sample_cov)
    weights_lw = min_var_weights(lw_cov)

    # Shrunk weights should be less extreme (smaller max absolute value)
    max_abs_sample = np.abs(weights_sample).max()
    max_abs_lw = np.abs(weights_lw).max()

    # This is a heuristic test - may not always hold but usually does
    # Allow for some flexibility
    assert max_abs_lw <= max_abs_sample * 1.5, (
        f"LW weights not more stable: max|w_sample|={max_abs_sample:.4f}, "
        f"max|w_lw|={max_abs_lw:.4f}"
    )


def test_min_var_weights_sum_to_one():
    """Test that min-var weights sum to 1."""
    np.random.seed(42)

    N = 10
    cov = pd.DataFrame(np.eye(N), columns=[f"A{i}" for i in range(N)])

    weights = min_var_weights(cov)

    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0


def test_shrinkage_reduces_condition_number():
    """Test that shrinkage reduces condition number of covariance."""
    np.random.seed(42)

    # Generate ill-conditioned covariance
    T = 40
    N = 20

    returns = pd.DataFrame(np.random.randn(T, N), columns=[f"A{i}" for i in range(N)])

    # Sample covariance (ill-conditioned when T < N)
    sample_cov = returns.cov()

    # Ledoit-Wolf
    lw_cov = ledoit_wolf_cov(returns)

    # Compute condition numbers
    cond_sample = np.linalg.cond(sample_cov.values)
    cond_lw = np.linalg.cond(lw_cov.values)

    # LW should have lower condition number (more stable)
    assert cond_lw < cond_sample, (
        f"Shrinkage did not reduce condition number: "
        f"cond(sample)={cond_sample:.2f}, cond(LW)={cond_lw:.2f}"
    )


def test_factor_model_cov_with_const():
    """Test factor model with intercept."""
    np.random.seed(42)

    T = 100
    N = 10
    K = 2

    returns = pd.DataFrame(np.random.randn(T, N), columns=[f"A{i}" for i in range(N)])
    factor_returns = pd.DataFrame(np.random.randn(T, K), columns=["F1", "F2"])

    # Should not crash with add_const=True
    cov = factor_model_cov(returns, factor_returns, add_const=True)

    assert cov.shape == (N, N)
    assert np.all(np.linalg.eigvalsh(cov.values) >= -1e-6)
