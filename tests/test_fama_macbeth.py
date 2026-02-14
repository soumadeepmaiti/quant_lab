"""
Tests for Fama-MacBeth cross-sectional regression.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.alpha.fama_macbeth import (
    cross_sectional_ols,
    fama_macbeth,
    summary_table,
)


def test_cross_sectional_ols_basic():
    """Test cross-sectional OLS on simple data."""
    np.random.seed(42)
    N = 100

    # Generate synthetic data
    X = pd.DataFrame({"factor1": np.random.randn(N), "factor2": np.random.randn(N)})
    true_betas = np.array([0.02, 0.01])
    y = pd.Series((X.values @ true_betas) + np.random.randn(N) * 0.01)

    # Run regression
    betas, r2, nobs = cross_sectional_ols(X, y, add_const=True, min_obs=50)

    # Check shape
    assert len(betas) == 3  # const + 2 factors
    assert "const" in betas.index
    assert "factor1" in betas.index
    assert "factor2" in betas.index

    # Check nobs
    assert nobs == N

    # Check R2 is reasonable
    assert 0 <= r2 <= 1


def test_cross_sectional_ols_insufficient_data():
    """Test handling of insufficient observations."""
    np.random.seed(42)
    N = 10  # Less than min_obs

    X = pd.DataFrame({"factor1": np.random.randn(N)})
    y = pd.Series(np.random.randn(N))

    betas, r2, nobs = cross_sectional_ols(X, y, add_const=True, min_obs=50)

    # Should return NaNs
    assert np.all(pd.isna(betas))
    assert np.isnan(r2)
    assert nobs == N


def test_fama_macbeth_recovers_premia():
    """Test that Fama-MacBeth recovers true risk premia from synthetic data."""
    np.random.seed(42)

    # Parameters
    T = 60  # 60 months
    N = 30  # 30 assets
    K = 2  # 2 factors

    # True risk premia (monthly)
    lambda_true = np.array([0.01, 0.005])  # 1% and 0.5% per month

    # Generate panel data
    dates = pd.date_range("2020-01-01", periods=T, freq="ME")
    tickers = [f"ASSET_{i}" for i in range(N)]

    # Factor exposures (z-scored per date)
    panel_data = []
    for t, date in enumerate(dates):
        # Cross-sectional factor exposures
        factor_mom = np.random.randn(N)
        factor_vol = np.random.randn(N)

        # Z-score
        factor_mom = (factor_mom - factor_mom.mean()) / factor_mom.std()
        factor_vol = (factor_vol - factor_vol.mean()) / factor_vol.std()

        # Generate returns: r = X @ lambda + noise
        X = np.column_stack([factor_mom, factor_vol])
        ret_fwd = X @ lambda_true + np.random.randn(N) * 0.02

        for i, ticker in enumerate(tickers):
            panel_data.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "ret_fwd_1m": ret_fwd[i],
                    "factor_mom": factor_mom[i],
                    "factor_vol": factor_vol[i],
                }
            )

    panel = pd.DataFrame(panel_data).set_index(["date", "ticker"])

    # Run Fama-MacBeth
    result = fama_macbeth(
        panel=panel,
        factor_cols=["factor_mom", "factor_vol"],
        ret_col="ret_fwd_1m",
        add_const=True,
        min_obs=20,
        nw_lags=3,
        freq="M",
    )

    # Check recovered premia
    lambda_hat = result["lambda_hat"]

    # Allow 50% error due to noise
    assert pytest.approx(lambda_hat["factor_mom"], rel=0.5) == lambda_true[0]
    assert pytest.approx(lambda_hat["factor_vol"], rel=0.5) == lambda_true[1]

    # Check t-stats are large (should be significant)
    tstats = result["tstats_hac"]
    assert abs(tstats["factor_mom"]) > 1.0  # At least some signal
    assert abs(tstats["factor_vol"]) > 0.5


def test_fama_macbeth_handles_missing_data():
    """Test that Fama-MacBeth handles missing data gracefully."""
    np.random.seed(42)

    T = 30
    N = 20

    dates = pd.date_range("2020-01-01", periods=T, freq="ME")
    tickers = [f"ASSET_{i}" for i in range(N)]

    panel_data = []
    for date in dates:
        for ticker in tickers:
            # Randomly drop 10% of observations
            if np.random.rand() > 0.1:
                panel_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "ret_fwd_1m": np.random.randn() * 0.02,
                        "factor_mom": np.random.randn(),
                    }
                )

    panel = pd.DataFrame(panel_data).set_index(["date", "ticker"])

    # Should not crash
    result = fama_macbeth(
        panel=panel,
        factor_cols=["factor_mom"],
        ret_col="ret_fwd_1m",
        add_const=True,
        min_obs=10,
        nw_lags=3,
        freq="M",
    )

    # Check result structure
    assert "lambda_hat" in result
    assert "se_hac" in result
    assert "tstats_hac" in result


def test_nw_lags_default_behavior():
    """Test that default Newey-West lags are set correctly."""
    np.random.seed(42)

    T = 24
    N = 15

    dates = pd.date_range("2020-01-01", periods=T, freq="ME")
    tickers = [f"ASSET_{i}" for i in range(N)]

    panel_data = []
    for date in dates:
        for ticker in tickers:
            panel_data.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "ret_fwd_1m": np.random.randn() * 0.02,
                    "factor1": np.random.randn(),
                }
            )

    panel = pd.DataFrame(panel_data).set_index(["date", "ticker"])

    # Test monthly default (should be 3)
    result_m = fama_macbeth(
        panel=panel,
        factor_cols=["factor1"],
        ret_col="ret_fwd_1m",
        freq="M",
    )
    assert result_m["settings"]["nw_lags"] == 3

    # Test daily default (should be 5)
    result_d = fama_macbeth(
        panel=panel,
        factor_cols=["factor1"],
        ret_col="ret_fwd_1m",
        freq="D",
    )
    assert result_d["settings"]["nw_lags"] == 5


def test_summary_table():
    """Test summary table generation."""
    # Create mock result
    result = {
        "lambda_hat": pd.Series({"const": 0.01, "factor1": 0.02}),
        "se_hac": pd.Series({"const": 0.005, "factor1": 0.008}),
        "tstats_hac": pd.Series({"const": 2.0, "factor1": 2.5}),
        "pvals": pd.Series({"const": 0.045, "factor1": 0.012}),
        "r2_by_date": pd.Series([0.05, 0.06, 0.07]),
        "nobs_by_date": pd.Series([25, 26, 24]),
    }

    summary = summary_table(result)

    # Check columns
    assert "lambda_hat" in summary.columns
    assert "se_hac" in summary.columns
    assert "tstat_hac" in summary.columns
    assert "pval" in summary.columns
    assert "mean_r2" in summary.columns
    assert "mean_nobs" in summary.columns

    # Check values
    assert pytest.approx(summary.loc["const", "lambda_hat"]) == 0.01
    assert pytest.approx(summary.loc["factor1", "tstat_hac"]) == 2.5
