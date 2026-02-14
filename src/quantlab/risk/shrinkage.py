"""
Covariance shrinkage estimation for robust portfolio construction.

Implements Ledoit-Wolf shrinkage, factor model covariance, and combined shrinkage
to reduce estimation error and improve out-of-sample stability.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ledoit-Wolf shrinkage covariance estimator.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix (T × N), should be demeaned

    Returns
    -------
    pd.DataFrame
        Shrinkage covariance matrix (N × N)
    """
    # Fit Ledoit-Wolf
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns.values)

    # Get shrinkage covariance
    cov_shrunk = lw.covariance_

    # Return as DataFrame
    cov_df = pd.DataFrame(cov_shrunk, index=returns.columns, columns=returns.columns)

    return cov_df


def shrink_to_identity(sample_cov: np.ndarray, delta: float) -> np.ndarray:
    """
    Shrink sample covariance toward identity matrix.

    Sigma_shrunk = (1 - delta) * sample_cov + delta * trace(sample_cov)/N * I

    Parameters
    ----------
    sample_cov : np.ndarray
        Sample covariance matrix (N × N)
    delta : float
        Shrinkage intensity [0, 1]

    Returns
    -------
    np.ndarray
        Shrunk covariance
    """
    N = sample_cov.shape[0]
    trace_avg = np.trace(sample_cov) / N

    target = trace_avg * np.eye(N)
    shrunk = (1 - delta) * sample_cov + delta * target

    return shrunk


def factor_model_cov(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    add_const: bool = True,
) -> pd.DataFrame:
    """
    Compute factor model covariance matrix.

    Model: R = B * F + epsilon

    Covariance: Sigma = B * Cov(F) * B' + Diag(Var(epsilon))

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T × N)
    factor_returns : pd.DataFrame
        Factor returns (T × K), must align with returns index
    add_const : bool
        Add intercept in regression

    Returns
    -------
    pd.DataFrame
        Factor model covariance (N × N)
    """
    # Align dates
    common_dates = returns.index.intersection(factor_returns.index)
    returns_aligned = returns.loc[common_dates]
    factors_aligned = factor_returns.loc[common_dates]

    N = returns_aligned.shape[1]
    K = factors_aligned.shape[1]

    # Estimate betas via time-series regression for each asset
    betas = np.zeros((N, K if not add_const else K + 1))
    residual_vars = np.zeros(N)

    X = factors_aligned.values
    if add_const:
        X = sm.add_constant(X)

    for i, asset in enumerate(returns_aligned.columns):
        y = returns_aligned[asset].values

        # OLS
        try:
            model = sm.OLS(y, X, missing="drop").fit()
            betas[i, :] = model.params
            residual_vars[i] = model.mse_resid
        except Exception:
            # If regression fails, use zeros
            betas[i, :] = 0
            residual_vars[i] = np.var(y)

    # Extract beta matrix (drop const column if present)
    if add_const:
        B = betas[:, 1:]  # Drop intercept
    else:
        B = betas

    # Factor covariance
    Cov_F = np.cov(factors_aligned.values, rowvar=False)

    # Factor model covariance: B * Cov(F) * B'
    factor_cov = B @ Cov_F @ B.T

    # Add idiosyncratic variance
    idio_cov = np.diag(residual_vars)

    total_cov = factor_cov + idio_cov

    # Return as DataFrame
    cov_df = pd.DataFrame(
        total_cov, index=returns_aligned.columns, columns=returns_aligned.columns
    )

    return cov_df


def shrink_cov(
    sample_cov: pd.DataFrame, target_cov: pd.DataFrame, delta: float
) -> pd.DataFrame:
    """
    Linearly shrink sample covariance toward target.

    Sigma_shrunk = (1 - delta) * sample_cov + delta * target_cov

    Parameters
    ----------
    sample_cov : pd.DataFrame
        Sample covariance (N × N)
    target_cov : pd.DataFrame
        Target covariance (N × N)
    delta : float
        Shrinkage intensity [0, 1]

    Returns
    -------
    pd.DataFrame
        Shrunk covariance
    """
    shrunk = (1 - delta) * sample_cov + delta * target_cov
    return shrunk


def min_var_weights(cov: pd.DataFrame, long_only: bool = False) -> pd.Series:
    """
    Compute minimum variance portfolio weights.

    Unconstrained: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix (N × N)
    long_only : bool
        If True, enforce non-negative weights (not implemented here, returns unconstrained)

    Returns
    -------
    pd.Series
        Portfolio weights
    """
    try:
        cov_inv = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov.values)

    ones = np.ones(len(cov))
    w = cov_inv @ ones
    w = w / np.sum(w)

    weights = pd.Series(w, index=cov.index)

    return weights


def rolling_shrinkage_backtest(
    returns: pd.DataFrame,
    train_window: int = 252,
    method: str = "ledoit_wolf",
    factor_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Rolling out-of-sample backtest comparing shrinkage methods.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (T × N)
    train_window : int
        Training window size
    method : str
        'sample', 'ledoit_wolf', or 'factor_shrink'
    factor_returns : pd.DataFrame, optional
        Factor returns for factor model (if method='factor_shrink')

    Returns
    -------
    pd.DataFrame
        Results with columns: date, weights, realized_vol, turnover
    """
    dates = returns.index[train_window:]
    results = []

    prev_weights = None

    for i, date in enumerate(dates):
        # Training window
        train_start = i
        train_end = i + train_window
        train_returns = returns.iloc[train_start:train_end]

        # Estimate covariance
        if method == "sample":
            cov = train_returns.cov()
        elif method == "ledoit_wolf":
            cov = ledoit_wolf_cov(train_returns)
        elif method == "factor_shrink":
            if factor_returns is None:
                raise ValueError("factor_returns required for factor_shrink")
            sample_cov = train_returns.cov()
            target_cov = factor_model_cov(train_returns, factor_returns)
            # Use LedoitWolf optimal delta as heuristic
            lw = LedoitWolf(assume_centered=False)
            lw.fit(train_returns.values)
            delta = lw.shrinkage_
            cov = shrink_cov(sample_cov, target_cov, delta)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute min-var weights
        weights = min_var_weights(cov)

        # Out-of-sample return (next day)
        if i + train_window < len(returns):
            oos_ret = returns.iloc[i + train_window]
            port_ret = (weights * oos_ret).sum()
        else:
            port_ret = np.nan

        # Turnover
        if prev_weights is not None:
            turnover = np.abs(weights - prev_weights).sum()
        else:
            turnover = np.nan

        prev_weights = weights.copy()

        results.append(
            {
                "date": date,
                "port_ret": port_ret,
                "turnover": turnover,
            }
        )

    results_df = pd.DataFrame(results).set_index("date")

    # Compute realized volatility
    results_df["realized_vol"] = results_df["port_ret"].rolling(21).std() * np.sqrt(252)

    return results_df
