"""
Fama-MacBeth cross-sectional regression with Newey-West HAC standard errors.

This module implements institutional-grade factor risk premia estimation using
two-stage cross-sectional regression, following Fama & MacBeth (1973).
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac


def cross_sectional_ols(
    X: pd.DataFrame,
    y: pd.Series,
    add_const: bool = True,
    min_obs: int = 50,
) -> tuple[pd.Series, float, int]:
    """
    Run single cross-sectional OLS regression.

    Parameters
    ----------
    X : pd.DataFrame
        Factor exposures (N assets × K factors)
    y : pd.Series
        Forward returns (N assets)
    add_const : bool
        Add intercept
    min_obs : int
        Minimum observations required

    Returns
    -------
    betas : pd.Series
        Factor loadings (includes 'const' if add_const=True)
    r2 : float
        R-squared
    nobs : int
        Number of observations
    """
    # Align and drop NaNs
    data = pd.concat([y, X], axis=1).dropna()

    if len(data) < min_obs:
        # Return NaNs if insufficient data
        cols = ["const"] + list(X.columns) if add_const else list(X.columns)
        return pd.Series(np.nan, index=cols), np.nan, len(data)

    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]

    if add_const:
        X_clean = sm.add_constant(X_clean)

    # Run OLS
    try:
        model = OLS(y_clean, X_clean).fit()
        betas = pd.Series(model.params, index=X_clean.columns)
        r2 = model.rsquared
        nobs = int(model.nobs)
    except Exception as e:
        warnings.warn(f"OLS failed: {e}")
        cols = X_clean.columns
        betas = pd.Series(np.nan, index=cols)
        r2 = np.nan
        nobs = len(data)

    return betas, r2, nobs


def fama_macbeth(
    panel: pd.DataFrame,
    factor_cols: list[str],
    ret_col: str = "ret_fwd_1m",
    add_const: bool = True,
    min_obs: int = 50,
    nw_lags: Optional[int] = None,
    freq: str = "M",
) -> dict:
    """
    Fama-MacBeth two-stage cross-sectional regression.

    Stage 1: For each date t, run cross-sectional regression:
        r_{i,t+1} = a_t + b_t' x_{i,t} + eps_{i,t+1}

    Stage 2: Compute time-series average risk premia and HAC standard errors.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with MultiIndex [date, ticker] containing:
        - ret_col: forward returns (strictly future)
        - factor_cols: factor exposures (lagged, no lookahead)
    factor_cols : list[str]
        Column names of factors
    ret_col : str
        Forward return column name
    add_const : bool
        Include intercept
    min_obs : int
        Minimum cross-sectional observations per date
    nw_lags : int, optional
        Newey-West lags. Default: 3 for monthly, 5 for daily
    freq : str
        Frequency: 'M' for monthly, 'D' for daily

    Returns
    -------
    dict with:
        - betas_by_date: DataFrame (date × factors)
        - lambda_hat: Series (mean premia)
        - se_hac: Series (HAC standard errors)
        - tstats_hac: Series (t-statistics)
        - pvals: Series (p-values, two-sided)
        - nobs_by_date: Series
        - r2_by_date: Series
        - settings: dict
    """
    # Set default lags
    if nw_lags is None:
        nw_lags = 3 if freq == "M" else 5

    # Ensure panel has the required columns
    required_cols = [ret_col] + factor_cols
    missing = set(required_cols) - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    # Get unique dates (level 0 of MultiIndex)
    dates = panel.index.get_level_values(0).unique().sort_values()

    # Storage
    betas_list = []
    r2_list = []
    nobs_list = []

    # Stage 1: Cross-sectional regressions
    for date in dates:
        date_data = panel.xs(date, level=0)

        y = date_data[ret_col]
        X = date_data[factor_cols]

        betas, r2, nobs = cross_sectional_ols(X, y, add_const=add_const, min_obs=min_obs)

        betas_list.append(betas)
        r2_list.append(r2)
        nobs_list.append(nobs)

    # Organize results
    betas_by_date = pd.DataFrame(betas_list, index=dates)
    r2_by_date = pd.Series(r2_list, index=dates, name="r2")
    nobs_by_date = pd.Series(nobs_list, index=dates, name="nobs")

    # Stage 2: Time-series average and HAC standard errors
    lambda_hat = betas_by_date.mean(axis=0)

    # Compute HAC standard errors using Newey-West
    se_hac = pd.Series(index=lambda_hat.index, dtype=float)
    for col in betas_by_date.columns:
        beta_series = betas_by_date[col].dropna()
        if len(beta_series) < 2:
            se_hac[col] = np.nan
            continue

        # Fit OLS with constant only (testing H0: mean = 0 after demeaning)
        T = len(beta_series)
        y_hac = beta_series.values
        X_hac = np.ones((T, 1))

        try:
            model = OLS(y_hac, X_hac).fit()
            # Compute HAC standard errors using Newey-West
            cov_hac_matrix = model.cov_HC0  # Start with heteroskedasticity-robust
            # Apply Newey-West adjustment
            from statsmodels.stats.sandwich_covariance import cov_hac as cov_hac_func

            cov_hac_matrix = cov_hac_func(model, nlags=nw_lags, use_correction=True)
            se_hac[col] = np.sqrt(cov_hac_matrix[0, 0])
        except Exception as e:
            warnings.warn(f"HAC failed for {col}: {e}")
            # Fall back to simple standard error
            se_hac[col] = beta_series.std() / np.sqrt(T)

    # T-statistics
    tstats_hac = lambda_hat / se_hac

    # Two-sided p-values (normal approximation)
    from scipy.stats import norm

    pvals = 2 * (1 - norm.cdf(np.abs(tstats_hac)))

    return {
        "betas_by_date": betas_by_date,
        "lambda_hat": lambda_hat,
        "se_hac": se_hac,
        "tstats_hac": tstats_hac,
        "pvals": pvals,
        "nobs_by_date": nobs_by_date,
        "r2_by_date": r2_by_date,
        "settings": {
            "factor_cols": factor_cols,
            "ret_col": ret_col,
            "add_const": add_const,
            "min_obs": min_obs,
            "nw_lags": nw_lags,
            "freq": freq,
            "n_dates": len(dates),
        },
    }


def summary_table(result: dict) -> pd.DataFrame:
    """
    Create summary table of Fama-MacBeth results.

    Parameters
    ----------
    result : dict
        Output from fama_macbeth()

    Returns
    -------
    pd.DataFrame
        Summary with lambda_hat, se_hac, tstat, pval, mean_r2, mean_nobs
    """
    summary = pd.DataFrame(
        {
            "lambda_hat": result["lambda_hat"],
            "se_hac": result["se_hac"],
            "tstat_hac": result["tstats_hac"],
            "pval": result["pvals"],
        }
    )

    # Add aggregate statistics
    summary["mean_r2"] = result["r2_by_date"].mean()
    summary["mean_nobs"] = result["nobs_by_date"].mean()

    return summary


def build_monthly_panel(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    forward_periods: int = 1,
) -> pd.DataFrame:
    """
    Build monthly panel with forward returns and lagged factors.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices (date × tickers)
    factors : pd.DataFrame
        Daily factor values (date × tickers)
    forward_periods : int
        Forward return horizon in months

    Returns
    -------
    pd.DataFrame
        Panel with MultiIndex [date, ticker]
        Columns: ret_fwd_1m, factor_*
    """
    # Resample to monthly (end of month)
    prices_monthly = prices.resample("M").last()
    factors_monthly = factors.resample("M").last()

    # Compute forward returns (strictly future)
    ret_fwd = prices_monthly.pct_change(periods=forward_periods).shift(-forward_periods)

    # Stack to panel format
    ret_panel = ret_fwd.stack().to_frame("ret_fwd_1m")
    factor_panel = factors_monthly.stack().to_frame()

    # Merge
    panel = ret_panel.join(factor_panel, how="inner")

    # Ensure MultiIndex names
    panel.index.names = ["date", "ticker"]

    return panel
