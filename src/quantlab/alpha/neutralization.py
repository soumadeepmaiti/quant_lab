"""
Factor Neutralization & Orthogonalization Module.

Implements research-grade factor analysis:
- Cross-sectional neutralization (remove beta, sector, size exposures)
- Factor orthogonalization (Gram-Schmidt decorrelation)
- Pure alpha isolation

References:
- Fama-French (1993) - Factor models
- Asness et al. (2013) - Value and Momentum Everywhere
"""

from typing import Dict

import numpy as np
import pandas as pd

from quantlab.config import get_logger

logger = get_logger(__name__)


def cross_sectional_neutralize(
    factor: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    method: str = 'regression'
) -> pd.DataFrame:
    """
    Neutralize factor against control variables using cross-sectional regression.
    
    For each time period t, runs:
        factor_i = α + Σ β_k * control_k_i + ε_i
    
    Returns residuals ε as the neutralized factor.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates × stocks)
    controls : dict
        Dictionary of control DataFrames {'beta': beta_df, 'size': size_df, ...}
    method : str
        'regression' (OLS) or 'rank' (rank-based neutralization)
    
    Returns
    -------
    pd.DataFrame
        Neutralized factor (residuals from cross-sectional regression)
    
    Example
    -------
    >>> neutralized = cross_sectional_neutralize(
    ...     momentum,
    ...     {'beta': market_beta, 'size': log_market_cap}
    ... )
    """
    if method not in ['regression', 'rank']:
        raise ValueError(f"method must be 'regression' or 'rank', got {method}")

    neutralized = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)

    for date in factor.index:
        # Get factor values for this date
        y = factor.loc[date].dropna()

        if len(y) < 10:
            continue

        # Build control matrix
        X_data = {'const': np.ones(len(y))}
        valid_stocks = y.index

        for name, control_df in controls.items():
            if date in control_df.index:
                ctrl_vals = control_df.loc[date].reindex(valid_stocks)
                # Only include if we have enough non-NaN values
                if ctrl_vals.notna().sum() > len(valid_stocks) * 0.5:
                    X_data[name] = ctrl_vals.values

        # Check if we have any controls
        if len(X_data) == 1:  # Only constant
            neutralized.loc[date, y.index] = y - y.mean()
            continue

        X = pd.DataFrame(X_data, index=valid_stocks)

        # Drop rows with NaN in controls
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_mask]
        y_clean = y.loc[valid_mask]

        if len(y_clean) < 10:
            continue

        if method == 'regression':
            # OLS regression
            try:
                X_mat = X_clean.values
                y_vec = y_clean.values

                # Solve normal equations: β = (X'X)^(-1) X'y
                beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
                residuals = y_vec - X_mat @ beta

                neutralized.loc[date, y_clean.index] = residuals

            except np.linalg.LinAlgError:
                # Fallback to demeaning
                neutralized.loc[date, y_clean.index] = y_clean - y_clean.mean()

        else:  # rank method
            # Rank-based: subtract median within each control bucket
            y_ranked = y_clean.rank(pct=True)
            neutralized.loc[date, y_clean.index] = y_ranked - y_ranked.mean()

    return neutralized


def neutralize_sector(
    factor: pd.DataFrame,
    sector_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Neutralize factor within sectors (demean within each sector).
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates × stocks)
    sector_map : dict
        Mapping of ticker to sector {'AAPL': 'Technology', ...}
    
    Returns
    -------
    pd.DataFrame
        Sector-neutralized factor
    """
    neutralized = factor.copy()

    # Get sectors for available stocks
    stocks = factor.columns
    sectors = pd.Series({s: sector_map.get(s, 'Other') for s in stocks})

    for date in factor.index:
        row = factor.loc[date].dropna()

        for sector in sectors.unique():
            sector_stocks = sectors[sectors == sector].index
            sector_stocks = [s for s in sector_stocks if s in row.index]

            if len(sector_stocks) > 1:
                sector_mean = row[sector_stocks].mean()
                neutralized.loc[date, sector_stocks] = row[sector_stocks] - sector_mean

    return neutralized


def orthogonalize_factors(
    factors: Dict[str, pd.DataFrame],
    method: str = 'gram_schmidt'
) -> Dict[str, pd.DataFrame]:
    """
    Orthogonalize multiple factors to remove cross-factor correlation.
    
    Uses Gram-Schmidt process:
        f1_orth = f1
        f2_orth = f2 - proj(f2, f1_orth)
        f3_orth = f3 - proj(f3, f1_orth) - proj(f3, f2_orth)
        ...
    
    Parameters
    ----------
    factors : dict
        Dictionary of factor DataFrames {'momentum': mom_df, 'value': val_df}
    method : str
        'gram_schmidt' or 'symmetric' (Löwdin)
    
    Returns
    -------
    dict
        Dictionary of orthogonalized factors
    
    Example
    -------
    >>> factors = {'momentum': mom, 'value': val, 'quality': qual}
    >>> orth_factors = orthogonalize_factors(factors)
    """
    factor_names = list(factors.keys())
    n_factors = len(factor_names)

    if n_factors < 2:
        return factors.copy()

    # Align all factors to common dates and stocks
    common_dates = factors[factor_names[0]].index
    common_stocks = factors[factor_names[0]].columns

    for name in factor_names[1:]:
        common_dates = common_dates.intersection(factors[name].index)
        common_stocks = common_stocks.intersection(factors[name].columns)

    orthogonalized = {}

    if method == 'gram_schmidt':
        # Gram-Schmidt orthogonalization
        orth_list = []

        for i, name in enumerate(factor_names):
            f = factors[name].loc[common_dates, common_stocks].copy()

            # Subtract projections onto previous orthogonalized factors
            for j in range(i):
                f_prev = orth_list[j]

                # Compute projection coefficient for each date
                for date in common_dates:
                    f_row = f.loc[date].dropna()
                    f_prev_row = f_prev.loc[date].reindex(f_row.index).dropna()

                    common = f_row.index.intersection(f_prev_row.index)
                    if len(common) < 5:
                        continue

                    # proj(f, f_prev) = (f · f_prev) / (f_prev · f_prev) * f_prev
                    dot_product = (f_row[common] * f_prev_row[common]).sum()
                    norm_sq = (f_prev_row[common] ** 2).sum()

                    if norm_sq > 1e-10:
                        proj_coef = dot_product / norm_sq
                        f.loc[date, common] -= proj_coef * f_prev_row[common]

            orth_list.append(f)
            orthogonalized[name] = f

    elif method == 'symmetric':
        # Symmetric orthogonalization (Löwdin)
        # This preserves variance better but is computationally heavier
        for date in common_dates:
            # Stack factors into matrix
            F = np.zeros((len(common_stocks), n_factors))

            for i, name in enumerate(factor_names):
                F[:, i] = factors[name].loc[date, common_stocks].fillna(0).values

            # Compute correlation matrix
            C = np.corrcoef(F.T)

            if np.any(np.isnan(C)):
                continue

            # Symmetric orthogonalization: F_orth = F @ C^(-1/2)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)  # Numerical stability
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                F_orth = F @ C_inv_sqrt

                for i, name in enumerate(factor_names):
                    if name not in orthogonalized:
                        orthogonalized[name] = pd.DataFrame(
                            index=common_dates, columns=common_stocks, dtype=float
                        )
                    orthogonalized[name].loc[date] = F_orth[:, i]

            except np.linalg.LinAlgError:
                continue

    else:
        raise ValueError(f"method must be 'gram_schmidt' or 'symmetric', got {method}")

    return orthogonalized


def compute_factor_exposures(
    returns: pd.DataFrame,
    factors: Dict[str, pd.DataFrame],
    window: int = 60
) -> Dict[str, pd.DataFrame]:
    """
    Compute rolling factor exposures (betas) for each stock.
    
    For each stock i and time t, estimates:
        r_i,t = α_i + Σ β_i,k * factor_k,t + ε_i,t
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns (dates × stocks)
    factors : dict
        Factor return series
    window : int
        Rolling window for beta estimation
    
    Returns
    -------
    dict
        Dictionary of beta DataFrames for each factor
    """
    exposures = {name: pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
                 for name in factors.keys()}

    for stock in returns.columns:
        stock_ret = returns[stock].dropna()

        if len(stock_ret) < window:
            continue

        for i in range(window, len(stock_ret)):
            date = stock_ret.index[i]
            ret_window = stock_ret.iloc[i-window:i]

            # Build factor matrix for this window
            X_data = {'const': np.ones(window)}

            for name, factor_df in factors.items():
                if stock in factor_df.columns:
                    factor_vals = factor_df[stock].reindex(ret_window.index)
                    if factor_vals.notna().sum() > window * 0.8:
                        X_data[name] = factor_vals.fillna(0).values

            if len(X_data) <= 1:
                continue

            X = np.column_stack(list(X_data.values()))
            y = ret_window.values

            try:
                betas = np.linalg.lstsq(X, y, rcond=None)[0]

                for j, name in enumerate(list(X_data.keys())[1:], 1):
                    exposures[name].loc[date, stock] = betas[j]

            except np.linalg.LinAlgError:
                continue

    return exposures


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: Dict[str, pd.Series],
    rf_rate: float = 0.0
) -> pd.DataFrame:
    """
    Decompose portfolio returns into factor contributions.
    
    Runs regression:
        r_p - r_f = α + Σ β_k * (f_k - r_f) + ε
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio return series
    factor_returns : dict
        Dictionary of factor return series
    rf_rate : float
        Risk-free rate (annualized, converted internally)
    
    Returns
    -------
    pd.DataFrame
        Attribution results with alpha, betas, R², and contribution
    """
    # Align dates
    common_dates = portfolio_returns.dropna().index
    for f in factor_returns.values():
        common_dates = common_dates.intersection(f.dropna().index)

    # Excess returns
    rf_daily = rf_rate / 252
    y = portfolio_returns.loc[common_dates] - rf_daily

    # Build factor matrix
    X_data = {'const': np.ones(len(common_dates))}
    for name, f in factor_returns.items():
        X_data[name] = (f.loc[common_dates] - rf_daily).values

    X = np.column_stack(list(X_data.values()))
    y_vec = y.values

    # OLS
    betas, residuals, rank, s = np.linalg.lstsq(X, y_vec, rcond=None)

    # Compute statistics
    y_pred = X @ betas
    ss_res = np.sum((y_vec - y_pred) ** 2)
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard errors
    n = len(y_vec)
    k = X.shape[1]
    mse = ss_res / (n - k) if n > k else 0

    try:
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_beta = np.sqrt(np.maximum(var_beta, 0))
    except np.linalg.LinAlgError:
        se_beta = np.zeros(k)

    t_stats = betas / (se_beta + 1e-10)

    # Build results
    results = []
    factor_names = list(X_data.keys())

    for i, name in enumerate(factor_names):
        contrib = betas[i] * np.mean(X[:, i]) * 252 if i > 0 else betas[i] * 252

        results.append({
            'factor': name if name != 'const' else 'alpha',
            'beta': betas[i],
            'std_error': se_beta[i],
            't_stat': t_stats[i],
            'contribution_annual': contrib
        })

    df = pd.DataFrame(results)
    df['r_squared'] = r_squared

    return df


def pure_factor_portfolio(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    top_pct: float = 0.2,
    bottom_pct: float = 0.2
) -> pd.Series:
    """
    Construct factor-mimicking portfolio with control neutralization.
    
    1. Neutralize factor against controls
    2. Form long-short portfolio on neutralized factor
    3. Return portfolio returns
    
    This gives "pure" factor exposure independent of control variables.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Raw factor values
    returns : pd.DataFrame
        Stock returns
    controls : dict
        Control variables to neutralize against
    top_pct : float
        Top percentile for long leg
    bottom_pct : float
        Bottom percentile for short leg
    
    Returns
    -------
    pd.Series
        Pure factor portfolio returns
    """
    # Neutralize factor
    neutral_factor = cross_sectional_neutralize(factor, controls)

    # Construct long-short portfolio
    portfolio_returns = []

    for date in neutral_factor.index[:-1]:
        next_date_idx = neutral_factor.index.get_loc(date) + 1
        if next_date_idx >= len(neutral_factor.index):
            break
        next_date = neutral_factor.index[next_date_idx]

        # Get factor values
        factor_vals = neutral_factor.loc[date].dropna()

        if len(factor_vals) < 10:
            continue

        # Rank and select
        ranks = factor_vals.rank(pct=True)

        long_stocks = ranks[ranks >= (1 - top_pct)].index
        short_stocks = ranks[ranks <= bottom_pct].index

        if len(long_stocks) == 0 or len(short_stocks) == 0:
            continue

        # Get next period returns
        if next_date not in returns.index:
            continue

        next_returns = returns.loc[next_date]

        long_ret = next_returns.reindex(long_stocks).mean()
        short_ret = next_returns.reindex(short_stocks).mean()

        if pd.notna(long_ret) and pd.notna(short_ret):
            portfolio_returns.append({
                'date': next_date,
                'return': long_ret - short_ret
            })

    if not portfolio_returns:
        return pd.Series(dtype=float)

    return pd.DataFrame(portfolio_returns).set_index('date')['return']
