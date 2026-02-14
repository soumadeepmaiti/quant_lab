"""
Factor risk decomposition and marginal risk contribution analysis.
"""

import numpy as np
import pandas as pd


def factor_risk_decomposition(
    weights: pd.Series, cov: pd.DataFrame
) -> dict:
    """
    Decompose portfolio risk into marginal and total risk contributions.

    Formulas:
    - Portfolio variance: sigma_p^2 = w' Sigma w
    - Portfolio std dev: sigma_p = sqrt(w' Sigma w)
    - Marginal risk contribution: MRC_i = (Sigma w)_i
    - Total risk contribution: RC_i = w_i * MRC_i
    - Percent risk contribution: PctRC_i = RC_i / sigma_p

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (N assets)
    cov : pd.DataFrame
        Covariance matrix (N × N)

    Returns
    -------
    dict with:
        - port_var : float
        - port_sd : float
        - mrc : pd.Series (marginal risk contributions)
        - rc : pd.Series (total risk contributions)
        - pct_rc : pd.Series (percent risk contributions)
    """
    # Align
    weights = weights.reindex(cov.index, fill_value=0)

    w = weights.values
    Sigma = cov.values

    # Portfolio variance
    port_var = w @ Sigma @ w

    # Portfolio std dev
    port_sd = np.sqrt(max(port_var, 0))

    # Marginal risk contribution: Sigma * w
    mrc = Sigma @ w

    # Total risk contribution: w_i * MRC_i
    rc = w * mrc

    # Percent risk contribution: RC_i / sigma_p
    if port_sd > 1e-8:
        pct_rc = rc / port_sd
    else:
        pct_rc = np.zeros_like(rc)

    return {
        "port_var": port_var,
        "port_sd": port_sd,
        "mrc": pd.Series(mrc, index=cov.index, name="MRC"),
        "rc": pd.Series(rc, index=cov.index, name="RC"),
        "pct_rc": pd.Series(pct_rc, index=cov.index, name="PctRC"),
    }


def compare_risk_contributions(
    weights: pd.Series, sample_cov: pd.DataFrame, shrunk_cov: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare risk contributions under sample vs shrinkage covariance.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights
    sample_cov : pd.DataFrame
        Sample covariance
    shrunk_cov : pd.DataFrame
        Shrunk covariance

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: pct_rc_sample, pct_rc_shrunk, diff
    """
    # Compute risk decompositions
    risk_sample = factor_risk_decomposition(weights, sample_cov)
    risk_shrunk = factor_risk_decomposition(weights, shrunk_cov)

    # Build comparison
    comparison = pd.DataFrame(
        {
            "pct_rc_sample": risk_sample["pct_rc"],
            "pct_rc_shrunk": risk_shrunk["pct_rc"],
        }
    )

    comparison["diff"] = comparison["pct_rc_shrunk"] - comparison["pct_rc_sample"]

    # Add portfolio-level stats
    comparison.attrs["port_sd_sample"] = risk_sample["port_sd"]
    comparison.attrs["port_sd_shrunk"] = risk_shrunk["port_sd"]

    return comparison


def plot_risk_contrib_comparison(
    comparison: pd.DataFrame, save_path: str = "reports/figures/risk_contrib_sample_vs_shrinkage.png"
):
    """
    Plot risk contribution comparison.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output from compare_risk_contributions()
    save_path : str
        Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 10 contributors for each
    top_sample = comparison["pct_rc_sample"].abs().nlargest(10)
    top_shrunk = comparison["pct_rc_shrunk"].abs().nlargest(10)

    # Plot 1: Sample covariance
    top_sample.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Top 10 Risk Contributors (Sample Cov)")
    axes[0].set_xlabel("% Risk Contribution")
    axes[0].axvline(0, color="black", linewidth=0.8)

    # Plot 2: Shrunk covariance
    top_shrunk.plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Top 10 Risk Contributors (Shrunk Cov)")
    axes[1].set_xlabel("% Risk Contribution")
    axes[1].axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved risk contribution comparison to {save_path}")
