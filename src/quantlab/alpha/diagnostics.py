"""
Diagnostic plots for Fama-MacBeth and asset pricing analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_premia_rolling(
    betas_by_date: pd.DataFrame,
    window: int = 12,
    save_path: str = "reports/figures/fmb_premia_rolling.png",
):
    """
    Plot rolling mean of factor risk premia.

    Parameters
    ----------
    betas_by_date : pd.DataFrame
        Time series of cross-sectional betas (date × factors)
    window : int
        Rolling window size (months)
    save_path : str
        Path to save figure
    """
    rolling_mean = betas_by_date.rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in rolling_mean.columns:
        if col != "const":  # Skip intercept
            ax.plot(rolling_mean.index, rolling_mean[col], label=col, linewidth=2)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk Premium (Rolling Mean)")
    ax.set_title(f"Fama-MacBeth Risk Premia (Rolling {window}-Period Average)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved rolling premia plot to {save_path}")


def plot_premia_timeseries(
    betas_by_date: pd.DataFrame,
    top_n: int = 3,
    save_path: str = "reports/figures/fmb_premia_timeseries.png",
):
    """
    Plot time series of factor risk premia for top factors.

    Parameters
    ----------
    betas_by_date : pd.DataFrame
        Time series of cross-sectional betas
    top_n : int
        Number of top factors to plot (by mean absolute premium)
    save_path : str
        Path to save figure
    """
    # Select top factors by mean absolute value
    mean_abs = betas_by_date.drop(columns=["const"], errors="ignore").abs().mean()
    top_factors = mean_abs.nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(top_n, 1, figsize=(12, 3 * top_n), sharex=True)

    if top_n == 1:
        axes = [axes]

    for i, factor in enumerate(top_factors):
        ax = axes[i]
        series = betas_by_date[factor]

        ax.plot(series.index, series.values, color="steelblue", linewidth=1.5)
        ax.axhline(
            series.mean(), color="red", linestyle="--", linewidth=1, label="Mean"
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        ax.set_ylabel(f"{factor}")
        ax.set_title(f"Risk Premium: {factor}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved premia time series to {save_path}")


def plot_r2_histogram(
    r2_by_date: pd.Series, save_path: str = "reports/figures/fmb_r2_hist.png"
):
    """
    Plot histogram of cross-sectional R-squared values.

    Parameters
    ----------
    r2_by_date : pd.Series
        Time series of R-squared values
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    r2_clean = r2_by_date.dropna()

    ax.hist(r2_clean, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(
        r2_clean.mean(), color="red", linestyle="--", linewidth=2, label="Mean R²"
    )

    ax.set_xlabel("Cross-Sectional R²")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Cross-Sectional R² (Fama-MacBeth)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved R² histogram to {save_path}")


def plot_shrinkage_oos_vol(
    results_dict: dict, save_path: str = "reports/figures/shrinkage_oos_vol.png"
):
    """
    Plot out-of-sample volatility comparison across shrinkage methods.

    Parameters
    ----------
    results_dict : dict
        Dict with keys: method names, values: results DataFrames with 'realized_vol'
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, results in results_dict.items():
        vol_series = results["realized_vol"].dropna()
        ax.plot(vol_series.index, vol_series.values, label=method, linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Volatility (Annualized)")
    ax.set_title("Out-of-Sample Portfolio Volatility: Shrinkage Methods")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved OOS volatility comparison to {save_path}")


def plot_shrinkage_turnover(
    results_dict: dict, save_path: str = "reports/figures/shrinkage_turnover.png"
):
    """
    Plot turnover comparison across shrinkage methods.

    Parameters
    ----------
    results_dict : dict
        Dict with keys: method names, values: results DataFrames with 'turnover'
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, results in results_dict.items():
        turnover_series = results["turnover"].dropna()
        # Plot cumulative turnover
        cumulative = turnover_series.cumsum()
        ax.plot(cumulative.index, cumulative.values, label=method, linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Turnover")
    ax.set_title("Cumulative Portfolio Turnover: Shrinkage Methods")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved turnover comparison to {save_path}")
