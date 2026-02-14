"""
Run Fama-MacBeth cross-sectional regression analysis.

This script estimates factor risk premia using two-stage cross-sectional regression
with Newey-West HAC standard errors.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.alpha.fama_macbeth import fama_macbeth, summary_table
from quantlab.alpha.diagnostics import (
    plot_premia_rolling,
    plot_premia_timeseries,
    plot_r2_histogram,
)
from quantlab.alpha.factors import momentum, volatility, rsi
from quantlab.data.universe import DOW30
from quantlab.data.loaders import load_prices


def build_factor_panel(
    prices: pd.DataFrame, monthly: bool = True
) -> pd.DataFrame:
    """
    Build factor panel with forward returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices (date × tickers)
    monthly : bool
        If True, resample to monthly

    Returns
    -------
    pd.DataFrame
        Panel with MultiIndex [date, ticker]
    """
    print("Computing factors...")
    
    # Compute returns
    returns = prices.pct_change()

    # Compute factors (lagged to avoid lookahead)
    factor_mom = momentum(prices, lookback=252, skip_recent=21)  # 12M momentum, skip 1M
    factor_vol = volatility(returns, window=60)  # 60-day volatility
    factor_rsi = rsi(prices, window=14) / 100 - 0.5  # Normalize RSI to [-0.5, 0.5]

    # Size proxy: use simple rank as placeholder (in production: use market cap)
    factor_size = prices.rank(axis=1, pct=True) - 0.5

    # Quality proxy: inverse volatility rank
    factor_quality = (1 / (factor_vol + 1e-6)).rank(axis=1, pct=True) - 0.5

    if monthly:
        # Resample to month-end
        prices_m = prices.resample("M").last()
        factor_mom_m = factor_mom.resample("M").last()
        factor_vol_m = factor_vol.resample("M").last()
        factor_rsi_m = factor_rsi.resample("M").last()
        factor_size_m = factor_size.resample("M").last()
        factor_quality_m = factor_quality.resample("M").last()

        # Forward returns (strictly future)
        ret_fwd = prices_m.pct_change(periods=1).shift(-1)

        # Stack to panel
        panel_data = {
            "ret_fwd_1m": ret_fwd.stack(),
            "factor_mom": factor_mom_m.stack(),
            "factor_vol": factor_vol_m.stack(),
            "factor_rsi": factor_rsi_m.stack(),
            "factor_size": factor_size_m.stack(),
            "factor_quality": factor_quality_m.stack(),
        }
    else:
        # Daily
        ret_fwd = prices.pct_change(periods=1).shift(-1)

        panel_data = {
            "ret_fwd_1d": ret_fwd.stack(),
            "factor_mom": factor_mom.stack(),
            "factor_vol": factor_vol.stack(),
            "factor_rsi": factor_rsi.stack(),
            "factor_size": factor_size.stack(),
            "factor_quality": factor_quality.stack(),
        }

    panel = pd.DataFrame(panel_data)
    panel.index.names = ["date", "ticker"]

    # Cross-sectional z-score each factor per date
    print("Standardizing factors...")
    for col in panel.columns:
        if col.startswith("factor_"):
            panel[col] = panel.groupby(level=0)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

    print(f"Panel shape: {panel.shape}")
    print(f"Date range: {panel.index.get_level_values(0).min()} to {panel.index.get_level_values(0).max()}")

    return panel


def main():
    parser = argparse.ArgumentParser(description="Fama-MacBeth risk premia estimation")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument(
        "--freq", type=str, default="M", choices=["M", "D"], help="Frequency"
    )
    parser.add_argument(
        "--ret-col", type=str, default="ret_fwd_1m", help="Return column name"
    )
    parser.add_argument("--min-obs", type=int, default=15, help="Min observations")
    parser.add_argument("--nw-lags", type=int, default=None, help="Newey-West lags")

    args = parser.parse_args()

    print("=" * 80)
    print("FAMA-MACBETH CROSS-SECTIONAL REGRESSION")
    print("=" * 80)
    print(f"Period: {args.start} to {args.end}")
    print(f"Frequency: {args.freq}")
    print()

    # Load data
    print("Loading prices...")
    prices = load_prices(DOW30, start=args.start, end=args.end)

    # Build panel
    monthly = args.freq == "M"
    panel = build_factor_panel(prices, monthly=monthly)

    # Factor columns
    factor_cols = [col for col in panel.columns if col.startswith("factor_")]
    print(f"Factors: {factor_cols}")
    print()

    # Run Fama-MacBeth
    print("Running Fama-MacBeth regression...")
    result = fama_macbeth(
        panel=panel,
        factor_cols=factor_cols,
        ret_col=args.ret_col,
        add_const=True,
        min_obs=args.min_obs,
        nw_lags=args.nw_lags,
        freq=args.freq,
    )

    # Summary table
    print("\n" + "=" * 80)
    print("RISK PREMIA ESTIMATES (Annualized)")
    print("=" * 80)
    
    summary = summary_table(result)
    
    # Annualize if monthly
    if args.freq == "M":
        summary["lambda_hat"] *= 12
        summary["se_hac"] *= np.sqrt(12)
    elif args.freq == "D":
        summary["lambda_hat"] *= 252
        summary["se_hac"] *= np.sqrt(252)
    
    summary["tstat_hac"] = summary["lambda_hat"] / summary["se_hac"]
    
    print(summary.to_string())
    print()

    # Save results
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "fmb_premia.csv")
    result["betas_by_date"].to_csv(output_dir / "fmb_betas_by_date.csv")

    print(f"Saved results to {output_dir}/")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_premia_rolling(result["betas_by_date"], window=12)
    plot_premia_timeseries(result["betas_by_date"], top_n=3)
    plot_r2_histogram(result["r2_by_date"])

    print("\n" + "=" * 80)
    print("FAMA-MACBETH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"Figures: {fig_dir}/")


if __name__ == "__main__":
    main()
