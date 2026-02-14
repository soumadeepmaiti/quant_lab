"""
Run covariance shrinkage study with out-of-sample evaluation.

This script compares sample covariance vs shrinkage estimators for
portfolio construction and evaluates realized volatility and turnover.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.risk.shrinkage import (
    ledoit_wolf_cov,
    factor_model_cov,
    shrink_cov,
    min_var_weights,
    rolling_shrinkage_backtest,
)
from quantlab.risk.factor_risk import (
    factor_risk_decomposition,
    compare_risk_contributions,
    plot_risk_contrib_comparison,
)
from quantlab.alpha.diagnostics import plot_shrinkage_oos_vol, plot_shrinkage_turnover
from quantlab.data.universe import DOW30
from quantlab.data.loaders import load_prices


def compute_oos_metrics(weights_series: pd.Series, returns: pd.DataFrame) -> dict:
    """
    Compute out-of-sample performance metrics.

    Parameters
    ----------
    weights_series : pd.Series
        Time series of weights (index: date)
    returns : pd.DataFrame
        Returns (date × assets)

    Returns
    -------
    dict with metrics
    """
    # Forward-fill weights for daily rebalance simulation
    # In practice, weights change monthly
    
    # Compute portfolio returns (simplified: assumes monthly rebalance)
    # This is a placeholder - real implementation needs careful alignment
    
    return {
        "mean_return": 0.0,  # Placeholder
        "volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Covariance shrinkage study")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--train-window", type=int, default=252, help="Training days")

    args = parser.parse_args()

    print("=" * 80)
    print("COVARIANCE SHRINKAGE STUDY")
    print("=" * 80)
    print(f"Period: {args.start} to {args.end}")
    print(f"Training window: {args.train_window} days")
    print()

    # Load data
    print("Loading prices...")
    prices = load_prices(DOW30, start=args.start, end=args.end)
    returns = prices.pct_change().dropna()

    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print()

    # Run rolling backtests for different methods
    print("Running rolling out-of-sample backtests...")
    print("This may take a few minutes...")
    print()

    methods = ["sample", "ledoit_wolf"]
    results_dict = {}

    for method in methods:
        print(f"  Method: {method}")
        results = rolling_shrinkage_backtest(
            returns, train_window=args.train_window, method=method
        )
        results_dict[method] = results

    # Compute summary statistics
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE PERFORMANCE SUMMARY")
    print("=" * 80)

    summary_data = []
    for method, results in results_dict.items():
        # Compute annualized metrics
        port_rets = results["port_ret"].dropna()
        realized_vol = port_rets.std() * np.sqrt(252)
        sharpe = port_rets.mean() / port_rets.std() * np.sqrt(252) if port_rets.std() > 0 else 0
        avg_turnover = results["turnover"].mean()

        summary_data.append(
            {
                "Method": method,
                "Ann. Vol": f"{realized_vol:.2%}",
                "Sharpe": f"{sharpe:.2f}",
                "Avg Turnover": f"{avg_turnover:.2%}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()

    # Save results
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, results in results_dict.items():
        results.to_csv(output_dir / f"shrinkage_{method}.csv")

    summary_df.to_csv(output_dir / "shrinkage_study.csv", index=False)
    print(f"Saved results to {output_dir}/")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_shrinkage_oos_vol(results_dict)
    plot_shrinkage_turnover(results_dict)

    # Risk contribution comparison (use last training window)
    print("\nComputing risk contribution comparison...")
    train_end = len(returns) - 1
    train_start = max(0, train_end - args.train_window)
    train_returns = returns.iloc[train_start:train_end]

    sample_cov = train_returns.cov()
    shrunk_cov = ledoit_wolf_cov(train_returns)

    # Get min-var weights
    weights_sample = min_var_weights(sample_cov)
    weights_shrunk = min_var_weights(shrunk_cov)

    # Compare risk contributions
    comparison = compare_risk_contributions(weights_sample, sample_cov, shrunk_cov)
    print(f"\nPortfolio Std Dev (Sample): {comparison.attrs['port_sd_sample']:.4f}")
    print(f"Portfolio Std Dev (Shrunk): {comparison.attrs['port_sd_shrunk']:.4f}")
    print()

    plot_risk_contrib_comparison(comparison)

    print("\n" + "=" * 80)
    print("SHRINKAGE STUDY COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"Figures: {fig_dir}/")
    print()
    print("KEY INSIGHTS:")
    print("- Shrinkage reduces estimation error in covariance matrix")
    print("- Should see lower turnover and more stable weights")
    print("- Out-of-sample volatility typically more predictable")


if __name__ == "__main__":
    main()
