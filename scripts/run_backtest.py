#!/usr/bin/env python3
"""
Run alpha factor backtest.

Usage:
    python scripts/run_backtest.py --universe sp500 --start 2020-01-01
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.alpha import (
    calculate_forward_returns,
    composite_factor,
    ic_analysis,
    mean_reversion,
    momentum,
    rsi,
    run_long_short,
    volatility,
)
from quantlab.config import settings, setup_logging
from quantlab.data import clean_prices, get_universe, load_prices


def main():
    parser = argparse.ArgumentParser(description="Run alpha factor backtest")
    parser.add_argument("--universe", type=str, default="dow30",
                        help="Universe: dow30, sp500")
    parser.add_argument("--start", type=str, default=settings.default_start_date)
    parser.add_argument("--end", type=str, default=settings.default_end_date)
    parser.add_argument("--top-pct", type=float, default=0.2,
                        help="Top percentage for long positions")
    parser.add_argument("--rebalance", type=str, default="M",
                        help="Rebalance frequency: D, W, M")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting alpha backtest")

    # Get universe
    tickers = get_universe(args.universe)
    logger.info(f"Universe: {args.universe} ({len(tickers)} stocks)")

    # Load data
    logger.info(f"Loading data from {args.start} to {args.end}")
    prices = load_prices(tickers, args.start, args.end)
    prices = clean_prices(prices)
    returns = prices.pct_change().dropna()

    logger.info(f"Data loaded: {len(prices)} days, {len(prices.columns)} stocks")

    # Calculate factors
    logger.info("Calculating factors...")

    factors = {
        'Momentum': momentum(prices, lookback=252),
        'RSI': -rsi(prices, window=14),  # Inverted for contrarian
        'Low_Vol': -volatility(returns, window=60),
        'Mean_Rev': mean_reversion(prices, window=20)
    }

    # IC Analysis
    logger.info("Running IC analysis...")

    fwd_returns = calculate_forward_returns(prices, periods=[21])['21D']

    ic_results = {}
    for name, factor in factors.items():
        ic_stats = ic_analysis(factor, fwd_returns)
        ic_results[name] = ic_stats
        logger.info(f"  {name}: Mean IC = {ic_stats.get('mean_ic', 0):.4f}")

    # Run backtests
    logger.info("Running backtests...")

    backtest_results = {}
    for name, factor in factors.items():
        result = run_long_short(
            factor=factor,
            returns=returns,
            top_pct=args.top_pct,
            bottom_pct=args.top_pct,
            rebalance_freq=args.rebalance
        )
        backtest_results[name] = result
        logger.info(f"  {name}: Sharpe = {result.metrics.get('sharpe', 0):.2f}")

    # Composite factor
    logger.info("Building composite factor...")

    composite = composite_factor(factors, method='rank_average')
    composite_result = run_long_short(
        factor=composite,
        returns=returns,
        top_pct=args.top_pct,
        bottom_pct=args.top_pct,
        rebalance_freq=args.rebalance
    )
    backtest_results['Composite'] = composite_result

    # Summary
    print("\n" + "="*70)
    print("                    BACKTEST RESULTS")
    print("="*70)
    print(f"\nUniverse: {args.universe} | Period: {args.start} to {args.end}")
    print(f"Rebalance: {args.rebalance} | Long/Short: {args.top_pct:.0%}")

    print("\n" + "-"*70)
    print(f"{'Factor':<15} {'Sharpe':>10} {'Return':>12} {'Max DD':>10} {'Mean IC':>10}")
    print("-"*70)

    for name, result in backtest_results.items():
        sharpe = result.metrics.get('sharpe', 0)
        ret = result.metrics.get('annual_return', 0) * 100
        dd = result.metrics.get('max_drawdown', 0) * 100
        ic = ic_results.get(name, {}).get('mean_ic', 0)

        print(f"{name:<15} {sharpe:>10.2f} {ret:>11.1f}% {dd:>9.1f}% {ic:>10.4f}")

    print("="*70)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        settings.ensure_dirs()
        output_path = settings.processed_data_dir / "backtest_results.csv"

    summary_df = pd.DataFrame([
        {
            'factor': name,
            **result.metrics,
            'mean_ic': ic_results.get(name, {}).get('mean_ic', 0)
        }
        for name, result in backtest_results.items()
    ])

    summary_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    return backtest_results


if __name__ == "__main__":
    main()
