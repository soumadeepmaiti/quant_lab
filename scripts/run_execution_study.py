#!/usr/bin/env python3
"""
Run execution study: LOB simulation, market impact, VWAP analysis.

Usage:
    python scripts/run_execution_study.py --size 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.config import settings, setup_logging
from quantlab.execution import VWAPExecutor, compare_strategies
from quantlab.microstructure import (
    impact_by_size,
    initialize_book,
)


def main():
    parser = argparse.ArgumentParser(description="Run execution study")
    parser.add_argument("--size", type=int, default=10000,
                        help="Order size for analysis")
    parser.add_argument("--slices", type=int, default=10,
                        help="VWAP slices")
    parser.add_argument("--mid-price", type=float, default=150.0,
                        help="Initial mid price")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    np.random.seed(42)

    logger = setup_logging()
    logger.info("Starting execution study")

    # Book factory
    def create_book():
        return initialize_book(
            symbol='AAPL',
            mid_price=args.mid_price,
            spread=0.02,
            levels=20,
            base_size=1000
        )

    # Market Impact Analysis
    logger.info("Analyzing market impact by order size...")

    sizes = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    impact_df = impact_by_size(create_book, sizes)

    print("\n" + "="*70)
    print("                    MARKET IMPACT ANALYSIS")
    print("="*70)
    print(f"\nInitial Mid Price: ${args.mid_price:.2f}")

    print("\n" + "-"*70)
    print("IMPACT BY ORDER SIZE")
    print("-"*70)
    print(f"{'Size':>10} {'Slippage (bps)':>15} {'Impact (bps)':>15} {'VWAP':>12}")
    print("-"*52)

    for _, row in impact_df.iterrows():
        print(f"{int(row['size']):>10,} {row['slippage_bps']:>15.1f} {row['impact_bps']:>15.1f} ${row['vwap']:>11.4f}")

    # Execution Strategy Comparison
    logger.info("Comparing execution strategies...")

    comparison_df = compare_strategies(create_book, args.size, 'BUY', args.slices)

    print("\n" + "-"*70)
    print(f"EXECUTION STRATEGY COMPARISON ({args.size:,} shares)")
    print("-"*70)
    print(f"{'Strategy':<15} {'Slippage (bps)':>15} {'VWAP':>12} {'Savings (bps)':>15}")
    print("-"*57)

    for _, row in comparison_df.iterrows():
        print(f"{row['strategy']:<15} {row['slippage_bps']:>15.1f} ${row['vwap']:>11.4f} {row['savings_vs_aggressive']:>15.1f}")

    # Detailed VWAP Execution
    logger.info("Running detailed VWAP execution...")

    book = create_book()
    vwap_exec = VWAPExecutor(book)
    vwap_result = vwap_exec.execute(args.size, 'BUY', args.slices)

    print("\n" + "-"*70)
    print(f"VWAP EXECUTION DETAIL ({args.slices} slices)")
    print("-"*70)
    print(f"{'Slice':>6} {'VWAP':>12}")
    print("-"*18)

    for i, vwap in enumerate(vwap_result.slice_vwaps):
        print(f"{i+1:>6} ${vwap:>11.4f}")

    # Summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)

    aggressive = comparison_df[comparison_df['strategy'] == 'Aggressive'].iloc[0]
    vwap_row = comparison_df[comparison_df['strategy'] == 'VWAP'].iloc[0]

    savings_bps = aggressive['slippage_bps'] - vwap_row['slippage_bps']
    savings_dollar = savings_bps / 10000 * args.size * args.mid_price

    print(f"\n  Order Size:           {args.size:,} shares")
    print(f"  Initial Mid:          ${args.mid_price:.2f}")
    print(f"  Aggressive VWAP:      ${aggressive['vwap']:.4f}")
    print(f"  VWAP ({args.slices} slices):     ${vwap_row['vwap']:.4f}")
    print(f"\n  Aggressive Slippage:  {aggressive['slippage_bps']:.1f} bps")
    print(f"  VWAP Slippage:        {vwap_row['slippage_bps']:.1f} bps")
    print(f"  Savings:              {savings_bps:.1f} bps (${savings_dollar:,.2f})")
    print(f"  Cost Reduction:       {savings_bps/aggressive['slippage_bps']*100:.1f}%")

    print("="*70)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        settings.ensure_dirs()
        output_path = settings.processed_data_dir / "execution_results.csv"

    comparison_df.to_csv(output_path, index=False)

    impact_output = output_path.parent / "impact_analysis.csv"
    impact_df.to_csv(impact_output, index=False)

    logger.info(f"Results saved to {output_path}")

    return comparison_df, impact_df


if __name__ == "__main__":
    main()
