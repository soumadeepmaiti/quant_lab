#!/usr/bin/env python3
"""
Run risk analysis: VaR, ES, GARCH, stress testing.

Usage:
    python scripts/run_risk.py --portfolio-value 1000000
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.config import settings, setup_logging
from quantlab.data import get_universe, load_returns, clean_prices, load_prices
from quantlab.risk import (
    var_historical, var_parametric, var_monte_carlo, var_all,
    es_historical, es_all, var_es_comparison,
    garch_fit, garch_forecast, garch_var,
    stress_run_all, compare_to_var, SCENARIOS
)


# Default portfolio weights
DEFAULT_PORTFOLIO = {
    'AAPL': 0.08, 'MSFT': 0.08, 'NVDA': 0.05, 'GOOGL': 0.04,
    'JNJ': 0.06, 'UNH': 0.05, 'PFE': 0.05, 'MRK': 0.04,
    'JPM': 0.06, 'BAC': 0.05, 'GS': 0.05, 'V': 0.04,
    'WMT': 0.05, 'PG': 0.05, 'KO': 0.05, 'MCD': 0.05,
    'XOM': 0.05, 'CVX': 0.04, 'CAT': 0.03, 'BA': 0.03,
}


def main():
    parser = argparse.ArgumentParser(description="Run risk analysis")
    parser.add_argument("--portfolio-value", type=float, 
                        default=settings.default_portfolio_value)
    parser.add_argument("--start", type=str, default=settings.default_start_date)
    parser.add_argument("--end", type=str, default=settings.default_end_date)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting risk analysis")
    
    # Load data
    tickers = list(DEFAULT_PORTFOLIO.keys())
    weights = np.array(list(DEFAULT_PORTFOLIO.values()))
    
    logger.info(f"Loading data for {len(tickers)} stocks")
    prices = load_prices(tickers, args.start, args.end)
    prices = clean_prices(prices)
    returns = prices.pct_change().dropna()
    
    # Align weights with available data
    available = [t for t in tickers if t in returns.columns]
    weights_available = np.array([DEFAULT_PORTFOLIO[t] for t in available])
    weights_available = weights_available / weights_available.sum()  # Renormalize
    
    returns = returns[available]
    portfolio_returns = (returns * weights_available).sum(axis=1)
    
    logger.info(f"Data loaded: {len(returns)} days")
    
    # VaR Analysis
    logger.info("Calculating VaR...")
    
    var_results = var_all(returns, weights=weights_available)
    
    # ES Analysis
    logger.info("Calculating Expected Shortfall...")
    
    es_results = es_all(returns, weights=weights_available)
    
    # GARCH
    logger.info("Fitting GARCH model...")
    
    garch_result, garch_params = garch_fit(portfolio_returns)
    garch_vol = garch_forecast(garch_result)['volatility'].iloc[0]
    garch_var_95 = garch_var(garch_vol, 0.95)
    garch_var_99 = garch_var(garch_vol, 0.99)
    
    # Stress Testing
    logger.info("Running stress tests...")
    
    stress_results = stress_run_all(DEFAULT_PORTFOLIO, args.portfolio_value)
    
    # Print Results
    print("\n" + "="*70)
    print("                    RISK ANALYSIS RESULTS")
    print("="*70)
    print(f"\nPortfolio Value: ${args.portfolio_value:,.0f}")
    print(f"Period: {args.start} to {args.end}")
    
    # VaR Table
    print("\n" + "-"*70)
    print("VALUE AT RISK")
    print("-"*70)
    print(f"{'Method':<15} {'95% VaR':>15} {'99% VaR':>15}")
    print("-"*45)
    
    for method in ['historical', 'parametric', 'monte_carlo']:
        v95 = var_results.loc['95%', method] * 100
        v99 = var_results.loc['99%', method] * 100
        print(f"{method.title():<15} {v95:>14.2f}% {v99:>14.2f}%")
    
    print(f"{'GARCH':<15} {garch_var_95*100:>14.2f}% {garch_var_99*100:>14.2f}%")
    
    # ES Table
    print("\n" + "-"*70)
    print("EXPECTED SHORTFALL (CVaR)")
    print("-"*70)
    print(f"{'Method':<15} {'95% ES':>15} {'99% ES':>15}")
    print("-"*45)
    
    for method in ['historical', 'parametric', 'monte_carlo']:
        e95 = es_results.loc['95%', method] * 100
        e99 = es_results.loc['99%', method] * 100
        print(f"{method.title():<15} {e95:>14.2f}% {e99:>14.2f}%")
    
    # GARCH
    print("\n" + "-"*70)
    print("GARCH(1,1) MODEL")
    print("-"*70)
    print(f"  Alpha (shock impact):      {garch_params['alpha']:.4f}")
    print(f"  Beta (persistence):        {garch_params['beta']:.4f}")
    print(f"  Persistence (α+β):         {garch_params['persistence']:.4f}")
    print(f"  Next-day Vol Forecast:     {garch_vol*100:.2f}%")
    
    # Stress Tests
    print("\n" + "-"*70)
    print("STRESS TESTS")
    print("-"*70)
    print(f"{'Scenario':<30} {'Return':>10} {'Loss ($)':>15}")
    print("-"*55)
    
    var_99_dollar = var_results.loc['99%', 'historical'] * args.portfolio_value
    
    for name, result in stress_results.items():
        ret = result.portfolio_return * 100
        loss = result.dollar_loss
        print(f"{result.scenario_name:<30} {ret:>9.1f}% ${loss:>14,.0f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    hist_var_95 = var_results.loc['95%', 'historical']
    hist_es_95 = es_results.loc['95%', 'historical']
    worst_stress = max(stress_results.values(), key=lambda x: x.dollar_loss)
    
    print(f"\n  95% 1-day VaR:        {hist_var_95*100:.2f}% (${hist_var_95*args.portfolio_value:,.0f})")
    print(f"  95% 1-day ES:         {hist_es_95*100:.2f}% (${hist_es_95*args.portfolio_value:,.0f})")
    print(f"  ES/VaR Ratio:         {hist_es_95/hist_var_95:.2f}x")
    print(f"  Worst Stress Loss:    ${worst_stress.dollar_loss:,.0f} ({worst_stress.scenario_name})")
    print(f"  Stress/VaR Ratio:     {worst_stress.dollar_loss/var_99_dollar:.1f}x")
    
    print("="*70)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        settings.ensure_dirs()
        output_path = settings.processed_data_dir / "risk_results.csv"
    
    summary = pd.DataFrame({
        'metric': ['VaR_95_hist', 'VaR_99_hist', 'ES_95_hist', 'ES_99_hist', 
                   'GARCH_vol', 'worst_stress_loss'],
        'value': [
            var_results.loc['95%', 'historical'],
            var_results.loc['99%', 'historical'],
            es_results.loc['95%', 'historical'],
            es_results.loc['99%', 'historical'],
            garch_vol,
            worst_stress.dollar_loss
        ]
    })
    
    summary.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return var_results, es_results, stress_results


if __name__ == "__main__":
    main()
