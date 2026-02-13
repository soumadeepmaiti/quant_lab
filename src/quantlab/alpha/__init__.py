"""Alpha module - factor research, backtesting, and evaluation."""

from quantlab.alpha.backtest import (
    BacktestResult,
    compare_strategies,
    run_long_only,
    run_long_short,
)
from quantlab.alpha.evaluation import (
    calculate_alpha_beta,
    calculate_drawdown_series,
    calculate_metrics,
    monthly_returns,
    rolling_sharpe,
)
from quantlab.alpha.factors import (
    beta,
    composite_factor,
    mean_reversion,
    momentum,
    quality_proxy,
    rsi,
    size,
    value_proxy,
    volatility,
)
from quantlab.alpha.ic import (
    calculate_forward_returns,
    calculate_ic,
    ic_analysis,
    ic_decay,
    quantile_returns,
)
from quantlab.alpha.neutralization import (
    compute_factor_exposures,
    cross_sectional_neutralize,
    factor_attribution,
    neutralize_sector,
    orthogonalize_factors,
    pure_factor_portfolio,
)
from quantlab.alpha.portfolio import (
    calculate_portfolio_returns,
    construct_long_only,
    construct_long_short,
    rebalance_weights,
)
from quantlab.alpha.regime import (
    REGIMES,
    create_sensitivity_heatmap,
    ic_decay_analysis,
    parameter_sensitivity,
    regime_performance,
    rolling_ic_analysis,
    split_by_regime,
    transaction_cost_impact,
    turnover_analysis,
    walk_forward_analysis,
)
from quantlab.alpha.statistics import (
    bootstrap_confidence_interval,
    bootstrap_hypothesis_test,
    comprehensive_factor_statistics,
    factor_decay_test,
    ic_significance_test,
    jarque_bera_test,
    multiple_testing_correction,
    newey_west_se,
    sharpe_ratio_test,
)

__all__ = [
    # Factors
    "momentum",
    "rsi",
    "volatility",
    "mean_reversion",
    "beta",
    "value_proxy",
    "size",
    "quality_proxy",
    "composite_factor",
    # IC Analysis
    "calculate_ic",
    "ic_analysis",
    "ic_decay",
    "quantile_returns",
    "calculate_forward_returns",
    # Portfolio
    "construct_long_short",
    "construct_long_only",
    "calculate_portfolio_returns",
    "rebalance_weights",
    # Backtest
    "run_long_short",
    "run_long_only",
    "compare_strategies",
    "BacktestResult",
    # Evaluation
    "calculate_metrics",
    "calculate_drawdown_series",
    "rolling_sharpe",
    "calculate_alpha_beta",
    "monthly_returns",
    # Neutralization (Research-Grade)
    "cross_sectional_neutralize",
    "neutralize_sector",
    "orthogonalize_factors",
    "compute_factor_exposures",
    "factor_attribution",
    "pure_factor_portfolio",
    # Regime Analysis (Research-Grade)
    "REGIMES",
    "split_by_regime",
    "regime_performance",
    "rolling_ic_analysis",
    "ic_decay_analysis",
    "parameter_sensitivity",
    "create_sensitivity_heatmap",
    "walk_forward_analysis",
    "turnover_analysis",
    "transaction_cost_impact",
    # Statistics (Research-Grade)
    "bootstrap_confidence_interval",
    "bootstrap_hypothesis_test",
    "newey_west_se",
    "sharpe_ratio_test",
    "ic_significance_test",
    "multiple_testing_correction",
    "jarque_bera_test",
    "factor_decay_test",
    "comprehensive_factor_statistics",
]
