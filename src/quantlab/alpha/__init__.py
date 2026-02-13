"""Alpha module - factor research, backtesting, and evaluation."""

from quantlab.alpha.factors import (
    momentum,
    rsi,
    volatility,
    mean_reversion,
    beta,
    value_proxy,
    size,
    quality_proxy,
    composite_factor
)
from quantlab.alpha.ic import (
    calculate_ic,
    ic_analysis,
    ic_decay,
    quantile_returns,
    calculate_forward_returns
)
from quantlab.alpha.portfolio import (
    construct_long_short,
    construct_long_only,
    calculate_portfolio_returns,
    rebalance_weights
)
from quantlab.alpha.backtest import (
    run_long_short,
    run_long_only,
    compare_strategies,
    BacktestResult
)
from quantlab.alpha.evaluation import (
    calculate_metrics,
    calculate_drawdown_series,
    rolling_sharpe,
    calculate_alpha_beta,
    monthly_returns
)
from quantlab.alpha.neutralization import (
    cross_sectional_neutralize,
    neutralize_sector,
    orthogonalize_factors,
    compute_factor_exposures,
    factor_attribution,
    pure_factor_portfolio
)
from quantlab.alpha.regime import (
    REGIMES,
    split_by_regime,
    regime_performance,
    rolling_ic_analysis,
    ic_decay_analysis,
    parameter_sensitivity,
    create_sensitivity_heatmap,
    walk_forward_analysis,
    turnover_analysis,
    transaction_cost_impact
)
from quantlab.alpha.statistics import (
    bootstrap_confidence_interval,
    bootstrap_hypothesis_test,
    newey_west_se,
    sharpe_ratio_test,
    ic_significance_test,
    multiple_testing_correction,
    jarque_bera_test,
    factor_decay_test,
    comprehensive_factor_statistics
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
