"""Execution module - VWAP, TWAP, POV algorithms."""

from quantlab.execution.strategies import (
    VWAPExecutor,
    TWAPExecutor,
    POVExecutor,
    ExecutionResult,
    compare_strategies
)
from quantlab.execution.shortfall import (
    implementation_shortfall as is_decomposition,
    vwap_benchmark,
    arrival_price_benchmark,
    execution_quality_metrics,
    cost_curve_analysis,
    compare_execution_strategies,
    is_attribution_report,
    liquidity_regime_analysis,
    ISDecomposition
)

__all__ = [
    # Strategies
    "VWAPExecutor",
    "TWAPExecutor",
    "POVExecutor",
    "ExecutionResult",
    "compare_strategies",
    # Implementation Shortfall (Research-Grade)
    "is_decomposition",
    "vwap_benchmark",
    "arrival_price_benchmark",
    "execution_quality_metrics",
    "cost_curve_analysis",
    "compare_execution_strategies",
    "is_attribution_report",
    "liquidity_regime_analysis",
    "ISDecomposition",
]
