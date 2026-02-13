"""Execution module - VWAP, TWAP, POV algorithms."""

from quantlab.execution.shortfall import (
    ISDecomposition,
    arrival_price_benchmark,
    compare_execution_strategies,
    cost_curve_analysis,
    execution_quality_metrics,
    is_attribution_report,
    liquidity_regime_analysis,
    vwap_benchmark,
)
from quantlab.execution.shortfall import implementation_shortfall as is_decomposition
from quantlab.execution.strategies import (
    ExecutionResult,
    POVExecutor,
    TWAPExecutor,
    VWAPExecutor,
    compare_strategies,
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
