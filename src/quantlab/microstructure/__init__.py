"""Microstructure module - LOB, signals, and impact analysis."""

from quantlab.microstructure.lob import (
    LimitOrderBook,
    Order,
    Trade,
    initialize_book
)
from quantlab.microstructure.signals import (
    order_flow_imbalance,
    calculate_ofi_from_book,
    relative_spread,
    depth_imbalance,
    vpin,
    kyle_lambda,
    trade_flow_toxicity
)
from quantlab.microstructure.impact import (
    analyze_market_order,
    square_root_impact,
    linear_impact,
    almgren_chriss_impact,
    implementation_shortfall,
    impact_by_size
)
from quantlab.microstructure.regression import (
    estimate_kyle_lambda,
    estimate_power_law_impact,
    permanent_transitory_decomposition,
    adverse_selection_measure,
    toxicity_index,
    rolling_impact_estimation,
    ImpactRegressionResult
)

__all__ = [
    # LOB
    "LimitOrderBook",
    "Order",
    "Trade",
    "initialize_book",
    # Signals
    "order_flow_imbalance",
    "calculate_ofi_from_book",
    "relative_spread",
    "depth_imbalance",
    "vpin",
    "kyle_lambda",
    "trade_flow_toxicity",
    # Impact
    "analyze_market_order",
    "square_root_impact",
    "linear_impact",
    "almgren_chriss_impact",
    "implementation_shortfall",
    "impact_by_size",
    # Regression (Research-Grade)
    "estimate_kyle_lambda",
    "estimate_power_law_impact",
    "permanent_transitory_decomposition",
    "adverse_selection_measure",
    "toxicity_index",
    "rolling_impact_estimation",
    "ImpactRegressionResult",
]
