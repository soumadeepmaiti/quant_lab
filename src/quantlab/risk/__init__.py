"""Risk module - VaR, ES, GARCH, and stress testing."""

from quantlab.risk.var import (
    historical as var_historical,
    parametric as var_parametric,
    monte_carlo as var_monte_carlo,
    calculate_all as var_all,
    rolling_var,
    backtest_var
)
from quantlab.risk.es import (
    historical as es_historical,
    parametric as es_parametric,
    monte_carlo as es_monte_carlo,
    calculate_all as es_all,
    var_es_comparison
)
from quantlab.risk.garch import (
    fit as garch_fit,
    forecast as garch_forecast,
    fit_predict as garch_fit_predict,
    conditional_volatility,
    garch_var,
    compare_models as garch_compare
)
from quantlab.risk.stress import (
    run_scenario,
    run_all as stress_run_all,
    create_custom_scenario,
    sensitivity_analysis,
    compare_to_var,
    SCENARIOS,
    StressTestResult
)
from quantlab.risk.backtest import (
    kupiec_test,
    christoffersen_independence_test,
    conditional_coverage_test,
    basel_traffic_light,
    backtest_var as var_backtest_full,
    rolling_var_backtest,
    var_model_comparison,
    violation_clustering_analysis,
    BacktestResult as VaRBacktestResult
)
from quantlab.risk.evt import (
    fit_gpd,
    gpd_var,
    gpd_es,
    hill_estimator,
    evt_var_comparison,
    tail_dependence_coefficient,
    stress_correlation_analysis,
    GPDFitResult
)
from quantlab.risk.distributions import (
    fit_normal,
    fit_student_t,
    fit_empirical,
    compare_distributions,
    qq_data,
    tail_ratio,
    kurtosis_analysis,
    fat_tail_impact_on_var,
    volatility_model_comparison
)

__all__ = [
    # VaR
    "var_historical",
    "var_parametric",
    "var_monte_carlo",
    "var_all",
    "rolling_var",
    "backtest_var",
    # ES
    "es_historical",
    "es_parametric",
    "es_monte_carlo",
    "es_all",
    "var_es_comparison",
    # GARCH
    "garch_fit",
    "garch_forecast",
    "garch_fit_predict",
    "conditional_volatility",
    "garch_var",
    "garch_compare",
    # Stress
    "run_scenario",
    "stress_run_all",
    "create_custom_scenario",
    "sensitivity_analysis",
    "compare_to_var",
    "SCENARIOS",
    "StressTestResult",
    # VaR Backtesting (Research-Grade)
    "kupiec_test",
    "christoffersen_independence_test",
    "conditional_coverage_test",
    "basel_traffic_light",
    "var_backtest_full",
    "rolling_var_backtest",
    "var_model_comparison",
    "violation_clustering_analysis",
    "VaRBacktestResult",
    # EVT (Research-Grade)
    "fit_gpd",
    "gpd_var",
    "gpd_es",
    "hill_estimator",
    "evt_var_comparison",
    "tail_dependence_coefficient",
    "stress_correlation_analysis",
    "GPDFitResult",
    # Distributions (Research-Grade)
    "fit_normal",
    "fit_student_t",
    "fit_empirical",
    "compare_distributions",
    "qq_data",
    "tail_ratio",
    "kurtosis_analysis",
    "fat_tail_impact_on_var",
    "volatility_model_comparison",
]
