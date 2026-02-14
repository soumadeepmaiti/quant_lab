"""Risk module - VaR, ES, GARCH, stress testing, and covariance shrinkage."""

from quantlab.risk.backtest import BacktestResult as VaRBacktestResult
from quantlab.risk.backtest import backtest_var as var_backtest_full
from quantlab.risk.backtest import (
    basel_traffic_light,
    christoffersen_independence_test,
    conditional_coverage_test,
    kupiec_test,
    rolling_var_backtest,
    var_model_comparison,
    violation_clustering_analysis,
)
from quantlab.risk.factor_risk import (
    compare_risk_contributions,
    factor_risk_decomposition,
    plot_risk_contrib_comparison,
)
from quantlab.risk.shrinkage import (
    factor_model_cov,
    ledoit_wolf_cov,
    min_var_weights,
    rolling_shrinkage_backtest,
    shrink_cov,
    shrink_to_identity,
)
from quantlab.risk.distributions import (
    compare_distributions,
    fat_tail_impact_on_var,
    fit_empirical,
    fit_normal,
    fit_student_t,
    kurtosis_analysis,
    qq_data,
    tail_ratio,
    volatility_model_comparison,
)
from quantlab.risk.es import calculate_all as es_all
from quantlab.risk.es import historical as es_historical
from quantlab.risk.es import monte_carlo as es_monte_carlo
from quantlab.risk.es import parametric as es_parametric
from quantlab.risk.es import var_es_comparison
from quantlab.risk.evt import (
    GPDFitResult,
    evt_var_comparison,
    fit_gpd,
    gpd_es,
    gpd_var,
    hill_estimator,
    stress_correlation_analysis,
    tail_dependence_coefficient,
)
from quantlab.risk.garch import compare_models as garch_compare
from quantlab.risk.garch import conditional_volatility, garch_var
from quantlab.risk.garch import fit as garch_fit
from quantlab.risk.garch import fit_predict as garch_fit_predict
from quantlab.risk.garch import forecast as garch_forecast
from quantlab.risk.stress import (
    SCENARIOS,
    StressTestResult,
    compare_to_var,
    create_custom_scenario,
    run_scenario,
    sensitivity_analysis,
)
from quantlab.risk.stress import run_all as stress_run_all
from quantlab.risk.var import backtest_var, rolling_var
from quantlab.risk.var import calculate_all as var_all
from quantlab.risk.var import historical as var_historical
from quantlab.risk.var import monte_carlo as var_monte_carlo
from quantlab.risk.var import parametric as var_parametric

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
    # Covariance Shrinkage (Institutional-Grade)
    "ledoit_wolf_cov",
    "shrink_to_identity",
    "factor_model_cov",
    "shrink_cov",
    "min_var_weights",
    "rolling_shrinkage_backtest",
    # Factor Risk Decomposition
    "factor_risk_decomposition",
    "compare_risk_contributions",
    "plot_risk_contrib_comparison",
]
