"""
Distribution Comparison Module.

Implements research-grade distribution analysis:
- Fit and compare Normal, Student-t, and Empirical distributions
- Goodness-of-fit testing (KS, Anderson-Darling)
- QQ plots and probability plots
- Fat-tail impact quantification

References:
- Cont (2001) - "Empirical Properties of Asset Returns"
- Bollerslev (1987) - "A Conditionally Heteroskedastic Time Series Model"
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from quantlab.config import get_logger

logger = get_logger(__name__)


@dataclass
class DistributionFit:
    """Container for distribution fit results."""
    name: str
    params: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float


def fit_normal(
    data: np.ndarray
) -> DistributionFit:
    """
    Fit normal distribution via MLE.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    
    Returns
    -------
    DistributionFit
        Fit results
    """
    n = len(data)

    # MLE estimates
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)  # MLE variance

    # Log-likelihood
    ll = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

    # Information criteria
    k = 2  # Number of parameters
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # KS test
    ks_stat, ks_pvalue = stats.kstest(data, 'norm', args=(mu, sigma))

    # Risk measures
    var_95 = -stats.norm.ppf(0.05, loc=mu, scale=sigma)
    var_99 = -stats.norm.ppf(0.01, loc=mu, scale=sigma)

    # ES for normal: ES = μ + σ * φ(z_α) / (1 - α)
    z_95 = stats.norm.ppf(0.05)
    z_99 = stats.norm.ppf(0.01)
    es_95 = -(mu + sigma * stats.norm.pdf(z_95) / 0.05)
    es_99 = -(mu + sigma * stats.norm.pdf(z_99) / 0.01)

    return DistributionFit(
        name='Normal',
        params={'mu': mu, 'sigma': sigma},
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def fit_student_t(
    data: np.ndarray,
    df_bounds: Tuple[float, float] = (2.1, 30.0)
) -> DistributionFit:
    """
    Fit Student-t distribution via MLE.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    df_bounds : tuple
        Bounds for degrees of freedom
    
    Returns
    -------
    DistributionFit
        Fit results
    """
    n = len(data)

    # Initial estimates
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    df_init = 5.0

    def neg_log_likelihood(params):
        df, loc, scale = params
        if df <= 2 or scale <= 0:
            return 1e10
        return -np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))

    # Optimize
    try:
        result = minimize(
            neg_log_likelihood,
            x0=[df_init, mu_init, sigma_init],
            method='L-BFGS-B',
            bounds=[(df_bounds[0], df_bounds[1]), (None, None), (1e-6, None)]
        )
        df, mu, sigma = result.x
    except:
        df, mu, sigma = df_init, mu_init, sigma_init

    # Log-likelihood
    ll = -neg_log_likelihood([df, mu, sigma])

    # Information criteria
    k = 3  # Number of parameters
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # KS test
    ks_stat, ks_pvalue = stats.kstest(data, 't', args=(df, mu, sigma))

    # Risk measures
    var_95 = -stats.t.ppf(0.05, df=df, loc=mu, scale=sigma)
    var_99 = -stats.t.ppf(0.01, df=df, loc=mu, scale=sigma)

    # ES for Student-t
    def t_es(alpha, df, loc, scale):
        q = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
        # ES = -loc - scale * (df + q^2) / (df - 1) * t_pdf(q) / alpha
        pdf_q = stats.t.pdf(q, df=df, loc=0, scale=1)
        return -(loc + scale * (df + (q - loc/scale)**2) / (df - 1) * pdf_q / alpha)

    es_95 = t_es(0.05, df, mu, sigma)
    es_99 = t_es(0.01, df, mu, sigma)

    return DistributionFit(
        name='Student-t',
        params={'df': df, 'mu': mu, 'sigma': sigma},
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def fit_empirical(
    data: np.ndarray
) -> DistributionFit:
    """
    Non-parametric empirical distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    
    Returns
    -------
    DistributionFit
        Fit results (using historical simulation)
    """
    n = len(data)

    # VaR from empirical quantiles
    var_95 = -np.percentile(data, 5)
    var_99 = -np.percentile(data, 1)

    # ES: average of returns worse than VaR
    es_95 = -np.mean(data[data <= np.percentile(data, 5)])
    es_99 = -np.mean(data[data <= np.percentile(data, 1)])

    return DistributionFit(
        name='Empirical',
        params={},
        log_likelihood=np.nan,
        aic=np.nan,
        bic=np.nan,
        ks_stat=np.nan,
        ks_pvalue=np.nan,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def compare_distributions(
    data: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    """
    Compare Normal, Student-t, and Empirical distributions.
    
    Parameters
    ----------
    data : array-like
        Return data
    
    Returns
    -------
    pd.DataFrame
        Comparison of distribution fits
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    results = []

    # Fit all distributions
    normal_fit = fit_normal(data)
    t_fit = fit_student_t(data)
    emp_fit = fit_empirical(data)

    for fit in [normal_fit, t_fit, emp_fit]:
        results.append({
            'distribution': fit.name,
            'log_likelihood': fit.log_likelihood,
            'aic': fit.aic,
            'bic': fit.bic,
            'ks_statistic': fit.ks_stat,
            'ks_pvalue': fit.ks_pvalue,
            'var_95': fit.var_95,
            'var_99': fit.var_99,
            'es_95': fit.es_95,
            'es_99': fit.es_99
        })

    df = pd.DataFrame(results)

    # Add fat-tail impact
    normal_var_99 = df.loc[df['distribution'] == 'Normal', 'var_99'].values[0]
    df['var_99_vs_normal'] = df['var_99'] / normal_var_99 - 1
    df['es_99_vs_normal'] = df['es_99'] / df.loc[df['distribution'] == 'Normal', 'es_99'].values[0] - 1

    return df


def qq_data(
    data: np.ndarray,
    distribution: str = 'normal'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate QQ plot data.
    
    Parameters
    ----------
    data : np.ndarray
        Sample data
    distribution : str
        'normal' or 't' (with df=5)
    
    Returns
    -------
    tuple
        (theoretical_quantiles, sample_quantiles)
    """
    data = np.sort(data)
    n = len(data)

    # Theoretical quantiles
    p = (np.arange(1, n + 1) - 0.5) / n

    if distribution == 'normal':
        theoretical = stats.norm.ppf(p)
    elif distribution == 't':
        theoretical = stats.t.ppf(p, df=5)
    else:
        theoretical = stats.norm.ppf(p)

    return theoretical, data


def tail_ratio(
    data: np.ndarray,
    quantiles: List[float] = [0.01, 0.05, 0.10]
) -> pd.DataFrame:
    """
    Compare empirical tail quantiles to normal distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    quantiles : list
        Quantiles to compare
    
    Returns
    -------
    pd.DataFrame
        Tail comparison
    """
    data = np.asarray(data)
    mu = np.mean(data)
    sigma = np.std(data)

    results = []

    for q in quantiles:
        emp_q = np.percentile(data, q * 100)
        norm_q = stats.norm.ppf(q, loc=mu, scale=sigma)

        results.append({
            'quantile': f'{q*100:.1f}%',
            'empirical': emp_q,
            'normal': norm_q,
            'ratio': emp_q / norm_q if norm_q != 0 else np.nan,
            'excess_loss': emp_q - norm_q
        })

    return pd.DataFrame(results)


def kurtosis_analysis(
    data: np.ndarray
) -> Dict[str, float]:
    """
    Analyze excess kurtosis and its implications.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    
    Returns
    -------
    dict
        Kurtosis statistics
    """
    # Sample statistics
    n = len(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # Excess kurtosis

    # Standard error for kurtosis
    se_kurt = np.sqrt(24 / n)
    t_stat = kurt / se_kurt

    # Jarque-Bera test
    jb_stat = n / 6 * (skew ** 2 + kurt ** 2 / 4)
    jb_pvalue = 1 - stats.chi2.cdf(jb_stat, df=2)

    # Implied df for Student-t (if excess kurtosis > 0)
    # For t-distribution: excess_kurtosis = 6/(df-4) for df > 4
    if kurt > 0:
        implied_df = 4 + 6 / kurt
    else:
        implied_df = np.inf

    return {
        'skewness': skew,
        'excess_kurtosis': kurt,
        'kurtosis_se': se_kurt,
        'kurtosis_t_stat': t_stat,
        'kurtosis_significant': abs(t_stat) > 1.96,
        'jarque_bera': jb_stat,
        'jb_pvalue': jb_pvalue,
        'is_normal': jb_pvalue > 0.05,
        'implied_t_df': implied_df
    }


def fat_tail_impact_on_var(
    data: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99, 0.995, 0.999]
) -> pd.DataFrame:
    """
    Quantify how fat tails impact VaR and ES estimates.
    
    Shows underestimation of risk when assuming normality.
    
    Parameters
    ----------
    data : np.ndarray
        Return data
    confidence_levels : list
        Confidence levels
    
    Returns
    -------
    pd.DataFrame
        Impact analysis
    """
    data = np.asarray(data)
    mu = np.mean(data)
    sigma = np.std(data)

    # Fit Student-t
    t_fit = fit_student_t(data)
    df = t_fit.params['df']

    results = []

    for conf in confidence_levels:
        alpha = 1 - conf

        # Normal VaR
        norm_var = -stats.norm.ppf(alpha, loc=mu, scale=sigma)

        # Student-t VaR
        t_var = -stats.t.ppf(alpha, df=df, loc=t_fit.params['mu'], scale=t_fit.params['sigma'])

        # Empirical VaR
        emp_var = -np.percentile(data, alpha * 100)

        # Underestimation by normal
        normal_underest_pct = (emp_var - norm_var) / emp_var * 100 if emp_var > 0 else 0

        results.append({
            'confidence': conf,
            'normal_var': norm_var,
            'student_t_var': t_var,
            'empirical_var': emp_var,
            'normal_underestimation_%': normal_underest_pct,
            't_vs_normal_%': (t_var - norm_var) / norm_var * 100 if norm_var > 0 else 0
        })

    return pd.DataFrame(results)


def volatility_model_comparison(
    returns: pd.Series,
    window: int = 60
) -> pd.DataFrame:
    """
    Compare volatility estimation methods.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window
    
    Returns
    -------
    pd.DataFrame
        Volatility estimates over time
    """
    results = []

    # Constant variance (full sample)
    const_vol = returns.std() * np.sqrt(252)

    # Rolling variance
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

    # EWMA variance (lambda = 0.94)
    lambda_ewma = 0.94
    ewma_var = returns.iloc[:window].var()
    ewma_vol = [np.sqrt(ewma_var) * np.sqrt(252)]

    for i in range(window, len(returns)):
        ewma_var = lambda_ewma * ewma_var + (1 - lambda_ewma) * returns.iloc[i-1] ** 2
        ewma_vol.append(np.sqrt(ewma_var) * np.sqrt(252))

    ewma_series = pd.Series(
        [np.nan] * (window - 1) + ewma_vol,
        index=returns.index
    )

    return pd.DataFrame({
        'date': returns.index,
        'constant': const_vol,
        'rolling': rolling_vol.values,
        'ewma': ewma_series.values
    }).set_index('date')
