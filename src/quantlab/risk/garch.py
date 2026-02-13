"""
GARCH volatility modeling and forecasting.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from arch import arch_model


def fit(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    rescale: bool = True
) -> Tuple[object, Dict]:
    """
    Fit a GARCH(p,q) model.
    
    GARCH(1,1):
    σ²_t = ω + α × r²_{t-1} + β × σ²_{t-1}
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    p : int
        ARCH order (lagged squared returns)
    q : int
        GARCH order (lagged variances)
    dist : str
        Distribution: 'normal', 't', 'skewt'
    rescale : bool
        Scale returns for numerical stability
    
    Returns
    -------
    Tuple[object, Dict]
        (model_result, parameters_dict)
    """
    # Scale returns
    if rescale:
        scaled = returns * 100
    else:
        scaled = returns

    model = arch_model(
        scaled,
        vol='Garch',
        p=p,
        q=q,
        dist=dist
    )

    result = model.fit(disp='off')

    # Extract parameters
    params = result.params.to_dict()

    # Calculate persistence
    alpha = params.get('alpha[1]', 0)
    beta = params.get('beta[1]', 0)
    persistence = alpha + beta

    # Long-run volatility
    omega = params.get('omega', 0)
    if rescale:
        omega = omega / 10000  # Convert back

    long_run_var = omega / (1 - persistence) if persistence < 1 else np.nan
    long_run_vol = np.sqrt(long_run_var) * np.sqrt(252) if not np.isnan(long_run_var) else np.nan

    parameters = {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'persistence': persistence,
        'long_run_variance': long_run_var,
        'long_run_volatility': long_run_vol,
        'aic': result.aic,
        'bic': result.bic
    }

    return result, parameters


def forecast(
    model_result: object,
    horizon: int = 1,
    rescale: bool = True
) -> pd.DataFrame:
    """
    Forecast volatility from fitted GARCH model.
    
    Parameters
    ----------
    model_result : object
        Fitted GARCH model
    horizon : int
        Forecast horizon in days
    rescale : bool
        Whether returns were scaled
    
    Returns
    -------
    pd.DataFrame
        Volatility forecasts
    """
    forecast_result = model_result.forecast(horizon=horizon)

    variance = forecast_result.variance.iloc[-1]

    if rescale:
        variance = variance / 10000

    volatility = np.sqrt(variance)

    return pd.DataFrame({
        'variance': variance,
        'volatility': volatility,
        'volatility_annual': volatility * np.sqrt(252)
    })


def fit_predict(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    horizon: int = 1
) -> float:
    """
    Fit GARCH and predict next-day volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    p, q : int
        GARCH orders
    horizon : int
        Forecast horizon
    
    Returns
    -------
    float
        Forecasted daily volatility
    """
    result, _ = fit(returns, p, q)
    forecast_df = forecast(result, horizon)
    return forecast_df['volatility'].iloc[0]


def conditional_volatility(
    model_result: object,
    rescale: bool = True
) -> pd.Series:
    """
    Extract conditional volatility series from fitted model.
    
    Parameters
    ----------
    model_result : object
        Fitted GARCH model
    rescale : bool
        Whether returns were scaled
    
    Returns
    -------
    pd.Series
        Conditional volatility time series
    """
    cond_vol = model_result.conditional_volatility

    if rescale:
        cond_vol = cond_vol / 100

    return cond_vol


def garch_var(
    vol_forecast: float,
    confidence: float = 0.95,
    mean: float = 0
) -> float:
    """
    Calculate VaR using GARCH volatility forecast.
    
    Parameters
    ----------
    vol_forecast : float
        GARCH forecasted volatility
    confidence : float
        Confidence level
    mean : float
        Expected return (often 0 for short horizons)
    
    Returns
    -------
    float
        VaR as positive number
    """
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)
    return -(mean + z * vol_forecast)


def compare_models(
    returns: pd.Series,
    models: Dict[str, Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Compare different GARCH specifications.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    models : Dict[str, Tuple[int, int]]
        {model_name: (p, q)}
    
    Returns
    -------
    pd.DataFrame
        Comparison of models
    """
    if models is None:
        models = {
            'GARCH(1,1)': (1, 1),
            'GARCH(1,2)': (1, 2),
            'GARCH(2,1)': (2, 1),
            'GARCH(2,2)': (2, 2)
        }

    results = []

    for name, (p, q) in models.items():
        try:
            _, params = fit(returns, p, q)
            results.append({
                'model': name,
                'p': p,
                'q': q,
                'persistence': params['persistence'],
                'aic': params['aic'],
                'bic': params['bic']
            })
        except Exception as e:
            results.append({
                'model': name,
                'p': p,
                'q': q,
                'error': str(e)
            })

    df = pd.DataFrame(results)

    # Rank by AIC
    if 'aic' in df.columns:
        df = df.sort_values('aic')

    return df
