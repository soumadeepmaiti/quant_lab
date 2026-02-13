"""
Stress testing scenarios and engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# Pre-defined stress scenarios
SCENARIOS = {
    '1987_black_monday': {
        'name': '1987 Black Monday',
        'description': 'Market-wide crash of 20%+',
        'sector_shocks': {
            'technology': -0.22,
            'healthcare': -0.15,
            'financials': -0.25,
            'consumer': -0.15,
            'energy': -0.18,
            'industrials': -0.22
        },
        'market_shock': -0.20
    },
    '2008_financial_crisis': {
        'name': '2008 Financial Crisis',
        'description': 'Banking crisis with sector dispersion',
        'sector_shocks': {
            'technology': -0.10,
            'healthcare': -0.05,
            'financials': -0.40,
            'consumer': -0.05,
            'energy': -0.12,
            'industrials': -0.20
        },
        'market_shock': -0.15
    },
    'covid_march_2020': {
        'name': 'COVID-19 March 2020',
        'description': 'Pandemic shock with tech resilience',
        'sector_shocks': {
            'technology': -0.08,
            'healthcare': -0.10,
            'financials': -0.20,
            'consumer': -0.10,
            'energy': -0.25,
            'industrials': -0.20
        },
        'market_shock': -0.12
    },
    'interest_rate_spike': {
        'name': 'Interest Rate Spike',
        'description': '200bp rate hike scenario',
        'sector_shocks': {
            'technology': -0.15,
            'healthcare': -0.05,
            'financials': 0.05,
            'consumer': -0.08,
            'energy': 0.00,
            'industrials': -0.10
        },
        'market_shock': -0.08
    },
    'volatility_spike': {
        'name': 'Volatility Spike (VIX 80)',
        'description': 'Extreme volatility scenario',
        'sector_shocks': {
            'technology': -0.15,
            'healthcare': -0.10,
            'financials': -0.18,
            'consumer': -0.08,
            'energy': -0.15,
            'industrials': -0.15
        },
        'market_shock': -0.15
    }
}


@dataclass
class StressTestResult:
    """Container for stress test results."""
    scenario_name: str
    description: str
    portfolio_return: float
    dollar_loss: float
    contributions: Dict[str, float]
    
    def __repr__(self):
        return (
            f"StressTestResult(\n"
            f"  scenario='{self.scenario_name}',\n"
            f"  return={self.portfolio_return:.1%},\n"
            f"  loss=${self.dollar_loss:,.0f}\n"
            f")"
        )


def run_scenario(
    portfolio: Dict[str, float],
    scenario: Dict[str, Any],
    portfolio_value: float = 1_000_000,
    ticker_sectors: Optional[Dict[str, str]] = None
) -> StressTestResult:
    """
    Run a single stress test scenario.
    
    Parameters
    ----------
    portfolio : Dict[str, float]
        {ticker: weight}
    scenario : Dict[str, Any]
        Scenario definition with shocks
    portfolio_value : float
        Portfolio value in dollars
    ticker_sectors : Dict[str, str], optional
        {ticker: sector} mapping
    
    Returns
    -------
    StressTestResult
        Stress test results
    """
    # Default sector mapping
    if ticker_sectors is None:
        ticker_sectors = _default_sector_mapping()
    
    sector_shocks = scenario.get('sector_shocks', {})
    market_shock = scenario.get('market_shock', -0.10)
    
    portfolio_return = 0.0
    contributions = {}
    
    for ticker, weight in portfolio.items():
        sector = ticker_sectors.get(ticker, 'other')
        shock = sector_shocks.get(sector, market_shock)
        
        contribution = weight * shock
        portfolio_return += contribution
        contributions[ticker] = contribution
    
    dollar_loss = -portfolio_return * portfolio_value
    
    return StressTestResult(
        scenario_name=scenario.get('name', 'Unknown'),
        description=scenario.get('description', ''),
        portfolio_return=portfolio_return,
        dollar_loss=dollar_loss,
        contributions=contributions
    )


def run_all(
    portfolio: Dict[str, float],
    portfolio_value: float = 1_000_000,
    ticker_sectors: Optional[Dict[str, str]] = None,
    scenarios: Optional[Dict[str, Dict]] = None
) -> Dict[str, StressTestResult]:
    """
    Run all stress test scenarios.
    
    Parameters
    ----------
    portfolio : Dict[str, float]
        {ticker: weight}
    portfolio_value : float
        Portfolio value
    ticker_sectors : Dict[str, str], optional
        Sector mapping
    scenarios : Dict[str, Dict], optional
        Custom scenarios (defaults to SCENARIOS)
    
    Returns
    -------
    Dict[str, StressTestResult]
        Results by scenario name
    """
    if scenarios is None:
        scenarios = SCENARIOS
    
    results = {}
    
    for scenario_key, scenario in scenarios.items():
        result = run_scenario(
            portfolio, scenario, portfolio_value, ticker_sectors
        )
        results[scenario_key] = result
    
    return results


def create_custom_scenario(
    name: str,
    description: str,
    market_shock: float,
    sector_shocks: Optional[Dict[str, float]] = None,
    ticker_shocks: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create a custom stress test scenario.
    
    Parameters
    ----------
    name : str
        Scenario name
    description : str
        Scenario description
    market_shock : float
        Default market-wide shock
    sector_shocks : Dict[str, float], optional
        Sector-specific shocks
    ticker_shocks : Dict[str, float], optional
        Ticker-specific shocks (override sector)
    
    Returns
    -------
    Dict[str, Any]
        Scenario definition
    """
    return {
        'name': name,
        'description': description,
        'market_shock': market_shock,
        'sector_shocks': sector_shocks or {},
        'ticker_shocks': ticker_shocks or {}
    }


def sensitivity_analysis(
    portfolio: Dict[str, float],
    shock_range: List[float] = None,
    portfolio_value: float = 1_000_000
) -> pd.DataFrame:
    """
    Analyze portfolio sensitivity to market shocks.
    
    Parameters
    ----------
    portfolio : Dict[str, float]
        {ticker: weight}
    shock_range : List[float]
        Range of shocks to test
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    pd.DataFrame
        Loss for each shock level
    """
    if shock_range is None:
        shock_range = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    
    results = []
    
    for shock in shock_range:
        portfolio_return = sum(w * shock for w in portfolio.values())
        dollar_loss = -portfolio_return * portfolio_value
        
        results.append({
            'market_shock': shock,
            'portfolio_return': portfolio_return,
            'dollar_loss': dollar_loss
        })
    
    return pd.DataFrame(results)


def compare_to_var(
    stress_results: Dict[str, StressTestResult],
    var_99: float,
    portfolio_value: float = 1_000_000
) -> pd.DataFrame:
    """
    Compare stress test results to VaR.
    
    Parameters
    ----------
    stress_results : Dict[str, StressTestResult]
        Stress test results
    var_99 : float
        99% VaR (as decimal, e.g., 0.03 for 3%)
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    var_dollar = var_99 * portfolio_value
    
    rows = []
    for name, result in stress_results.items():
        rows.append({
            'scenario': result.scenario_name,
            'return': result.portfolio_return,
            'dollar_loss': result.dollar_loss,
            'vs_var_99': result.dollar_loss / var_dollar if var_dollar > 0 else 0
        })
    
    return pd.DataFrame(rows)


def _default_sector_mapping() -> Dict[str, str]:
    """Default ticker to sector mapping."""
    return {
        # Technology
        'AAPL': 'technology', 'MSFT': 'technology', 'NVDA': 'technology',
        'GOOGL': 'technology', 'META': 'technology', 'AVGO': 'technology',
        'ORCL': 'technology', 'CRM': 'technology', 'AMD': 'technology',
        'ADBE': 'technology', 'CSCO': 'technology', 'INTC': 'technology',
        'IBM': 'technology', 'QCOM': 'technology', 'ACN': 'technology',
        # Healthcare
        'UNH': 'healthcare', 'JNJ': 'healthcare', 'LLY': 'healthcare',
        'PFE': 'healthcare', 'MRK': 'healthcare', 'ABBV': 'healthcare',
        'TMO': 'healthcare', 'ABT': 'healthcare', 'DHR': 'healthcare',
        'BMY': 'healthcare', 'AMGN': 'healthcare', 'GILD': 'healthcare',
        # Financials
        'JPM': 'financials', 'BAC': 'financials', 'WFC': 'financials',
        'GS': 'financials', 'MS': 'financials', 'BLK': 'financials',
        'SCHW': 'financials', 'AXP': 'financials', 'C': 'financials',
        'V': 'financials', 'USB': 'financials',
        # Consumer
        'AMZN': 'consumer', 'TSLA': 'consumer', 'HD': 'consumer',
        'MCD': 'consumer', 'NKE': 'consumer', 'SBUX': 'consumer',
        'WMT': 'consumer', 'COST': 'consumer', 'TGT': 'consumer',
        'LOW': 'consumer', 'KO': 'consumer', 'PG': 'consumer',
        # Energy
        'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
        # Industrials
        'CAT': 'industrials', 'DE': 'industrials', 'BA': 'industrials',
        'HON': 'industrials', 'UPS': 'industrials', 'RTX': 'industrials',
        'LMT': 'industrials', 'GE': 'industrials',
        # Communications
        'DIS': 'consumer', 'NFLX': 'consumer', 'VZ': 'consumer',
        'T': 'consumer', 'CMCSA': 'consumer'
    }
