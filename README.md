# Quantitative Finance Research Portfolio

Multi-Factor Alpha | Risk Modeling | Market Microstructure

---

## Executive Summary

This repository presents three interconnected quantitative finance studies demonstrating production-quality research methodology in systematic trading, risk management, and market microstructure.

| Study | Objective | Key Result |
|-------|-----------|------------|
| **Factor Alpha** | Construct momentum-based long-short equity portfolios | IC = 0.021 (t=2.3), Sharpe = 0.33 |
| **Risk Engine** | Validate VaR models with fat-tail awareness | Basel GREEN zone, excess kurtosis = 4.2 |
| **Microstructure** | Estimate market impact and optimal execution | Square-root law confirmed (α = 0.48) |

**Universe:** DOW 30 constituents | **Period:** 2018–2024 (1,740 trading days) | **Tests:** 83 unit tests (100% pass rate)

---

## Project 1: Multi-Factor Alpha Research

### Motivation
Systematic factor investing requires rigorous validation beyond simple backtests. This study implements institutional-grade alpha research with proper statistical inference.

### Methodology
- **Factor:** 12-month momentum (skip last month)
- **Portfolio:** Long top 20% / Short bottom 20%, monthly rebalance
- **Validation:** Rolling IC analysis, regime decomposition, parameter sensitivity

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean IC | 0.021 | Low but significant cross-sectional predictability |
| t-statistic | 2.3 | Statistically significant (p < 0.05) |
| Information Ratio | 0.23 | Modest signal-to-noise |
| Optimal Horizon | 21 days | IC peaks at monthly forward returns |

**Regime Analysis:**
- Pre-COVID (2018-2020): Sharpe 0.65
- COVID Crash: Sharpe -1.82 (momentum crash)
- Post-COVID Rally: Sharpe 1.45
- Rate Hike 2022: Sharpe -0.73

### Key Insight
> Momentum exhibits significant but unstable predictive power. Performance varies dramatically across market regimes, emphasizing the need for regime-aware portfolio construction.

---

## Project 2: Quantitative Risk Engine

### Motivation
Standard VaR models assume normally distributed returns, systematically underestimating tail risk. This study quantifies fat-tail effects and validates risk models using regulatory standards.

### Methodology
- **VaR Methods:** Historical simulation, Parametric (Normal), Monte Carlo
- **Advanced Models:** GARCH(1,1), Student-t, EVT/GPD
- **Validation:** Kupiec unconditional coverage test

### Results

**Distribution Analysis:**
| Metric | Value | Implication |
|--------|-------|-------------|
| Excess Kurtosis | 4.21 | Heavy tails (Normal = 0) |
| Jarque-Bera | 847 (p<0.001) | Normality rejected |
| Implied t-df | 4.8 | Student-t better fit |

**VaR Comparison (95% confidence):**
| Method | VaR | Underestimation vs Empirical |
|--------|-----|------------------------------|
| Parametric (Normal) | 1.49% | -8.6% |
| Historical | 1.63% | baseline |
| EVT/GPD (99.9%) | 5.87% | +30% vs Historical |

**Backtest Result:** 14 violations in 250 days (expected: 12.5)  
**Kupiec LR:** 0.42, p-value = 0.52  
**Basel Zone:** GREEN ✓

### Key Insight
> Normal VaR underestimates 99% risk by 24% and 99.9% risk by 36%. EVT-based methods capture extreme tails more accurately—critical for capital adequacy and stress testing.

---

## Project 3: Market Microstructure & Execution

### Motivation
Large institutional orders move prices. Understanding market impact mechanics enables optimal execution strategy design and transaction cost reduction.

### Methodology
- **Impact Model:** Kyle's lambda (ΔP = λ × OFI)
- **Scaling Law:** Power law regression (ΔP = k × Q^α)
- **Execution:** VWAP vs aggressive market order comparison

### Results

**Power Law Impact:**
| Parameter | Estimate | SE | Interpretation |
|-----------|----------|-----|----------------|
| α | 0.48 | 0.03 | Square-root law (α ≈ 0.5) |
| R² | 0.72 | — | Good model fit |
| t-stat vs 0.5 | -0.67 | — | Cannot reject √Q hypothesis |

**Execution Strategy Comparison (10,000 shares):**
| Strategy | Slippage | Savings vs Aggressive |
|----------|----------|----------------------|
| Aggressive | 3.5 bps | — |
| VWAP (10 slices) | 2.6 bps | 26% |
| TWAP (10 slices) | 2.5 bps | 29% |

**Implementation Shortfall Decomposition:**
- Spread cost: 1.0 bps
- Market impact: 1.8 bps  
- Timing cost: 0.5 bps
- **Total:** 3.3 bps

### Key Insight
> Market impact follows the theoretically predicted square-root scaling. Algorithmic execution (VWAP/TWAP) reduces costs by ~25-30% compared to aggressive trading—significant at institutional scale.

---

## Technical Implementation

### Architecture
```
quantlab/
├── alpha/           # Factor construction, IC analysis, portfolio optimization
├── risk/            # VaR/ES, GARCH, EVT, backtesting
├── microstructure/  # LOB simulation, impact modeling
├── execution/       # VWAP/TWAP, implementation shortfall
├── config/          # Configuration and logging setup
├── data/            # Data loaders and preprocessing
└── utils/           # Utility functions
```

### Installation

**Prerequisites:** Python 3.10 or higher

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantlab
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Base installation
   pip install -r requirements.txt

   # Development setup (with testing tools)
   pip install -e ".[dev]"

   # Jupyter notebooks support
   pip install -e ".[notebooks]"

   # With Polygon API support for market data
   pip install -e ".[polygon]"

   # All features
   pip install -e ".[all]"
   ```

### Usage

```bash
# Factor backtest with performance metrics
python scripts/run_backtest.py

# Risk analysis (VaR, GARCH, EVT)
python scripts/run_risk.py

# Market impact and execution study
python scripts/run_execution_study.py

# Run tests
pytest tests/ -v --cov=src/quantlab
```

### Key Dependencies
- **Data & Computation:** `pandas` ≥2.0, `numpy` ≥1.24, `scipy` ≥1.10
- **Finance:** `yfinance` ≥0.2, `arch` ≥6.0
- **Statistics:** `statsmodels` ≥0.14, `scikit-learn` ≥1.3
- **Visualization:** `matplotlib` ≥3.7, `seaborn` ≥0.12, `plotly` ≥5.15
- **Data I/O:** `pyarrow` ≥14.0, `requests` ≥2.31

---

## Research Methodology

### Statistical Rigor
- All IC estimates include Newey-West standard errors (HAC)
- Bootstrap confidence intervals for Sharpe ratios
- Multiple testing awareness (avoid p-hacking)

### Validation Standards
- VaR backtested using Kupiec (1995) likelihood ratio test
- Basel regulatory traffic light zones applied
- Out-of-sample regime analysis

### Reproducibility
- 83 unit tests ensure code correctness
- Deterministic random seeds where applicable
- All data sourced from public APIs

---

## Figures

| Figure | Description |
|--------|-------------|
| Rolling IC | 12-month IC with ±2SE confidence bands |
| IC Decay | Predictability across forward return horizons |
| Sharpe Heatmap | Parameter sensitivity (lookback × quantile) |
| VaR Backtest | Violations vs Basel thresholds |
| Distribution Fit | Normal vs Student-t QQ comparison |
| EVT Analysis | GPD tail vs historical VaR |
| Market Impact | Power law regression (log-log) |
| GARCH Volatility | Conditional volatility time series |

---

## References

1. Asness, C., Moskowitz, T., & Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of Finance*
2. Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*
3. Kyle, A. (1985). Continuous Auctions and Insider Trading. *Econometrica*
4. McNeil, A., & Frey, R. (2000). Estimation of Tail-Related Risk Measures. *Journal of Empirical Finance*

---

## Contributing

Contributions are welcome! Please:
1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Run tests and ensure coverage (`pytest --cov`)
3. Format code with black and lint with ruff
4. Commit with clear messages
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details

## Contact

For questions about the research methodology or implementation, please open an issue in the repository.

*Built with Python 3.10+ | Production-grade quantitative research*
