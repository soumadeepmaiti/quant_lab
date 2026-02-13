# Quantlab: Quantitative Finance Research Platform

[![CI/CD](https://github.com/soumadeepmaiti/quant_lab/workflows/CI/badge.svg)](https://github.com/soumadeepmaiti/quant_lab/actions) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-grade research platform** combining systematic factor investing, risk modeling, and market microstructure analysis.

## 🎯 Three Research Pillars

| **Study** | **Focus** | **Key Result** |
|-----------|-----------|--------------|
| **Alpha** | Momentum factor strategy validation | IC = 0.021 (t=2.3, p<0.05) |
| **Risk** | Fat-tail aware VaR modeling | Basel GREEN zone (Kupiec LR=0.42) |
| **Microstructure** | Market impact & optimal execution | Square-root law confirmed (α=0.48±0.03) |

**Dataset:** DOW 30 (2018–2024) | **Validation:** 83 unit tests (100% pass)

---

## 📖 Quick Start

### Installation (Python 3.10+)

```bash
git clone https://github.com/soumadeepmaiti/quant_lab.git
cd quant_lab
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For development/testing:
```bash
pip install -e ".[dev]"      # Testing & code quality tools
pip install -e ".[polygon]"  # Real intraday market data
pip install -e ".[all]"      # Everything
```

### Run Analysis

```bash
# Factor backtest with statistical inference
python scripts/run_backtest.py

# Risk model validation (VaR, GARCH, EVT)
python scripts/run_risk.py

# Market impact & execution optimization
python scripts/run_execution_study.py

# Test suite
pytest tests/ -v --cov=src/quantlab
```

---

## 📚 Research Overview

For **detailed methodology, mathematical formulation, and empirical results**, see [METHODOLOGY.md](docs/METHODOLOGY.md) (8 pages).

### 1. Multi-Factor Alpha Research
- **Strategy:** Momentum (12M lookback, skip last month) long-short portfolio
- **Validation:** Information Coefficient analysis with Newey-West HAC standard errors
- **Insight:** Significant but regime-dependent factor; performance varies across market conditions

### 2. Quantitative Risk Engine  
- **Models:** VaR (Historical, Parametric, Student-t, GARCH), EVT/GPD
- **Finding:** Student-t and EVT outperform Normal VaR by 8-10% in tail estimation
- **Application:** Basel III traffic light zone validation using Kupiec backtests

### 3. Market Microstructure & Execution
- **Theory:** Kyle's λ model, power-law market impact scaling
- **Result:** Square-root law empirically validated (α=0.48 vs theoretical 0.50)
- **Optimization:** VWAP/TWAP algorithms reduce execution costs by 25-30%

---

## 🏗️ Architecture

```
quantlab/
├── alpha/           # Factor modeling, IC analysis, backtesting
├── risk/            # VaR, GARCH, EVT, stress testing
├── microstructure/  # LOB matching engine, market impact analysis
├── execution/       # VWAP/TWAP, implementation shortfall
├── data/            # Polygon API client, data loaders
├── config/          # Settings, logging
└── utils/           # Helper functions
```

---

## 🔧 Key Features

 **Real Intraday Data:** Polygon.io API integration for tick-by-tick quotes & trades  
 **Production LOB:** Full matching engine with FIFO priority matching  
 **Statistical Rigor:** HAC robust errors, bootstrap CI, multiple testing correction  
 **Regulatory Validation:** Kupiec backtesting, Basel zones, stress scenarios  
 **Reproducible:** 83 unit tests, deterministic seeds, CI/CD on GitHub Actions  

---

## 📊 Results Summary

**Alpha Performance (2018–2024):**
- Sharpe Ratio: 0.33 | Max Drawdown: -18.3% | Win Rate: 58.3%
- Regime analysis shows momentum crashes during COVID, thrives post-crisis

**Risk Model Validation:**
| Model | 95% VaR | Violations | Basel Zone |
|-------|---------|-----------|-----------|
| Normal | 1.49% | 24 | RED ✗ |
| EVT/GPD | 1.71% | 12 | GREEN ✓ |

**Execution Efficiency:**
- Aggressive (full market order): 3.5 bps slippage
- VWAP (10 slices): 2.6 bps (**26% savings**)
- TWAP (10 slices): 2.5 bps (**29% savings**)

---

## 📦 Dependencies

Core: `pandas`, `numpy`, `scipy`, `yfinance`, `arch`, `statsmodels`, `scikit-learn`  
Viz: `matplotlib`, `seaborn`, `plotly`  
API: `polygon-api-client` (optional, for real intraday data)  
Dev: `pytest`, `black`, `ruff`, `mypy`, `jupyter`

See [`requirements.txt`](requirements.txt) for complete list.

---

## 🧪 Testing & Quality

```bash
# Run tests with coverage
pytest tests/ -v --cov=src/quantlab

# Format code
black src/ tests/ scripts/

# Lint
ruff check src/ --fix

# Type check
mypy src/quantlab --ignore-missing-imports
```

**Coverage:** 83 tests across alpha, risk, microstructure, execution modules  
**CI/CD:** GitHub Actions (Python 3.10–3.12)

---

## 📖 Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** – Detailed technical writeup (8 pages) with equations, validation results, references
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** – Development guidelines  
- **[INSTALL.md](docs/INSTALL.md)** – Detailed installation instructions
- **Notebooks:** Interactive analysis in `/notebooks/`

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

Quick workflow:
```bash
git checkout -b feature/your-feature
pip install -e ".[dev]"       # Development tools
black src/ tests/             # Format
ruff check src/ --fix         # Lint
pytest tests/ --cov           # Test
git commit -am "feat: description"
git push origin feature/your-feature
```

---

## 📄 License

MIT License – See [LICENSE](LICENSE) for details.

---

## 📧 Contact & Citation

Questions about methodology or implementation? Open an issue.

**Citation:**
```bibtex
@software{quantlab2026,
  author = {Maiti, Soumadeep},
  title = {Quantlab: Production-Grade Quantitative Finance Research Platform},
  year = {2026},
  url = {https://github.com/soumadeepmaiti/quant_lab}
}
```

---

*Built with Python 3.10+ | Production-grade quantitative research*
