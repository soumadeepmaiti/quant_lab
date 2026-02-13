````markdown
# Quantlab: Quantitative Finance Research Platform

[![CI/CD](https://github.com/soumadeepmaiti/quant_lab/workflows/CI/badge.svg)](https://github.com/soumadeepmaiti/quant_lab/actions) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-grade research platform** combining systematic factor investing, risk modeling, and market microstructure analysis.

## 🎯 Three Research Pillars

| **Study** | **Focus** | **Key Result** |
|-----------|-----------|--------------|
| **Alpha** | Momentum factor strategy validation | Representative IC analysis (see reports) |
| **Risk** | Fat-tail aware VaR modeling | EVT/GPD improves tail estimation versus Normal |
| **Microstructure** | Market impact & optimal execution | Empirical impact scaling observed in studies |

**Dataset:** DOW 30 (2018–2024) | **Validation:** Core correctness test suite with CI (expanding)

---

## 📖 Quick Start

### Installation (Python 3.11+)

```bash
git clone https://github.com/soumadeepmaiti/quant_lab.git
cd quant_lab
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

For optional realtime intraday data support:
```bash
pip install -e ".[polygon]"
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
pytest
```

---

## 📚 Research Overview

For **detailed methodology, mathematical formulation, and empirical results**, see [METHODOLOGY.md](docs/METHODOLOGY.md) (8 pages).

### 1. Multi-Factor Alpha Research
- **Strategy:** Momentum (12M lookback, skip last month) long-short portfolio
- **Validation:** Information Coefficient analysis with Newey-West HAC standard errors

### 2. Quantitative Risk Engine  
- **Models:** VaR (Historical, Parametric, Student-t, GARCH), EVT/GPD

### 3. Market Microstructure & Execution
- **Theory:** Kyle's λ model, power-law market impact scaling

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
 **Production LOB:** Full matching engine with FIFO priority matching (simplified in current implementation)  
 **Statistical Rigor:** HAC robust errors, bootstrap CI, multiple testing correction  
 **Regulatory Validation:** Kupiec backtesting, Basel zones, stress scenarios  
 **Reproducible:** Core correctness suite (run `pytest`), deterministic seeds, CI on GitHub Actions

---

## 📊 Results Summary

Representative performance and risk summaries are produced by the scripts in `scripts/` and saved to `reports/` and `data/processed/` when running the pipeline. Exact numbers in the README were removed to ensure all reported metrics are reproducible from the code and data processing pipeline.

---

## 📦 Dependencies

Core and dev dependencies are defined in `pyproject.toml` and can be installed with the development extras shown above (`pip install -e ".[dev]"`).

---

## 🧪 Testing & Quality

```bash
# Run tests
pytest

# Format code
black src/ tests/ scripts/

# Lint
ruff check src/ --fix

# Type check
mypy src/quantlab --ignore-missing-imports
```

---

**Reproducibility**:

1. Install: `pip install -e "[dev]"`
2. Run alpha: `python scripts/run_backtest.py`
3. Run risk: `python scripts/run_risk.py`
4. Run execution: `python scripts/run_execution_study.py`
5. Run tests: `pytest`

Notes on research hygiene: factors are lagged; forward returns use strictly future data; results are reported net of specified transaction cost assumptions.

**Limitations:** Microstructure analysis currently uses a simplified/synthetic LOB model and limited tick-level sampling; full-production trade/quote pipelines (e.g., Polygon) are optional and require API credentials. Use results as representative analyses rather than production signals.

---

## 📖 Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** – Detailed technical writeup (8 pages) with equations, validation results, references
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** – Development guidelines  
- **[INSTALL.md](docs/INSTALL.md)** – Detailed installation instructions

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License – See [LICENSE](LICENSE) for details.

---

*Built with Python 3.11+ | Reproducible research-oriented quantitative toolkit*

````
