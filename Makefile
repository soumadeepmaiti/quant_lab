.PHONY: help install install-dev test lint clean backtest risk execution viz notebooks

help:
	@echo "Quant Lab - Commands"
	@echo "===================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo ""
	@echo "Analysis:"
	@echo "  make backtest      Run alpha factor backtest"
	@echo "  make risk          Run risk analysis (VaR, ES, GARCH)"
	@echo "  make execution     Run execution study (LOB, VWAP)"
	@echo "  make viz           Generate all visualizations"
	@echo "  make all           Run all analyses"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linter"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebooks     Start Jupyter Lab"

# Setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks]"

# Analysis
backtest:
	python scripts/run_backtest.py

risk:
	python scripts/run_risk.py

execution:
	python scripts/run_execution_study.py

viz:
	python scripts/generate_visualizations.py

all: backtest risk execution viz

# Development
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Notebooks
notebooks:
	jupyter lab notebooks/
