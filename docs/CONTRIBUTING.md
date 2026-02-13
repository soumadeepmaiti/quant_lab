# Contributing Guide

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/quant_lab.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Code Style

We use `black` for formatting and `ruff` for linting. Before committing:

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ --fix
```

### Testing

Write tests for all new functionality:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/quantlab --cov-report=html
```

### Type Hints

Add type hints to function signatures:

```python
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series."""
    return np.diff(np.log(prices))
```

## Commit Messages

Follow conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for test additions
- `refactor:` for code restructuring

Example: `feat: add GARCH volatility estimation`

## Pull Request Process

1. Ensure all tests pass locally
2. Update relevant documentation
3. Add entry to CHANGELOG.md
4. Create PR with clear description
5. Address any review comments

## Questions?

Open an issue or start a discussion in the repository.
