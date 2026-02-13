# Installation Guide

## Requirements

- Python 3.10 or higher
- pip or conda package manager

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/soumadeepmaiti/quant_lab.git
cd quant_lab
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Package

#### Base Installation
```bash
pip install -r requirements.txt
```

#### Development Setup (with testing tools)
```bash
pip install -e ".[dev]"
```

#### With Jupyter Notebooks
```bash
pip install -e ".[notebooks]"
```

#### With Polygon API Support
```bash
pip install -e ".[polygon]"
```

#### All Features
```bash
pip install -e ".[all]"
```

## Verification

Run the test suite to verify installation:

```bash
pytest tests/ -v
```

All tests should pass with no warnings.
