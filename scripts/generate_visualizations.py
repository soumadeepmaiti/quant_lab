#!/usr/bin/env python
"""
Generate visualization outputs for quantitative research.

This script creates figures from analysis notebooks and saves them
to the reports/figures directory.
"""
import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_output_dir(base_dir: Path = None) -> Path:
    """Create output directory if it doesn't exist."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "reports" / "figures"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {base_dir}")
    return base_dir


def generate_sample_returns(n_periods: int = 252) -> np.ndarray:
    """Generate sample return data."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_periods)
    return returns


def plot_cumulative_returns(returns: np.ndarray, output_dir: Path) -> None:
    """Plot cumulative returns."""
    cumulative = np.cumprod(1 + returns) - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative, linewidth=2, label='Cumulative Returns')
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Returns Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'cumulative_returns.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_return_distribution(returns: np.ndarray, output_dir: Path) -> None:
    """Plot return distribution with normal overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(returns, bins=50, density=True, alpha=0.7, label='Returns')
    
    # Add normal distribution overlay
    mu, sigma = np.mean(returns), np.std(returns)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2)),
            'r-', linewidth=2, label='Normal Distribution')
    
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title('Return Distribution Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'return_distribution.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_rolling_volatility(returns: np.ndarray, output_dir: Path, window: int = 20) -> None:
    """Plot rolling volatility."""
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rolling_vol, linewidth=2, label=f'{window}-day Rolling Volatility')
    ax.axhline(rolling_vol.mean(), color='r', linestyle='--', label='Mean Volatility')
    ax.fill_between(range(len(rolling_vol)), rolling_vol, alpha=0.3)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Annualized Volatility')
    ax.set_title('Rolling Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'rolling_volatility.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_qq_plot(returns: np.ndarray, output_dir: Path) -> None:
    """Plot Q-Q plot against normal distribution."""
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 10))
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Returns vs Normal Distribution')
    
    output_path = output_dir / 'qq_plot.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate quantitative analysis visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all plots'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    
    # Generate sample data
    logger.info("Generating sample data...")
    returns = generate_sample_returns()
    
    # Generate plots
    logger.info("Generating plots...")
    if args.all or True:  # Default: generate all
        plot_cumulative_returns(returns, output_dir)
        plot_return_distribution(returns, output_dir)
        plot_rolling_volatility(returns, output_dir)
        plot_qq_plot(returns, output_dir)
    
    logger.info("Visualization generation complete!")


if __name__ == '__main__':
    main()
