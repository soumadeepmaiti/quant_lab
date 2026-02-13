"""
Logging configuration for Quant Lab.
"""

import logging
import sys
from typing import Optional

from quantlab.config.settings import settings


def setup_logging(
    level: Optional[str] = None,
    format_style: str = "simple"
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        Defaults to settings.log_level
    format_style : str
        'simple', 'detailed', or 'json'
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    level = level or settings.log_level
    
    # Format strings
    formats = {
        "simple": "%(levelname)s | %(message)s",
        "detailed": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        "json": '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    }
    
    log_format = formats.get(format_style, formats["simple"])
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger for quantlab
    logger = logging.getLogger("quantlab")
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Parameters
    ----------
    name : str
        Module name (e.g., 'quantlab.alpha.factors')
    
    Returns
    -------
    logging.Logger
        Module-specific logger
    """
    return logging.getLogger(name)
