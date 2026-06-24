"""
Trading Bot - Core Module
Contient les principaux composants du bot de trading.
"""

from typing import Final

__version__: Final[str] = "1.0.0"
__author__: Final[str] = "Trading Team"
__all__ = [
    "RiskManager",
    "PositionSizer",
    "CircuitBreaker",
    "PortfolioRiskCalculator",
]