"""
Risk Management Module - Package d'initialisation

Ce module encapsule tout le système de gestion des risques pour le trading bot.
Exporte les classes principales pour une utilisation simplifiée.
"""

from trading_bot.core.risk.models import (
    TradeSignal,
    Position,
    RiskState,
    RiskReport,
    RiskValidationResult,
)
from trading_bot.core.risk.position_sizing import PositionSizer
from trading_bot.core.risk.circuit_breaker import CircuitBreaker
from trading_bot.core.risk.portfolio_risk import PortfolioRiskCalculator
from trading_bot.core.risk.risk_manager import RiskManager

__all__ = [
    "TradeSignal",
    "Position",
    "RiskState",
    "RiskReport",
    "RiskValidationResult",
    "PositionSizer",
    "CircuitBreaker",
    "PortfolioRiskCalculator",
    "RiskManager",
]

__version__ = "1.0.0"