"""
Position Sizing - Calcul des tailles de position

Implémente :
- Kelly Criterion (full + fractional)
- ATR-based sizing
- Gestion des contraintes de risque
"""

import logging
import json
from decimal import Decimal
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KellyCriterionResult(BaseModel):
    """Résultat du calcul Kelly Criterion."""
    position_size_pct: float = Field(..., description="Taille de position en % du capital")
    kelly_fraction: float = Field(default=0.25, description="Fraction de Kelly utilisée")
    win_rate: float = Field(..., description="Taux de victoire")
    avg_win: float = Field(..., description="Gain moyen")
    avg_loss: float = Field(..., description="Perte moyenne")
    confidence: float = Field(..., description="Confiance du calcul")


class ATRSizingResult(BaseModel):
    """Résultat du sizing basé sur ATR."""
    position_size: Decimal = Field(..., description="Taille de position")
    atr_value: Decimal = Field(..., description="Valeur ATR")
    stop_loss_price: Decimal = Field(..., description="Stop loss basé ATR")
    risk_amount: Decimal = Field(..., description="Montant à risquer")


class PositionSizer:
    """
    Gestionnaire de dimensionnement des positions.

    Implements Kelly Criterion, ATR-based sizing, et respecte les contraintes de risque.
    """

    def __init__(
        self,
        initial_capital: Decimal,
        kelly_fraction: float = 0.25,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_risk_per_trade_pct: float = 2.0,
        max_position_size_pct: float = 5.0,
        max_concurrent_positions: int = 6,
        min_position_size_pct: float = 0.1,
        max_leverage: float = 3.0,
    ) -> None:
        """
        Initialiser le Position Sizer.

        Parameters
        ----------
        initial_capital : Decimal
            Capital initial
        kelly_fraction : float
            Fraction de Kelly à utiliser (0.25 = 25% Kelly)
        atr_period : int
            Période pour le calcul ATR
        atr_multiplier : float
            Multiplicateur ATR pour le stop loss
        max_risk_per_trade_pct : float
            Max risque par trade en % du capital
        max_position_size_pct : float
            Max taille position en % du capital
        max_concurrent_positions : int
            Max positions simultanées
        min_position_size_pct : float
            Min taille position en % du capital
        max_leverage : float
            Leverage maximum autorisé
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_size_pct = max_position_size_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.min_position_size_pct = min_position_size_pct
        self.max_leverage = max_leverage

        self.trade_history: List[Dict] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._log_initialization()

    def _log_initialization(self) -> None:
        """Logger l'initialisation."""
        config = {
            "initial_capital": str(self.initial_capital),
            "kelly_fraction": self.kelly_fraction,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "max_position_size_pct": self.max_position_size_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
        }
        self.logger.info(json.dumps({
            "event": "position_sizer_initialized",
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0,
        min_sample_size: int = 20,
    ) -> Optional[KellyCriterionResult]:
        """
        Calculer la fraction de Kelly pour le sizing.

        Parameters
        ----------
        win_rate : float
            Taux de victoire (0-1)
        avg_win : float
            Gain moyen
        avg_loss : float
            Perte moyenne (positive)
        confidence : float
            Confiance du calcul (0-1)
        min_sample_size : int
            Nombre minimal de trades pour valider

        Returns
        -------
        Optional[KellyCriterionResult]
            Résultat du calcul Kelly ou None si invalide

        Notes
        -----
        Kelly Criterion : f = (bp - q) / b
        où : f = fraction à risquer
             b = ratio gain/perte
             p = probabilité de gain
             q = probabilité de perte = 1 - p
        """
        if not (0 < win_rate < 1):
            self.logger.warning(json.dumps({
                "event": "invalid_kelly_parameters",
                "reason": "invalid_win_rate",
                "win_rate": win_rate
            }))
            return None

        if avg_win <= 0 or avg_loss <= 0:
            self.logger.warning(json.dumps({
                "event": "invalid_kelly_parameters",
                "reason": "invalid_win_loss_amounts"
            }))
            return None

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, kelly_fraction * confidence)
        final_fraction = kelly_fraction * self.kelly_fraction
        final_fraction = max(0, min(final_fraction, self.max_position_size_pct / 100))

        result = KellyCriterionResult(
            position_size_pct=float(final_fraction * 100),
            kelly_fraction=self.kelly_fraction,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            confidence=confidence,
        )

        self.logger.info(json.dumps({
            "event": "kelly_criterion_calculated",
            "kelly_fraction": kelly_fraction,
            "final_fraction": final_fraction,
            "position_size_pct": result.position_size_pct,
            "confidence": confidence
        }))

        return result

    def calculate_atr_sizing(
        self,
        current_price: Decimal,
        atr_value: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
    ) -> ATRSizingResult:
        """
        Calculer la taille de position basée sur ATR.

        Parameters
        ----------
        current_price : Decimal
            Prix courant du marché
        atr_value : Decimal
            Valeur ATR (Average True Range)
        entry_price : Decimal
            Prix d'entrée proposé
        stop_loss_price : Decimal
            Prix de stop loss proposé

        Returns
        -------
        ATRSizingResult
            Résultat du sizing ATR

        Notes
        -----
        Position Size = (Capital * Risk %) / (Entry Price - Stop Loss)
        Stop Loss est ajusté à : Entry Price - (ATR * Multiplier)
        """
        adjusted_stop_loss = entry_price - (atr_value * Decimal(str(self.atr_multiplier)))
        final_stop_loss = max(adjusted_stop_loss, stop_loss_price)
        max_risk_amount = self.current_capital * Decimal(str(self.max_risk_per_trade_pct / 100))
        risk_distance = entry_price - final_stop_loss

        if risk_distance <= 0:
            self.logger.error(json.dumps({
                "event": "invalid_atr_sizing",
                "reason": "risk_distance_non_positive",
                "entry_price": str(entry_price),
                "stop_loss": str(final_stop_loss)
            }))
            raise ValueError("Risk distance must be positive")

        position_size = max_risk_amount / risk_distance

        result = ATRSizingResult(
            position_size=position_size.quantize(Decimal("0.00000001")),
            atr_value=atr_value,
            stop_loss_price=final_stop_loss,
            risk_amount=max_risk_amount,
        )

        self.logger.info(json.dumps({
            "event": "atr_sizing_calculated",
            "position_size": str(result.position_size),
            "atr_value": str(atr_value),
            "stop_loss": str(final_stop_loss)
        }))

        return result

    def calculate_constrained_size(
        self,
        proposed_quantity: Decimal,
        entry_price: Decimal,
        current_positions: int,
        sector_exposure_pct: float = 0.0,
    ) -> Tuple[Decimal, List[str]]:
        """
        Calculer la taille de position finale avec toutes les contraintes.

        Parameters
        ----------
        proposed_quantity : Decimal
            Quantité proposée
        entry_price : Decimal
            Prix d'entrée
        current_positions : int
            Nombre de positions actuelles
        sector_exposure_pct : float
            Exposition sectorielle actuelle en %

        Returns
        -------
        Tuple[Decimal, List[str]]
            (Taille approuvée, Liste des réductions appliquées)

        Notes
        -----
        Applique les contraintes dans cet ordre :
        1. Limite de positions concurrentes
        2. Limite d'exposition sectorielle
        3. Limite de taille de position
        4. Limite de risque par trade
        5. Taille minimale de position
        """
        approved_quantity = proposed_quantity
        constraints_applied = []

        if current_positions >= self.max_concurrent_positions:
            approved_quantity = Decimal("0")
            constraints_applied.append(
                f"Maximum concurrent positions ({self.max_concurrent_positions}) reached"
            )
            self.logger.warning(json.dumps({
                "event": "position_limit_reached",
                "current_positions": current_positions,
                "max_positions": self.max_concurrent_positions
            }))
            return approved_quantity, constraints_applied

        if sector_exposure_pct > 25.0:
            max_exposure_addition = Decimal(str((25.0 - sector_exposure_pct) / 100))
            position_value_addition = approved_quantity * entry_price
            additional_exposure = (position_value_addition / self.current_capital) * 100

            if additional_exposure + sector_exposure_pct > 30.0:
                reduction_factor = (30.0 - sector_exposure_pct) / additional_exposure
                approved_quantity = (approved_quantity * Decimal(str(reduction_factor))).quantize(
                    Decimal("0.00000001")
                )
                constraints_applied.append("Sector exposure limit applied")

        position_value = approved_quantity * entry_price
        max_position_value = self.current_capital * Decimal(str(self.max_position_size_pct / 100))

        if position_value > max_position_value:
            approved_quantity = (max_position_value / entry_price).quantize(Decimal("0.00000001"))
            constraints_applied.append(
                f"Position size limited to {self.max_position_size_pct}% of capital"
            )

        max_risk_amount = self.current_capital * Decimal(str(self.max_risk_per_trade_pct / 100))
        if position_value > max_risk_amount:
            approved_quantity = (max_risk_amount / entry_price).quantize(Decimal("0.00000001"))
            constraints_applied.append(
                f"Risk per trade limited to {self.max_risk_per_trade_pct}% of capital"
            )

        min_position_value = self.current_capital * Decimal(str(self.min_position_size_pct / 100))
        final_position_value = approved_quantity * entry_price

        if final_position_value < min_position_value and approved_quantity > 0:
            approved_quantity = Decimal("0")
            constraints_applied.append(
                f"Position below minimum size ({self.min_position_size_pct}% of capital)"
            )

        if constraints_applied:
            self.logger.info(json.dumps({
                "event": "position_constraints_applied",
                "proposed_quantity": str(proposed_quantity),
                "approved_quantity": str(approved_quantity),
                "constraints": constraints_applied
            }))

        return approved_quantity, constraints_applied

    def update_capital(self, pnl: Decimal) -> None:
        """
        Mettre à jour le capital après un trade.

        Parameters
        ----------
        pnl : Decimal
            P&L du trade (positif ou négatif)
        """
        self.current_capital += pnl
        self.logger.debug(json.dumps({
            "event": "capital_updated",
            "pnl": str(pnl),
            "new_capital": str(self.current_capital),
            "timestamp": datetime.utcnow().isoformat()
        }))

    def record_trade(
        self,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
    ) -> None:
        """
        Enregistrer un trade pour l'historique.

        Parameters
        ----------
        symbol : str
            Symbole du titre
        quantity : Decimal
            Quantité tradée
        entry_price : Decimal
            Prix d'entrée
        exit_price : Decimal
            Prix de sortie
        pnl : Decimal
            P&L du trade
        """
        trade_record = {
            "symbol": symbol,
            "quantity": str(quantity),
            "entry_price": str(entry_price),
            "exit_price": str(exit_price),
            "pnl": str(pnl),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.trade_history.append(trade_record)

        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]

        self.logger.debug(json.dumps({
            "event": "trade_recorded",
            "trade": trade_record
        }))

    def get_statistics(self) -> Dict:
        """
        Obtenir les statistiques de trading.

        Returns
        -------
        Dict
            Statistiques du trading (win rate, avg win/loss, etc.)
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if Decimal(t["pnl"]) > 0)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        wins = [Decimal(t["pnl"]) for t in self.trade_history if Decimal(t["pnl"]) > 0]
        losses = [Decimal(t["pnl"]) for t in self.trade_history if Decimal(t["pnl"]) < 0]

        avg_win = float(sum(wins) / len(wins)) if wins else 0.0
        avg_loss = float(abs(sum(losses)) / len(losses)) if losses else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "current_capital": str(self.current_capital),
            "capital_growth_pct": float(
                ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            ),
        }
