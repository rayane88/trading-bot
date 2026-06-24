"""
Risk Manager - Orchestrateur central du système de gestion des risques

Le Risk Manager a TOUJOURS le dernier mot.
Il valide, approuve, réduit ou bloque chaque trade.
"""

import logging
import json
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from trading_bot.core.risk.models import (
    TradeSignal,
    TradeSignalType,
    Position,
    RiskState,
    RiskDecision,
    RiskValidationResult,
    RiskReport,
    PositionStatus,
)
from trading_bot.core.risk.position_sizing import PositionSizer
from trading_bot.core.risk.circuit_breaker import CircuitBreaker
from trading_bot.core.risk.portfolio_risk import PortfolioRiskCalculator


class RiskManager:
    """
    Gestionnaire de risque central pour le trading bot.

    Responsabilités :
    - Valider tous les signaux de trading
    - Approuver/Réduire/Bloquer les trades
    - Gérer le circuit breaker
    - Calculer les expositions et risques
    - Générer des rapports de risque
    """

    def __init__(
        self,
        initial_capital: Decimal,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialiser le Risk Manager.

        Parameters
        ----------
        initial_capital : Decimal
            Capital initial du portefeuille
        config : Optional[Dict]
            Configuration optionnelle (sinon utilise les valeurs par défaut)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio_value = initial_capital

        # Configuration par défaut
        self.config = config or self._get_default_config()

        # Composants
        self.position_sizer = PositionSizer(
            initial_capital=initial_capital,
            kelly_fraction=self.config.get("kelly_fraction", 0.25),
            atr_period=self.config.get("atr_period", 14),
            atr_multiplier=self.config.get("atr_multiplier", 2.0),
            max_risk_per_trade_pct=self.config.get("max_risk_per_trade_pct", 2.0),
            max_position_size_pct=self.config.get("max_position_size_pct", 5.0),
            max_concurrent_positions=self.config.get("max_concurrent_positions", 6),
            min_position_size_pct=self.config.get("min_position_size_pct", 0.1),
        )

        self.circuit_breaker = CircuitBreaker(
            initial_capital=initial_capital,
            daily_drawdown_threshold_pct=self.config.get("daily_drawdown_threshold_pct", 5.0),
            monthly_drawdown_threshold_pct=self.config.get("monthly_drawdown_threshold_pct", 15.0),
            daily_reset_time=self.config.get("daily_reset_time", "16:00"),
            monthly_reset_day=self.config.get("monthly_reset_day", 1),
            consecutive_loss_trigger=self.config.get("consecutive_loss_trigger", 5),
            loss_ratio_trigger=self.config.get("loss_ratio_trigger", 0.30),
            cooldown_duration_minutes=self.config.get("cooldown_duration_minutes", 30),
            max_daily_restarts=self.config.get("max_daily_restarts", 3),
        )

        self.portfolio_risk = PortfolioRiskCalculator(
            portfolio_value=initial_capital,
            var_lookback_days=self.config.get("var_lookback_days", 252),
            var_min_observations=self.config.get("var_min_observations", 20),
            correlation_lookback_days=self.config.get("correlation_lookback_days", 60),
            correlation_threshold=self.config.get("correlation_threshold", 0.7),
            max_sector_exposure_pct=self.config.get("max_sector_exposure_pct", 30.0),
            herfindahl_threshold=self.config.get("herfindahl_threshold", 0.15),
        )

        # État
        self.open_positions: Dict[str, Position] = {}
        self.daily_pnl = Decimal("0")
        self.monthly_pnl = Decimal("0")
        self.validation_history: List[RiskValidationResult] = []

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._log_initialization()

    @staticmethod
    def _get_default_config() -> Dict:
        """Obtenir la configuration par défaut."""
        return {
            "kelly_fraction": 0.25,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "max_risk_per_trade_pct": 2.0,
            "max_position_size_pct": 5.0,
            "max_concurrent_positions": 6,
            "min_position_size_pct": 0.1,
            "daily_drawdown_threshold_pct": 5.0,
            "monthly_drawdown_threshold_pct": 15.0,
            "daily_reset_time": "16:00",
            "monthly_reset_day": 1,
            "consecutive_loss_trigger": 5,
            "loss_ratio_trigger": 0.30,
            "cooldown_duration_minutes": 30,
            "max_daily_restarts": 3,
            "var_lookback_days": 252,
            "var_min_observations": 20,
            "correlation_lookback_days": 60,
            "correlation_threshold": 0.7,
            "max_sector_exposure_pct": 30.0,
            "herfindahl_threshold": 0.15,
            "signal_quality_threshold": 0.70,
            "liquidity_min_daily_volume_usd": 1000000,
        }

    def _log_initialization(self) -> None:
        """Logger l'initialisation."""
        self.logger.info(json.dumps({
            "event": "risk_manager_initialized",
            "initial_capital": str(self.initial_capital),
            "timestamp": datetime.utcnow().isoformat()
        }))

    def validate_trade(self, signal: TradeSignal) -> RiskValidationResult:
        """
        Valider un signal de trading.

        C'est la fonction PRINCIPALE du Risk Manager.
        Le Risk Manager a TOUJOURS le dernier mot.

        Parameters
        ----------
        signal : TradeSignal
            Signal à valider

        Returns
        -------
        RiskValidationResult
            Résultat de la validation (GO/REDUCE/BLOCK)
        """
        validation_id = str(uuid4())
        checks = {}
        reasons = []
        risk_score = 0.0

        # CHECK 1 : Circuit breaker
        if not self.circuit_breaker.can_trade():
            reasons.append("Circuit breaker is active - NO TRADING")
            checks["circuit_breaker"] = False
            self.logger.warning(json.dumps({
                "event": "trade_blocked_circuit_breaker",
                "signal_id": signal.signal_id,
                "timestamp": datetime.utcnow().isoformat()
            }))

            result = RiskValidationResult(
                signal_id=signal.signal_id,
                decision=RiskDecision.BLOCK,
                approved_quantity=Decimal("0"),
                rejection_reasons=reasons,
                required_checks=checks,
                risk_score=100.0,
                timestamp=datetime.utcnow(),
            )
            self.validation_history.append(result)
            return result

        # CHECK 2 : Qualité du signal
        if signal.signal_quality < self.config["signal_quality_threshold"]:
            reasons.append(
                f"Signal quality {signal.signal_quality} below threshold "
                f"{self.config['signal_quality_threshold']}"
            )
            checks["signal_quality"] = False
            risk_score += 20.0
        else:
            checks["signal_quality"] = True

        # CHECK 3 : Limites de position
        current_positions = len(self.open_positions)
        if current_positions >= self.config["max_concurrent_positions"]:
            reasons.append(
                f"Max concurrent positions ({self.config['max_concurrent_positions']}) reached"
            )
            checks["position_limits"] = False
            risk_score += 30.0
        else:
            checks["position_limits"] = True

        # CHECK 4 : Liquidity
        # (Simplifié - en production, vérifier le volume réel)
        checks["liquidity_check"] = True

        # CHECK 5 : Calcul de la taille de position avec contraintes
        proposed_quantity = signal.quantity
        approved_quantity, constraints = self.position_sizer.calculate_constrained_size(
            proposed_quantity=proposed_quantity,
            entry_price=signal.entry_price,
            current_positions=current_positions,
            sector_exposure_pct=0.0,  # Simplifié
        )

        if len(constraints) > 0:
            reasons.extend(constraints)
            checks["position_constraints"] = False
            risk_score += 15.0
        else:
            checks["position_constraints"] = True

        # CHECK 6 : Correlation check
        # (Simplifié - en production, vérifier les corrélations du portefeuille)
        checks["correlation_check"] = True

        # DECISION FINALE
        if approved_quantity == Decimal("0"):
            decision = RiskDecision.BLOCK
            self.logger.warning(json.dumps({
                "event": "trade_blocked",
                "signal_id": signal.signal_id,
                "reasons": reasons,
                "risk_score": risk_score,
                "timestamp": datetime.utcnow().isoformat()
            }))
        elif approved_quantity < proposed_quantity:
            decision = RiskDecision.REDUCE
            reduction_pct = float((1 - approved_quantity / proposed_quantity) * 100)
            self.logger.info(json.dumps({
                "event": "trade_reduced",
                "signal_id": signal.signal_id,
                "reduction_pct": reduction_pct,
                "approved_quantity": str(approved_quantity),
                "timestamp": datetime.utcnow().isoformat()
            }))
        else:
            decision = RiskDecision.GO
            self.logger.info(json.dumps({
                "event": "trade_approved",
                "signal_id": signal.signal_id,
                "quantity": str(approved_quantity),
                "timestamp": datetime.utcnow().isoformat()
            }))

        result = RiskValidationResult(
            signal_id=signal.signal_id,
            decision=decision,
            approved_quantity=approved_quantity,
            rejection_reasons=reasons,
            required_checks=checks,
            risk_score=risk_score,
            timestamp=datetime.utcnow(),
        )

        self.validation_history.append(result)
        return result

    def execute_trade(
        self,
        signal: TradeSignal,
        validation_result: RiskValidationResult,
    ) -> Optional[Position]:
        """
        Exécuter un trade après validation.

        Parameters
        ----------
        signal : TradeSignal
            Signal original
        validation_result : RiskValidationResult
            Résultat de validation

        Returns
        -------
        Optional[Position]
            Position ouverte ou None si rejetée
        """
        if validation_result.decision != RiskDecision.GO:
            self.logger.warning(json.dumps({
                "event": "trade_not_executed",
                "signal_id": signal.signal_id,
                "decision": validation_result.decision.value,
                "timestamp": datetime.utcnow().isoformat()
            }))
            return None

        position_id = f"pos_{signal.signal_id}_{datetime.utcnow().timestamp()}"

        position = Position(
            position_id=position_id,
            symbol=signal.symbol,
            quantity=validation_result.approved_quantity,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            unrealized_pnl=Decimal("0"),
            stop_loss=signal.stop_loss_price,
            take_profit=signal.take_profit_price,
            status=PositionStatus.OPEN,
            opened_at=datetime.utcnow(),
            commission_paid=Decimal("0"),
        )

        self.open_positions[position_id] = position
        self.portfolio_risk.add_position(
            symbol=signal.symbol,
            quantity=validation_result.approved_quantity,
            current_price=signal.entry_price,
            sector="Unknown",  # À obtenir de source externe
        )

        self.logger.info(json.dumps({
            "event": "trade_executed",
            "position_id": position_id,
            "symbol": signal.symbol,
            "quantity": str(validation_result.approved_quantity),
            "entry_price": str(signal.entry_price),
            "timestamp": datetime.utcnow().isoformat()
        }))

        return position

    def update_position(
        self,
        position_id: str,
        current_price: Decimal,
    ) -> Optional[Position]:
        """
        Mettre à jour une position avec le prix courant.

        Parameters
        ----------
        position_id : str
            ID de la position
        current_price : Decimal
            Prix courant du marché

        Returns
        -------
        Optional[Position]
            Position mise à jour ou None
        """
        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]
        position.current_price = current_price

        # Calculer P&L
        price_diff = current_price - position.entry_price
        position.unrealized_pnl = price_diff * position.quantity - position.commission_paid

        # Vérifier stop loss
        if current_price <= position.stop_loss:
            self.close_position(position_id, current_price, reason="Stop loss hit")
            return position

        # Vérifier take profit
        if position.take_profit and current_price >= position.take_profit:
            self.close_position(position_id, current_price, reason="Take profit hit")
            return position

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        reason: str = "Manual close",
    ) -> Optional[Position]:
        """
        Fermer une position.

        Parameters
        ----------
        position_id : str
            ID de la position
        exit_price : Decimal
            Prix de sortie
        reason : str
            Raison de la fermeture

        Returns
        -------
        Optional[Position]
            Position fermée ou None
        """
        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.utcnow()

        # Calculer P&L réalisé
        price_diff = exit_price - position.entry_price
        position.realized_pnl = (price_diff * position.quantity) - position.commission_paid

        # Mettre à jour capital
        self.current_capital += position.realized_pnl
        self.daily_pnl += position.realized_pnl
        self.monthly_pnl += position.realized_pnl

        # Mettre à jour circuit breaker
        self.circuit_breaker.record_trade_result(
            pnl=position.realized_pnl,
            current_capital=self.current_capital,
        )

        # Enregistrer le trade
        self.position_sizer.record_trade(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=position.realized_pnl,
        )

        del self.open_positions[position_id]

        self.logger.info(json.dumps({
            "event": "position_closed",
            "position_id": position_id,
            "symbol": position.symbol,
            "entry_price": str(position.entry_price),
            "exit_price": str(exit_price),
            "realized_pnl": str(position.realized_pnl),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }))

        return position

    def update_pnl(self, current_time: datetime) -> None:
        """
        Mettre à jour le P&L et vérifier les limites.

        Parameters
        ----------
        current_time : datetime
            Temps courant
        """
        # Mettre à jour le circuit breaker
        self.circuit_breaker.update_capital(
            current_capital=self.current_capital,
            current_time=current_time,
        )

        # Calculer la valeur du portefeuille
        position_value = sum(
            p.unrealized_pnl for p in self.open_positions.values()
        )
        self.portfolio_value = self.current_capital + position_value

        self.logger.debug(json.dumps({
            "event": "pnl_updated",
            "current_capital": str(self.current_capital),
            "portfolio_value": str(self.portfolio_value),
            "daily_pnl": str(self.daily_pnl),
            "monthly_pnl": str(self.monthly_pnl),
            "timestamp": current_time.isoformat()
        }))

    def get_risk_state(self) -> RiskState:
        """
        Obtenir l'état actuel du risque.

        Returns
        -------
        RiskState
            État complet du risque
        """
        # Calculer drawdowns
        daily_drawdown = ((self.initial_capital - self.portfolio_value) / self.initial_capital) * 100
        monthly_drawdown = daily_drawdown  # Simplifié

        # Calculer risques
        var_95 = self.portfolio_risk.calculate_var_historical(0.95)
        var_99 = self.portfolio_risk.calculate_var_historical(0.99)
        cvar_95 = self.portfolio_risk.calculate_cvar(0.95)

        # Exposition sectorielle
        sector_exposure = self.portfolio_risk.calculate_sector_exposure()
        max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0.0

        # Concentration
        herfindahl = self.portfolio_risk.calculate_herfindahl_index()

        return RiskState(
            timestamp=datetime.utcnow(),
            portfolio_value=self.portfolio_value,
            cash_available=self.current_capital,
            total_positions=len(self.open_positions),
            daily_pnl=self.daily_pnl,
            monthly_pnl=self.monthly_pnl,
            drawdown_daily_pct=-max(0, daily_drawdown),
            drawdown_monthly_pct=-max(0, monthly_drawdown),
            circuit_breaker_active=not self.circuit_breaker.can_trade(),
            circuit_breaker_reason=None,
            var_95=var_95.var_value if var_95 else 0.0,
            var_99=var_99.var_value if var_99 else 0.0,
            cvar_95=cvar_95.cvar_value if cvar_95 else 0.0,
            max_sector_exposure_pct=float(max_sector_exposure),
            max_concentration=herfindahl * 100,
        )

    def get_risk_report(self) -> RiskReport:
        """
        Générer un rapport complet de risque.

        Returns
        -------
        RiskReport
            Rapport complet
        """
        report_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H_%M')}"
        risk_state = self.get_risk_state()
        positions = list(self.open_positions.values())
        sector_exposure = self.portfolio_risk.calculate_sector_exposure()
        correlation_matrix = self.portfolio_risk.calculate_correlation_matrix()

        recommendations = []
        alerts = []

        # Générer recommandations
        if risk_state.circuit_breaker_active:
            alerts.append("Circuit breaker is active - no new trades allowed")
        if risk_state.max_concentration > 15.0:
            recommendations.append("Portfolio concentration is high - consider diversifying")
        if risk_state.max_sector_exposure_pct > 25.0:
            recommendations.append("Sector exposure is high - consider rebalancing")

        return RiskReport(
            report_id=report_id,
            timestamp=datetime.utcnow(),
            risk_state=risk_state,
            positions=positions,
            sector_exposures=sector_exposure,
            correlation_matrix=correlation_matrix,
            stress_test_results={},
            recommendations=recommendations,
            alerts=alerts,
            metrics=self.position_sizer.get_statistics(),
        )

    def reset_daily_pnl(self) -> None:
        """Réinitialiser le P&L journalier."""
        self.daily_pnl = Decimal("0")
        self.logger.info(json.dumps({
            "event": "daily_pnl_reset",
            "timestamp": datetime.utcnow().isoformat()
        }))

    def reset_monthly_pnl(self) -> None:
        """Réinitialiser le P&L mensuel."""
        self.monthly_pnl = Decimal("0")
        self.logger.info(json.dumps({
            "event": "monthly_pnl_reset",
            "timestamp": datetime.utcnow().isoformat()
        }))
