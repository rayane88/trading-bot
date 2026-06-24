"""
Circuit Breaker - Système d'arrêt automatique

Implémente :
- Drawdown limits (daily 5%, monthly 15%)
- Auto kill switch
- Cooldown periods
- Reset manuel
"""

import logging
import json
from decimal import Decimal
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List
from enum import Enum

from pydantic import BaseModel, Field


class CircuitBreakerStatus(str, Enum):
    """Statut du circuit breaker."""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    COOLDOWN = "COOLDOWN"
    RESET = "RESET"


class CircuitBreakerEvent(BaseModel):
    """Evénement de circuit breaker."""
    status: CircuitBreakerStatus = Field(..., description="Statut du circuit breaker")
    timestamp: datetime = Field(..., description="Quand l'événement s'est produit")
    reason: str = Field(..., description="Raison du déclenchement")
    drawdown_pct: Optional[float] = Field(None, description="Drawdown au moment du trigger")
    portfolio_value: Decimal = Field(..., description="Valeur du portefeuille")


class CircuitBreaker:
    """
    Gestionnaire du circuit breaker.

    Arrête le trading automatiquement si :
    - Drawdown journalier > 5%
    - Drawdown mensuel > 15%
    - 5 pertes consécutives
    - Loss ratio > 30%
    """

    def __init__(
        self,
        initial_capital: Decimal,
        daily_drawdown_threshold_pct: float = 5.0,
        monthly_drawdown_threshold_pct: float = 15.0,
        daily_reset_time: str = "16:00",
        monthly_reset_day: int = 1,
        consecutive_loss_trigger: int = 5,
        loss_ratio_trigger: float = 0.30,
        cooldown_duration_minutes: int = 30,
        max_daily_restarts: int = 3,
    ) -> None:
        """
        Initialiser le Circuit Breaker.

        Parameters
        ----------
        initial_capital : Decimal
            Capital initial
        daily_drawdown_threshold_pct : float
            Seuil de drawdown journalier en %
        monthly_drawdown_threshold_pct : float
            Seuil de drawdown mensuel en %
        daily_reset_time : str
            Heure de réinitialisation quotidienne (HH:MM)
        monthly_reset_day : int
            Jour de réinitialisation mensuelle (1-31)
        consecutive_loss_trigger : int
            Nombre de pertes consécutives pour déclencher
        loss_ratio_trigger : float
            Ratio de perte pour déclencher
        cooldown_duration_minutes : int
            Durée du cooldown après trigger
        max_daily_restarts : int
            Max de restarts par jour
        """
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.monthly_start_capital = initial_capital

        self.daily_drawdown_threshold = daily_drawdown_threshold_pct
        self.monthly_drawdown_threshold = monthly_drawdown_threshold_pct

        self.daily_reset_time = self._parse_time(daily_reset_time)
        self.monthly_reset_day = monthly_reset_day

        self.consecutive_loss_trigger = consecutive_loss_trigger
        self.loss_ratio_trigger = loss_ratio_trigger

        self.cooldown_duration = timedelta(minutes=cooldown_duration_minutes)
        self.max_daily_restarts = max_daily_restarts

        self.status = CircuitBreakerStatus.ACTIVE
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.cooldown_start_time: Optional[datetime] = None

        self.daily_restarts = 0
        self.last_restart_date: Optional[datetime] = None

        self.recent_trades: List[Dict] = []
        self.events: List[CircuitBreakerEvent] = []

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._log_initialization()

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parser une chaîne de temps HH:MM."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _log_initialization(self) -> None:
        """Logger l'initialisation."""
        config = {
            "initial_capital": str(self.initial_capital),
            "daily_drawdown_threshold_pct": self.daily_drawdown_threshold,
            "monthly_drawdown_threshold_pct": self.monthly_drawdown_threshold,
            "consecutive_loss_trigger": self.consecutive_loss_trigger,
            "loss_ratio_trigger": self.loss_ratio_trigger,
        }
        self.logger.info(json.dumps({
            "event": "circuit_breaker_initialized",
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def _should_reset_daily(self, current_time: datetime) -> bool:
        """Vérifier si le reset journalier doit être appliqué."""
        current_time_only = current_time.time()
        return current_time_only >= self.daily_reset_time

    def _should_reset_monthly(self, current_time: datetime) -> bool:
        """Vérifier si le reset mensuel doit être appliqué."""
        return current_time.day >= self.monthly_reset_day

    def update_capital(self, current_capital: Decimal, current_time: datetime) -> None:
        """
        Mettre à jour le capital et vérifier les seuils.

        Parameters
        ----------
        current_capital : Decimal
            Capital actuel du portefeuille
        current_time : datetime
            Temps actuel
        """
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        self._check_daily_reset(current_time)
        self._check_monthly_reset(current_time)
        self._check_cooldown(current_time)
        self._check_drawdowns(current_capital)

    def _check_daily_reset(self, current_time: datetime) -> None:
        """Vérifier et appliquer le reset journalier."""
        if self._should_reset_daily(current_time):
            if (self.last_restart_date is None or
                self.last_restart_date.date() != current_time.date()):

                self.daily_start_capital = self.peak_capital
                self.daily_restarts = 0
                self.last_restart_date = current_time

                self.logger.info(json.dumps({
                    "event": "daily_reset",
                    "timestamp": current_time.isoformat(),
                    "reset_capital": str(self.daily_start_capital)
                }))

    def _check_monthly_reset(self, current_time: datetime) -> None:
        """Vérifier et appliquer le reset mensuel."""
        if self._should_reset_monthly(current_time):
            self.monthly_start_capital = self.peak_capital
            self.logger.info(json.dumps({
                "event": "monthly_reset",
                "timestamp": current_time.isoformat(),
                "reset_capital": str(self.monthly_start_capital)
            }))

    def _check_cooldown(self, current_time: datetime) -> None:
        """Vérifier et sortir du cooldown si nécessaire."""
        if self.status == CircuitBreakerStatus.COOLDOWN:
            if self.cooldown_start_time is None:
                return

            if current_time - self.cooldown_start_time >= self.cooldown_duration:
                self._exit_cooldown(current_time)

    def _check_drawdowns(self, current_capital: Decimal) -> None:
        """Vérifier les seuils de drawdown."""
        daily_drawdown = (
            (self.daily_start_capital - current_capital) / self.daily_start_capital * 100
        )

        if daily_drawdown > self.daily_drawdown_threshold:
            self._trigger_circuit_breaker(
                f"Daily drawdown {daily_drawdown:.2f}% exceeded threshold "
                f"{self.daily_drawdown_threshold}%",
                daily_drawdown,
                current_capital,
            )
            return

        monthly_drawdown = (
            (self.monthly_start_capital - current_capital) / self.monthly_start_capital * 100
        )

        if monthly_drawdown > self.monthly_drawdown_threshold:
            self._trigger_circuit_breaker(
                f"Monthly drawdown {monthly_drawdown:.2f}% exceeded threshold "
                f"{self.monthly_drawdown_threshold}%",
                monthly_drawdown,
                current_capital,
            )

    def record_trade_result(self, pnl: Decimal, current_capital: Decimal) -> None:
        """
        Enregistrer le résultat d'un trade et vérifier les triggers.

        Parameters
        ----------
        pnl : Decimal
            P&L du trade
        current_capital : Decimal
            Capital après le trade
        """
        self.recent_trades.append({
            "pnl": pnl,
            "timestamp": datetime.utcnow(),
            "is_loss": pnl < 0,
        })

        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]

        self._check_consecutive_losses()
        self._check_loss_ratio()

    def _check_consecutive_losses(self) -> None:
        """Vérifier les pertes consécutives."""
        if len(self.recent_trades) < self.consecutive_loss_trigger:
            return

        recent = self.recent_trades[-self.consecutive_loss_trigger:]
        if all(t["is_loss"] for t in recent):
            self._trigger_circuit_breaker(
                f"Consecutive losses ({self.consecutive_loss_trigger}) detected",
                None,
                None,
            )

    def _check_loss_ratio(self) -> None:
        """Vérifier le ratio de perte."""
        if len(self.recent_trades) < 10:
            return

        recent = self.recent_trades[-20:]
        loss_count = sum(1 for t in recent if t["is_loss"])
        loss_ratio = loss_count / len(recent)

        if loss_ratio > self.loss_ratio_trigger:
            self._trigger_circuit_breaker(
                f"Loss ratio {loss_ratio:.2f} exceeded threshold {self.loss_ratio_trigger}",
                None,
                None,
            )

    def _trigger_circuit_breaker(
        self,
        reason: str,
        drawdown_pct: Optional[float],
        current_capital: Optional[Decimal],
    ) -> None:
        """Déclencher le circuit breaker."""
        if self.is_triggered:
            return

        self.is_triggered = True
        self.status = CircuitBreakerStatus.TRIGGERED
        self.trigger_time = datetime.utcnow()

        event = CircuitBreakerEvent(
            status=CircuitBreakerStatus.TRIGGERED,
            timestamp=self.trigger_time,
            reason=reason,
            drawdown_pct=drawdown_pct,
            portfolio_value=current_capital or self.initial_capital,
        )
        self.events.append(event)

        self.logger.error(json.dumps({
            "event": "circuit_breaker_triggered",
            "reason": reason,
            "drawdown_pct": drawdown_pct,
            "timestamp": self.trigger_time.isoformat()
        }))

    def enter_cooldown(self, current_time: datetime) -> None:
        """
        Entrer en phase cooldown après un trigger.

        Parameters
        ----------
        current_time : datetime
            Temps actuel
        """
        self.status = CircuitBreakerStatus.COOLDOWN
        self.cooldown_start_time = current_time
        self.daily_restarts += 1

        self.logger.info(json.dumps({
            "event": "circuit_breaker_cooldown",
            "duration_minutes": self.cooldown_duration.total_seconds() / 60,
            "daily_restarts": self.daily_restarts,
            "timestamp": current_time.isoformat()
        }))

    def _exit_cooldown(self, current_time: datetime) -> None:
        """Sortir de la phase cooldown."""
        if self.daily_restarts >= self.max_daily_restarts:
            self.logger.warning(json.dumps({
                "event": "circuit_breaker_max_restarts_reached",
                "daily_restarts": self.daily_restarts,
                "timestamp": current_time.isoformat()
            }))
            return

        self.status = CircuitBreakerStatus.ACTIVE
        self.is_triggered = False
        self.cooldown_start_time = None

        event = CircuitBreakerEvent(
            status=CircuitBreakerStatus.ACTIVE,
            timestamp=current_time,
            reason="Cooldown period completed",
            portfolio_value=self.initial_capital,
        )
        self.events.append(event)

        self.logger.info(json.dumps({
            "event": "circuit_breaker_cooldown_exit",
            "timestamp": current_time.isoformat()
        }))

    def manual_reset(self, current_time: datetime) -> bool:
        """
        Réinitialisation manuelle du circuit breaker.

        Parameters
        ----------
        current_time : datetime
            Temps actuel

        Returns
        -------
        bool
            True si reset réussi, False sinon
        """
        if not self.is_triggered:
            self.logger.warning(json.dumps({
                "event": "manual_reset_not_triggered",
                "timestamp": current_time.isoformat()
            }))
            return False

        if self.daily_restarts >= self.max_daily_restarts:
            self.logger.error(json.dumps({
                "event": "manual_reset_max_restarts",
                "timestamp": current_time.isoformat()
            }))
            return False

        self.status = CircuitBreakerStatus.RESET
        self.is_triggered = False
        self.cooldown_start_time = None
        self.trigger_time = None
        self.recent_trades.clear()

        event = CircuitBreakerEvent(
            status=CircuitBreakerStatus.RESET,
            timestamp=current_time,
            reason="Manual reset",
            portfolio_value=self.initial_capital,
        )
        self.events.append(event)

        self.logger.info(json.dumps({
            "event": "circuit_breaker_manual_reset",
            "timestamp": current_time.isoformat()
        }))

        return True

    def can_trade(self) -> bool:
        """
        Vérifier si le trading est autorisé.

        Returns
        -------
        bool
            True si le trading peut continuer, False sinon
        """
        return self.status == CircuitBreakerStatus.ACTIVE

    def get_status_report(self) -> Dict:
        """
        Obtenir un rapport du statut du circuit breaker.

        Returns
        -------
        Dict
            Rapport du statut
        """
        return {
            "status": self.status.value,
            "is_triggered": self.is_triggered,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "can_trade": self.can_trade(),
            "cooldown_active": self.status == CircuitBreakerStatus.COOLDOWN,
            "daily_restarts": self.daily_restarts,
            "recent_events": [
                {
                    "status": e.status.value,
                    "timestamp": e.timestamp.isoformat(),
                    "reason": e.reason,
                }
                for e in self.events[-5:]
            ],
        }
