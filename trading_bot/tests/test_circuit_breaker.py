"""
Tests pour le module Circuit Breaker

Couverture:
- Drawdown limits (daily 5%, monthly 15%)
- Auto kill switch
- Cooldown periods
- Manual reset
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from trading_bot.core.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerStatus,
)


class TestCircuitBreaker:
    """Tests du Circuit Breaker."""

    @pytest.fixture
    def breaker(self):
        """Fixture pour un Circuit Breaker initialisé."""
        return CircuitBreaker(
            initial_capital=Decimal("1000000"),
            daily_drawdown_threshold_pct=5.0,
            monthly_drawdown_threshold_pct=15.0,
            daily_reset_time="16:00",
            monthly_reset_day=1,
            consecutive_loss_trigger=5,
            loss_ratio_trigger=0.30,
            cooldown_duration_minutes=30,
            max_daily_restarts=3,
        )

    def test_initialization(self, breaker):
        """Test l'initialisation du Circuit Breaker."""
        assert breaker.initial_capital == Decimal("1000000")
        assert breaker.status == CircuitBreakerStatus.ACTIVE
        assert breaker.is_triggered is False
        assert breaker.can_trade() is True

    def test_daily_drawdown_trigger(self, breaker):
        """Test le déclenchement du drawdown journalier."""
        breaker.daily_start_capital = Decimal("1000000")
        # 6% de drawdown > 5% threshold
        current_capital = Decimal("940000")
        current_time = datetime.utcnow()

        breaker.update_capital(current_capital, current_time)

        assert breaker.is_triggered is True
        assert breaker.status == CircuitBreakerStatus.TRIGGERED
        assert breaker.can_trade() is False

    def test_monthly_drawdown_trigger(self, breaker):
        """Test le déclenchement du drawdown mensuel."""
        breaker.monthly_start_capital = Decimal("1000000")
        # 16% de drawdown > 15% threshold
        current_capital = Decimal("840000")
        current_time = datetime.utcnow()

        breaker.update_capital(current_capital, current_time)

        assert breaker.is_triggered is True

    def test_consecutive_losses_trigger(self, breaker):
        """Test le déclenchement par pertes consécutives."""
        for i in range(5):
            breaker.record_trade_result(
                pnl=Decimal("-1000"),
                current_capital=Decimal(f"{1000000 - (i+1)*1000}"),
            )

        assert breaker.is_triggered is True
        assert breaker.status == CircuitBreakerStatus.TRIGGERED

    def test_loss_ratio_trigger(self, breaker):
        """Test le déclenchement par ratio de perte."""
        # 20 trades avec 70% de pertes
        for i in range(20):
            pnl = Decimal("-100") if i % 10 < 7 else Decimal("100")
            breaker.record_trade_result(pnl, Decimal("1000000"))

        assert breaker.is_triggered is True

    def test_enter_cooldown(self, breaker):
        """Test l'entrée en phase cooldown."""
        breaker.is_triggered = True
        breaker.status = CircuitBreakerStatus.TRIGGERED
        current_time = datetime.utcnow()

        breaker.enter_cooldown(current_time)

        assert breaker.status == CircuitBreakerStatus.COOLDOWN
        assert breaker.can_trade() is False
        assert breaker.daily_restarts == 1

    def test_exit_cooldown_after_duration(self, breaker):
        """Test la sortie du cooldown après expiration."""
        breaker.status = CircuitBreakerStatus.COOLDOWN
        breaker.daily_restarts = 1
        start_time = datetime.utcnow()
        breaker.cooldown_start_time = start_time

        # Simuler le passage du temps
        exit_time = start_time + timedelta(minutes=31)
        breaker._check_cooldown(exit_time)

        assert breaker.status == CircuitBreakerStatus.ACTIVE
        assert breaker.can_trade() is True

    def test_max_daily_restarts_limit(self, breaker):
        """Test la limite de restarts quotidiens."""
        breaker.daily_restarts = 3
        breaker.status = CircuitBreakerStatus.COOLDOWN
        breaker.cooldown_start_time = datetime.utcnow() - timedelta(minutes=31)

        # Tenter de sortir du cooldown après 3 restarts
        exit_time = datetime.utcnow()
        breaker._check_cooldown(exit_time)

        # Devrait rester en cooldown
        assert breaker.status == CircuitBreakerStatus.COOLDOWN

    def test_manual_reset_when_triggered(self, breaker):
        """Test le reset manuel quand déclenché."""
        breaker.is_triggered = True
        breaker.status = CircuitBreakerStatus.TRIGGERED
        breaker.daily_restarts = 1

        result = breaker.manual_reset(datetime.utcnow())

        assert result is True
        assert breaker.status == CircuitBreakerStatus.RESET
        assert breaker.is_triggered is False

    def test_manual_reset_when_not_triggered(self, breaker):
        """Test le reset manuel quand non déclenché."""
        result = breaker.manual_reset(datetime.utcnow())
        assert result is False

    def test_manual_reset_max_restarts_reached(self, breaker):
        """Test le reset manuel avec max restarts atteint."""
        breaker.is_triggered = True
        breaker.daily_restarts = 3

        result = breaker.manual_reset(datetime.utcnow())
        assert result is False

    def test_status_report(self, breaker):
        """Test le rapport d'état."""
        report = breaker.get_status_report()

        assert "status" in report
        assert "is_triggered" in report
        assert "can_trade" in report
        assert "daily_restarts" in report
        assert report["can_trade"] is True
        assert report["is_triggered"] is False

    def test_daily_reset(self, breaker):
        """Test le reset journalier."""
        breaker.daily_start_capital = Decimal("900000")
        breaker.peak_capital = Decimal("1000000")
        current_time = datetime.utcnow().replace(hour=17, minute=0)  # Après 16:00

        breaker._check_daily_reset(current_time)

        assert breaker.daily_start_capital == Decimal("1000000")
        assert breaker.daily_restarts == 0

    def test_monthly_reset(self, breaker):
        """Test le reset mensuel."""
        breaker.monthly_start_capital = Decimal("900000")
        breaker.peak_capital = Decimal("1000000")
        current_time = datetime.utcnow().replace(day=2)  # Après le 1er

        breaker._check_monthly_reset(current_time)

        assert breaker.monthly_start_capital == Decimal("1000000")

    def test_edge_case_zero_drawdown(self, breaker):
        """Test cas limite avec zéro drawdown."""
        breaker.daily_start_capital = Decimal("1000000")
        current_capital = Decimal("1000000")
        current_time = datetime.utcnow()

        breaker.update_capital(current_capital, current_time)

        assert breaker.is_triggered is False
        assert breaker.can_trade() is True

    def test_edge_case_exactly_at_threshold(self, breaker):
        """Test cas limite exactement au seuil."""
        breaker.daily_start_capital = Decimal("1000000")
        # Exactement 5% de drawdown
        current_capital = Decimal("950000")
        current_time = datetime.utcnow()

        breaker.update_capital(current_capital, current_time)

        # Ne devrait pas déclencher (seuil > 5%, pas >=)
        assert breaker.is_triggered is False
