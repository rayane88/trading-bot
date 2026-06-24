"""
Tests pour le module Position Sizing

Couverture:
- Kelly Criterion (full + fractional)
- ATR-based sizing
- Contraintes de sizing
- Historique des trades
"""

import pytest
from decimal import Decimal
from datetime import datetime

from trading_bot.core.risk.position_sizing import PositionSizer


class TestPositionSizer:
    """Tests du Position Sizer."""

    @pytest.fixture
    def sizer(self):
        """Fixture pour un Position Sizer initialisé."""
        return PositionSizer(
            initial_capital=Decimal("1000000"),
            kelly_fraction=0.25,
            atr_period=14,
            atr_multiplier=2.0,
            max_risk_per_trade_pct=2.0,
            max_position_size_pct=5.0,
            max_concurrent_positions=6,
            min_position_size_pct=0.1,
        )

    def test_initialization(self, sizer):
        """Test l'initialisation du Position Sizer."""
        assert sizer.initial_capital == Decimal("1000000")
        assert sizer.current_capital == Decimal("1000000")
        assert sizer.kelly_fraction == 0.25
        assert len(sizer.trade_history) == 0

    def test_kelly_criterion_valid(self, sizer):
        """Test le calcul Kelly Criterion avec paramétres valides."""
        result = sizer.calculate_kelly_criterion(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=80.0,
            confidence=1.0,
        )

        assert result is not None
        assert result.win_rate == 0.55
        assert result.avg_win == 100.0
        assert result.avg_loss == 80.0
        assert 0 < result.position_size_pct <= 5.0

    def test_kelly_criterion_invalid_win_rate(self, sizer):
        """Test Kelly Criterion avec taux de victoire invalide."""
        result = sizer.calculate_kelly_criterion(
            win_rate=1.5,
            avg_win=100.0,
            avg_loss=80.0,
        )
        assert result is None

    def test_kelly_criterion_invalid_amounts(self, sizer):
        """Test Kelly Criterion avec montants invalides."""
        result = sizer.calculate_kelly_criterion(
            win_rate=0.5,
            avg_win=-100.0,
            avg_loss=80.0,
        )
        assert result is None

    def test_atr_sizing(self, sizer):
        """Test le sizing basé sur ATR."""
        result = sizer.calculate_atr_sizing(
            current_price=Decimal("100"),
            atr_value=Decimal("5"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
        )

        assert result is not None
        assert result.position_size > 0
        assert result.atr_value == Decimal("5")
        assert result.stop_loss_price >= Decimal("90")

    def test_atr_sizing_invalid_risk_distance(self, sizer):
        """Test ATR sizing avec distance de risque invalide."""
        with pytest.raises(ValueError):
            sizer.calculate_atr_sizing(
                current_price=Decimal("100"),
                atr_value=Decimal("10"),
                entry_price=Decimal("100"),
                stop_loss_price=Decimal("100"),
            )

    def test_constrained_size_no_constraints(self, sizer):
        """Test le sizing avec contraintes (aucune contrainte)."""
        approved, constraints = sizer.calculate_constrained_size(
            proposed_quantity=Decimal("100"),
            entry_price=Decimal("100"),
            current_positions=0,
            sector_exposure_pct=0.0,
        )

        assert approved == Decimal("100")
        assert len(constraints) == 0

    def test_constrained_size_max_positions_reached(self, sizer):
        """Test avec maximum de positions atteint."""
        approved, constraints = sizer.calculate_constrained_size(
            proposed_quantity=Decimal("100"),
            entry_price=Decimal("100"),
            current_positions=6,  # Max atteint
            sector_exposure_pct=0.0,
        )

        assert approved == Decimal("0")
        assert len(constraints) > 0
        assert "Maximum concurrent positions" in constraints[0]

    def test_constrained_size_position_size_limit(self, sizer):
        """Test avec limite de taille de position."""
        approved, constraints = sizer.calculate_constrained_size(
            proposed_quantity=Decimal("100000"),  # Très grand
            entry_price=Decimal("100"),
            current_positions=2,
            sector_exposure_pct=0.0,
        )

        assert approved < Decimal("100000")
        # Max position = 5% of 1M = 50k, so max qty = 50k/100 = 500
        assert approved <= Decimal("500")

    def test_update_capital(self, sizer):
        """Test la mise à jour du capital."""
        initial = sizer.current_capital
        pnl = Decimal("10000")
        sizer.update_capital(pnl)

        assert sizer.current_capital == initial + pnl

    def test_record_trade(self, sizer):
        """Test l'enregistrement d'un trade."""
        sizer.record_trade(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            exit_price=Decimal("155"),
            pnl=Decimal("500"),
        )

        assert len(sizer.trade_history) == 1
        trade = sizer.trade_history[0]
        assert trade["symbol"] == "AAPL"
        assert trade["quantity"] == "100"
        assert trade["pnl"] == "500"

    def test_get_statistics_empty_history(self, sizer):
        """Test les statistiques sans historique."""
        stats = sizer.get_statistics()

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_win"] == 0.0

    def test_get_statistics_with_trades(self, sizer):
        """Test les statistiques avec historique."""
        sizer.record_trade(
            symbol="AAPL",
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
            exit_price=Decimal("155"),
            pnl=Decimal("500"),
        )
        sizer.record_trade(
            symbol="MSFT",
            quantity=Decimal("50"),
            entry_price=Decimal("300"),
            exit_price=Decimal("295"),
            pnl=Decimal("-250"),
        )

        stats = sizer.get_statistics()

        assert stats["total_trades"] == 2
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.5
        assert stats["avg_win"] == 500.0
        assert stats["avg_loss"] == 250.0

    def test_fractional_kelly_vs_full_kelly(self):
        """Test la comparaison entre fractional et full Kelly."""
        sizer_full = PositionSizer(
            initial_capital=Decimal("1000000"),
            kelly_fraction=1.0,
        )
        sizer_frac = PositionSizer(
            initial_capital=Decimal("1000000"),
            kelly_fraction=0.25,
        )

        result_full = sizer_full.calculate_kelly_criterion(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=80.0,
        )
        result_frac = sizer_frac.calculate_kelly_criterion(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=80.0,
        )

        assert result_frac.position_size_pct < result_full.position_size_pct

    def test_edge_case_zero_quantity(self, sizer):
        """Test cas limite avec quantité zéro."""
        approved, constraints = sizer.calculate_constrained_size(
            proposed_quantity=Decimal("0"),
            entry_price=Decimal("100"),
            current_positions=2,
            sector_exposure_pct=0.0,
        )

        assert approved == Decimal("0")

    def test_edge_case_very_small_position(self, sizer):
        """Test cas limite avec très petite position."""
        approved, constraints = sizer.calculate_constrained_size(
            proposed_quantity=Decimal("0.00001"),
            entry_price=Decimal("100"),
            current_positions=2,
            sector_exposure_pct=0.0,
        )

        # Devrait être bloqué par minimum size
        assert approved == Decimal("0") or len(constraints) > 0

    def test_capital_growth_calculation(self, sizer):
        """Test le calcul de la croissance du capital."""
        sizer.update_capital(Decimal("100000"))
        stats = sizer.get_statistics()

        expected_growth = (100000 / 1000000) * 100
        assert stats["capital_growth_pct"] == expected_growth
