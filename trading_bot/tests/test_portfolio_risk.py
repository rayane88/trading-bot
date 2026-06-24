"""
Tests pour le module Portfolio Risk

Couverture:
- VaR historique et paramétrique
- CVaR
- Matrice de corrélation
- Exposition sectorielle
- Indice Herfindahl
"""

import pytest
import numpy as np
from decimal import Decimal
from typing import List

from trading_bot.core.risk.portfolio_risk import PortfolioRiskCalculator


class TestPortfolioRiskCalculator:
    """Tests du Portfolio Risk Calculator."""

    @pytest.fixture
    def calculator(self):
        """Fixture pour un calculateur de risque initialisé."""
        return PortfolioRiskCalculator(
            portfolio_value=Decimal("1000000"),
            var_lookback_days=252,
            var_min_observations=20,
            correlation_lookback_days=60,
            correlation_threshold=0.7,
            max_sector_exposure_pct=30.0,
            herfindahl_threshold=0.15,
        )

    @pytest.fixture
    def sample_returns(self) -> List[float]:
        """Générer des returns historiques."""
        np.random.seed(42)
        return list(np.random.normal(0.0005, 0.01, 252))

    def test_initialization(self, calculator):
        """Test l'initialisation du calculateur."""
        assert calculator.portfolio_value == Decimal("1000000")
        assert calculator.var_lookback_days == 252
        assert len(calculator.positions) == 0

    def test_add_position(self, calculator):
        """Test l'ajout d'une position."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )

        assert "AAPL" in calculator.positions
        assert calculator.positions["AAPL"]["quantity"] == 100.0
        assert calculator.positions["AAPL"]["sector"] == "Technology"

    def test_add_multiple_positions(self, calculator):
        """Test l'ajout de plusieurs positions."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.add_position(
            symbol="MSFT",
            quantity=Decimal("200"),
            current_price=Decimal("300"),
            sector="Technology",
        )

        assert len(calculator.positions) == 2
        assert calculator.positions["AAPL"]["position_pct"] > 0
        assert calculator.positions["MSFT"]["position_pct"] > 0

    def test_update_price_history(self, calculator, sample_returns):
        """Test la mise à jour de l'historique des prix."""
        calculator.update_price_history("AAPL", sample_returns)

        assert len(calculator.price_history["AAPL"]) == 252
        assert calculator.price_history["AAPL"][0] == sample_returns[0]

    def test_var_historical_sufficient_data(self, calculator, sample_returns):
        """Test le calcul VaR historique avec données suffisantes."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)

        result = calculator.calculate_var_historical(0.95)

        assert result is not None
        assert result.confidence_level == 0.95
        assert result.var_value > 0
        assert result.var_pct > 0
        assert result.method == "historical"

    def test_var_historical_insufficient_data(self, calculator):
        """Test VaR historique avec données insuffisantes."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        # Ajouter seulement 5 observations
        short_returns = [0.001 * (i % 2 - 0.5) for i in range(5)]
        calculator.update_price_history("AAPL", short_returns)

        result = calculator.calculate_var_historical(0.95)

        assert result is None

    def test_var_parametric(self, calculator, sample_returns):
        """Test le calcul VaR paramétrique."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)

        result = calculator.calculate_var_parametric(0.95)

        assert result is not None
        assert result.method == "parametric"
        assert result.var_value > 0

    def test_cvar_calculation(self, calculator, sample_returns):
        """Test le calcul CVaR."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)

        result = calculator.calculate_cvar(0.95)

        assert result is not None
        assert result.cvar_value > 0
        assert result.confidence_level == 0.95

    def test_cvar_greater_than_var(self, calculator, sample_returns):
        """Test que CVaR > VaR (ce qui devrait toujours être vrai)."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)

        var_result = calculator.calculate_var_historical(0.95)
        cvar_result = calculator.calculate_cvar(0.95)

        if var_result and cvar_result:
            assert cvar_result.cvar_value >= var_result.var_value

    def test_correlation_matrix_single_position(self, calculator):
        """Test la matrice de corrélation avec une seule position."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        correlation = calculator.calculate_correlation_matrix()

        assert len(correlation) == 0

    def test_correlation_matrix_two_positions(self, calculator, sample_returns):
        """Test la matrice de corrélation avec deux positions."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.add_position(
            symbol="MSFT",
            quantity=Decimal("100"),
            current_price=Decimal("300"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)
        calculator.update_price_history("MSFT", sample_returns)

        correlation = calculator.calculate_correlation_matrix()

        assert len(correlation) >= 1
        # Les corrélations avec les mêmes returns doivent être proches de 1.0
        if "AAPL" in correlation and "MSFT" in correlation["AAPL"]:
            assert abs(correlation["AAPL"]["MSFT"] - 1.0) < 0.1

    def test_sector_exposure(self, calculator):
        """Test le calcul de l'exposition sectorielle."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.add_position(
            symbol="JPM",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Finance",
        )

        exposure = calculator.calculate_sector_exposure()

        assert "Technology" in exposure
        assert "Finance" in exposure
        assert exposure["Technology"] > 0
        assert exposure["Finance"] > 0

    def test_herfindahl_index_single_position(self, calculator):
        """Test l'indice Herfindahl avec une seule position."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )

        herfindahl = calculator.calculate_herfindahl_index()

        # Une seule position = 100% de concentration
        assert herfindahl == 1.0

    def test_herfindahl_index_two_equal_positions(self, calculator):
        """Test l'indice Herfindahl avec deux positions égales."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.add_position(
            symbol="MSFT",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )

        herfindahl = calculator.calculate_herfindahl_index()

        # 50% + 50% = 0.5^2 + 0.5^2 = 0.5
        assert abs(herfindahl - 0.5) < 0.01

    def test_herfindahl_index_concentrated_portfolio(self, calculator):
        """Test l'indice Herfindahl avec portefeuille concentré."""
        # 80% AAPL, 20% MSFT
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("800"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.add_position(
            symbol="MSFT",
            quantity=Decimal("100"),
            current_price=Decimal("300"),
            sector="Technology",
        )

        herfindahl = calculator.calculate_herfindahl_index()

        # 0.8^2 + 0.2^2 = 0.64 + 0.04 = 0.68
        assert 0.60 < herfindahl < 0.75

    def test_risk_summary(self, calculator, sample_returns):
        """Test le résumé complet du risque."""
        calculator.add_position(
            symbol="AAPL",
            quantity=Decimal("100"),
            current_price=Decimal("150"),
            sector="Technology",
        )
        calculator.update_price_history("AAPL", sample_returns)

        summary = calculator.get_risk_summary()

        assert "portfolio_value" in summary
        assert "total_positions" in summary
        assert "sector_exposure" in summary
        assert "herfindahl_index" in summary
        assert summary["total_positions"] == 1

    def test_edge_case_no_positions(self, calculator):
        """Test cas limite sans positions."""
        var_result = calculator.calculate_var_historical(0.95)
        assert var_result is None

    def test_edge_case_zero_portfolio_value(self):
        """Test cas limite avec valeur portefeuille zéro."""
        calculator = PortfolioRiskCalculator(
            portfolio_value=Decimal("0.01"),  # Très petite valeur
        )
        assert calculator.portfolio_value == Decimal("0.01")
