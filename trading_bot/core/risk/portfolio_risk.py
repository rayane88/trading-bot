"""
Portfolio Risk - Calcul des risques de portefeuille

Implémente :
- VaR 95% et 99% (historique + parametrique)
- CVaR (Expected Shortfall)
- Matrice de corrélation
- Exposition sectorielle
- Concentration risk
"""

import logging
import json
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from pydantic import BaseModel, Field


class VaRResult(BaseModel):
    """Résultat du calcul VaR."""
    confidence_level: float = Field(..., description="Niveau de confiance")
    var_value: float = Field(..., description="Valeur à risque")
    var_pct: float = Field(..., description="VaR en % du portefeuille")
    method: str = Field(..., description="Méthode utilisée (historical/parametric)")
    timestamp: datetime = Field(..., description="Quand le calcul a été effectué")


class CVaRResult(BaseModel):
    """Résultat du calcul CVaR."""
    confidence_level: float = Field(..., description="Niveau de confiance")
    cvar_value: float = Field(..., description="Perte moyenne dépassant VaR")
    cvar_pct: float = Field(..., description="CVaR en % du portefeuille")
    tail_average_return: float = Field(..., description="Moyenne des returns queue")
    timestamp: datetime = Field(..., description="Quand le calcul a été effectué")


class PortfolioRiskCalculator:
    """
    Calculateur de risque de portefeuille.

    Calcule : VaR, CVaR, corrélations, concentration, exposition sectorielle.
    """

    def __init__(
        self,
        portfolio_value: Decimal,
        var_lookback_days: int = 252,
        var_min_observations: int = 20,
        correlation_lookback_days: int = 60,
        correlation_threshold: float = 0.7,
        max_sector_exposure_pct: float = 30.0,
        herfindahl_threshold: float = 0.15,
    ) -> None:
        """
        Initialiser le calculateur de risque de portefeuille.

        Parameters
        ----------
        portfolio_value : Decimal
            Valeur actuelle du portefeuille
        var_lookback_days : int
            Nombre de jours historiques pour VaR
        var_min_observations : int
            Nombre minimal d'observations
        correlation_lookback_days : int
            Jours pour calculer les corrélations
        correlation_threshold : float
            Seuil d'alerte pour corrélations
        max_sector_exposure_pct : float
            Max exposition sectorielle en %
        herfindahl_threshold : float
            Seuil Herfindahl pour concentration
        """
        self.portfolio_value = portfolio_value
        self.var_lookback_days = var_lookback_days
        self.var_min_observations = var_min_observations
        self.correlation_lookback_days = correlation_lookback_days
        self.correlation_threshold = correlation_threshold
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.herfindahl_threshold = herfindahl_threshold

        self.positions: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.sector_map: Dict[str, str] = {}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_position(
        self,
        symbol: str,
        quantity: Decimal,
        current_price: Decimal,
        sector: str,
    ) -> None:
        """
        Ajouter une position au portefeuille.

        Parameters
        ----------
        symbol : str
            Symbole du titre
        quantity : Decimal
            Quantité détenue
        current_price : Decimal
            Prix courant
        sector : str
            Secteur du titre
        """
        position_value = float(quantity * current_price)
        position_pct = (position_value / float(self.portfolio_value)) * 100

        self.positions[symbol] = {
            "quantity": float(quantity),
            "current_price": float(current_price),
            "position_value": position_value,
            "position_pct": position_pct,
            "sector": sector,
        }
        self.sector_map[symbol] = sector

        self.logger.debug(json.dumps({
            "event": "position_added",
            "symbol": symbol,
            "position_pct": position_pct,
            "sector": sector
        }))

    def update_price_history(
        self,
        symbol: str,
        returns: List[float],
    ) -> None:
        """
        Mettre à jour l'historique des prix/returns.

        Parameters
        ----------
        symbol : str
            Symbole du titre
        returns : List[float]
            Liste des returns historiques
        """
        self.price_history[symbol] = returns[-self.var_lookback_days:]

    def calculate_var_historical(
        self,
        confidence_level: float = 0.95,
    ) -> Optional[VaRResult]:
        """
        Calculer la VaR avec la méthode historique.

        Parameters
        ----------
        confidence_level : float
            Niveau de confiance (0.95 ou 0.99)

        Returns
        -------
        Optional[VaRResult]
            Résultat VaR ou None si données insuffisantes

        Notes
        -----
        VaR Historique utilise le percentile de la distribution empirique des losses.
        """
        if not self.positions:
            return None

        portfolio_returns = self._calculate_portfolio_returns()

        if len(portfolio_returns) < self.var_min_observations:
            self.logger.warning(json.dumps({
                "event": "insufficient_observations_var",
                "observations": len(portfolio_returns),
                "required": self.var_min_observations
            }))
            return None

        returns_array = np.array(portfolio_returns)
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns_array, percentile)
        var_value = float(self.portfolio_value) * abs(var_return)
        var_pct = abs(var_return) * 100

        result = VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            var_pct=var_pct,
            method="historical",
            timestamp=datetime.utcnow(),
        )

        self.logger.info(json.dumps({
            "event": "var_calculated_historical",
            "confidence_level": confidence_level,
            "var_pct": var_pct,
            "var_value": var_value
        }))

        return result

    def calculate_var_parametric(
        self,
        confidence_level: float = 0.95,
    ) -> Optional[VaRResult]:
        """
        Calculer la VaR avec la méthode paramétrique (normale).

        Parameters
        ----------
        confidence_level : float
            Niveau de confiance

        Returns
        -------
        Optional[VaRResult]
            Résultat VaR

        Notes
        -----
        Assume une distribution normale des returns.
        VaR = Mean - (StdDev * Z-score)
        """
        if not self.positions:
            return None

        portfolio_returns = self._calculate_portfolio_returns()

        if len(portfolio_returns) < self.var_min_observations:
            return None

        returns_array = np.array(portfolio_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var_return = mean_return - (z_score * std_return)
        var_value = float(self.portfolio_value) * abs(var_return)
        var_pct = abs(var_return) * 100

        result = VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            var_pct=var_pct,
            method="parametric",
            timestamp=datetime.utcnow(),
        )

        self.logger.info(json.dumps({
            "event": "var_calculated_parametric",
            "confidence_level": confidence_level,
            "var_pct": var_pct,
            "var_value": var_value
        }))

        return result

    def calculate_cvar(
        self,
        confidence_level: float = 0.95,
    ) -> Optional[CVaRResult]:
        """
        Calculer la CVaR (Conditional VaR / Expected Shortfall).

        Parameters
        ----------
        confidence_level : float
            Niveau de confiance

        Returns
        -------
        Optional[CVaRResult]
            Résultat CVaR

        Notes
        -----
        CVaR = Moyenne des returns pire que VaR
        """
        if not self.positions:
            return None

        portfolio_returns = self._calculate_portfolio_returns()

        if len(portfolio_returns) < self.var_min_observations:
            return None

        returns_array = np.array(portfolio_returns)
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns_array, percentile)
        tail_returns = returns_array[returns_array <= var_return]

        if len(tail_returns) == 0:
            tail_returns = returns_array[returns_array <= np.min(returns_array)]

        cvar_return = np.mean(tail_returns)
        cvar_value = float(self.portfolio_value) * abs(cvar_return)
        cvar_pct = abs(cvar_return) * 100

        result = CVaRResult(
            confidence_level=confidence_level,
            cvar_value=cvar_value,
            cvar_pct=cvar_pct,
            tail_average_return=float(cvar_return),
            timestamp=datetime.utcnow(),
        )

        self.logger.info(json.dumps({
            "event": "cvar_calculated",
            "confidence_level": confidence_level,
            "cvar_pct": cvar_pct,
            "cvar_value": cvar_value
        }))

        return result

    def _calculate_portfolio_returns(self) -> List[float]:
        """
        Calculer les returns du portefeuille.

        Returns
        -------
        List[float]
            Liste des returns du portefeuille
        """
        if not self.positions or not self.price_history:
            return []

        max_history_len = min(
            [len(returns) for returns in self.price_history.values()]
            if self.price_history.values()
            else [0]
        )

        if max_history_len == 0:
            return []

        portfolio_returns = []
        symbols = list(self.positions.keys())

        for i in range(1, max_history_len):
            daily_return = 0.0
            for symbol in symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) > i:
                    position_weight = self.positions[symbol]["position_pct"] / 100
                    symbol_return = self.price_history[symbol][i]
                    daily_return += position_weight * symbol_return
            portfolio_returns.append(daily_return)

        return portfolio_returns

    def calculate_correlation_matrix(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculer la matrice de corrélation entre positions.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Matrice de corrélation
        """
        if not self.positions or len(self.positions) < 2:
            return {}

        symbols = list(self.positions.keys())
        correlation_matrix = {}

        for i, sym1 in enumerate(symbols):
            correlation_matrix[sym1] = {}
            if sym1 not in self.price_history:
                continue

            for sym2 in symbols:
                if sym2 not in self.price_history:
                    continue

                returns1 = np.array(self.price_history[sym1][-self.correlation_lookback_days:])
                returns2 = np.array(self.price_history[sym2][-self.correlation_lookback_days:])

                if len(returns1) > 1 and len(returns2) > 1:
                    correlation = float(np.corrcoef(returns1, returns2)[0, 1])
                    correlation_matrix[sym1][sym2] = correlation

                    if abs(correlation) > self.correlation_threshold and sym1 != sym2:
                        self.logger.warning(json.dumps({
                            "event": "high_correlation_detected",
                            "symbol1": sym1,
                            "symbol2": sym2,
                            "correlation": correlation
                        }))

        return correlation_matrix

    def calculate_sector_exposure(self) -> Dict[str, float]:
        """
        Calculer l'exposition par secteur.

        Returns
        -------
        Dict[str, float]
            Exposition par secteur en %
        """
        sector_exposure = defaultdict(float)

        for symbol, position in self.positions.items():
            sector = self.sector_map.get(symbol, "Unknown")
            sector_exposure[sector] += position["position_pct"]

        for sector in sector_exposure:
            if sector_exposure[sector] > self.max_sector_exposure_pct:
                self.logger.warning(json.dumps({
                    "event": "sector_exposure_exceeded",
                    "sector": sector,
                    "exposure_pct": sector_exposure[sector],
                    "max_pct": self.max_sector_exposure_pct
                }))

        return dict(sector_exposure)

    def calculate_herfindahl_index(self) -> float:
        """
        Calculer l'indice Herfindahl pour mesurer la concentration.

        Returns
        -------
        float
            Indice Herfindahl (0-1)

        Notes
        -----
        HHI = Sum(position_weight^2)
        0 = diversifié, 1 = concentré (une seule position)
        """
        if not self.positions:
            return 0.0

        herfindahl = 0.0
        for position in self.positions.values():
            weight = position["position_pct"] / 100
            herfindahl += weight ** 2

        if herfindahl > self.herfindahl_threshold:
            self.logger.warning(json.dumps({
                "event": "high_concentration",
                "herfindahl_index": herfindahl,
                "threshold": self.herfindahl_threshold
            }))

        return herfindahl

    def get_risk_summary(self) -> Dict:
        """
        Obtenir un résumé complet du risque du portefeuille.

        Returns
        -------
        Dict
            Résumé du risque
        """
        var_95 = self.calculate_var_historical(0.95)
        var_99 = self.calculate_var_historical(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        correlation_matrix = self.calculate_correlation_matrix()
        sector_exposure = self.calculate_sector_exposure()
        herfindahl = self.calculate_herfindahl_index()

        return {
            "portfolio_value": str(self.portfolio_value),
            "total_positions": len(self.positions),
            "var_95": var_95.var_value if var_95 else None,
            "var_99": var_99.var_value if var_99 else None,
            "cvar_95": cvar_95.cvar_value if cvar_95 else None,
            "sector_exposure": sector_exposure,
            "herfindahl_index": herfindahl,
            "correlation_matrix": correlation_matrix,
            "timestamp": datetime.utcnow().isoformat(),
        }
