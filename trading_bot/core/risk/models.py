"""
Models - Modèles Pydantic v2 pour le système de risk management

Définit toutes les structures de données utilisées par le trading bot.
Utilise Pydantic v2 pour la validation et la sérialisation.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict, field_serializer


class TradeSignalType(str, Enum):
    """Types de signaux de trading."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


class RiskDecision(str, Enum):
    """Décisions possibles du Risk Manager."""
    GO = "GO"
    REDUCE = "REDUCE"
    BLOCK = "BLOCK"


class PositionStatus(str, Enum):
    """Statut d'une position."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


class TradeSignal(BaseModel):
    """
    Modèle représentant un signal de trading.

    Attributes
    ----------
    signal_id : str
        Identifiant unique du signal
    timestamp : datetime
        Quand le signal a été généré
    symbol : str
        Symbole du titre (ex: AAPL, BTC/USD)
    signal_type : TradeSignalType
        Type de signal (BUY, SELL, CLOSE)
    quantity : Decimal
        Quantité proposée
    entry_price : Decimal
        Prix d'entrée proposé
    stop_loss_price : Decimal
        Prix de stop loss
    take_profit_price : Optional[Decimal]
        Prix de take profit (optionnel)
    signal_quality : float
        Score de qualité du signal (0-1)
    confidence : float
        Niveau de confiance (0-1)
    strategy_name : str
        Nom de la stratégie source
    metadata : Dict[str, Any]
        Données supplémentaires
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signal_id": "sig_20240101_001",
                "timestamp": "2024-01-01T10:30:00Z",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "quantity": "100",
                "entry_price": "150.25",
                "stop_loss_price": "145.00",
                "take_profit_price": "160.00",
                "signal_quality": 0.85,
                "confidence": 0.92,
                "strategy_name": "momentum_breakout",
                "metadata": {"timeframe": "1H", "pattern": "bullish_engulfing"}
            }
        }
    )

    signal_id: str = Field(..., description="Identifiant unique du signal")
    timestamp: datetime = Field(..., description="Timestamp du signal")
    symbol: str = Field(..., description="Symbole du titre")
    signal_type: TradeSignalType = Field(..., description="Type de signal")
    quantity: Decimal = Field(..., description="Quantité proposée", decimal_places=8)
    entry_price: Decimal = Field(..., description="Prix d'entrée", decimal_places=8)
    stop_loss_price: Decimal = Field(..., description="Stop loss", decimal_places=8)
    take_profit_price: Optional[Decimal] = Field(None, description="Take profit optionnel", decimal_places=8)
    signal_quality: float = Field(..., ge=0.0, le=1.0, description="Qualité du signal (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance (0-1)")
    strategy_name: str = Field(..., description="Nom de la stratégie")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Données supplémentaires")

    @field_validator("quantity", "entry_price", "stop_loss_price", "take_profit_price", mode="before")
    @classmethod
    def validate_positive_decimal(cls, v: Any) -> Decimal:
        """Valider que les valeurs décimales sont positives."""
        if v is None:
            return None
        d = Decimal(str(v))
        if d <= 0:
            raise ValueError("Les valeurs décimales doivent être positives")
        return d

    @field_validator("stop_loss_price")
    @classmethod
    def validate_stop_loss(cls, v: Decimal, info) -> Decimal:
        """Valider que le stop loss est inférieur au prix d'entrée pour les BUY."""
        if info.data.get("signal_type") == TradeSignalType.BUY:
            entry_price = info.data.get("entry_price")
            if entry_price and v >= entry_price:
                raise ValueError("Stop loss doit être inférieur au prix d'entrée pour BUY")
        return v


class Position(BaseModel):
    """
    Modèle représentant une position ouverte.

    Attributes
    ----------
    position_id : str
        Identifiant unique
    symbol : str
        Symbole du titre
    quantity : Decimal
        Quantité détenue
    entry_price : Decimal
        Prix d'entrée moyen
    current_price : Decimal
        Prix courant du marché
    unrealized_pnl : Decimal
        P&L non réalisé
    realized_pnl : Decimal
        P&L réalisé
    stop_loss : Decimal
        Niveau de stop loss
    take_profit : Optional[Decimal]
        Niveau de take profit
    status : PositionStatus
        Statut de la position
    opened_at : datetime
        Quand la position a été ouverte
    closed_at : Optional[datetime]
        Quand la position a été fermée
    duration_minutes : int
        Durée de la position en minutes
    commission_paid : Decimal
        Commissions payées
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "position_id": "pos_20240101_001",
                "symbol": "AAPL",
                "quantity": "100",
                "entry_price": "150.25",
                "current_price": "152.00",
                "unrealized_pnl": "175.00",
                "realized_pnl": "0",
                "stop_loss": "145.00",
                "take_profit": "160.00",
                "status": "OPEN",
                "opened_at": "2024-01-01T10:30:00Z",
                "duration_minutes": 120,
                "commission_paid": "10.00"
            }
        }
    )

    position_id: str = Field(..., description="ID unique de la position")
    symbol: str = Field(..., description="Symbole du titre")
    quantity: Decimal = Field(..., description="Quantité", decimal_places=8)
    entry_price: Decimal = Field(..., description="Prix d'entrée", decimal_places=8)
    current_price: Decimal = Field(..., description="Prix courant", decimal_places=8)
    unrealized_pnl: Decimal = Field(..., description="P&L non réalisé", decimal_places=2)
    realized_pnl: Decimal = Field(default=Decimal("0"), description="P&L réalisé", decimal_places=2)
    stop_loss: Decimal = Field(..., description="Stop loss", decimal_places=8)
    take_profit: Optional[Decimal] = Field(None, description="Take profit", decimal_places=8)
    status: PositionStatus = Field(default=PositionStatus.OPEN, description="Statut")
    opened_at: datetime = Field(..., description="Ouverture")
    closed_at: Optional[datetime] = Field(None, description="Fermeture")
    duration_minutes: int = Field(default=0, description="Durée en minutes")
    commission_paid: Decimal = Field(default=Decimal("0"), description="Commissions", decimal_places=2)

    @property
    def roi_pct(self) -> float:
        """Calculer le ROI en pourcentage."""
        if self.entry_price == 0:
            return 0.0
        return float((self.unrealized_pnl / (self.entry_price * self.quantity)) * 100)

    @property
    def max_loss_pct(self) -> float:
        """Calculer la perte max potentielle en pourcentage."""
        if self.entry_price == 0:
            return 0.0
        max_loss = self.entry_price - self.stop_loss
        return float((max_loss / self.entry_price) * 100)


class RiskState(BaseModel):
    """
    État actuel du système de risk management.

    Attributes
    ----------
    timestamp : datetime
        Timestamp de l'état
    portfolio_value : Decimal
        Valeur totale du portefeuille
    cash_available : Decimal
        Cash disponible
    total_positions : int
        Nombre de positions ouvertes
    daily_pnl : Decimal
        P&L du jour
    monthly_pnl : Decimal
        P&L du mois
    drawdown_daily_pct : float
        Drawdown journalier en %
    drawdown_monthly_pct : float
        Drawdown mensuel en %
    circuit_breaker_active : bool
        Si le circuit breaker est activé
    circuit_breaker_reason : Optional[str]
        Raison de l'activation
    var_95 : float
        VaR 95%
    var_99 : float
        VaR 99%
    cvar_95 : float
        CVaR 95%
    max_sector_exposure_pct : float
        Exposition sectorielle max
    max_concentration : float
        Concentration max
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-01T16:00:00Z",
                "portfolio_value": "1000000",
                "cash_available": "500000",
                "total_positions": 3,
                "daily_pnl": "5000",
                "monthly_pnl": "25000",
                "drawdown_daily_pct": -2.5,
                "drawdown_monthly_pct": -5.0,
                "circuit_breaker_active": False,
                "var_95": 25000,
                "var_99": 40000,
                "cvar_95": 30000,
                "max_sector_exposure_pct": 28.5,
                "max_concentration": 15.0
            }
        }
    )

    timestamp: datetime = Field(..., description="Timestamp")
    portfolio_value: Decimal = Field(..., description="Valeur du portefeuille", decimal_places=2)
    cash_available: Decimal = Field(..., description="Cash disponible", decimal_places=2)
    total_positions: int = Field(..., ge=0, description="Nombre de positions")
    daily_pnl: Decimal = Field(..., description="P&L journalier", decimal_places=2)
    monthly_pnl: Decimal = Field(..., description="P&L mensuel", decimal_places=2)
    drawdown_daily_pct: float = Field(..., ge=-100, le=0, description="Drawdown journalier %")
    drawdown_monthly_pct: float = Field(..., ge=-100, le=0, description="Drawdown mensuel %")
    circuit_breaker_active: bool = Field(default=False, description="Circuit breaker actif")
    circuit_breaker_reason: Optional[str] = Field(None, description="Raison du circuit breaker")
    var_95: float = Field(..., description="VaR 95%")
    var_99: float = Field(..., description="VaR 99%")
    cvar_95: float = Field(..., description="CVaR 95%")
    max_sector_exposure_pct: float = Field(..., ge=0, le=100, description="Expo secteur max %")
    max_concentration: float = Field(..., ge=0, le=100, description="Concentration max %")


class RiskValidationResult(BaseModel):
    """
    Résultat de la validation d'un trade par le Risk Manager.

    Attributes
    ----------
    signal_id : str
        ID du signal validé
    decision : RiskDecision
        Décision (GO, REDUCE, BLOCK)
    approved_quantity : Decimal
        Quantité approuvée
    rejection_reasons : List[str]
        Raisons de rejet/réduction
    required_checks : Dict[str, bool]
        Résultats des vérifications
    risk_score : float
        Score de risque (0-100)
    timestamp : datetime
        Quand la validation a été effectuée
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signal_id": "sig_20240101_001",
                "decision": "REDUCE",
                "approved_quantity": "50",
                "rejection_reasons": ["Portfolio concentration limit exceeded"],
                "required_checks": {
                    "signal_quality": True,
                    "position_limits": True,
                    "sector_limits": False,
                    "correlation_check": True
                },
                "risk_score": 75.5,
                "timestamp": "2024-01-01T10:30:05Z"
            }
        }
    )

    signal_id: str = Field(..., description="ID du signal")
    decision: RiskDecision = Field(..., description="Décision du Risk Manager")
    approved_quantity: Decimal = Field(..., description="Quantité approuvée", decimal_places=8)
    rejection_reasons: List[str] = Field(default_factory=list, description="Raisons de rejet")
    required_checks: Dict[str, bool] = Field(..., description="Résultats des checks")
    risk_score: float = Field(..., ge=0, le=100, description="Score de risque (0-100)")
    timestamp: datetime = Field(..., description="Timestamp de validation")


class RiskReport(BaseModel):
    """
    Rapport de risque complet généré par le Risk Manager.

    Attributes
    ----------
    report_id : str
        ID unique du rapport
    timestamp : datetime
        Quand le rapport a été généré
    risk_state : RiskState
        État actuel du risque
    positions : List[Position]
        Positions ouvertes
    sector_exposures : Dict[str, float]
        Expositions par secteur
    correlation_matrix : Dict[str, Dict[str, float]]
        Matrice de corrélation
    stress_test_results : Dict[str, float]
        Résultats des stress tests
    recommendations : List[str]
        Recommandations d'action
    alerts : List[str]
        Alertes actives
    metrics : Dict[str, Any]
        Métriques supplémentaires
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report_id": "report_20240101_16_00",
                "timestamp": "2024-01-01T16:00:00Z",
                "risk_state": {
                    "timestamp": "2024-01-01T16:00:00Z",
                    "portfolio_value": "1000000",
                    "cash_available": "500000",
                    "total_positions": 3,
                    "daily_pnl": "5000",
                    "monthly_pnl": "25000",
                    "drawdown_daily_pct": -2.5,
                    "drawdown_monthly_pct": -5.0,
                    "circuit_breaker_active": False,
                    "var_95": 25000,
                    "var_99": 40000,
                    "cvar_95": 30000,
                    "max_sector_exposure_pct": 28.5,
                    "max_concentration": 15.0
                },
                "positions": [],
                "sector_exposures": {"Technology": 28.5, "Finance": 15.0},
                "correlation_matrix": {"AAPL": {"MSFT": 0.65}},
                "stress_test_results": {"market_crash_10pct": -100000},
                "recommendations": [],
                "alerts": [],
                "metrics": {}
            }
        }
    )

    report_id: str = Field(..., description="ID du rapport")
    timestamp: datetime = Field(..., description="Timestamp du rapport")
    risk_state: RiskState = Field(..., description="État du risque")
    positions: List[Position] = Field(default_factory=list, description="Positions")
    sector_exposures: Dict[str, float] = Field(default_factory=dict, description="Expositions sectorielles")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Corrélations")
    stress_test_results: Dict[str, float] = Field(default_factory=dict, description="Stress tests")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    alerts: List[str] = Field(default_factory=list, description="Alertes")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métriques supplémentaires")
