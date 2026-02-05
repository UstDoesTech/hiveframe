"""
HiveFrame Industry Solutions - Phase 5

Specialized modules for Healthcare, Finance, Retail, Manufacturing,
and Government sectors with industry-specific optimizations.
"""

from .finance import (
    FraudDetector,
    RegulatoryReporter,
    RiskScorer,
)
from .government import (
    ComplianceFramework,
    DataSovereigntyController,
    SecureMultiTenancy,
)
from .healthcare import (
    AuditLogger,
    DataEncryption,
    PrivacyPreservingAnalytics,
)
from .manufacturing import (
    PredictiveMaintenanceSystem,
    QualityControlAnalytics,
    SensorDataProcessor,
)
from .retail import (
    CustomerDataIntegrator,
    DemandForecaster,
    RecommendationEngine,
)

__all__ = [
    # Healthcare
    "DataEncryption",
    "AuditLogger",
    "PrivacyPreservingAnalytics",
    # Finance
    "FraudDetector",
    "RiskScorer",
    "RegulatoryReporter",
    # Retail
    "CustomerDataIntegrator",
    "DemandForecaster",
    "RecommendationEngine",
    # Manufacturing
    "SensorDataProcessor",
    "PredictiveMaintenanceSystem",
    "QualityControlAnalytics",
    # Government
    "DataSovereigntyController",
    "SecureMultiTenancy",
    "ComplianceFramework",
]
