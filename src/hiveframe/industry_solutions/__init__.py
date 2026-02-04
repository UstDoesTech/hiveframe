"""
HiveFrame Industry Solutions - Phase 5

Specialized modules for Healthcare, Finance, Retail, Manufacturing,
and Government sectors with industry-specific optimizations.
"""

from .healthcare import (
    DataEncryption,
    AuditLogger,
    PrivacyPreservingAnalytics,
)
from .finance import (
    FraudDetector,
    RiskScorer,
    RegulatoryReporter,
)
from .retail import (
    CustomerDataIntegrator,
    DemandForecaster,
    RecommendationEngine,
)
from .manufacturing import (
    SensorDataProcessor,
    PredictiveMaintenanceSystem,
    QualityControlAnalytics,
)
from .government import (
    DataSovereigntyController,
    SecureMultiTenancy,
    ComplianceFramework,
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
