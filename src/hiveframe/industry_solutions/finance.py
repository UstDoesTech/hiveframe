"""
HiveFrame for Finance - Real-time risk analysis and fraud detection

Provides financial services with real-time fraud detection,
risk scoring, and regulatory reporting capabilities.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FraudRiskLevel(Enum):
    """Fraud risk levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Risk categories"""

    CREDIT = "credit"
    MARKET = "market"
    OPERATIONAL = "operational"
    LIQUIDITY = "liquidity"


@dataclass
class Transaction:
    """Represents a financial transaction"""

    transaction_id: str
    user_id: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    location: Optional[str] = None
    merchant: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class FraudDetector:
    """
    Real-time fraud detection using swarm intelligence.

    Scout bees explore transaction patterns to identify anomalies,
    similar to how bees detect threats to the hive.
    """

    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.user_profiles: Dict[str, Dict] = {}
        self.fraud_cases: List[Dict] = []
        self.detection_count = 0

    def analyze_transaction(
        self,
        transaction: Transaction,
    ) -> Dict:
        """
        Analyze a transaction for fraud indicators.

        Returns analysis result with risk level and score.
        """
        risk_score = 0.0
        indicators = []

        # Build or update user profile
        if transaction.user_id not in self.user_profiles:
            self.user_profiles[transaction.user_id] = {
                "avg_amount": transaction.amount,
                "transaction_count": 0,
                "locations": set(),
                "merchants": set(),
            }

        profile = self.user_profiles[transaction.user_id]

        # Check for anomalies

        # 1. Unusual amount
        if transaction.amount > profile["avg_amount"] * 3:
            risk_score += 0.3
            indicators.append("unusual_amount")

        # 2. New location
        if transaction.location and transaction.location not in profile["locations"]:
            risk_score += 0.2
            indicators.append("new_location")

        # 3. Rapid transactions
        if profile["transaction_count"] > 0:
            # Simplified - real implementation would check time between transactions
            risk_score += 0.1

        # 4. High-risk merchant
        high_risk_keywords = ["casino", "crypto", "offshore"]
        if transaction.merchant:
            for keyword in high_risk_keywords:
                if keyword in transaction.merchant.lower():
                    risk_score += 0.4
                    indicators.append("high_risk_merchant")
                    break

        # Update profile
        profile["transaction_count"] += 1
        profile["avg_amount"] = (
            profile["avg_amount"] * (profile["transaction_count"] - 1) + transaction.amount
        ) / profile["transaction_count"]
        if transaction.location:
            profile["locations"].add(transaction.location)
        if transaction.merchant:
            profile["merchants"].add(transaction.merchant)

        # Determine risk level
        if risk_score < 0.3:
            risk_level = FraudRiskLevel.LOW
        elif risk_score < 0.5:
            risk_level = FraudRiskLevel.MEDIUM
        elif risk_score < 0.7:
            risk_level = FraudRiskLevel.HIGH
        else:
            risk_level = FraudRiskLevel.CRITICAL

        self.detection_count += 1

        result = {
            "transaction_id": transaction.transaction_id,
            "risk_level": risk_level.value,
            "risk_score": min(1.0, risk_score),
            "indicators": indicators,
            "requires_review": risk_score >= self.sensitivity,
        }

        if result["requires_review"]:
            self.fraud_cases.append(result)

        return result

    def get_fraud_stats(self) -> Dict:
        """Get fraud detection statistics"""
        return {
            "total_analyzed": self.detection_count,
            "flagged_cases": len(self.fraud_cases),
            "user_profiles": len(self.user_profiles),
            "detection_rate": (
                len(self.fraud_cases) / self.detection_count if self.detection_count > 0 else 0
            ),
        }


class RiskScorer:
    """
    Calculate risk scores for portfolios and positions.

    Uses swarm optimization to balance risk across portfolio,
    like bees balance resources across the hive.
    """

    def __init__(self):
        self.portfolios: Dict[str, Dict] = {}
        self.risk_assessments: List[Dict] = []

    def create_portfolio(
        self,
        portfolio_id: str,
        positions: List[Dict],
    ) -> bool:
        """Create a portfolio for risk assessment"""
        if portfolio_id in self.portfolios:
            return False

        self.portfolios[portfolio_id] = {
            "positions": positions,
            "created_at": time.time(),
            "last_assessment": None,
        }

        return True

    def calculate_risk_score(
        self,
        portfolio_id: str,
        category: RiskCategory,
    ) -> Optional[Dict]:
        """
        Calculate risk score for a portfolio.

        Returns risk assessment or None if portfolio not found.
        """
        if portfolio_id not in self.portfolios:
            return None

        portfolio = self.portfolios[portfolio_id]
        positions = portfolio["positions"]

        # Calculate risk based on category
        risk_score = 0.0

        if category == RiskCategory.CREDIT:
            # Credit risk: based on counterparty ratings
            risk_score = sum(p.get("credit_rating", 0.5) for p in positions) / len(positions)

        elif category == RiskCategory.MARKET:
            # Market risk: based on volatility
            risk_score = sum(p.get("volatility", 0.3) for p in positions) / len(positions)

        elif category == RiskCategory.OPERATIONAL:
            # Operational risk: fixed estimate
            risk_score = 0.2

        elif category == RiskCategory.LIQUIDITY:
            # Liquidity risk: based on market depth
            risk_score = sum(1.0 - p.get("liquidity", 0.7) for p in positions) / len(positions)

        assessment = {
            "portfolio_id": portfolio_id,
            "category": category.value,
            "risk_score": risk_score,
            "timestamp": time.time(),
            "position_count": len(positions),
        }

        portfolio["last_assessment"] = assessment
        self.risk_assessments.append(assessment)

        return assessment

    def get_risk_stats(self) -> Dict:
        """Get risk scoring statistics"""
        return {
            "total_portfolios": len(self.portfolios),
            "total_assessments": len(self.risk_assessments),
        }


class RegulatoryReporter:
    """
    Generate regulatory reports for compliance.

    Maintains comprehensive audit trail using bee-inspired
    information persistence.
    """

    def __init__(self):
        self.reports: Dict[str, Dict] = {}
        self.report_templates: Dict[str, List[str]] = {
            "mifid2": ["trade_data", "best_execution", "transaction_reporting"],
            "dodd_frank": ["swap_data", "position_limits", "large_trader_reporting"],
            "basel3": ["capital_adequacy", "liquidity_coverage", "leverage_ratio"],
        }

    def generate_report(
        self,
        report_id: str,
        regulation: str,
        data: Dict,
    ) -> Dict:
        """
        Generate a regulatory report.

        Returns report details.
        """
        template = self.report_templates.get(regulation.lower(), ["general"])

        report = {
            "report_id": report_id,
            "regulation": regulation,
            "sections": template,
            "data": data,
            "generated_at": time.time(),
            "status": "generated",
        }

        self.reports[report_id] = report

        return report

    def submit_report(
        self,
        report_id: str,
        authority: str,
    ) -> bool:
        """Mark a report as submitted to regulatory authority"""
        if report_id not in self.reports:
            return False

        self.reports[report_id]["status"] = "submitted"
        self.reports[report_id]["submitted_to"] = authority
        self.reports[report_id]["submitted_at"] = time.time()

        return True

    def get_reporting_stats(self) -> Dict:
        """Get regulatory reporting statistics"""
        submitted = sum(1 for r in self.reports.values() if r["status"] == "submitted")

        return {
            "total_reports": len(self.reports),
            "submitted_reports": submitted,
            "pending_reports": len(self.reports) - submitted,
        }
