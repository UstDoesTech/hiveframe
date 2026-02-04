"""
HiveFrame for Government - Secure, sovereign data processing

Provides government agencies with data sovereignty controls,
secure multi-tenancy, and compliance frameworks.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import hashlib


class JurisdictionLevel(Enum):
    """Government jurisdiction levels"""
    FEDERAL = "federal"
    STATE = "state"
    LOCAL = "local"
    INTERNATIONAL = "international"


class SecurityLevel(Enum):
    """Data security classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ComplianceStandard(Enum):
    """Government compliance standards"""
    FISMA = "fisma"
    FEDRAMP = "fedramp"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOC2 = "soc2"


@dataclass
class DataAsset:
    """Represents a government data asset"""
    asset_id: str
    jurisdiction: JurisdictionLevel
    security_level: SecurityLevel
    location: str  # Physical location
    owner_agency: str
    metadata: Dict = field(default_factory=dict)


class DataSovereigntyController:
    """
    Enforce data sovereignty and residency requirements.
    
    Ensures data stays within jurisdictional boundaries,
    similar to how bee territories are maintained.
    """
    
    def __init__(self):
        self.assets: Dict[str, DataAsset] = {}
        self.jurisdictions: Dict[str, Set[str]] = {}  # jurisdiction -> locations
        self.sovereignty_violations: List[Dict] = []
        
    def register_jurisdiction(
        self,
        jurisdiction: str,
        allowed_locations: List[str],
    ) -> bool:
        """Register a jurisdiction with allowed data locations"""
        if jurisdiction in self.jurisdictions:
            return False
        
        self.jurisdictions[jurisdiction] = set(allowed_locations)
        return True
    
    def register_asset(
        self,
        asset: DataAsset,
    ) -> bool:
        """Register a data asset with sovereignty constraints"""
        if asset.asset_id in self.assets:
            return False
        
        # Validate location is within jurisdiction
        jurisdiction_key = asset.jurisdiction.value
        if jurisdiction_key in self.jurisdictions:
            allowed_locations = self.jurisdictions[jurisdiction_key]
            if asset.location not in allowed_locations:
                self.sovereignty_violations.append({
                    "asset_id": asset.asset_id,
                    "violation": "location_outside_jurisdiction",
                    "location": asset.location,
                    "jurisdiction": jurisdiction_key,
                    "timestamp": time.time(),
                })
                return False
        
        self.assets[asset.asset_id] = asset
        return True
    
    def check_transfer(
        self,
        asset_id: str,
        target_location: str,
    ) -> Dict:
        """
        Check if a data transfer is allowed.
        
        Returns validation result.
        """
        if asset_id not in self.assets:
            return {"allowed": False, "reason": "asset_not_found"}
        
        asset = self.assets[asset_id]
        jurisdiction_key = asset.jurisdiction.value
        
        if jurisdiction_key not in self.jurisdictions:
            return {"allowed": True, "reason": "no_constraints"}
        
        allowed_locations = self.jurisdictions[jurisdiction_key]
        
        if target_location in allowed_locations:
            return {
                "allowed": True,
                "asset_id": asset_id,
                "target_location": target_location,
            }
        else:
            self.sovereignty_violations.append({
                "asset_id": asset_id,
                "violation": "unauthorized_transfer",
                "target_location": target_location,
                "jurisdiction": jurisdiction_key,
                "timestamp": time.time(),
            })
            return {
                "allowed": False,
                "reason": "location_outside_jurisdiction",
                "asset_id": asset_id,
                "target_location": target_location,
            }
    
    def get_sovereignty_stats(self) -> Dict:
        """Get data sovereignty statistics"""
        return {
            "total_assets": len(self.assets),
            "jurisdictions_managed": len(self.jurisdictions),
            "sovereignty_violations": len(self.sovereignty_violations),
        }


class SecureMultiTenancy:
    """
    Provide secure multi-tenant isolation for government agencies.
    
    Ensures complete isolation between tenants, like separate
    bee colonies sharing the same geographical area.
    """
    
    def __init__(self):
        self.tenants: Dict[str, Dict] = {}
        self.tenant_data: Dict[str, Set[str]] = {}  # tenant_id -> data_ids
        self.access_log: List[Dict] = []
        
    def create_tenant(
        self,
        tenant_id: str,
        agency_name: str,
        security_level: SecurityLevel,
    ) -> bool:
        """Create a new tenant for an agency"""
        if tenant_id in self.tenants:
            return False
        
        self.tenants[tenant_id] = {
            "agency_name": agency_name,
            "security_level": security_level,
            "created_at": time.time(),
            "quota_mb": 10000,  # Default quota
            "used_mb": 0,
        }
        
        self.tenant_data[tenant_id] = set()
        return True
    
    def store_data(
        self,
        tenant_id: str,
        data_id: str,
        size_mb: float,
    ) -> Dict:
        """
        Store data for a tenant with isolation.
        
        Returns storage result.
        """
        if tenant_id not in self.tenants:
            return {"success": False, "reason": "tenant_not_found"}
        
        tenant = self.tenants[tenant_id]
        
        # Check quota
        if tenant["used_mb"] + size_mb > tenant["quota_mb"]:
            return {"success": False, "reason": "quota_exceeded"}
        
        # Store data
        self.tenant_data[tenant_id].add(data_id)
        tenant["used_mb"] += size_mb
        
        # Log access
        self.access_log.append({
            "tenant_id": tenant_id,
            "action": "store",
            "data_id": data_id,
            "timestamp": time.time(),
        })
        
        return {
            "success": True,
            "data_id": data_id,
            "tenant_id": tenant_id,
        }
    
    def access_data(
        self,
        tenant_id: str,
        data_id: str,
    ) -> Dict:
        """
        Access data with tenant isolation enforcement.
        
        Returns access result.
        """
        if tenant_id not in self.tenants:
            return {"success": False, "reason": "tenant_not_found"}
        
        # Check if data belongs to tenant
        if data_id not in self.tenant_data[tenant_id]:
            self.access_log.append({
                "tenant_id": tenant_id,
                "action": "unauthorized_access_attempt",
                "data_id": data_id,
                "timestamp": time.time(),
            })
            return {"success": False, "reason": "unauthorized"}
        
        # Log successful access
        self.access_log.append({
            "tenant_id": tenant_id,
            "action": "access",
            "data_id": data_id,
            "timestamp": time.time(),
        })
        
        return {
            "success": True,
            "data_id": data_id,
            "tenant_id": tenant_id,
        }
    
    def get_tenant_stats(self, tenant_id: str) -> Optional[Dict]:
        """Get statistics for a tenant"""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "agency_name": tenant["agency_name"],
            "security_level": tenant["security_level"].value,
            "data_items": len(self.tenant_data[tenant_id]),
            "used_mb": tenant["used_mb"],
            "quota_mb": tenant["quota_mb"],
            "utilization": tenant["used_mb"] / tenant["quota_mb"],
        }
    
    def get_multitenancy_stats(self) -> Dict:
        """Get overall multi-tenancy statistics"""
        unauthorized_attempts = sum(
            1 for log in self.access_log
            if log["action"] == "unauthorized_access_attempt"
        )
        
        return {
            "total_tenants": len(self.tenants),
            "total_access_log_entries": len(self.access_log),
            "unauthorized_attempts": unauthorized_attempts,
        }


class ComplianceFramework:
    """
    Manage compliance with government regulations.
    
    Tracks compliance requirements and validates adherence,
    similar to how bees maintain hive standards.
    """
    
    def __init__(self):
        self.standards: Dict[ComplianceStandard, Dict] = {}
        self.compliance_checks: List[Dict] = []
        self.non_compliance_issues: List[Dict] = []
        
    def enable_standard(
        self,
        standard: ComplianceStandard,
        requirements: List[str],
    ) -> bool:
        """Enable a compliance standard"""
        if standard in self.standards:
            return False
        
        self.standards[standard] = {
            "requirements": requirements,
            "enabled_at": time.time(),
            "checks_performed": 0,
        }
        
        return True
    
    def perform_compliance_check(
        self,
        standard: ComplianceStandard,
        entity_id: str,
        check_results: Dict[str, bool],
    ) -> Dict:
        """
        Perform a compliance check.
        
        Returns check result.
        """
        if standard not in self.standards:
            return {"status": "error", "reason": "standard_not_enabled"}
        
        standard_info = self.standards[standard]
        requirements = standard_info["requirements"]
        
        # Check each requirement
        passed = []
        failed = []
        
        for requirement in requirements:
            if check_results.get(requirement, False):
                passed.append(requirement)
            else:
                failed.append(requirement)
        
        is_compliant = len(failed) == 0
        
        check = {
            "standard": standard.value,
            "entity_id": entity_id,
            "is_compliant": is_compliant,
            "passed": passed,
            "failed": failed,
            "timestamp": time.time(),
        }
        
        self.compliance_checks.append(check)
        standard_info["checks_performed"] += 1
        
        # Record non-compliance
        if not is_compliant:
            self.non_compliance_issues.append({
                "standard": standard.value,
                "entity_id": entity_id,
                "failed_requirements": failed,
                "timestamp": time.time(),
            })
        
        return check
    
    def get_compliance_status(
        self,
        entity_id: str,
    ) -> Dict:
        """Get compliance status for an entity"""
        entity_checks = [
            c for c in self.compliance_checks
            if c["entity_id"] == entity_id
        ]
        
        if not entity_checks:
            return {
                "entity_id": entity_id,
                "status": "no_checks_performed",
            }
        
        # Get most recent check for each standard
        by_standard = {}
        for check in entity_checks:
            standard = check["standard"]
            if standard not in by_standard or check["timestamp"] > by_standard[standard]["timestamp"]:
                by_standard[standard] = check
        
        overall_compliant = all(c["is_compliant"] for c in by_standard.values())
        
        return {
            "entity_id": entity_id,
            "overall_compliant": overall_compliant,
            "standards_checked": list(by_standard.keys()),
            "latest_checks": by_standard,
        }
    
    def get_compliance_stats(self) -> Dict:
        """Get compliance framework statistics"""
        return {
            "enabled_standards": len(self.standards),
            "total_checks": len(self.compliance_checks),
            "non_compliance_issues": len(self.non_compliance_issues),
        }
