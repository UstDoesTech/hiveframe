"""
HiveFrame Lakehouse Architecture (Phase 3)
==========================================

Unity Hive Catalog and lakehouse infrastructure for enterprise data management.

This module provides:
- Unity Hive Catalog: Unified governance and data discovery
- Fine-grained access control
- Data lineage tracking
- Automatic PII detection
- Delta Sharing: Secure data sharing across organizations

Key Components:
    - UnityHiveCatalog: Centralized metadata and governance
    - AccessControl: Fine-grained permissions management
    - LineageTracker: Track data transformations and dependencies
    - PIIDetector: Automatic detection and classification of sensitive data
    - DeltaSharing: Secure data sharing protocol

Example:
    from hiveframe.lakehouse import UnityHiveCatalog, AccessControl

    catalog = UnityHiveCatalog()
    catalog.register_table("users", schema, location="/data/users")

    acl = AccessControl(catalog)
    acl.grant("user@example.com", "users", ["SELECT", "INSERT"])
"""

__all__ = [
    "UnityHiveCatalog",
    "AccessControl",
    "LineageTracker",
    "PIIDetector",
    "DeltaSharing",
    "PermissionType",
    "PIISensitivity",
    "TableMetadata",
    "LineageNode",
    "ShareAccessLevel",
    "Share",
    "ShareRecipient",
]

from .catalog import (
    AccessControl,
    LineageNode,
    LineageTracker,
    PermissionType,
    PIIDetector,
    PIISensitivity,
    TableMetadata,
    UnityHiveCatalog,
)
from .delta_sharing import (
    DeltaSharing,
    Share,
    ShareAccessLevel,
    ShareRecipient,
)
