"""
HiveFrame Open Ecosystem - Phase 5

Components for marketplace, partner certification, academic research,
and open source governance to build a thriving community.
"""

from .marketplace import (
    PluginSystem,
    AppRegistry,
    VersionManager,
)
from .partners import (
    CertificationFramework,
    SkillsVerification,
    PartnerDirectory,
)
from .research import (
    GrantApplication,
    ResearchProjectTracker,
    PublicationManager,
)
from .governance import (
    GovernanceModel,
    ContributorGuidelines,
    DecisionMakingProcess,
)

__all__ = [
    # Marketplace
    "PluginSystem",
    "AppRegistry",
    "VersionManager",
    # Partners
    "CertificationFramework",
    "SkillsVerification",
    "PartnerDirectory",
    # Research
    "GrantApplication",
    "ResearchProjectTracker",
    "PublicationManager",
    # Governance
    "GovernanceModel",
    "ContributorGuidelines",
    "DecisionMakingProcess",
]
