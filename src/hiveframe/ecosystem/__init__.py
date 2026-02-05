"""
HiveFrame Open Ecosystem - Phase 5

Components for marketplace, partner certification, academic research,
and open source governance to build a thriving community.
"""

from .governance import (
    ContributorGuidelines,
    DecisionMakingProcess,
    GovernanceModel,
)
from .marketplace import (
    AppRegistry,
    PluginSystem,
    VersionManager,
)
from .partners import (
    CertificationFramework,
    PartnerDirectory,
    SkillsVerification,
)
from .research import (
    GrantApplication,
    PublicationManager,
    ResearchProjectTracker,
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
