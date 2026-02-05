"""
Partner Certification Program - Build a community of experts

Certification framework, skills verification, and partner directory
to grow the HiveFrame expert ecosystem.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CertificationLevel(Enum):
    """Partner certification levels"""

    ASSOCIATE = "associate"
    PROFESSIONAL = "professional"
    EXPERT = "expert"
    MASTER = "master"


class SkillCategory(Enum):
    """Skill categories"""

    DEVELOPMENT = "development"
    ARCHITECTURE = "architecture"
    OPERATIONS = "operations"
    DATA_SCIENCE = "data_science"
    CONSULTING = "consulting"


@dataclass
class Certification:
    """Represents a certification"""

    cert_id: str
    partner_id: str
    level: CertificationLevel
    skills: List[str]
    issued_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_valid: bool = True


class CertificationFramework:
    """
    Framework for certifying HiveFrame partners.

    Uses swarm-based evaluation where multiple assessors
    contribute to certification decisions like bee scouts
    evaluate new hive locations.
    """

    def __init__(self):
        self.certifications: Dict[str, Certification] = {}
        self.requirements: Dict[CertificationLevel, Dict] = {
            CertificationLevel.ASSOCIATE: {
                "min_experience_months": 3,
                "required_skills": 2,
                "exam_required": True,
            },
            CertificationLevel.PROFESSIONAL: {
                "min_experience_months": 12,
                "required_skills": 5,
                "exam_required": True,
            },
            CertificationLevel.EXPERT: {
                "min_experience_months": 24,
                "required_skills": 8,
                "exam_required": True,
            },
            CertificationLevel.MASTER: {
                "min_experience_months": 48,
                "required_skills": 10,
                "exam_required": True,
            },
        }

    def issue_certification(
        self,
        partner_id: str,
        level: CertificationLevel,
        skills: List[str],
        validity_months: int = 24,
    ) -> Optional[str]:
        """
        Issue a certification to a partner.

        Returns certification ID or None if failed.
        """
        import hashlib

        cert_id = hashlib.sha256(f"{partner_id}{level.value}{time.time()}".encode()).hexdigest()[
            :16
        ]

        expires_at = time.time() + (validity_months * 30 * 24 * 3600)

        cert = Certification(
            cert_id=cert_id,
            partner_id=partner_id,
            level=level,
            skills=skills,
            expires_at=expires_at,
        )

        self.certifications[cert_id] = cert
        return cert_id

    def verify_certification(
        self,
        cert_id: str,
    ) -> Dict:
        """
        Verify a certification.

        Returns verification result.
        """
        if cert_id not in self.certifications:
            return {"valid": False, "reason": "not_found"}

        cert = self.certifications[cert_id]

        # Check expiration
        if cert.expires_at and time.time() > cert.expires_at:
            cert.is_valid = False
            return {
                "valid": False,
                "reason": "expired",
                "cert_id": cert_id,
            }

        return {
            "valid": cert.is_valid,
            "cert_id": cert_id,
            "partner_id": cert.partner_id,
            "level": cert.level.value,
            "skills": cert.skills,
        }

    def revoke_certification(
        self,
        cert_id: str,
        reason: str = "",
    ) -> bool:
        """Revoke a certification"""
        if cert_id not in self.certifications:
            return False

        self.certifications[cert_id].is_valid = False
        return True

    def get_certification_stats(self) -> Dict:
        """Get certification statistics"""
        valid_certs = sum(1 for c in self.certifications.values() if c.is_valid)

        by_level: Dict[str, int] = {}
        for cert in self.certifications.values():
            if cert.is_valid:
                level = cert.level.value
                by_level[level] = by_level.get(level, 0) + 1

        return {
            "total_certifications": len(self.certifications),
            "valid_certifications": valid_certs,
            "by_level": by_level,
        }


@dataclass
class Skill:
    """Represents a verified skill"""

    skill_id: str
    partner_id: str
    skill_name: str
    category: SkillCategory
    proficiency_level: int  # 1-5
    verified_by: List[str] = field(default_factory=list)
    verified_at: float = field(default_factory=time.time)


class SkillsVerification:
    """
    Verify partner skills through peer review.

    Uses swarm wisdom where multiple partners verify
    each other's skills, similar to bee inspection rituals.
    """

    def __init__(self, min_verifiers: int = 3):
        self.skills: Dict[str, Skill] = {}
        self.min_verifiers = min_verifiers

    def add_skill(
        self,
        partner_id: str,
        skill_name: str,
        category: SkillCategory,
        proficiency_level: int,
    ) -> str:
        """Add a skill claim for a partner"""
        import hashlib

        skill_id = hashlib.sha256(f"{partner_id}{skill_name}{time.time()}".encode()).hexdigest()[
            :16
        ]

        skill = Skill(
            skill_id=skill_id,
            partner_id=partner_id,
            skill_name=skill_name,
            category=category,
            proficiency_level=max(1, min(5, proficiency_level)),
        )

        self.skills[skill_id] = skill
        return skill_id

    def verify_skill(
        self,
        skill_id: str,
        verifier_id: str,
    ) -> bool:
        """Add a verification for a skill"""
        if skill_id not in self.skills:
            return False

        skill = self.skills[skill_id]

        if verifier_id == skill.partner_id:
            return False  # Cannot verify own skill

        if verifier_id not in skill.verified_by:
            skill.verified_by.append(verifier_id)

        return True

    def is_skill_verified(self, skill_id: str) -> bool:
        """Check if a skill has sufficient verifications"""
        if skill_id not in self.skills:
            return False

        skill = self.skills[skill_id]
        return len(skill.verified_by) >= self.min_verifiers

    def get_partner_skills(
        self,
        partner_id: str,
        verified_only: bool = True,
    ) -> List[Skill]:
        """Get all skills for a partner"""
        partner_skills = [s for s in self.skills.values() if s.partner_id == partner_id]

        if verified_only:
            partner_skills = [s for s in partner_skills if len(s.verified_by) >= self.min_verifiers]

        return partner_skills

    def get_verification_stats(self) -> Dict:
        """Get skill verification statistics"""
        verified_skills = sum(
            1 for s in self.skills.values() if len(s.verified_by) >= self.min_verifiers
        )

        return {
            "total_skills": len(self.skills),
            "verified_skills": verified_skills,
            "verification_rate": verified_skills / len(self.skills) if self.skills else 0,
        }


@dataclass
class Partner:
    """Represents a certified partner"""

    partner_id: str
    name: str
    organization: str
    location: str
    specializations: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    projects_completed: int = 0
    average_rating: float = 0.0
    joined_at: float = field(default_factory=time.time)


class PartnerDirectory:
    """
    Directory of certified partners.

    Organizes partners like bees organize by role and
    specialization within the hive.
    """

    def __init__(self):
        self.partners: Dict[str, Partner] = {}
        self.by_location: Dict[str, List[str]] = {}
        self.by_specialization: Dict[str, List[str]] = {}

    def register_partner(
        self,
        partner: Partner,
    ) -> bool:
        """Register a partner in the directory"""
        if partner.partner_id in self.partners:
            return False

        self.partners[partner.partner_id] = partner

        # Index by location
        if partner.location not in self.by_location:
            self.by_location[partner.location] = []
        self.by_location[partner.location].append(partner.partner_id)

        # Index by specializations
        for spec in partner.specializations:
            if spec not in self.by_specialization:
                self.by_specialization[spec] = []
            self.by_specialization[spec].append(partner.partner_id)

        return True

    def find_partners(
        self,
        location: Optional[str] = None,
        specialization: Optional[str] = None,
        min_rating: float = 0.0,
    ) -> List[Partner]:
        """Find partners matching criteria"""
        results = list(self.partners.values())

        if location:
            location_partners = self.by_location.get(location, [])
            results = [p for p in results if p.partner_id in location_partners]

        if specialization:
            spec_partners = self.by_specialization.get(specialization, [])
            results = [p for p in results if p.partner_id in spec_partners]

        if min_rating > 0:
            results = [p for p in results if p.average_rating >= min_rating]

        # Sort by rating
        results.sort(key=lambda p: p.average_rating, reverse=True)

        return results

    def update_partner_rating(
        self,
        partner_id: str,
        rating: float,
    ) -> bool:
        """Update partner's average rating"""
        if partner_id not in self.partners:
            return False

        partner = self.partners[partner_id]

        # Simple moving average
        total_projects = partner.projects_completed + 1
        partner.average_rating = (
            partner.average_rating * partner.projects_completed + rating
        ) / total_projects
        partner.projects_completed = total_projects

        return True

    def get_directory_stats(self) -> Dict:
        """Get partner directory statistics"""
        return {
            "total_partners": len(self.partners),
            "locations": len(self.by_location),
            "specializations": len(self.by_specialization),
            "total_projects": sum(p.projects_completed for p in self.partners.values()),
        }
