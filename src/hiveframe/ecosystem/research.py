"""
Academic Research Grants - Fund swarm intelligence research

Grant application system, research project tracking, and
publication management to advance swarm intelligence research.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class GrantStatus(Enum):
    """Grant application status"""

    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    COMPLETED = "completed"


class ResearchPhase(Enum):
    """Research project phases"""

    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    WRITING = "writing"
    COMPLETED = "completed"


@dataclass
class Grant:
    """Represents a research grant"""

    grant_id: str
    researcher_id: str
    institution: str
    title: str
    description: str
    amount_requested: float
    status: GrantStatus = GrantStatus.SUBMITTED
    submitted_at: float = field(default_factory=time.time)
    research_area: str = "swarm_intelligence"
    duration_months: int = 12


class GrantApplication:
    """
    System for submitting and managing research grants.

    Uses peer review process inspired by bee collective
    decision-making where multiple reviewers evaluate proposals.
    """

    def __init__(self, total_budget: float = 1000000.0):
        self.grants: Dict[str, Grant] = {}
        self.total_budget = total_budget
        self.allocated_budget = 0.0
        self.reviews: Dict[str, List[Dict]] = {}  # grant_id -> reviews

    def submit_grant(
        self,
        researcher_id: str,
        institution: str,
        title: str,
        description: str,
        amount_requested: float,
        duration_months: int = 12,
    ) -> Optional[str]:
        """
        Submit a grant application.

        Returns grant ID or None if failed.
        """
        import hashlib

        if amount_requested <= 0 or amount_requested > self.total_budget:
            return None

        grant_id = hashlib.sha256(f"{researcher_id}{title}{time.time()}".encode()).hexdigest()[:16]

        grant = Grant(
            grant_id=grant_id,
            researcher_id=researcher_id,
            institution=institution,
            title=title,
            description=description,
            amount_requested=amount_requested,
            duration_months=duration_months,
        )

        self.grants[grant_id] = grant
        self.reviews[grant_id] = []

        return grant_id

    def add_review(
        self,
        grant_id: str,
        reviewer_id: str,
        score: int,  # 1-10
        comments: str = "",
    ) -> bool:
        """Add a peer review for a grant"""
        if grant_id not in self.grants:
            return False

        if score < 1 or score > 10:
            return False

        review = {
            "reviewer_id": reviewer_id,
            "score": score,
            "comments": comments,
            "timestamp": time.time(),
        }

        self.reviews[grant_id].append(review)

        # Auto-update status if enough reviews
        grant = self.grants[grant_id]
        if len(self.reviews[grant_id]) >= 3:  # Require 3 reviews
            avg_score = sum(r["score"] for r in self.reviews[grant_id]) / len(
                self.reviews[grant_id]
            )
            grant.status = GrantStatus.UNDER_REVIEW

        return True

    def approve_grant(
        self,
        grant_id: str,
    ) -> bool:
        """Approve a grant application"""
        if grant_id not in self.grants:
            return False

        grant = self.grants[grant_id]

        # Check budget availability
        if self.allocated_budget + grant.amount_requested > self.total_budget:
            return False

        grant.status = GrantStatus.APPROVED
        self.allocated_budget += grant.amount_requested

        return True

    def activate_grant(
        self,
        grant_id: str,
    ) -> bool:
        """Activate an approved grant"""
        if grant_id not in self.grants:
            return False

        grant = self.grants[grant_id]

        if grant.status != GrantStatus.APPROVED:
            return False

        grant.status = GrantStatus.ACTIVE
        return True

    def complete_grant(
        self,
        grant_id: str,
    ) -> bool:
        """Mark a grant as completed"""
        if grant_id not in self.grants:
            return False

        grant = self.grants[grant_id]
        grant.status = GrantStatus.COMPLETED

        return True

    def get_grant_stats(self) -> Dict:
        """Get grant application statistics"""
        approved = sum(
            1
            for g in self.grants.values()
            if g.status in (GrantStatus.APPROVED, GrantStatus.ACTIVE, GrantStatus.COMPLETED)
        )

        return {
            "total_applications": len(self.grants),
            "approved_grants": approved,
            "total_budget": self.total_budget,
            "allocated_budget": self.allocated_budget,
            "remaining_budget": self.total_budget - self.allocated_budget,
        }


@dataclass
class ResearchProject:
    """Represents a research project"""

    project_id: str
    grant_id: str
    title: str
    phase: ResearchPhase = ResearchPhase.PLANNING
    progress: float = 0.0  # 0.0 to 1.0
    milestones: List[Dict] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


class ResearchProjectTracker:
    """
    Track progress of funded research projects.

    Monitors project health like bees monitor hive conditions,
    enabling early intervention if projects struggle.
    """

    def __init__(self):
        self.projects: Dict[str, ResearchProject] = {}
        self.milestones: Dict[str, List[Dict]] = {}  # project_id -> milestones

    def create_project(
        self,
        grant_id: str,
        title: str,
    ) -> str:
        """Create a new research project"""
        import hashlib

        project_id = hashlib.sha256(f"{grant_id}{title}{time.time()}".encode()).hexdigest()[:16]

        project = ResearchProject(
            project_id=project_id,
            grant_id=grant_id,
            title=title,
        )

        self.projects[project_id] = project
        self.milestones[project_id] = []

        return project_id

    def add_milestone(
        self,
        project_id: str,
        milestone_name: str,
        target_date: float,
    ) -> bool:
        """Add a milestone to a project"""
        if project_id not in self.projects:
            return False

        milestone = {
            "name": milestone_name,
            "target_date": target_date,
            "completed": False,
            "completed_date": None,
        }

        self.milestones[project_id].append(milestone)
        return True

    def complete_milestone(
        self,
        project_id: str,
        milestone_name: str,
    ) -> bool:
        """Mark a milestone as complete"""
        if project_id not in self.milestones:
            return False

        for milestone in self.milestones[project_id]:
            if milestone["name"] == milestone_name and not milestone["completed"]:
                milestone["completed"] = True
                milestone["completed_date"] = time.time()

                # Update project progress
                project = self.projects[project_id]
                total = len(self.milestones[project_id])
                completed = sum(1 for m in self.milestones[project_id] if m["completed"])
                project.progress = completed / total if total > 0 else 0.0

                return True

        return False

    def update_phase(
        self,
        project_id: str,
        phase: ResearchPhase,
    ) -> bool:
        """Update project phase"""
        if project_id not in self.projects:
            return False

        self.projects[project_id].phase = phase
        return True

    def get_project_health(
        self,
        project_id: str,
    ) -> Optional[Dict]:
        """
        Assess project health.

        Returns health assessment or None if not found.
        """
        if project_id not in self.projects:
            return None

        project = self.projects[project_id]

        # Check milestone delays
        delayed_milestones = 0
        for milestone in self.milestones[project_id]:
            if not milestone["completed"] and time.time() > milestone["target_date"]:
                delayed_milestones += 1

        # Determine health status
        if delayed_milestones == 0:
            status = "healthy"
        elif delayed_milestones <= 2:
            status = "warning"
        else:
            status = "at_risk"

        return {
            "project_id": project_id,
            "status": status,
            "progress": project.progress,
            "phase": project.phase.value,
            "delayed_milestones": delayed_milestones,
        }

    def get_tracking_stats(self) -> Dict:
        """Get project tracking statistics"""
        completed = sum(1 for p in self.projects.values() if p.phase == ResearchPhase.COMPLETED)

        avg_progress = (
            sum(p.progress for p in self.projects.values()) / len(self.projects)
            if self.projects
            else 0.0
        )

        return {
            "total_projects": len(self.projects),
            "completed_projects": completed,
            "average_progress": avg_progress,
        }


@dataclass
class Publication:
    """Represents a research publication"""

    publication_id: str
    project_id: str
    title: str
    authors: List[str]
    venue: str
    published_date: float
    doi: Optional[str] = None
    citations: int = 0


class PublicationManager:
    """
    Manage research publications.

    Tracks publication output and impact, helping the swarm
    identify valuable research contributions.
    """

    def __init__(self):
        self.publications: Dict[str, Publication] = {}
        self.by_project: Dict[str, List[str]] = {}  # project_id -> publication_ids

    def add_publication(
        self,
        project_id: str,
        title: str,
        authors: List[str],
        venue: str,
        published_date: float,
        doi: Optional[str] = None,
    ) -> str:
        """Add a publication"""
        import hashlib

        pub_id = hashlib.sha256(f"{project_id}{title}{time.time()}".encode()).hexdigest()[:16]

        publication = Publication(
            publication_id=pub_id,
            project_id=project_id,
            title=title,
            authors=authors,
            venue=venue,
            published_date=published_date,
            doi=doi,
        )

        self.publications[pub_id] = publication

        if project_id not in self.by_project:
            self.by_project[project_id] = []
        self.by_project[project_id].append(pub_id)

        return pub_id

    def update_citations(
        self,
        publication_id: str,
        citations: int,
    ) -> bool:
        """Update citation count for a publication"""
        if publication_id not in self.publications:
            return False

        self.publications[publication_id].citations = citations
        return True

    def get_project_publications(
        self,
        project_id: str,
    ) -> List[Publication]:
        """Get all publications for a project"""
        pub_ids = self.by_project.get(project_id, [])
        return [self.publications[pid] for pid in pub_ids]

    def get_publication_stats(self) -> Dict:
        """Get publication statistics"""
        total_citations = sum(p.citations for p in self.publications.values())

        return {
            "total_publications": len(self.publications),
            "total_citations": total_citations,
            "projects_with_publications": len(self.by_project),
        }
