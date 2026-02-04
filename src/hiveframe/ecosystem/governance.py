"""
Open Source Governance - Foundation-based project governance

Governance model, contributor guidelines, and decision-making process
for community-driven HiveFrame development.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class ProposalType(Enum):
    """Types of governance proposals"""
    FEATURE = "feature"
    POLICY = "policy"
    BUDGET = "budget"
    MEMBERSHIP = "membership"
    PROCESS = "process"


class ProposalStatus(Enum):
    """Proposal lifecycle status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class VoteOption(Enum):
    """Vote options"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class Proposal:
    """Represents a governance proposal"""
    proposal_id: str
    title: str
    description: str
    proposal_type: ProposalType
    author_id: str
    status: ProposalStatus = ProposalStatus.DRAFT
    submitted_at: float = field(default_factory=time.time)
    voting_deadline: Optional[float] = None
    votes: Dict[str, VoteOption] = field(default_factory=dict)


class GovernanceModel:
    """
    Foundation-based governance model for HiveFrame.
    
    Implements swarm democracy where community members
    collectively make decisions, like bees voting on
    new hive locations through waggle dances.
    """
    
    def __init__(self, voting_period_days: int = 14):
        self.proposals: Dict[str, Proposal] = {}
        self.voting_period_days = voting_period_days
        self.voting_members: Set[str] = set()
        self.quorum_percentage = 0.3  # 30% participation required
        
    def add_voting_member(self, member_id: str) -> bool:
        """Add a member with voting rights"""
        if member_id in self.voting_members:
            return False
        
        self.voting_members.add(member_id)
        return True
    
    def submit_proposal(
        self,
        author_id: str,
        title: str,
        description: str,
        proposal_type: ProposalType,
    ) -> Optional[str]:
        """
        Submit a governance proposal.
        
        Returns proposal ID or None if failed.
        """
        if author_id not in self.voting_members:
            return None
        
        import hashlib
        
        proposal_id = hashlib.sha256(
            f"{author_id}{title}{time.time()}".encode()
        ).hexdigest()[:16]
        
        proposal = Proposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            author_id=author_id,
            status=ProposalStatus.SUBMITTED,
        )
        
        self.proposals[proposal_id] = proposal
        return proposal_id
    
    def start_voting(
        self,
        proposal_id: str,
    ) -> bool:
        """Start voting period for a proposal"""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.SUBMITTED:
            return False
        
        proposal.status = ProposalStatus.VOTING
        proposal.voting_deadline = time.time() + (self.voting_period_days * 24 * 3600)
        
        return True
    
    def cast_vote(
        self,
        proposal_id: str,
        member_id: str,
        vote: VoteOption,
    ) -> bool:
        """Cast a vote on a proposal"""
        if proposal_id not in self.proposals:
            return False
        
        if member_id not in self.voting_members:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.VOTING:
            return False
        
        if proposal.voting_deadline and time.time() > proposal.voting_deadline:
            return False
        
        proposal.votes[member_id] = vote
        return True
    
    def tally_votes(
        self,
        proposal_id: str,
    ) -> Optional[Dict]:
        """
        Tally votes for a proposal.
        
        Returns vote results or None if not found.
        """
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        
        # Count votes
        yes_votes = sum(1 for v in proposal.votes.values() if v == VoteOption.YES)
        no_votes = sum(1 for v in proposal.votes.values() if v == VoteOption.NO)
        abstain_votes = sum(1 for v in proposal.votes.values() if v == VoteOption.ABSTAIN)
        
        total_votes = len(proposal.votes)
        participation = total_votes / len(self.voting_members) if self.voting_members else 0
        
        # Check quorum
        quorum_met = participation >= self.quorum_percentage
        
        # Determine result
        if quorum_met and yes_votes > no_votes:
            result = "approved"
            proposal.status = ProposalStatus.APPROVED
        elif quorum_met:
            result = "rejected"
            proposal.status = ProposalStatus.REJECTED
        else:
            result = "quorum_not_met"
        
        return {
            "proposal_id": proposal_id,
            "result": result,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": abstain_votes,
            "total_votes": total_votes,
            "participation": participation,
            "quorum_met": quorum_met,
        }
    
    def get_governance_stats(self) -> Dict:
        """Get governance statistics"""
        approved = sum(1 for p in self.proposals.values() if p.status == ProposalStatus.APPROVED)
        rejected = sum(1 for p in self.proposals.values() if p.status == ProposalStatus.REJECTED)
        
        return {
            "total_proposals": len(self.proposals),
            "approved_proposals": approved,
            "rejected_proposals": rejected,
            "voting_members": len(self.voting_members),
        }


@dataclass
class Contributor:
    """Represents a project contributor"""
    contributor_id: str
    name: str
    email: str
    contributions: int = 0
    role: str = "contributor"
    joined_at: float = field(default_factory=time.time)


class ContributorGuidelines:
    """
    Contributor guidelines and management.
    
    Defines roles and responsibilities like bee castes,
    ensuring smooth collaboration.
    """
    
    def __init__(self):
        self.contributors: Dict[str, Contributor] = {}
        self.roles: Dict[str, Dict] = {
            "contributor": {
                "permissions": ["submit_pr", "comment"],
                "requirements": [],
            },
            "committer": {
                "permissions": ["submit_pr", "comment", "review", "merge"],
                "requirements": ["min_contributions:10", "active_months:3"],
            },
            "maintainer": {
                "permissions": ["submit_pr", "comment", "review", "merge", "release"],
                "requirements": ["min_contributions:50", "active_months:12"],
            },
        }
        
    def register_contributor(
        self,
        contributor_id: str,
        name: str,
        email: str,
    ) -> bool:
        """Register a new contributor"""
        if contributor_id in self.contributors:
            return False
        
        contributor = Contributor(
            contributor_id=contributor_id,
            name=name,
            email=email,
        )
        
        self.contributors[contributor_id] = contributor
        return True
    
    def record_contribution(
        self,
        contributor_id: str,
    ) -> bool:
        """Record a contribution"""
        if contributor_id not in self.contributors:
            return False
        
        self.contributors[contributor_id].contributions += 1
        return True
    
    def promote_contributor(
        self,
        contributor_id: str,
        new_role: str,
    ) -> bool:
        """Promote a contributor to a new role"""
        if contributor_id not in self.contributors:
            return False
        
        if new_role not in self.roles:
            return False
        
        self.contributors[contributor_id].role = new_role
        return True
    
    def check_permissions(
        self,
        contributor_id: str,
        permission: str,
    ) -> bool:
        """Check if contributor has a specific permission"""
        if contributor_id not in self.contributors:
            return False
        
        contributor = self.contributors[contributor_id]
        role_permissions = self.roles.get(contributor.role, {}).get("permissions", [])
        
        return permission in role_permissions
    
    def get_contributor_stats(self) -> Dict:
        """Get contributor statistics"""
        by_role = {}
        for contributor in self.contributors.values():
            role = contributor.role
            by_role[role] = by_role.get(role, 0) + 1
        
        total_contributions = sum(c.contributions for c in self.contributors.values())
        
        return {
            "total_contributors": len(self.contributors),
            "by_role": by_role,
            "total_contributions": total_contributions,
        }


class DecisionMakingProcess:
    """
    Decision-making process for project governance.
    
    Implements consensus-based decision making inspired by
    bee swarm intelligence where decisions emerge from
    collective input.
    """
    
    def __init__(self, consensus_threshold: float = 0.75):
        self.decisions: Dict[str, Dict] = {}
        self.consensus_threshold = consensus_threshold
        self.decision_history: List[Dict] = []
        
    def propose_decision(
        self,
        decision_id: str,
        topic: str,
        options: List[str],
        stakeholders: List[str],
    ) -> bool:
        """Propose a decision for consensus"""
        if decision_id in self.decisions:
            return False
        
        self.decisions[decision_id] = {
            "topic": topic,
            "options": options,
            "stakeholders": stakeholders,
            "responses": {},
            "status": "open",
            "created_at": time.time(),
        }
        
        return True
    
    def provide_input(
        self,
        decision_id: str,
        stakeholder_id: str,
        preferred_option: str,
        rationale: str = "",
    ) -> bool:
        """Provide input on a decision"""
        if decision_id not in self.decisions:
            return False
        
        decision = self.decisions[decision_id]
        
        if stakeholder_id not in decision["stakeholders"]:
            return False
        
        if preferred_option not in decision["options"]:
            return False
        
        decision["responses"][stakeholder_id] = {
            "option": preferred_option,
            "rationale": rationale,
            "timestamp": time.time(),
        }
        
        return True
    
    def check_consensus(
        self,
        decision_id: str,
    ) -> Optional[Dict]:
        """
        Check if consensus has been reached.
        
        Returns consensus result or None if not found.
        """
        if decision_id not in self.decisions:
            return None
        
        decision = self.decisions[decision_id]
        
        # Count votes for each option
        vote_counts = {}
        for response in decision["responses"].values():
            option = response["option"]
            vote_counts[option] = vote_counts.get(option, 0) + 1
        
        total_responses = len(decision["responses"])
        required_responses = len(decision["stakeholders"]) * self.consensus_threshold
        
        if total_responses < required_responses:
            return {
                "decision_id": decision_id,
                "consensus_reached": False,
                "reason": "insufficient_responses",
                "responses": total_responses,
                "required": required_responses,
            }
        
        # Find option with most votes
        if vote_counts:
            winning_option = max(vote_counts, key=vote_counts.get)
            winning_votes = vote_counts[winning_option]
            consensus_percentage = winning_votes / total_responses
            
            if consensus_percentage >= self.consensus_threshold:
                decision["status"] = "consensus_reached"
                
                # Record in history
                self.decision_history.append({
                    "decision_id": decision_id,
                    "topic": decision["topic"],
                    "chosen_option": winning_option,
                    "consensus_percentage": consensus_percentage,
                    "timestamp": time.time(),
                })
                
                return {
                    "decision_id": decision_id,
                    "consensus_reached": True,
                    "chosen_option": winning_option,
                    "consensus_percentage": consensus_percentage,
                    "vote_counts": vote_counts,
                }
        
        return {
            "decision_id": decision_id,
            "consensus_reached": False,
            "reason": "no_clear_consensus",
            "vote_counts": vote_counts,
        }
    
    def get_decision_stats(self) -> Dict:
        """Get decision-making statistics"""
        consensus_reached = sum(
            1 for d in self.decisions.values()
            if d["status"] == "consensus_reached"
        )
        
        return {
            "total_decisions": len(self.decisions),
            "consensus_reached": consensus_reached,
            "decision_history": len(self.decision_history),
        }
