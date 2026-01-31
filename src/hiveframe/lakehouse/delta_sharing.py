"""
Delta Sharing Protocol - Secure Data Sharing
============================================

Implementation of Delta Sharing protocol for secure, bee-inspired
data sharing across organizations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ShareAccessLevel(Enum):
    """Access levels for shared data."""

    READ = "READ"
    READ_WRITE = "READ_WRITE"
    ADMIN = "ADMIN"


@dataclass
class Share:
    """A share is a collection of tables shared with external parties."""

    share_id: str
    name: str
    owner: str
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tables: Set[str] = field(default_factory=set)


@dataclass
class ShareRecipient:
    """A recipient who has access to a share."""

    recipient_id: str
    email: str
    access_level: ShareAccessLevel
    share_id: str
    expires_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    token: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ShareToken:
    """Access token for a share."""

    token: str
    share_id: str
    recipient_id: str
    issued_at: datetime
    expires_at: datetime
    revoked: bool = False


class DeltaSharing:
    """
    Delta Sharing Protocol implementation.

    Enables secure data sharing across organizations using a REST-based
    protocol. Data is shared like nectar information between bee colonies -
    valuable resources are shared selectively with trusted partners.

    Example:
        sharing = DeltaSharing()

        # Create a share
        share = sharing.create_share(
            name="customer_data",
            owner="data_team@company.com",
            tables={"customers", "orders"}
        )

        # Add a recipient
        recipient = sharing.add_recipient(
            share_id=share.share_id,
            email="partner@external.com",
            access_level=ShareAccessLevel.READ,
            expires_in_days=90
        )

        # Recipient uses token to access data
        token = recipient.token
        tables = sharing.list_shared_tables(token)
    """

    def __init__(self):
        self._shares: Dict[str, Share] = {}
        self._recipients: Dict[str, ShareRecipient] = {}
        self._tokens: Dict[str, ShareToken] = {}

    def create_share(
        self,
        name: str,
        owner: str,
        tables: Optional[Set[str]] = None,
        description: Optional[str] = None,
    ) -> Share:
        """
        Create a new share.

        Args:
            name: Share name
            owner: Share owner identifier
            tables: Set of table names to include
            description: Optional description

        Returns:
            Created Share
        """
        share_id = str(uuid.uuid4())
        share = Share(
            share_id=share_id,
            name=name,
            owner=owner,
            description=description,
            tables=tables or set(),
        )

        self._shares[share_id] = share
        return share

    def add_recipient(
        self,
        share_id: str,
        email: str,
        access_level: ShareAccessLevel = ShareAccessLevel.READ,
        expires_in_days: Optional[int] = None,
    ) -> ShareRecipient:
        """
        Add a recipient to a share.

        Args:
            share_id: Share identifier
            email: Recipient email
            access_level: Access level to grant
            expires_in_days: Days until access expires (None = no expiration)

        Returns:
            Created ShareRecipient with access token
        """
        if share_id not in self._shares:
            raise ValueError(f"Share '{share_id}' not found")

        recipient_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        recipient = ShareRecipient(
            recipient_id=recipient_id,
            email=email,
            access_level=access_level,
            share_id=share_id,
            expires_at=expires_at,
        )

        self._recipients[recipient_id] = recipient

        # Create access token
        token = ShareToken(
            token=recipient.token,
            share_id=share_id,
            recipient_id=recipient_id,
            issued_at=datetime.now(),
            expires_at=expires_at or datetime.now() + timedelta(days=365),
        )
        self._tokens[recipient.token] = token

        return recipient

    def validate_token(self, token: str) -> bool:
        """
        Validate an access token.

        Args:
            token: Access token to validate

        Returns:
            True if token is valid and not expired
        """
        if token not in self._tokens:
            return False

        share_token = self._tokens[token]

        # Check if revoked
        if share_token.revoked:
            return False

        # Check expiration
        if datetime.now() > share_token.expires_at:
            return False

        return True

    def revoke_token(self, token: str) -> bool:
        """Revoke an access token."""
        if token in self._tokens:
            self._tokens[token].revoked = True
            return True
        return False

    def list_shared_tables(self, token: str) -> List[str]:
        """
        List tables available through a share token.

        Args:
            token: Access token

        Returns:
            List of table names
        """
        if not self.validate_token(token):
            raise PermissionError("Invalid or expired token")

        share_token = self._tokens[token]
        share = self._shares[share_token.share_id]

        return sorted(list(share.tables))

    def add_table_to_share(self, share_id: str, table: str) -> None:
        """Add a table to a share."""
        if share_id not in self._shares:
            raise ValueError(f"Share '{share_id}' not found")

        self._shares[share_id].tables.add(table)

    def remove_table_from_share(self, share_id: str, table: str) -> None:
        """Remove a table from a share."""
        if share_id not in self._shares:
            raise ValueError(f"Share '{share_id}' not found")

        self._shares[share_id].tables.discard(table)

    def get_share(self, share_id: str) -> Optional[Share]:
        """Get share by ID."""
        return self._shares.get(share_id)

    def list_shares(self, owner: Optional[str] = None) -> List[Share]:
        """
        List all shares, optionally filtered by owner.

        Args:
            owner: Optional owner filter

        Returns:
            List of shares
        """
        shares = list(self._shares.values())
        if owner:
            shares = [s for s in shares if s.owner == owner]
        return shares

    def delete_share(self, share_id: str) -> bool:
        """Delete a share and revoke all associated tokens."""
        if share_id not in self._shares:
            return False

        # Revoke all tokens for this share
        for token_obj in self._tokens.values():
            if token_obj.share_id == share_id:
                token_obj.revoked = True

        # Remove recipients
        recipients_to_remove = [
            rid for rid, r in self._recipients.items() if r.share_id == share_id
        ]
        for rid in recipients_to_remove:
            del self._recipients[rid]

        # Remove share
        del self._shares[share_id]
        return True

    def get_share_metadata(self, token: str) -> Dict[str, Any]:
        """
        Get metadata for a share using an access token.

        Args:
            token: Access token

        Returns:
            Share metadata dictionary
        """
        if not self.validate_token(token):
            raise PermissionError("Invalid or expired token")

        share_token = self._tokens[token]
        share = self._shares[share_token.share_id]

        return {
            "share_id": share.share_id,
            "name": share.name,
            "created_at": share.created_at.isoformat(),
            "description": share.description,
            "num_tables": len(share.tables),
        }
