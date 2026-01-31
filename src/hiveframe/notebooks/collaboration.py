"""
Collaboration Manager - Real-time Multi-user Collaboration
==========================================================

Manages collaborative editing sessions with bee-inspired coordination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid


class OperationType(Enum):
    """Types of collaborative operations."""

    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    CURSOR_MOVE = "cursor_move"


@dataclass
class User:
    """A collaborative session user."""

    user_id: str
    username: str
    email: str
    color: str  # Color for cursor/selection highlighting
    connected_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class Operation:
    """An operation in the collaborative session."""

    operation_id: str
    operation_type: OperationType
    user_id: str
    cell_id: str
    position: int
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborativeSession:
    """A collaborative editing session."""

    session_id: str
    notebook_id: str
    users: Dict[str, User] = field(default_factory=dict)
    operations: List[Operation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class CollaborationManager:
    """
    Manages real-time collaborative editing.

    Uses bee-inspired coordination where users are like worker bees
    contributing to the shared notebook (hive).

    Example:
        collab = CollaborationManager()

        # Create collaborative session
        session = collab.create_session("notebook_123")

        # Add users
        user1 = collab.add_user(
            session.session_id,
            username="alice",
            email="alice@example.com"
        )

        # Broadcast operation
        collab.broadcast_operation(
            session.session_id,
            user1.user_id,
            cell_id="cell_1",
            operation_type=OperationType.INSERT,
            position=0,
            data="print('Hello')"
        )

        # Get operations since timestamp
        ops = collab.get_operations_since(
            session.session_id,
            since=datetime.now()
        )
    """

    def __init__(self):
        """Initialize collaboration manager."""
        self._sessions: Dict[str, CollaborativeSession] = {}
        self._user_colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#FFA07A",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E2",
        ]
        self._next_color_idx = 0

    def create_session(self, notebook_id: str) -> CollaborativeSession:
        """
        Create a new collaborative session.

        Args:
            notebook_id: Notebook identifier

        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        session = CollaborativeSession(session_id=session_id, notebook_id=notebook_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CollaborativeSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def add_user(self, session_id: str, username: str, email: str) -> User:
        """
        Add a user to a collaborative session.

        Args:
            session_id: Session identifier
            username: User name
            email: User email

        Returns:
            Created User
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]
        user_id = str(uuid.uuid4())

        # Assign color
        color = self._user_colors[self._next_color_idx % len(self._user_colors)]
        self._next_color_idx += 1

        user = User(user_id=user_id, username=username, email=email, color=color)

        session.users[user_id] = user
        return user

    def remove_user(self, session_id: str, user_id: str) -> bool:
        """Remove a user from a session."""
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if user_id in session.users:
            del session.users[user_id]
            return True
        return False

    def broadcast_operation(
        self,
        session_id: str,
        user_id: str,
        cell_id: str,
        operation_type: OperationType,
        position: int,
        data: Any,
    ) -> Operation:
        """
        Broadcast an operation to all session participants.

        Args:
            session_id: Session identifier
            user_id: User performing the operation
            cell_id: Cell being modified
            operation_type: Type of operation
            position: Position in cell
            data: Operation data

        Returns:
            Created Operation
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]

        operation = Operation(
            operation_id=str(uuid.uuid4()),
            operation_type=operation_type,
            user_id=user_id,
            cell_id=cell_id,
            position=position,
            data=data,
        )

        session.operations.append(operation)

        # Update user activity
        if user_id in session.users:
            session.users[user_id].last_active = datetime.now()

        return operation

    def get_operations_since(
        self, session_id: str, since: datetime, user_id: Optional[str] = None
    ) -> List[Operation]:
        """
        Get operations since a timestamp.

        Args:
            session_id: Session identifier
            since: Timestamp to get operations after
            user_id: Optional user filter

        Returns:
            List of operations
        """
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        operations = [op for op in session.operations if op.timestamp > since]

        if user_id:
            operations = [op for op in operations if op.user_id == user_id]

        return operations

    def get_active_users(self, session_id: str) -> List[User]:
        """
        Get list of active users in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of active users
        """
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        return list(session.users.values())

    def close_session(self, session_id: str) -> bool:
        """Close a collaborative session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
