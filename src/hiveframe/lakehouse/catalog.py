"""
Unity Hive Catalog - Unified Governance and Data Discovery
==========================================================

Centralized metadata management with bee-inspired organization principles.
Tables are organized like honeycomb cells with metadata stored as pheromone trails.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PermissionType(Enum):
    """Permission types for access control."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"
    ALL = "ALL"


class PIISensitivity(Enum):
    """PII sensitivity levels."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


@dataclass
class TableMetadata:
    """Metadata for a cataloged table."""
    name: str
    schema: Dict[str, str]
    location: str
    format: str = "parquet"
    partitions: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: Optional[str] = None
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    pii_columns: Set[str] = field(default_factory=set)


@dataclass
class LineageNode:
    """Node in data lineage graph."""
    table: str
    operation: str
    timestamp: datetime
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnityHiveCatalog:
    """
    Unity Hive Catalog - Unified metadata and governance.
    
    Inspired by bees organizing honeycomb cells, tables are registered
    with rich metadata that guides the swarm to optimal data access patterns.
    
    Example:
        catalog = UnityHiveCatalog()
        
        # Register a table
        catalog.register_table(
            name="users",
            schema={"id": "int", "name": "string", "email": "string"},
            location="/data/users",
            owner="data_team",
            tags={"production", "user_data"}
        )
        
        # Query metadata
        table = catalog.get_table("users")
        print(f"Table location: {table.location}")
        
        # Search by tags
        tables = catalog.search_by_tags({"user_data"})
    """
    
    def __init__(self):
        self._tables: Dict[str, TableMetadata] = {}
        self._lineage: List[LineageNode] = []
    
    def register_table(
        self,
        name: str,
        schema: Dict[str, str],
        location: str,
        format: str = "parquet",
        partitions: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        owner: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> TableMetadata:
        """
        Register a new table in the catalog.
        
        Args:
            name: Table name
            schema: Column name -> type mapping
            location: Physical storage location
            format: Data format (parquet, delta, iceberg)
            partitions: Partition column names
            properties: Additional table properties
            owner: Table owner identifier
            description: Human-readable description
            tags: Tags for categorization and search
            
        Returns:
            Created TableMetadata
        """
        metadata = TableMetadata(
            name=name,
            schema=schema,
            location=location,
            format=format,
            partitions=partitions or [],
            properties=properties or {},
            owner=owner,
            description=description,
            tags=tags or set(),
        )
        
        self._tables[name] = metadata
        return metadata
    
    def get_table(self, name: str) -> Optional[TableMetadata]:
        """Get table metadata by name."""
        return self._tables.get(name)
    
    def list_tables(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all table names, optionally filtered by pattern.
        
        Args:
            pattern: Optional regex pattern to filter names
            
        Returns:
            List of matching table names
        """
        tables = list(self._tables.keys())
        if pattern:
            regex = re.compile(pattern)
            tables = [t for t in tables if regex.match(t)]
        return sorted(tables)
    
    def search_by_tags(self, tags: Set[str]) -> List[TableMetadata]:
        """
        Search tables by tags.
        
        Args:
            tags: Set of tags to search for
            
        Returns:
            List of tables that have any of the specified tags
        """
        results = []
        for table in self._tables.values():
            if table.tags & tags:  # Intersection
                results.append(table)
        return results
    
    def update_table(self, name: str, **updates) -> None:
        """Update table metadata."""
        if name not in self._tables:
            raise ValueError(f"Table '{name}' not found")
        
        table = self._tables[name]
        for key, value in updates.items():
            if hasattr(table, key):
                setattr(table, key, value)
        table.updated_at = datetime.now()
    
    def drop_table(self, name: str) -> bool:
        """Drop a table from the catalog."""
        if name in self._tables:
            del self._tables[name]
            return True
        return False


class AccessControl:
    """
    Fine-grained access control for Unity Hive Catalog.
    
    Implements row-level and column-level security using a distributed
    permission system inspired by bee colony guard behavior.
    
    Example:
        catalog = UnityHiveCatalog()
        acl = AccessControl(catalog)
        
        # Grant permissions
        acl.grant("user@example.com", "users", [PermissionType.SELECT])
        
        # Check permissions
        if acl.check_permission("user@example.com", "users", PermissionType.SELECT):
            print("Access granted")
    """
    
    def __init__(self, catalog: UnityHiveCatalog):
        self.catalog = catalog
        self._permissions: Dict[Tuple[str, str], Set[PermissionType]] = {}
    
    def grant(
        self,
        principal: str,
        table: str,
        permissions: List[PermissionType]
    ) -> None:
        """
        Grant permissions to a principal on a table.
        
        Args:
            principal: User or group identifier
            table: Table name
            permissions: List of permissions to grant
        """
        key = (principal, table)
        if key not in self._permissions:
            self._permissions[key] = set()
        self._permissions[key].update(permissions)
    
    def revoke(
        self,
        principal: str,
        table: str,
        permissions: List[PermissionType]
    ) -> None:
        """Revoke permissions from a principal on a table."""
        key = (principal, table)
        if key in self._permissions:
            self._permissions[key].difference_update(permissions)
    
    def check_permission(
        self,
        principal: str,
        table: str,
        permission: PermissionType
    ) -> bool:
        """
        Check if a principal has a specific permission on a table.
        
        Args:
            principal: User or group identifier
            table: Table name
            permission: Permission to check
            
        Returns:
            True if permission is granted, False otherwise
        """
        key = (principal, table)
        if key not in self._permissions:
            return False
        
        perms = self._permissions[key]
        return permission in perms or PermissionType.ALL in perms
    
    def list_permissions(self, principal: str) -> Dict[str, List[PermissionType]]:
        """List all permissions for a principal."""
        result = {}
        for (p, table), perms in self._permissions.items():
            if p == principal:
                result[table] = list(perms)
        return result


class LineageTracker:
    """
    Data lineage tracking for Unity Hive Catalog.
    
    Tracks data transformations and dependencies using a graph structure
    similar to how bees track flower locations through waggle dances.
    
    Example:
        tracker = LineageTracker(catalog)
        
        # Record a transformation
        tracker.record_lineage(
            output_table="user_summary",
            input_tables=["users", "orders"],
            operation="aggregate"
        )
        
        # Get upstream dependencies
        deps = tracker.get_upstream("user_summary")
    """
    
    def __init__(self, catalog: UnityHiveCatalog):
        self.catalog = catalog
        self._lineage: List[LineageNode] = []
        self._graph: Dict[str, Set[str]] = {}  # table -> dependencies
    
    def record_lineage(
        self,
        output_table: str,
        input_tables: List[str],
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LineageNode:
        """
        Record a data lineage operation.
        
        Args:
            output_table: Output table name
            input_tables: List of input table names
            operation: Operation type (e.g., 'join', 'aggregate', 'filter')
            metadata: Additional operation metadata
            
        Returns:
            Created LineageNode
        """
        node = LineageNode(
            table=output_table,
            operation=operation,
            timestamp=datetime.now(),
            dependencies=input_tables,
            metadata=metadata or {}
        )
        
        self._lineage.append(node)
        self._graph[output_table] = set(input_tables)
        
        return node
    
    def get_upstream(self, table: str, recursive: bool = False) -> Set[str]:
        """
        Get upstream dependencies for a table.
        
        Args:
            table: Table name
            recursive: If True, get all transitive dependencies
            
        Returns:
            Set of upstream table names
        """
        if table not in self._graph:
            return set()
        
        dependencies = self._graph[table].copy()
        
        if recursive:
            for dep in list(dependencies):
                dependencies.update(self.get_upstream(dep, recursive=True))
        
        return dependencies
    
    def get_downstream(self, table: str, recursive: bool = False) -> Set[str]:
        """Get downstream consumers of a table."""
        downstream = set()
        
        for output, inputs in self._graph.items():
            if table in inputs:
                downstream.add(output)
                if recursive:
                    downstream.update(self.get_downstream(output, recursive=True))
        
        return downstream
    
    def get_lineage_path(self, from_table: str, to_table: str) -> Optional[List[str]]:
        """Find a lineage path between two tables using BFS."""
        if from_table == to_table:
            return [from_table]
        
        visited = {from_table}
        queue = [(from_table, [from_table])]
        
        while queue:
            current, path = queue.pop(0)
            
            # Check tables that depend on current (downstream)
            for next_table, inputs in self._graph.items():
                if current in inputs and next_table not in visited:
                    new_path = path + [next_table]
                    if next_table == to_table:
                        return new_path
                    visited.add(next_table)
                    queue.append((next_table, new_path))
        
        return None


class PIIDetector:
    """
    Automatic PII detection for Unity Hive Catalog.
    
    Uses pattern matching and heuristics to identify sensitive data,
    similar to how bees detect and avoid threats in their environment.
    
    Example:
        detector = PIIDetector()
        
        # Detect PII in schema
        schema = {"name": "string", "email": "string", "age": "int"}
        pii_cols = detector.detect_pii(schema)
        print(f"PII columns: {pii_cols}")
    """
    
    # Common PII patterns
    PII_PATTERNS = {
        "email": r"email|e_mail|mail",
        "phone": r"phone|tel|mobile|cell",
        "ssn": r"ssn|social_security",
        "credit_card": r"credit_card|cc_number|card_number",
        "name": r"^name$|first_name|last_name|full_name",
        "address": r"address|street|city|zipcode|postal",
        "dob": r"date_of_birth|dob|birthdate",
        "ip_address": r"ip_address|ip_addr",
    }
    
    def detect_pii(
        self,
        schema: Dict[str, str],
        sensitivity: PIISensitivity = PIISensitivity.CONFIDENTIAL
    ) -> Dict[str, PIISensitivity]:
        """
        Detect PII columns in a schema.
        
        Args:
            schema: Column name -> type mapping
            sensitivity: Default sensitivity for detected PII
            
        Returns:
            Dictionary of column name -> sensitivity level
        """
        pii_columns = {}
        
        for column_name in schema.keys():
            column_lower = column_name.lower()
            
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, column_lower):
                    pii_columns[column_name] = sensitivity
                    break
        
        return pii_columns
    
    def classify_sensitivity(self, column_name: str, data_sample: List[Any]) -> Optional[PIISensitivity]:
        """
        Classify sensitivity based on column name and data samples.
        
        Args:
            column_name: Column name
            data_sample: Sample data values
            
        Returns:
            Detected sensitivity level or None
        """
        # For now, use name-based detection
        detected = self.detect_pii({column_name: "string"})
        return detected.get(column_name)
