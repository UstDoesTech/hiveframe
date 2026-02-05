"""
Intelligent Data Discovery

AI-suggested joins and relationship detection using swarm intelligence patterns.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class JoinSuggestion:
    """Suggested join between tables"""

    left_table: str
    right_table: str
    left_column: str
    right_column: str
    join_type: str  # 'inner', 'left', 'right', 'full'
    confidence: float
    evidence: List[str]


@dataclass
class Relationship:
    """Detected relationship between entities"""

    entity1: str
    entity2: str
    relationship_type: str  # 'one-to-one', 'one-to-many', 'many-to-many'
    foreign_key: Optional[str]
    confidence: float


@dataclass
class SchemaGraph:
    """Graph representation of schema relationships"""

    tables: List[str]
    relationships: List[Relationship]
    join_paths: Dict[Tuple[str, str], List[str]]


class RelationshipDetector:
    """
    Detect relationships between tables using swarm intelligence.

    Analyzes column names, data patterns, and cardinalities to infer
    relationships, similar to how bees map complex spatial relationships.
    """

    def __init__(self):
        self.known_relationships: List[Relationship] = []

    def detect_relationships(
        self,
        schema: Dict[str, List[str]],
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Relationship]:
        """
        Detect relationships between tables.

        Args:
            schema: Dictionary mapping table names to column lists
            sample_data: Optional sample data for validation

        Returns:
            List of detected relationships
        """
        relationships = []
        tables = list(schema.keys())

        # Check each pair of tables
        for i, table1 in enumerate(tables):
            for table2 in tables[i + 1 :]:
                cols1 = schema[table1]
                cols2 = schema[table2]

                # Look for common column names (FK candidates)
                for col1 in cols1:
                    for col2 in cols2:
                        if self._columns_match(col1, col2):
                            # Found potential FK relationship
                            rel_type, confidence = self._infer_relationship_type(
                                table1,
                                table2,
                                col1,
                                col2,
                                sample_data.get(table1) if sample_data else None,
                                sample_data.get(table2) if sample_data else None,
                            )

                            if confidence > 0.5:
                                relationships.append(
                                    Relationship(
                                        entity1=table1,
                                        entity2=table2,
                                        relationship_type=rel_type,
                                        foreign_key=col1,
                                        confidence=confidence,
                                    )
                                )

        self.known_relationships.extend(relationships)
        return relationships

    def _columns_match(self, col1: str, col2: str) -> bool:
        """Check if column names suggest a relationship"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()

        # Exact match
        if col1_lower == col2_lower:
            return True

        # Common FK patterns
        if col1_lower.endswith("_id") and col2_lower.endswith("_id"):
            return True

        if "id" in col1_lower and "id" in col2_lower:
            # Extract table name
            base1 = col1_lower.replace("_id", "").replace("id", "")
            base2 = col2_lower.replace("_id", "").replace("id", "")
            if base1 == base2:
                return True

        return False

    def _infer_relationship_type(
        self,
        table1: str,
        table2: str,
        col1: str,
        col2: str,
        data1: Optional[List[Dict[str, Any]]],
        data2: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, float]:
        """
        Infer relationship type and confidence.

        Returns:
            Tuple of (relationship_type, confidence)
        """
        if not data1 or not data2:
            # No data - use heuristics
            if col1.lower().endswith("_id") or col2.lower().endswith("_id"):
                return "one-to-many", 0.6
            return "one-to-many", 0.5

        # Count occurrences
        counts1 = {}
        for row in data1:
            val = row.get(col1)
            if val:
                counts1[val] = counts1.get(val, 0) + 1

        counts2 = {}
        for row in data2:
            val = row.get(col2)
            if val:
                counts2[val] = counts2.get(val, 0) + 1

        # Determine relationship type
        avg_count1 = statistics.mean(counts1.values()) if counts1 else 0
        avg_count2 = statistics.mean(counts2.values()) if counts2 else 0

        if avg_count1 <= 1.1 and avg_count2 <= 1.1:
            return "one-to-one", 0.8
        elif avg_count1 > 1.1 and avg_count2 > 1.1:
            return "many-to-many", 0.7
        else:
            return "one-to-many", 0.8


class JoinSuggester:
    """
    Suggest optimal joins between tables using bee-inspired path finding.

    Like bees that find optimal routes between hive and flowers,
    this suggester finds best join paths between tables.
    """

    def __init__(self):
        self.relationship_detector = RelationshipDetector()
        self.join_history: List[JoinSuggestion] = []

    def suggest_joins(
        self,
        target_table: str,
        available_tables: List[str],
        schema: Dict[str, List[str]],
        relationships: Optional[List[Relationship]] = None,
    ) -> List[JoinSuggestion]:
        """
        Suggest joins to reach target table from available tables.

        Args:
            target_table: Table to join to
            available_tables: Currently available tables
            schema: Schema information
            relationships: Known relationships (will detect if not provided)

        Returns:
            List of join suggestions
        """
        if not relationships:
            relationships = self.relationship_detector.detect_relationships(schema)

        suggestions = []

        for available_table in available_tables:
            if available_table == target_table:
                continue

            # Find direct relationships
            for rel in relationships:
                if (rel.entity1 == available_table and rel.entity2 == target_table) or (
                    rel.entity2 == available_table and rel.entity1 == target_table
                ):

                    # Determine join direction
                    if rel.entity1 == available_table:
                        left_table, right_table = available_table, target_table
                        left_col = right_col = rel.foreign_key or "id"
                    else:
                        left_table, right_table = target_table, available_table
                        left_col = right_col = rel.foreign_key or "id"

                    # Determine join type based on relationship
                    if rel.relationship_type == "one-to-one":
                        join_type = "inner"
                    elif rel.relationship_type == "one-to-many":
                        join_type = "left"
                    else:
                        join_type = "inner"

                    suggestions.append(
                        JoinSuggestion(
                            left_table=left_table,
                            right_table=right_table,
                            left_column=left_col,
                            right_column=right_col,
                            join_type=join_type,
                            confidence=rel.confidence,
                            evidence=[f"{rel.relationship_type} relationship detected"],
                        )
                    )

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        self.join_history.extend(suggestions)
        return suggestions

    def find_join_path(
        self,
        from_table: str,
        to_table: str,
        relationships: List[Relationship],
    ) -> Optional[List[JoinSuggestion]]:
        """
        Find a path of joins between two tables.

        Uses bee-inspired pathfinding (similar to waggle dance navigation).

        Args:
            from_table: Starting table
            to_table: Target table
            relationships: Known relationships

        Returns:
            List of joins forming a path, or None if no path exists
        """
        # Build adjacency list
        graph: Dict[str, List[Tuple[str, Relationship]]] = {}
        for rel in relationships:
            if rel.entity1 not in graph:
                graph[rel.entity1] = []
            if rel.entity2 not in graph:
                graph[rel.entity2] = []

            graph[rel.entity1].append((rel.entity2, rel))
            graph[rel.entity2].append((rel.entity1, rel))

        # BFS to find shortest path
        if from_table not in graph:
            return None

        queue = [(from_table, [])]
        visited = {from_table}

        while queue:
            current, path = queue.pop(0)

            if current == to_table:
                # Found path - convert to join suggestions
                return self._path_to_joins(path)

            for neighbor, rel in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(current, neighbor, rel)]))

        return None

    def _path_to_joins(self, path: List[Tuple[str, str, Relationship]]) -> List[JoinSuggestion]:
        """Convert a path to join suggestions"""
        joins = []

        for left_table, right_table, rel in path:
            joins.append(
                JoinSuggestion(
                    left_table=left_table,
                    right_table=right_table,
                    left_column=rel.foreign_key or "id",
                    right_column=rel.foreign_key or "id",
                    join_type="inner",
                    confidence=rel.confidence,
                    evidence=[f"Path through {rel.relationship_type} relationship"],
                )
            )

        return joins


class DataDiscovery:
    """
    Intelligent data discovery orchestrator.

    Helps users navigate and understand complex schemas through
    swarm-intelligence-powered relationship detection and join suggestions.
    """

    def __init__(self):
        self.relationship_detector = RelationshipDetector()
        self.join_suggester = JoinSuggester()
        self.schema_graph: Optional[SchemaGraph] = None

    def discover_schema(
        self,
        schema: Dict[str, List[str]],
        sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> SchemaGraph:
        """
        Discover and map schema relationships.

        Args:
            schema: Database schema
            sample_data: Optional sample data for validation

        Returns:
            SchemaGraph with discovered relationships
        """
        # Detect relationships
        relationships = self.relationship_detector.detect_relationships(schema, sample_data)

        # Build join paths
        tables = list(schema.keys())
        join_paths = {}

        for i, table1 in enumerate(tables):
            for table2 in tables[i + 1 :]:
                path = self.join_suggester.find_join_path(table1, table2, relationships)
                if path:
                    join_paths[(table1, table2)] = [j.right_table for j in path]

        self.schema_graph = SchemaGraph(
            tables=tables,
            relationships=relationships,
            join_paths=join_paths,
        )

        return self.schema_graph

    def suggest_for_analysis(
        self,
        goal: str,
        available_tables: List[str],
    ) -> Dict[str, Any]:
        """
        Suggest tables and joins for a specific analysis goal.

        Args:
            goal: Analysis goal description
            available_tables: Tables currently in scope

        Returns:
            Dictionary with suggestions
        """
        # Parse goal (simplified - would use NLP in production)
        goal_lower = goal.lower()

        # Identify potentially relevant tables based on keywords
        relevant_tables = []
        for table in available_tables:
            if any(word in table.lower() for word in goal_lower.split()):
                relevant_tables.append(table)

        return {
            "relevant_tables": relevant_tables,
            "confidence": 0.6,
            "explanation": f"Tables matching keywords in: {goal}",
        }
