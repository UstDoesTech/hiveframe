"""
Natural Language Query Interface

Translate natural language questions to SQL and HiveFrame operations.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QueryIntent:
    """Parsed query intent"""

    intent_type: str  # 'select', 'aggregate', 'filter', 'join', 'update'
    entities: List[str]  # Tables/columns mentioned
    conditions: List[str]  # Filter conditions
    aggregations: List[str]  # Aggregate functions
    confidence: float


@dataclass
class GeneratedQuery:
    """Generated SQL query"""

    sql: str
    dialect: str  # 'swarmql', 'standard_sql', 'hiveframe'
    parameters: Dict[str, Any]
    explanation: str
    confidence: float


class QueryTranslator:
    """
    Translate natural language to structured query intents.

    Uses pattern matching and keyword extraction (in production, would use
    a fine-tuned language model).
    """

    def __init__(self):
        # Simple keyword patterns (would be replaced with ML model)
        self.intent_patterns = {
            "select": ["show", "display", "list", "get", "find", "what", "which"],
            "aggregate": ["count", "sum", "average", "total", "max", "min", "avg"],
            "filter": ["where", "with", "having", "that have", "greater than", "less than"],
            "join": ["join", "combine", "merge", "with", "from both"],
            "sort": ["sort", "order", "rank", "top", "highest", "lowest"],
        }

        self.aggregation_keywords = {
            "count": "COUNT",
            "total": "SUM",
            "sum": "SUM",
            "average": "AVG",
            "avg": "AVG",
            "maximum": "MAX",
            "max": "MAX",
            "minimum": "MIN",
            "min": "MIN",
        }

    def parse_intent(self, natural_query: str) -> QueryIntent:
        """
        Parse natural language query into structured intent.

        Args:
            natural_query: Natural language question

        Returns:
            QueryIntent with parsed information
        """
        query_lower = natural_query.lower()

        # Detect intent type
        intent_type = "select"  # default
        for intent, keywords in self.intent_patterns.items():
            if any(kw in query_lower for kw in keywords):
                intent_type = intent
                break

        # Extract entities (simplified - would use NER in production)
        # Look for capitalized words or quoted strings
        entities = re.findall(r'\b[A-Z][a-z]+\b|"([^"]+)"', natural_query)
        entities = [e if isinstance(e, str) else e for e in entities]

        # Extract conditions
        conditions = []
        if "where" in query_lower:
            where_part = (
                query_lower.split("where")[1] if len(query_lower.split("where")) > 1 else ""
            )
            conditions.append(where_part.strip())

        # Extract aggregations
        aggregations = []
        for keyword, agg_func in self.aggregation_keywords.items():
            if keyword in query_lower:
                aggregations.append(agg_func)

        # Calculate confidence (simplified)
        confidence = 0.7 if entities and intent_type else 0.4

        return QueryIntent(
            intent_type=intent_type,
            entities=entities,
            conditions=conditions,
            aggregations=aggregations,
            confidence=confidence,
        )


class SQLGenerator:
    """
    Generate SQL from query intent.

    Transforms structured intent into executable SQL, with swarm-inspired
    query optimization patterns.
    """

    def __init__(self, default_schema: Optional[Dict[str, List[str]]] = None):
        """
        Initialize SQL generator.

        Args:
            default_schema: Dictionary mapping table names to column lists
        """
        self.default_schema = default_schema or {}

    def generate_sql(self, intent: QueryIntent) -> GeneratedQuery:
        """
        Generate SQL from query intent.

        Args:
            intent: Parsed query intent

        Returns:
            GeneratedQuery with SQL and metadata
        """
        # Extract table and column info
        table_name = intent.entities[0] if intent.entities else "data"
        columns = intent.entities[1:] if len(intent.entities) > 1 else ["*"]

        # Build SELECT clause
        if intent.aggregations:
            # Aggregate query
            agg_func = intent.aggregations[0]
            col = columns[0] if columns and columns[0] != "*" else "value"
            select_clause = f"SELECT {agg_func}({col})"
        else:
            # Regular select
            cols_str = ", ".join(columns) if columns else "*"
            select_clause = f"SELECT {cols_str}"

        # Build FROM clause
        from_clause = f"FROM {table_name}"

        # Build WHERE clause
        where_clause = ""
        if intent.conditions:
            where_clause = f" WHERE {intent.conditions[0]}"

        # Combine into SQL
        sql = f"{select_clause} {from_clause}{where_clause}"

        # Generate explanation
        explanation = self._generate_explanation(intent, sql)

        return GeneratedQuery(
            sql=sql,
            dialect="swarmql",
            parameters={},
            explanation=explanation,
            confidence=intent.confidence,
        )

    def _generate_explanation(self, intent: QueryIntent, sql: str) -> str:
        """Generate human-readable explanation of the query"""
        if intent.aggregations:
            return f"Calculating {intent.aggregations[0]} over {intent.entities[0] if intent.entities else 'data'}"
        else:
            return f"Retrieving data from {intent.entities[0] if intent.entities else 'table'}"


class NaturalLanguageQuery:
    """
    Natural language query interface for HiveFrame.

    Allows users to ask questions in plain English and get results,
    similar to how bees communicate complex information through dances.
    """

    def __init__(self, schema: Optional[Dict[str, List[str]]] = None):
        """
        Initialize NL query interface.

        Args:
            schema: Database schema for query generation
        """
        self.translator = QueryTranslator()
        self.sql_generator = SQLGenerator(schema)
        self.query_history: List[Dict[str, Any]] = []

    def query(self, natural_language: str) -> Dict[str, Any]:
        """
        Process a natural language query.

        Args:
            natural_language: Question in natural language

        Returns:
            Dictionary with query results and metadata
        """
        # Parse intent
        intent = self.translator.parse_intent(natural_language)

        # Generate SQL
        generated = self.sql_generator.generate_sql(intent)

        # Store in history (for learning)
        self.query_history.append(
            {
                "natural_language": natural_language,
                "intent": intent,
                "sql": generated.sql,
                "confidence": generated.confidence,
            }
        )

        return {
            "sql": generated.sql,
            "explanation": generated.explanation,
            "confidence": generated.confidence,
            "intent": {
                "type": intent.intent_type,
                "entities": intent.entities,
                "aggregations": intent.aggregations,
            },
        }

    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input.

        Args:
            partial_query: Partial natural language query

        Returns:
            List of suggested completions
        """
        suggestions = []

        # Simple prefix matching (would use ML model in production)
        query_lower = partial_query.lower()

        if "show" in query_lower or "list" in query_lower:
            suggestions.extend(
                [
                    "show all records",
                    "show records where",
                    "show top 10",
                ]
            )

        if "count" in query_lower:
            suggestions.extend(
                [
                    "count all records",
                    "count records where",
                    "count by category",
                ]
            )

        return suggestions[:5]  # Top 5 suggestions

    def learn_from_feedback(
        self,
        natural_query: str,
        generated_sql: str,
        correct_sql: str,
    ) -> None:
        """
        Learn from user corrections (would train model in production).

        Args:
            natural_query: Original natural language query
            generated_sql: SQL that was generated
            correct_sql: User's correction
        """
        # In production, this would update the model
        # For now, just store for future reference
        self.query_history.append(
            {
                "type": "feedback",
                "natural_query": natural_query,
                "generated": generated_sql,
                "correct": correct_sql,
            }
        )
