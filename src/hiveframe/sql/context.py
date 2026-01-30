"""
SwarmQL Context
===============

Main entry point for SwarmQL SQL engine.
"""

from typing import Any, Dict, List, Optional

from ..core import create_hive
from ..dataframe import HiveDataFrame
from .executor import SQLCatalog, SQLExecutor
from .parser import SQLParser, SQLStatement, SQLTokenizer


class SwarmQLContext:
    """
    SwarmQL Context
    ---------------

    Main entry point for the SwarmQL SQL engine.
    Provides SQL query execution over HiveDataFrames.

    Usage:
        ctx = SwarmQLContext()

        # Register tables
        ctx.register_table("users", users_df)
        ctx.register_table("orders", orders_df)

        # Execute SQL
        result = ctx.sql("SELECT * FROM users WHERE age > 21")
        result.show()

        # Join tables
        result = ctx.sql('''
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name
        ''')
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize SwarmQL context.

        Args:
            num_workers: Number of worker bees for processing
        """
        self.catalog = SQLCatalog()
        self._hive = create_hive(num_workers=num_workers)
        self._executor = SQLExecutor(self.catalog)

    def register_table(self, name: str, data: Any) -> None:
        """
        Register a table in the catalog.

        Args:
            name: Table name
            data: HiveDataFrame, list of dicts, or path to file
        """
        if isinstance(data, HiveDataFrame):
            df = data
        elif isinstance(data, list):
            df = HiveDataFrame(data, self._hive)
        elif isinstance(data, str):
            # Assume it's a file path
            if data.endswith(".json"):
                df = HiveDataFrame.from_json(data, self._hive)
            elif data.endswith(".csv"):
                df = HiveDataFrame.from_csv(data, self._hive)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        self.catalog.register_table(name, df)

    def drop_table(self, name: str) -> None:
        """
        Remove a table from the catalog.

        Args:
            name: Table name to drop
        """
        self.catalog.drop_table(name)

    def table(self, name: str) -> Optional[HiveDataFrame]:
        """
        Get a table as a HiveDataFrame.

        Args:
            name: Table name

        Returns:
            HiveDataFrame or None if not found
        """
        return self.catalog.get_table(name)

    def tables(self) -> List[str]:
        """
        List all registered tables.

        Returns:
            List of table names
        """
        return self.catalog.list_tables()

    def sql(self, query: str) -> HiveDataFrame:
        """
        Execute a SQL query.

        Args:
            query: SQL query string

        Returns:
            HiveDataFrame with query results

        Example:
            result = ctx.sql("SELECT name, age FROM users WHERE age > 21")
        """
        # Tokenize
        tokenizer = SQLTokenizer(query)
        tokens = tokenizer.tokenize()

        # Parse
        parser = SQLParser(tokens)
        stmt = parser.parse()

        # Execute
        return self._executor.execute(stmt)

    def explain(self, query: str) -> str:
        """
        Get execution plan for a query.

        Args:
            query: SQL query string

        Returns:
            String representation of query plan
        """
        # Tokenize
        tokenizer = SQLTokenizer(query)
        tokens = tokenizer.tokenize()

        # Parse
        parser = SQLParser(tokens)
        stmt = parser.parse()

        # Build plan
        plan = self._executor._build_plan(stmt)

        return self._format_plan(plan, stmt)

    def _format_plan(self, plan, stmt: SQLStatement) -> str:
        """Format query plan as string."""
        lines = ["== Physical Plan =="]

        def format_node(node, indent=0):
            prefix = "  " * indent
            node_lines = [f"{prefix}* {node.type.name}"]

            if node.metadata:
                for key, value in node.metadata.items():
                    if value is not None:
                        node_lines.append(f"{prefix}  - {key}: {value}")

            if node.predicates:
                node_lines.append(f"{prefix}  - predicates: {len(node.predicates)}")

            if node.columns:
                node_lines.append(f"{prefix}  - columns: {node.columns}")

            for child in node.input_nodes:
                node_lines.extend(format_node(child, indent + 1))

            return node_lines

        if plan.root:
            lines.extend(format_node(plan.root))
        else:
            lines.append("  (empty plan)")

        return "\n".join(lines)

    def create_dataframe(self, data: List[Dict[str, Any]]) -> HiveDataFrame:
        """
        Create a HiveDataFrame from data.

        Args:
            data: List of dictionaries

        Returns:
            HiveDataFrame
        """
        return HiveDataFrame(data, self._hive)
