"""
SQL Executor
============

Executes SQL statements by converting them to HiveDataFrame operations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from ..dataframe import Column, HiveDataFrame, col, lit
from ..dataframe.aggregations import avg, count, count_all, max_agg, min_agg, sum_agg
from .parser import (
    BetweenExpr,
    BinaryOp,
    CaseExpr,
    ColumnRef,
    CommonTableExpression,
    ExistsExpr,
    Expression,
    FunctionCall,
    InExpr,
    JoinClause,
    Literal,
    OrderByItem,
    SelectColumn,
    SetOperation,
    SQLStatement,
    SubqueryExpr,
    UnaryOp,
    WindowFunction,
    WindowSpec,
)


class PlanNodeType(Enum):
    """Types of query plan nodes."""

    SCAN = auto()
    FILTER = auto()
    PROJECT = auto()
    AGGREGATE = auto()
    JOIN = auto()
    SORT = auto()
    LIMIT = auto()
    DISTINCT = auto()


@dataclass
class PlanNode:
    """
    Query Plan Node
    ---------------
    Represents a step in the query execution plan.
    """

    type: PlanNodeType
    input_nodes: List["PlanNode"] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    predicates: List[Expression] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_rows: int = 0


@dataclass
class QueryPlan:
    """
    Query Plan
    ----------
    Complete execution plan for a SQL query.
    """

    root: PlanNode
    tables: Dict[str, str] = field(default_factory=dict)  # alias -> name
    total_cost: float = 0.0


class SQLExecutor:
    """
    SQL Executor
    ------------
    Converts SQL statements to DataFrame operations and executes them.

    This is the bridge between SwarmQL parsing and HiveFrame execution.
    """

    # Supported aggregate functions
    AGGREGATE_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "COLLECT_LIST", "COLLECT_SET"}

    # Window functions
    WINDOW_FUNCTIONS = {
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "LAG",
        "LEAD",
        "FIRST_VALUE",
        "LAST_VALUE",
    }

    # String functions
    STRING_FUNCTIONS = {"SUBSTRING", "SUBSTR", "CONCAT", "UPPER", "LOWER", "TRIM", "LENGTH"}

    # Date functions
    DATE_FUNCTIONS = {
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
        "NOW",
        "DATE_ADD",
        "DATE_SUB",
        "DATE_DIFF",
        "EXTRACT",
    }

    # Other functions
    OTHER_FUNCTIONS = {"COALESCE", "NULLIF"}

    def __init__(self, catalog: "SQLCatalog"):
        """
        Initialize executor with catalog.

        Args:
            catalog: Table catalog for resolving table references
        """
        self.catalog = catalog

    def execute(self, stmt: SQLStatement) -> HiveDataFrame:
        """
        Execute a SQL statement.

        Args:
            stmt: Parsed SQL statement

        Returns:
            HiveDataFrame with query results
        """
        # Handle set operations (UNION, INTERSECT, EXCEPT)
        if stmt.set_operation:
            return self._execute_set_operation(stmt.set_operation)

        # Process CTEs (WITH clause)
        cte_catalog = {}
        for cte in stmt.ctes:
            cte_result = self.execute(cte.query)
            cte_catalog[cte.name.lower()] = cte_result

        # Build query plan
        plan = self._build_plan(stmt)

        # Execute plan with CTE context
        return self._execute_plan(plan, stmt, cte_catalog)

    def _build_plan(self, stmt: SQLStatement) -> QueryPlan:
        """Build execution plan from SQL statement."""
        nodes = []
        tables = {}

        # Scan node (FROM clause)
        if stmt.from_table:
            tables[stmt.from_table.alias or stmt.from_table.name] = stmt.from_table.name
            scan_node = PlanNode(
                type=PlanNodeType.SCAN,
                metadata={"table": stmt.from_table.name, "alias": stmt.from_table.alias},
            )
            nodes.append(scan_node)

        # Join nodes
        for join in stmt.joins:
            tables[join.table.alias or join.table.name] = join.table.name
            join_node = PlanNode(
                type=PlanNodeType.JOIN,
                input_nodes=nodes.copy(),
                metadata={
                    "table": join.table.name,
                    "alias": join.table.alias,
                    "type": join.type,
                    "condition": join.condition,
                },
            )
            nodes = [join_node]

        # Filter node (WHERE clause)
        if stmt.where_clause:
            filter_node = PlanNode(
                type=PlanNodeType.FILTER, input_nodes=nodes.copy(), predicates=[stmt.where_clause]
            )
            nodes = [filter_node]

        # Aggregate node (GROUP BY)
        if stmt.group_by or self._has_aggregates(stmt):
            agg_node = PlanNode(
                type=PlanNodeType.AGGREGATE,
                input_nodes=nodes.copy(),
                metadata={
                    "group_by": stmt.group_by,
                    "having": stmt.having_clause,
                    "aggregates": self._extract_aggregates(stmt),
                },
            )
            nodes = [agg_node]

        # Project node (SELECT)
        project_node = PlanNode(
            type=PlanNodeType.PROJECT,
            input_nodes=nodes.copy(),
            columns=[self._column_name(sc) for sc in stmt.select_columns],
        )
        nodes = [project_node]

        # Distinct
        if stmt.distinct:
            distinct_node = PlanNode(type=PlanNodeType.DISTINCT, input_nodes=nodes.copy())
            nodes = [distinct_node]

        # Sort node (ORDER BY)
        if stmt.order_by:
            sort_node = PlanNode(
                type=PlanNodeType.SORT,
                input_nodes=nodes.copy(),
                metadata={"order_by": stmt.order_by},
            )
            nodes = [sort_node]

        # Limit node
        if stmt.limit is not None:
            limit_node = PlanNode(
                type=PlanNodeType.LIMIT,
                input_nodes=nodes.copy(),
                metadata={"limit": stmt.limit, "offset": stmt.offset},
            )
            nodes = [limit_node]

        default_node = PlanNode(type=PlanNodeType.SCAN)
        root = nodes[0] if nodes else default_node
        return QueryPlan(root=root, tables=tables)

    def _execute_plan(
        self, plan: QueryPlan, stmt: SQLStatement, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> HiveDataFrame:
        """Execute the query plan."""
        if cte_catalog is None:
            cte_catalog = {}

        # Get base DataFrame
        if not stmt.from_table:
            # SELECT without FROM (e.g., SELECT 1+1)
            return self._execute_expression_only(stmt)

        # Check if table is a CTE
        table_name = stmt.from_table.name.lower()
        if table_name in cte_catalog:
            df = cte_catalog[table_name]
        elif stmt.from_table.is_subquery and stmt.from_table.subquery:
            # Subquery in FROM clause
            df = self.execute(stmt.from_table.subquery)
        else:
            df = self.catalog.get_table(stmt.from_table.name)
            if df is None:
                raise ValueError(f"Table '{stmt.from_table.name}' not found")

        # Handle JOINs
        for join in stmt.joins:
            # Check for bee-inspired WAGGLE JOIN hint
            if join.type == "WAGGLE":
                # WAGGLE JOIN: use quality-weighted join execution
                # In production, this would trigger adaptive join strategy selection
                # For now, treat as inner join with hint metadata
                join.type = "INNER"

            # Check if join table is CTE
            join_table_name = join.table.name.lower()
            if join_table_name in cte_catalog:
                other_df = cte_catalog[join_table_name]
            elif join.table.is_subquery and join.table.subquery:
                other_df = self.execute(join.table.subquery)
            else:
                other_df = self.catalog.get_table(join.table.name)
                if other_df is None:
                    raise ValueError(f"Table '{join.table.name}' not found")
            df = self._execute_join(df, other_df, join)

        # Handle WHERE
        if stmt.where_clause:
            df = self._execute_filter(df, stmt.where_clause, cte_catalog)

        # Handle GROUP BY and aggregations
        if stmt.group_by or self._has_aggregates(stmt):
            df = self._execute_aggregate(df, stmt, cte_catalog)
        else:
            # Handle SELECT (projection)
            df = self._execute_projection(df, stmt.select_columns, cte_catalog)

        # Handle DISTINCT
        if stmt.distinct:
            df = df.distinct()

        # Handle ORDER BY
        if stmt.order_by:
            df = self._execute_order_by(df, stmt.order_by)

        # Handle LIMIT
        if stmt.limit is not None:
            offset = stmt.offset or 0
            if offset > 0:
                # Skip offset rows
                data = df.collect()[offset:]
                df = HiveDataFrame(data, df._hive)
            df = df.limit(stmt.limit)

        return df

    def _execute_expression_only(self, stmt: SQLStatement) -> HiveDataFrame:
        """Execute SELECT without FROM clause."""
        # Create single row with computed expressions
        row = {}
        for sc in stmt.select_columns:
            name = sc.alias or self._expr_name(sc.expression)
            # Evaluate the expression as if on an empty row
            value = self._eval_expr_constant_or_function(sc.expression)
            row[name] = value
        return HiveDataFrame([row])

    def _execute_join(
        self, left: HiveDataFrame, right: HiveDataFrame, join: JoinClause
    ) -> HiveDataFrame:
        """Execute JOIN operation."""
        # Extract join columns from condition
        join_cols = self._extract_join_columns(join.condition)

        how_map = {
            "INNER": "inner",
            "LEFT": "left",
            "RIGHT": "right",
            "FULL": "outer",
            "CROSS": "cross",
        }
        how = how_map.get(join.type, "inner")

        if how == "cross":
            return left.crossJoin(right)

        if join_cols:
            return left.join(right, on=join_cols, how=how)
        else:
            # Cross join if no condition
            return left.crossJoin(right)

    def _execute_filter(
        self, df: HiveDataFrame, expr: Expression, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> HiveDataFrame:
        """Execute WHERE filter."""
        if cte_catalog is None:
            cte_catalog = {}
        filter_col = self._expr_to_column(expr, cte_catalog)
        return df.filter(filter_col)

    def _execute_aggregate(
        self, df: HiveDataFrame, stmt: SQLStatement, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> HiveDataFrame:
        """Execute GROUP BY with aggregations."""
        if cte_catalog is None:
            cte_catalog = {}
        # Extract group by columns
        group_cols = []
        for expr in stmt.group_by:
            if isinstance(expr, ColumnRef):
                group_cols.append(expr.name)
            else:
                # For complex expressions, need to compute first
                group_cols.append(self._expr_name(expr))

        # Extract aggregations from SELECT
        aggs = []
        for sc in stmt.select_columns:
            agg = self._extract_agg_from_expr(sc.expression, sc.alias)
            if agg:
                aggs.append(agg)

        if group_cols:
            grouped = df.groupBy(*group_cols)
            if aggs:
                result = grouped.agg(*aggs)
            else:
                # Just group by without aggregations
                result = grouped.agg(count(col(group_cols[0])))
        else:
            # Global aggregation
            if aggs:
                # Compute aggregations over all data
                data = df.collect()
                row = {}
                for agg in aggs:
                    row[agg.name] = agg.apply(data)
                result = HiveDataFrame([row], df._hive)
            else:
                result = df

        # Handle HAVING
        if stmt.having_clause:
            having_col = self._expr_to_column(stmt.having_clause, cte_catalog)
            result = result.filter(having_col)

        return result

    def _execute_projection(
        self,
        df: HiveDataFrame,
        select_cols: List[SelectColumn],
        cte_catalog: Dict[str, HiveDataFrame] = None,
    ) -> HiveDataFrame:
        """Execute SELECT projection."""
        if cte_catalog is None:
            cte_catalog = {}
        # Handle SELECT *
        if len(select_cols) == 1 and isinstance(select_cols[0].expression, ColumnRef):
            if select_cols[0].expression.name == "*":
                return df

        cols = []
        for sc in select_cols:
            expr_col = self._expr_to_column(sc.expression, cte_catalog)
            if sc.alias:
                expr_col = expr_col.alias(sc.alias)
            cols.append(expr_col)

        return df.select(*cols)

    def _execute_order_by(self, df: HiveDataFrame, order_items: List[OrderByItem]) -> HiveDataFrame:
        """Execute ORDER BY."""
        # For simplicity, support single column ordering
        if order_items:
            item = order_items[0]
            if isinstance(item.expression, ColumnRef):
                return df.orderBy(item.expression.name, ascending=item.ascending)
            col_expr = self._expr_to_column(item.expression)
            return df.orderBy(col_expr, ascending=item.ascending)
        return df

    def _expr_to_column(
        self, expr: Expression, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Column:
        """Convert SQL expression to Column."""
        if cte_catalog is None:
            cte_catalog = {}

        if isinstance(expr, ColumnRef):
            return col(expr.name)

        if isinstance(expr, Literal):
            return lit(expr.value)

        if isinstance(expr, FunctionCall):
            return self._function_to_column(expr, cte_catalog)

        if isinstance(expr, WindowFunction):
            return self._window_function_to_column(expr, cte_catalog)

        if isinstance(expr, BinaryOp):
            left = self._expr_to_column(expr.left, cte_catalog)
            right = self._expr_to_column(expr.right, cte_catalog)

            ops = {
                "+": lambda lhs, rhs: lhs + rhs,
                "-": lambda lhs, rhs: lhs - rhs,
                "*": lambda lhs, rhs: lhs * rhs,
                "/": lambda lhs, rhs: lhs / rhs,
                "=": lambda lhs, rhs: lhs == rhs,
                "!=": lambda lhs, rhs: lhs != rhs,
                "<": lambda lhs, rhs: lhs < rhs,
                "<=": lambda lhs, rhs: lhs <= rhs,
                ">": lambda lhs, rhs: lhs > rhs,
                ">=": lambda lhs, rhs: lhs >= rhs,
                "AND": lambda lhs, rhs: lhs & rhs,
                "OR": lambda lhs, rhs: lhs | rhs,
                "LIKE": lambda lhs, rhs: lhs.contains(
                    rhs._value if hasattr(rhs, "_value") else str(rhs)
                ),
            }

            if expr.op in ops:
                return ops[expr.op](left, right)  # type: ignore
            raise ValueError(f"Unsupported operator: {expr.op}")

        if isinstance(expr, UnaryOp):
            operand = self._expr_to_column(expr.operand, cte_catalog)

            if expr.op == "NOT":
                return ~operand
            if expr.op == "-":
                return operand * lit(-1)  # type: ignore
            if expr.op == "IS NULL":
                return operand.isNull()
            if expr.op == "IS NOT NULL":
                return operand.isNotNull()
            raise ValueError(f"Unsupported unary operator: {expr.op}")

        if isinstance(expr, BetweenExpr):
            col_expr = self._expr_to_column(expr.expr, cte_catalog)
            low = self._expr_to_column(expr.low, cte_catalog)
            high = self._expr_to_column(expr.high, cte_catalog)
            return (col_expr >= low) & (col_expr <= high)

        if isinstance(expr, InExpr):
            col_expr = self._expr_to_column(expr.expr, cte_catalog)

            # Handle IN with subquery
            if expr.subquery:
                # Execute subquery and get values
                subquery_df = self.execute(expr.subquery)
                subquery_data = subquery_df.collect()
                # Assume subquery returns single column
                if subquery_data:
                    first_col = list(subquery_data[0].keys())[0]
                    values = [row[first_col] for row in subquery_data]
                else:
                    values = []
            else:
                # For IN, we need to evaluate against constant list
                values = [self._eval_constant_expr(v) for v in expr.values]

            # Use a custom function since Column doesn't have isin
            def check_in(row: Dict) -> bool:
                val = col_expr.eval(row)
                return val in values

            return Column("in_check", _expr=check_in)  # type: ignore

        if isinstance(expr, ExistsExpr):
            # EXISTS subquery
            subquery_df = self.execute(expr.subquery)
            exists = subquery_df.count() > 0
            result = not exists if expr.negated else exists
            return lit(result)

        if isinstance(expr, SubqueryExpr):
            # Scalar subquery - should return single value
            subquery_df = self.execute(expr.query)
            data = subquery_df.collect()
            if len(data) == 1:
                first_col = list(data[0].keys())[0]
                return lit(data[0][first_col])
            elif len(data) == 0:
                return lit(None)
            else:
                raise ValueError("Scalar subquery returned more than one row")

        if isinstance(expr, CaseExpr):
            # CASE expressions need special handling
            return self._case_to_column(expr, cte_catalog)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _function_to_column(
        self, func: FunctionCall, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Column:
        """Convert function call to Column."""
        if cte_catalog is None:
            cte_catalog = {}
        name = func.name.upper()

        # String functions
        if name == "UPPER":
            arg = self._expr_to_column(func.args[0], cte_catalog)

            def upper_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return str(val).upper() if val is not None else None

            return Column("upper", _expr=upper_fn)  # type: ignore
        if name == "LOWER":
            arg = self._expr_to_column(func.args[0], cte_catalog)

            def lower_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return str(val).lower() if val is not None else None

            return Column("lower", _expr=lower_fn)  # type: ignore
        if name == "TRIM":
            arg = self._expr_to_column(func.args[0], cte_catalog)

            def trim_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return str(val).strip() if val is not None else None

            return Column("trim", _expr=trim_fn)  # type: ignore
        if name == "LENGTH":
            arg = self._expr_to_column(func.args[0], cte_catalog)

            def length_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return len(str(val)) if val is not None else None

            return Column("length", _expr=length_fn)  # type: ignore
        if name == "CONCAT":
            args = [self._expr_to_column(a, cte_catalog) for a in func.args]

            def concat_fn(row: Dict) -> Any:
                parts = []
                for arg in args:
                    val = arg.eval(row)
                    parts.append(str(val) if val is not None else "")
                return "".join(parts)

            return Column("concat", _expr=concat_fn)  # type: ignore
        if name == "SUBSTRING" or name == "SUBSTR":
            s = self._expr_to_column(func.args[0], cte_catalog)
            start = self._eval_constant_expr(func.args[1])
            length = self._eval_constant_expr(func.args[2]) if len(func.args) > 2 else None

            def substr_fn(row: Dict) -> Any:
                val = s.eval(row)
                if val is None:
                    return None
                str_val = str(val)
                if length is not None:
                    return str_val[start : start + length]
                return str_val[start:]

            return Column("substring", _expr=substr_fn)  # type: ignore

        # Math functions
        if name == "ABS":
            arg = self._expr_to_column(func.args[0], cte_catalog)

            def abs_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return abs(val) if val is not None else None

            return Column("abs", _expr=abs_fn)  # type: ignore
        if name == "ROUND":
            arg = self._expr_to_column(func.args[0], cte_catalog)
            decimals = self._eval_constant_expr(func.args[1]) if len(func.args) > 1 else 0

            def round_fn(row: Dict) -> Any:
                val = arg.eval(row)
                return round(val, decimals) if val is not None else None

            return Column("round", _expr=round_fn)  # type: ignore

        # Date/Time functions
        if name == "NOW" or name == "CURRENT_TIMESTAMP":
            from datetime import datetime

            return lit(datetime.now())
        if name == "CURRENT_DATE":
            from datetime import date

            return lit(date.today())
        if name == "DATE_ADD":
            date_arg = self._expr_to_column(func.args[0], cte_catalog)
            days = self._eval_constant_expr(func.args[1])

            def date_add_fn(row: Dict) -> Any:
                from datetime import timedelta

                val = date_arg.eval(row)
                if val is not None:
                    return val + timedelta(days=days)
                return None

            return Column("date_add", _expr=date_add_fn)  # type: ignore
        if name == "DATE_SUB":
            date_arg = self._expr_to_column(func.args[0], cte_catalog)
            days = self._eval_constant_expr(func.args[1])

            def date_sub_fn(row: Dict) -> Any:
                from datetime import timedelta

                val = date_arg.eval(row)
                if val is not None:
                    return val - timedelta(days=days)
                return None

            return Column("date_sub", _expr=date_sub_fn)  # type: ignore
        if name == "DATE_DIFF":
            date1 = self._expr_to_column(func.args[0], cte_catalog)
            date2 = self._expr_to_column(func.args[1], cte_catalog)

            def date_diff_fn(row: Dict) -> Any:
                val1 = date1.eval(row)
                val2 = date2.eval(row)
                if val1 is not None and val2 is not None:
                    return (val1 - val2).days
                return None

            return Column("date_diff", _expr=date_diff_fn)  # type: ignore
        if name == "EXTRACT":
            # EXTRACT(field FROM source)
            # For simplicity, assume args[0] is field name, args[1] is source
            field = func.args[0]
            if isinstance(field, ColumnRef):
                field_name = field.name.upper()
            else:
                field_name = str(field)
            source = self._expr_to_column(func.args[1], cte_catalog)

            def extract_fn(row: Dict) -> Any:
                val = source.eval(row)
                if val is not None:
                    if field_name == "YEAR":
                        return val.year
                    elif field_name == "MONTH":
                        return val.month
                    elif field_name == "DAY":
                        return val.day
                    elif field_name == "HOUR":
                        return val.hour if hasattr(val, "hour") else None
                    elif field_name == "MINUTE":
                        return val.minute if hasattr(val, "minute") else None
                    elif field_name == "SECOND":
                        return val.second if hasattr(val, "second") else None
                return None

            return Column("extract", _expr=extract_fn)  # type: ignore

        # Coalesce and NULLIF
        if name == "COALESCE":
            return self._coalesce_column(func.args, cte_catalog)
        if name == "NULLIF":
            arg1 = self._expr_to_column(func.args[0], cte_catalog)
            arg2 = self._expr_to_column(func.args[1], cte_catalog)

            def nullif_fn(row: Dict) -> Any:
                val1 = arg1.eval(row)
                val2 = arg2.eval(row)
                if val1 == val2:
                    return None
                return val1

            return Column("nullif", _expr=nullif_fn)  # type: ignore

        # Aggregations (these return Column for use in select)
        if name in self.AGGREGATE_FUNCTIONS:
            arg = self._expr_to_column(func.args[0], cte_catalog) if func.args else col("*")
            return arg  # The aggregation is handled separately

        raise ValueError(f"Unsupported function: {name}")

    def _case_to_column(
        self, case: CaseExpr, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Column:
        """Convert CASE expression to Column."""
        if cte_catalog is None:
            cte_catalog = {}

        # Create a custom column that evaluates the CASE
        def eval_case(row: Dict) -> Any:
            for when_expr, then_expr in case.when_clauses:
                if case.operand:
                    # Simple CASE: compare operand to when_expr
                    operand_val = self._eval_expr_on_row(case.operand, row, cte_catalog)
                    when_val = self._eval_expr_on_row(when_expr, row, cte_catalog)
                    if operand_val == when_val:
                        return self._eval_expr_on_row(then_expr, row, cte_catalog)
                else:
                    # Searched CASE: evaluate when_expr as boolean
                    when_val = self._eval_expr_on_row(when_expr, row, cte_catalog)
                    if when_val:
                        return self._eval_expr_on_row(then_expr, row, cte_catalog)
            if case.else_clause:
                return self._eval_expr_on_row(case.else_clause, row, cte_catalog)
            return None

        return Column("case", _expr=eval_case)  # type: ignore

    def _coalesce_column(
        self, args: List[Expression], cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Column:
        """Create COALESCE column."""
        if cte_catalog is None:
            cte_catalog = {}

        def eval_coalesce(row: Dict) -> Any:
            for arg in args:
                val = self._eval_expr_on_row(arg, row, cte_catalog)
                if val is not None:
                    return val
            return None

        return Column("coalesce", _expr=eval_coalesce)  # type: ignore

    def _eval_expr_on_row(
        self, expr: Expression, row: Dict, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Any:
        """Evaluate expression on a row."""
        if cte_catalog is None:
            cte_catalog = {}
        if isinstance(expr, ColumnRef):
            return row.get(expr.name)
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, BinaryOp):
            left = self._eval_expr_on_row(expr.left, row, cte_catalog)
            right = self._eval_expr_on_row(expr.right, row, cte_catalog)
            return self._eval_binary_op(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_expr_on_row(expr.operand, row, cte_catalog)
            return self._eval_unary_op(expr.op, val)
        return None

    def _eval_binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate binary operation."""
        if left is None or right is None:
            return None
        ops = {
            "+": lambda lhs, rhs: lhs + rhs,
            "-": lambda lhs, rhs: lhs - rhs,
            "*": lambda lhs, rhs: lhs * rhs,
            "/": lambda lhs, rhs: lhs / rhs if rhs != 0 else None,
            "=": lambda lhs, rhs: lhs == rhs,
            "!=": lambda lhs, rhs: lhs != rhs,
            "<": lambda lhs, rhs: lhs < rhs,
            "<=": lambda lhs, rhs: lhs <= rhs,
            ">": lambda lhs, rhs: lhs > rhs,
            ">=": lambda lhs, rhs: lhs >= rhs,
            "AND": lambda lhs, rhs: lhs and rhs,
            "OR": lambda lhs, rhs: lhs or rhs,
        }
        if op in ops:
            return ops[op](left, right)
        return None

    def _eval_unary_op(self, op: str, val: Any) -> Any:
        """Evaluate unary operation."""
        if op == "NOT":
            return not val
        if op == "-":
            return -val if val is not None else None
        if op == "IS NULL":
            return val is None
        if op == "IS NOT NULL":
            return val is not None
        return None

    def _eval_constant_expr(self, expr: Expression) -> Any:
        """Evaluate constant expression."""
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, BinaryOp):
            left = self._eval_constant_expr(expr.left)
            right = self._eval_constant_expr(expr.right)
            return self._eval_binary_op(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_constant_expr(expr.operand)
            return self._eval_unary_op(expr.op, val)
        return None

    def _eval_expr_constant_or_function(self, expr: Expression) -> Any:
        """Evaluate constant expression or function call."""
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, FunctionCall):
            # Handle functions that work on constants
            name = expr.name.upper()
            if name == "UPPER":
                arg_val = self._eval_expr_constant_or_function(expr.args[0])
                return str(arg_val).upper() if arg_val is not None else None
            if name == "LOWER":
                arg_val = self._eval_expr_constant_or_function(expr.args[0])
                return str(arg_val).lower() if arg_val is not None else None
            if name == "TRIM":
                arg_val = self._eval_expr_constant_or_function(expr.args[0])
                return str(arg_val).strip() if arg_val is not None else None
            if name == "LENGTH":
                arg_val = self._eval_expr_constant_or_function(expr.args[0])
                return len(str(arg_val)) if arg_val is not None else None
            if name == "CONCAT":
                parts = []
                for arg in expr.args:
                    val = self._eval_expr_constant_or_function(arg)
                    parts.append(str(val) if val is not None else "")
                return "".join(parts)
            if name == "SUBSTRING" or name == "SUBSTR":
                s_val = self._eval_expr_constant_or_function(expr.args[0])
                start = self._eval_constant_expr(expr.args[1])
                length = self._eval_constant_expr(expr.args[2]) if len(expr.args) > 2 else None
                if s_val is None:
                    return None
                str_val = str(s_val)
                if length is not None:
                    return str_val[start : start + length]
                return str_val[start:]
            if name == "COALESCE":
                for arg in expr.args:
                    val = self._eval_expr_constant_or_function(arg)
                    if val is not None:
                        return val
                return None
            if name == "CURRENT_DATE":
                from datetime import date

                return date.today()
            if name == "CURRENT_TIMESTAMP" or name == "NOW":
                from datetime import datetime

                return datetime.now()
            # For other functions, return None
            return None
        if isinstance(expr, BinaryOp):
            left = self._eval_expr_constant_or_function(expr.left)
            right = self._eval_expr_constant_or_function(expr.right)
            return self._eval_binary_op(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_expr_constant_or_function(expr.operand)
            return self._eval_unary_op(expr.op, val)
        return None

    def _has_aggregates(self, stmt: SQLStatement) -> bool:
        """Check if statement has aggregate functions."""
        for sc in stmt.select_columns:
            if self._expr_has_aggregate(sc.expression):
                return True
        return False

    def _expr_has_aggregate(self, expr: Expression) -> bool:
        """Check if expression contains aggregates."""
        if isinstance(expr, FunctionCall):
            if expr.name.upper() in self.AGGREGATE_FUNCTIONS:
                return True
            return any(self._expr_has_aggregate(a) for a in expr.args)
        if isinstance(expr, BinaryOp):
            return self._expr_has_aggregate(expr.left) or self._expr_has_aggregate(expr.right)
        if isinstance(expr, UnaryOp):
            return self._expr_has_aggregate(expr.operand)
        return False

    def _extract_aggregates(self, stmt: SQLStatement) -> List[Tuple[str, Expression]]:
        """Extract aggregate functions from statement."""
        aggs: List[Tuple[str, Expression]] = []
        for sc in stmt.select_columns:
            if isinstance(sc.expression, FunctionCall):
                if sc.expression.name.upper() in self.AGGREGATE_FUNCTIONS:
                    aggs.append((sc.alias or sc.expression.name, sc.expression))
        return aggs

    def _extract_agg_from_expr(self, expr: Expression, alias: Optional[str]):
        """Extract aggregation from expression."""
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name == "COUNT":
                if expr.args and isinstance(expr.args[0], ColumnRef):
                    if expr.args[0].name == "*":
                        agg = count_all()
                    else:
                        agg = count(col(expr.args[0].name))
                else:
                    agg = count_all()
                if alias:
                    agg = agg.alias(alias)
                return agg
            if name == "SUM":
                if expr.args:
                    agg = sum_agg(
                        col(expr.args[0].name)
                        if isinstance(expr.args[0], ColumnRef)
                        else lit(self._eval_constant_expr(expr.args[0]))
                    )
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == "AVG":
                if expr.args:
                    agg = avg(
                        col(expr.args[0].name)
                        if isinstance(expr.args[0], ColumnRef)
                        else lit(self._eval_constant_expr(expr.args[0]))
                    )
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == "MIN":
                if expr.args:
                    agg = min_agg(
                        col(expr.args[0].name)
                        if isinstance(expr.args[0], ColumnRef)
                        else lit(self._eval_constant_expr(expr.args[0]))
                    )
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == "MAX":
                if expr.args:
                    agg = max_agg(
                        col(expr.args[0].name)
                        if isinstance(expr.args[0], ColumnRef)
                        else lit(self._eval_constant_expr(expr.args[0]))
                    )
                    if alias:
                        agg = agg.alias(alias)
                    return agg
        return None

    def _extract_join_columns(self, condition: Optional[Expression]) -> List[str]:
        """Extract column names from join condition."""
        if condition is None:
            return []
        if isinstance(condition, BinaryOp) and condition.op == "=":
            cols = []
            if isinstance(condition.left, ColumnRef):
                cols.append(condition.left.name)
            if isinstance(condition.right, ColumnRef):
                # Only add if it's the same column name (common join pattern)
                if condition.right.name in cols:
                    return cols
                cols.append(condition.right.name)
            return cols[:1] if cols else []  # Return first column for simple joins
        return []

    def _column_name(self, sc: SelectColumn) -> str:
        """Get column name from SelectColumn."""
        if sc.alias:
            return sc.alias
        return self._expr_name(sc.expression)

    def _expr_name(self, expr: Expression) -> str:
        """Get name for expression."""
        if isinstance(expr, ColumnRef):
            return expr.name
        if isinstance(expr, Literal):
            return str(expr.value)
        if isinstance(expr, FunctionCall):
            return f"{expr.name.lower()}_{self._expr_name(expr.args[0]) if expr.args else ''}"
        if isinstance(expr, BinaryOp):
            return f"{self._expr_name(expr.left)}_{expr.op}_{self._expr_name(expr.right)}"
        return "expr"

    def _execute_set_operation(self, set_op: SetOperation) -> HiveDataFrame:
        """
        Execute set operations (UNION, INTERSECT, EXCEPT).

        Set operations combine results from two SELECT statements.
        """
        left_df = self.execute(set_op.left)
        right_df = self.execute(set_op.right)

        if set_op.type == "UNION":
            if set_op.all:
                # UNION ALL: keep duplicates
                left_data = left_df.collect()
                right_data = right_df.collect()
                return HiveDataFrame(left_data + right_data, left_df._hive)
            else:
                # UNION: remove duplicates
                return left_df.union(right_df).distinct()

        elif set_op.type == "INTERSECT":
            # Return rows that appear in both datasets
            left_data = left_df.collect()
            right_data = right_df.collect()

            # Convert to hashable tuples for set intersection
            left_set = {tuple(sorted(row.items())) for row in left_data}
            right_set = {tuple(sorted(row.items())) for row in right_data}

            intersect = left_set & right_set
            result_data = [dict(item) for item in intersect]
            return HiveDataFrame(result_data, left_df._hive)

        elif set_op.type == "EXCEPT":
            # Return rows in left that are not in right
            left_data = left_df.collect()
            right_data = right_df.collect()

            left_set = {tuple(sorted(row.items())) for row in left_data}
            right_set = {tuple(sorted(row.items())) for row in right_data}

            except_set = left_set - right_set
            result_data = [dict(item) for item in except_set]
            return HiveDataFrame(result_data, left_df._hive)

        raise ValueError(f"Unsupported set operation: {set_op.type}")

    def _window_function_to_column(
        self, window_func: WindowFunction, cte_catalog: Dict[str, HiveDataFrame] = None
    ) -> Column:
        """
        Convert window function to Column.

        Window functions perform calculations across a set of rows related to the current row.
        Supports: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, etc.
        """
        if cte_catalog is None:
            cte_catalog = {}

        func_name = window_func.function.name.upper()

        # For window functions, we need access to all data
        # This is a simplified implementation
        # In production, this would be more efficient

        if func_name == "ROW_NUMBER":
            # Assign sequential number to each row within partition
            def row_number_fn(row: Dict) -> Any:
                # This is simplified - proper implementation would need partition context
                return 1

            return Column("row_number", _expr=row_number_fn)  # type: ignore

        elif func_name == "RANK":
            # Assign rank with gaps for ties
            def rank_fn(row: Dict) -> Any:
                return 1

            return Column("rank", _expr=rank_fn)  # type: ignore

        elif func_name == "DENSE_RANK":
            # Assign rank without gaps for ties
            def dense_rank_fn(row: Dict) -> Any:
                return 1

            return Column("dense_rank", _expr=dense_rank_fn)  # type: ignore

        elif func_name == "LAG":
            # Access previous row value
            def lag_fn(row: Dict) -> Any:
                return None

            return Column("lag", _expr=lag_fn)  # type: ignore

        elif func_name == "LEAD":
            # Access next row value
            def lead_fn(row: Dict) -> Any:
                return None

            return Column("lead", _expr=lead_fn)  # type: ignore

        raise ValueError(f"Unsupported window function: {func_name}")


class SQLCatalog:
    """
    SQL Catalog
    -----------
    Manages table registrations for SwarmQL.
    """

    def __init__(self):
        self._tables: Dict[str, HiveDataFrame] = {}

    def register_table(self, name: str, df: HiveDataFrame) -> None:
        """Register a DataFrame as a table."""
        self._tables[name.lower()] = df

    def drop_table(self, name: str) -> None:
        """Remove a table from the catalog."""
        self._tables.pop(name.lower(), None)

    def get_table(self, name: str) -> Optional[HiveDataFrame]:
        """Get a registered table."""
        return self._tables.get(name.lower())

    def list_tables(self) -> List[str]:
        """List all registered tables."""
        return list(self._tables.keys())

    def table_exists(self, name: str) -> bool:
        """Check if table exists."""
        return name.lower() in self._tables
