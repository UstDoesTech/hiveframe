"""
SQL Executor
============

Executes SQL statements by converting them to HiveDataFrame operations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

from ..dataframe import HiveDataFrame, Column, col, lit, GroupedData
from ..dataframe.aggregations import (
    sum_agg, avg, count, count_all, min_agg, max_agg, collect_list
)
from .parser import (
    SQLStatement, Expression, ColumnRef, Literal, FunctionCall,
    BinaryOp, UnaryOp, BetweenExpr, InExpr, CaseExpr, SelectColumn,
    JoinClause, OrderByItem
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
    input_nodes: List['PlanNode'] = field(default_factory=list)
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
    AGGREGATE_FUNCTIONS = {
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 
        'COLLECT_LIST', 'COLLECT_SET'
    }
    
    def __init__(self, catalog: 'SQLCatalog'):
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
        # Build query plan
        plan = self._build_plan(stmt)
        
        # Execute plan
        return self._execute_plan(plan, stmt)
        
    def _build_plan(self, stmt: SQLStatement) -> QueryPlan:
        """Build execution plan from SQL statement."""
        nodes = []
        tables = {}
        
        # Scan node (FROM clause)
        if stmt.from_table:
            tables[stmt.from_table.alias or stmt.from_table.name] = stmt.from_table.name
            scan_node = PlanNode(
                type=PlanNodeType.SCAN,
                metadata={'table': stmt.from_table.name, 'alias': stmt.from_table.alias}
            )
            nodes.append(scan_node)
            
        # Join nodes
        for join in stmt.joins:
            tables[join.table.alias or join.table.name] = join.table.name
            join_node = PlanNode(
                type=PlanNodeType.JOIN,
                input_nodes=nodes.copy(),
                metadata={
                    'table': join.table.name,
                    'alias': join.table.alias,
                    'type': join.type,
                    'condition': join.condition
                }
            )
            nodes = [join_node]
            
        # Filter node (WHERE clause)
        if stmt.where_clause:
            filter_node = PlanNode(
                type=PlanNodeType.FILTER,
                input_nodes=nodes.copy(),
                predicates=[stmt.where_clause]
            )
            nodes = [filter_node]
            
        # Aggregate node (GROUP BY)
        if stmt.group_by or self._has_aggregates(stmt):
            agg_node = PlanNode(
                type=PlanNodeType.AGGREGATE,
                input_nodes=nodes.copy(),
                metadata={
                    'group_by': stmt.group_by,
                    'having': stmt.having_clause,
                    'aggregates': self._extract_aggregates(stmt)
                }
            )
            nodes = [agg_node]
            
        # Project node (SELECT)
        project_node = PlanNode(
            type=PlanNodeType.PROJECT,
            input_nodes=nodes.copy(),
            columns=[self._column_name(sc) for sc in stmt.select_columns]
        )
        nodes = [project_node]
        
        # Distinct
        if stmt.distinct:
            distinct_node = PlanNode(
                type=PlanNodeType.DISTINCT,
                input_nodes=nodes.copy()
            )
            nodes = [distinct_node]
            
        # Sort node (ORDER BY)
        if stmt.order_by:
            sort_node = PlanNode(
                type=PlanNodeType.SORT,
                input_nodes=nodes.copy(),
                metadata={'order_by': stmt.order_by}
            )
            nodes = [sort_node]
            
        # Limit node
        if stmt.limit is not None:
            limit_node = PlanNode(
                type=PlanNodeType.LIMIT,
                input_nodes=nodes.copy(),
                metadata={'limit': stmt.limit, 'offset': stmt.offset}
            )
            nodes = [limit_node]
            
        return QueryPlan(root=nodes[0] if nodes else None, tables=tables)
        
    def _execute_plan(self, plan: QueryPlan, stmt: SQLStatement) -> HiveDataFrame:
        """Execute the query plan."""
        # Get base DataFrame
        if not stmt.from_table:
            # SELECT without FROM (e.g., SELECT 1+1)
            return self._execute_expression_only(stmt)
            
        df = self.catalog.get_table(stmt.from_table.name)
        if df is None:
            raise ValueError(f"Table '{stmt.from_table.name}' not found")
            
        # Handle JOINs
        for join in stmt.joins:
            other_df = self.catalog.get_table(join.table.name)
            if other_df is None:
                raise ValueError(f"Table '{join.table.name}' not found")
            df = self._execute_join(df, other_df, join)
            
        # Handle WHERE
        if stmt.where_clause:
            df = self._execute_filter(df, stmt.where_clause)
            
        # Handle GROUP BY and aggregations
        if stmt.group_by or self._has_aggregates(stmt):
            df = self._execute_aggregate(df, stmt)
        else:
            # Handle SELECT (projection)
            df = self._execute_projection(df, stmt.select_columns)
            
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
            value = self._eval_constant_expr(sc.expression)
            row[name] = value
        return HiveDataFrame([row])
        
    def _execute_join(self, 
                      left: HiveDataFrame, 
                      right: HiveDataFrame,
                      join: JoinClause) -> HiveDataFrame:
        """Execute JOIN operation."""
        # Extract join columns from condition
        join_cols = self._extract_join_columns(join.condition)
        
        how_map = {
            'INNER': 'inner',
            'LEFT': 'left',
            'RIGHT': 'right',
            'FULL': 'outer',
            'CROSS': 'cross'
        }
        how = how_map.get(join.type, 'inner')
        
        if how == 'cross':
            return left.crossJoin(right)
            
        if join_cols:
            return left.join(right, on=join_cols, how=how)
        else:
            # Cross join if no condition
            return left.crossJoin(right)
            
    def _execute_filter(self, df: HiveDataFrame, expr: Expression) -> HiveDataFrame:
        """Execute WHERE filter."""
        filter_col = self._expr_to_column(expr)
        return df.filter(filter_col)
        
    def _execute_aggregate(self, df: HiveDataFrame, stmt: SQLStatement) -> HiveDataFrame:
        """Execute GROUP BY with aggregations."""
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
            having_col = self._expr_to_column(stmt.having_clause)
            result = result.filter(having_col)
            
        return result
        
    def _execute_projection(self, 
                            df: HiveDataFrame, 
                            select_cols: List[SelectColumn]) -> HiveDataFrame:
        """Execute SELECT projection."""
        # Handle SELECT *
        if len(select_cols) == 1 and isinstance(select_cols[0].expression, ColumnRef):
            if select_cols[0].expression.name == '*':
                return df
                
        cols = []
        for sc in select_cols:
            expr_col = self._expr_to_column(sc.expression)
            if sc.alias:
                expr_col = expr_col.alias(sc.alias)
            cols.append(expr_col)
            
        return df.select(*cols)
        
    def _execute_order_by(self, 
                          df: HiveDataFrame, 
                          order_items: List[OrderByItem]) -> HiveDataFrame:
        """Execute ORDER BY."""
        # For simplicity, support single column ordering
        if order_items:
            item = order_items[0]
            if isinstance(item.expression, ColumnRef):
                return df.orderBy(item.expression.name, ascending=item.ascending)
            col_expr = self._expr_to_column(item.expression)
            return df.orderBy(col_expr, ascending=item.ascending)
        return df
        
    def _expr_to_column(self, expr: Expression) -> Column:
        """Convert SQL expression to Column."""
        if isinstance(expr, ColumnRef):
            return col(expr.name)
            
        if isinstance(expr, Literal):
            return lit(expr.value)
            
        if isinstance(expr, FunctionCall):
            return self._function_to_column(expr)
            
        if isinstance(expr, BinaryOp):
            left = self._expr_to_column(expr.left)
            right = self._expr_to_column(expr.right)
            
            ops = {
                '+': lambda l, r: l + r,
                '-': lambda l, r: l - r,
                '*': lambda l, r: l * r,
                '/': lambda l, r: l / r,
                '=': lambda l, r: l == r,
                '!=': lambda l, r: l != r,
                '<': lambda l, r: l < r,
                '<=': lambda l, r: l <= r,
                '>': lambda l, r: l > r,
                '>=': lambda l, r: l >= r,
                'AND': lambda l, r: l & r,
                'OR': lambda l, r: l | r,
                'LIKE': lambda l, r: l.contains(r._value if hasattr(r, '_value') else str(r)),
            }
            
            if expr.op in ops:
                return ops[expr.op](left, right)
            raise ValueError(f"Unsupported operator: {expr.op}")
            
        if isinstance(expr, UnaryOp):
            operand = self._expr_to_column(expr.operand)
            
            if expr.op == 'NOT':
                return ~operand
            if expr.op == '-':
                return -operand
            if expr.op == 'IS NULL':
                return operand.isNull()
            if expr.op == 'IS NOT NULL':
                return operand.isNotNull()
            raise ValueError(f"Unsupported unary operator: {expr.op}")
            
        if isinstance(expr, BetweenExpr):
            col_expr = self._expr_to_column(expr.expr)
            low = self._expr_to_column(expr.low)
            high = self._expr_to_column(expr.high)
            return (col_expr >= low) & (col_expr <= high)
            
        if isinstance(expr, InExpr):
            col_expr = self._expr_to_column(expr.expr)
            # For IN, we need to evaluate against constant list
            values = [self._eval_constant_expr(v) for v in expr.values]
            return col_expr.isin(values)
            
        if isinstance(expr, CaseExpr):
            # CASE expressions need special handling
            return self._case_to_column(expr)
            
        raise ValueError(f"Unsupported expression type: {type(expr)}")
        
    def _function_to_column(self, func: FunctionCall) -> Column:
        """Convert function call to Column."""
        name = func.name.upper()
        
        # String functions
        if name == 'UPPER':
            arg = self._expr_to_column(func.args[0])
            return arg.upper()
        if name == 'LOWER':
            arg = self._expr_to_column(func.args[0])
            return arg.lower()
        if name == 'LENGTH':
            arg = self._expr_to_column(func.args[0])
            return arg.length()
        if name == 'CONCAT':
            result = self._expr_to_column(func.args[0])
            for arg in func.args[1:]:
                result = result.concat(self._expr_to_column(arg))
            return result
        if name == 'SUBSTRING' or name == 'SUBSTR':
            s = self._expr_to_column(func.args[0])
            start = self._eval_constant_expr(func.args[1])
            length = self._eval_constant_expr(func.args[2]) if len(func.args) > 2 else None
            return s.substring(start, length)
            
        # Math functions
        if name == 'ABS':
            arg = self._expr_to_column(func.args[0])
            return arg.abs()
        if name == 'ROUND':
            arg = self._expr_to_column(func.args[0])
            decimals = self._eval_constant_expr(func.args[1]) if len(func.args) > 1 else 0
            return arg.round(decimals)
            
        # Date functions
        if name == 'NOW' or name == 'CURRENT_TIMESTAMP':
            from datetime import datetime
            return lit(datetime.now())
        if name == 'CURRENT_DATE':
            from datetime import date
            return lit(date.today())
            
        # Coalesce
        if name == 'COALESCE':
            return self._coalesce_column(func.args)
            
        # Aggregations (these return Column for use in select)
        if name in self.AGGREGATE_FUNCTIONS:
            arg = self._expr_to_column(func.args[0]) if func.args else col('*')
            return arg  # The aggregation is handled separately
            
        raise ValueError(f"Unsupported function: {name}")
        
    def _case_to_column(self, case: CaseExpr) -> Column:
        """Convert CASE expression to Column."""
        # Create a custom column that evaluates the CASE
        def eval_case(row: Dict) -> Any:
            for when_expr, then_expr in case.when_clauses:
                if case.operand:
                    # Simple CASE: compare operand to when_expr
                    operand_val = self._eval_expr_on_row(case.operand, row)
                    when_val = self._eval_expr_on_row(when_expr, row)
                    if operand_val == when_val:
                        return self._eval_expr_on_row(then_expr, row)
                else:
                    # Searched CASE: evaluate when_expr as boolean
                    when_val = self._eval_expr_on_row(when_expr, row)
                    if when_val:
                        return self._eval_expr_on_row(then_expr, row)
            if case.else_clause:
                return self._eval_expr_on_row(case.else_clause, row)
            return None
            
        return Column('case', eval_fn=eval_case)
        
    def _coalesce_column(self, args: List[Expression]) -> Column:
        """Create COALESCE column."""
        def eval_coalesce(row: Dict) -> Any:
            for arg in args:
                val = self._eval_expr_on_row(arg, row)
                if val is not None:
                    return val
            return None
        return Column('coalesce', eval_fn=eval_coalesce)
        
    def _eval_expr_on_row(self, expr: Expression, row: Dict) -> Any:
        """Evaluate expression on a row."""
        if isinstance(expr, ColumnRef):
            return row.get(expr.name)
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, BinaryOp):
            left = self._eval_expr_on_row(expr.left, row)
            right = self._eval_expr_on_row(expr.right, row)
            return self._eval_binary_op(expr.op, left, right)
        if isinstance(expr, UnaryOp):
            val = self._eval_expr_on_row(expr.operand, row)
            return self._eval_unary_op(expr.op, val)
        return None
        
    def _eval_binary_op(self, op: str, left: Any, right: Any) -> Any:
        """Evaluate binary operation."""
        if left is None or right is None:
            return None
        ops = {
            '+': lambda l, r: l + r,
            '-': lambda l, r: l - r,
            '*': lambda l, r: l * r,
            '/': lambda l, r: l / r if r != 0 else None,
            '=': lambda l, r: l == r,
            '!=': lambda l, r: l != r,
            '<': lambda l, r: l < r,
            '<=': lambda l, r: l <= r,
            '>': lambda l, r: l > r,
            '>=': lambda l, r: l >= r,
            'AND': lambda l, r: l and r,
            'OR': lambda l, r: l or r,
        }
        if op in ops:
            return ops[op](left, right)
        return None
        
    def _eval_unary_op(self, op: str, val: Any) -> Any:
        """Evaluate unary operation."""
        if op == 'NOT':
            return not val
        if op == '-':
            return -val if val is not None else None
        if op == 'IS NULL':
            return val is None
        if op == 'IS NOT NULL':
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
            return (self._expr_has_aggregate(expr.left) or 
                    self._expr_has_aggregate(expr.right))
        if isinstance(expr, UnaryOp):
            return self._expr_has_aggregate(expr.operand)
        return False
        
    def _extract_aggregates(self, stmt: SQLStatement) -> List[Tuple[str, Expression]]:
        """Extract aggregate functions from statement."""
        aggs = []
        for sc in stmt.select_columns:
            if isinstance(sc.expression, FunctionCall):
                if sc.expression.name.upper() in self.AGGREGATE_FUNCTIONS:
                    aggs.append((sc.alias or sc.expression.name, sc.expression))
        return aggs
        
    def _extract_agg_from_expr(self, expr: Expression, alias: Optional[str]):
        """Extract aggregation from expression."""
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name == 'COUNT':
                if expr.args and isinstance(expr.args[0], ColumnRef):
                    if expr.args[0].name == '*':
                        agg = count_all()
                    else:
                        agg = count(col(expr.args[0].name))
                else:
                    agg = count_all()
                if alias:
                    agg = agg.alias(alias)
                return agg
            if name == 'SUM':
                if expr.args:
                    agg = sum_agg(col(expr.args[0].name) 
                                  if isinstance(expr.args[0], ColumnRef) 
                                  else lit(self._eval_constant_expr(expr.args[0])))
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == 'AVG':
                if expr.args:
                    agg = avg(col(expr.args[0].name)
                             if isinstance(expr.args[0], ColumnRef)
                             else lit(self._eval_constant_expr(expr.args[0])))
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == 'MIN':
                if expr.args:
                    agg = min_agg(col(expr.args[0].name)
                                 if isinstance(expr.args[0], ColumnRef)
                                 else lit(self._eval_constant_expr(expr.args[0])))
                    if alias:
                        agg = agg.alias(alias)
                    return agg
            if name == 'MAX':
                if expr.args:
                    agg = max_agg(col(expr.args[0].name)
                                 if isinstance(expr.args[0], ColumnRef)
                                 else lit(self._eval_constant_expr(expr.args[0])))
                    if alias:
                        agg = agg.alias(alias)
                    return agg
        return None
        
    def _extract_join_columns(self, condition: Optional[Expression]) -> List[str]:
        """Extract column names from join condition."""
        if condition is None:
            return []
        if isinstance(condition, BinaryOp) and condition.op == '=':
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
