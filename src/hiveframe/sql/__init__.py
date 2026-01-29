"""
SwarmQL: HiveFrame SQL Engine
=============================

A bee-inspired SQL engine that translates SQL queries into
HiveFrame DataFrame operations.

Key Features:
- Standard SQL syntax support (SELECT, FROM, WHERE, GROUP BY, etc.)
- Integration with HiveFrame's bee-inspired processing engine
- Swarm-based query optimization
- Table registration and catalog management

Usage:
    from hiveframe.sql import SwarmQLContext

    ctx = SwarmQLContext()
    ctx.register_table("users", df)
    
    result = ctx.sql("SELECT name, age FROM users WHERE age > 21")
    result.show()
"""

from .context import SwarmQLContext, SQLCatalog
from .parser import SQLParser, SQLStatement, SQLTokenizer
from .executor import SQLExecutor, QueryPlan, PlanNode
from .types import (
    SQLType,
    IntegerType,
    FloatType,
    StringType,
    BooleanType,
    DateType,
    TimestampType,
    ArrayType,
    MapType,
)

__all__ = [
    # Context
    'SwarmQLContext',
    'SQLCatalog',
    # Parser
    'SQLParser',
    'SQLStatement',
    'SQLTokenizer',
    # Executor
    'SQLExecutor',
    'QueryPlan',
    'PlanNode',
    # Types
    'SQLType',
    'IntegerType',
    'FloatType',
    'StringType',
    'BooleanType',
    'DateType',
    'TimestampType',
    'ArrayType',
    'MapType',
]
