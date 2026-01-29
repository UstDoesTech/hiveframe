"""
Cost Model
==========

Bee-inspired cost estimation for query optimization.
Uses fitness functions instead of traditional cost models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

from .rules import PlanNode, NodeType


@dataclass
class Statistics:
    """
    Table/Column Statistics
    -----------------------
    Statistics used for cost estimation.
    """
    row_count: int = 0
    column_count: int = 0
    distinct_values: Dict[str, int] = field(default_factory=dict)
    null_count: Dict[str, int] = field(default_factory=dict)
    min_values: Dict[str, Any] = field(default_factory=dict)
    max_values: Dict[str, Any] = field(default_factory=dict)
    avg_row_size: int = 100  # bytes
    
    def selectivity(self, column: str, predicate_type: str) -> float:
        """
        Estimate selectivity of a predicate.
        
        Args:
            column: Column name
            predicate_type: Type of predicate (=, <, >, LIKE, etc.)
            
        Returns:
            Estimated fraction of rows that match
        """
        if self.row_count == 0:
            return 0.5
            
        distinct = self.distinct_values.get(column, self.row_count)
        
        # Standard selectivity estimates
        if predicate_type == '=':
            return 1.0 / distinct if distinct > 0 else 0.5
        elif predicate_type in ('<', '>', '<=', '>='):
            return 0.33  # Assume 1/3 for range predicates
        elif predicate_type == 'LIKE':
            return 0.1  # Assume 10% for pattern matching
        elif predicate_type == 'IS NULL':
            null_count = self.null_count.get(column, 0)
            return null_count / self.row_count if self.row_count > 0 else 0.05
        elif predicate_type == 'IN':
            return 0.2  # Assume 20% for IN lists
        else:
            return 0.5  # Default


@dataclass
class CostEstimate:
    """
    Cost Estimate
    -------------
    Estimated cost of executing a query plan.
    """
    cpu_cost: float = 0.0  # Processing time units
    io_cost: float = 0.0   # I/O operations
    network_cost: float = 0.0  # Network transfer
    memory_cost: float = 0.0  # Memory usage
    estimated_rows: int = 0
    estimated_bytes: int = 0
    
    @property
    def total_cost(self) -> float:
        """Calculate total weighted cost."""
        # Weights can be tuned based on environment
        return (
            self.cpu_cost * 1.0 +
            self.io_cost * 10.0 +  # I/O is expensive
            self.network_cost * 5.0 +  # Network is moderately expensive
            self.memory_cost * 0.1  # Memory is relatively cheap
        )
    
    @property
    def fitness(self) -> float:
        """
        Calculate fitness score (bee-inspired).
        Higher fitness = better plan.
        
        In the ABC algorithm, fitness = 1 / (1 + cost).
        This transforms minimization (cost) into maximization (fitness).
        """
        return 1.0 / (1.0 + self.total_cost)
    
    def __add__(self, other: 'CostEstimate') -> 'CostEstimate':
        """Add two cost estimates."""
        return CostEstimate(
            cpu_cost=self.cpu_cost + other.cpu_cost,
            io_cost=self.io_cost + other.io_cost,
            network_cost=self.network_cost + other.network_cost,
            memory_cost=self.memory_cost + other.memory_cost,
            estimated_rows=max(self.estimated_rows, other.estimated_rows),
            estimated_bytes=self.estimated_bytes + other.estimated_bytes
        )


class CostModel(ABC):
    """
    Cost Model Base Class
    ---------------------
    Abstract base for cost estimation models.
    """
    
    @abstractmethod
    def estimate(self, 
                 node: PlanNode, 
                 stats: Dict[str, Statistics]) -> CostEstimate:
        """
        Estimate cost of executing a plan node.
        
        Args:
            node: Plan node to estimate
            stats: Statistics for tables involved
            
        Returns:
            Cost estimate
        """
        pass


class SwarmCostModel(CostModel):
    """
    Swarm Cost Model
    ----------------
    Bee-inspired cost model using fitness functions.
    
    Key concepts:
    - Fitness functions replace traditional cost formulas
    - Continuous adaptation based on execution feedback
    - Quality signals (like waggle dances) improve estimates
    
    The model maintains historical execution data and uses it
    to refine estimates, similar to how bees learn from experience.
    """
    
    def __init__(self):
        # Historical execution data for learning
        self._history: Dict[str, List[float]] = {}
        # Adaptive weights
        self._weights = {
            'cpu': 1.0,
            'io': 10.0,
            'network': 5.0,
            'memory': 0.1
        }
        
    def estimate(self, 
                 node: PlanNode, 
                 stats: Dict[str, Statistics]) -> CostEstimate:
        """Estimate cost using swarm-inspired approach."""
        # Recursive estimation
        child_costs = [
            self.estimate(child, stats) 
            for child in node.children
        ]
        
        # Base cost from children
        if child_costs:
            base_cost = sum(child_costs, CostEstimate())
        else:
            base_cost = CostEstimate()
            
        # Add node-specific cost
        node_cost = self._estimate_node(node, stats, child_costs)
        
        return base_cost + node_cost
        
    def _estimate_node(self, 
                       node: PlanNode,
                       stats: Dict[str, Statistics],
                       child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate cost for specific node type."""
        if node.node_type == NodeType.SCAN:
            return self._estimate_scan(node, stats)
        elif node.node_type == NodeType.FILTER:
            return self._estimate_filter(node, stats, child_costs)
        elif node.node_type == NodeType.PROJECT:
            return self._estimate_project(node, child_costs)
        elif node.node_type == NodeType.JOIN:
            return self._estimate_join(node, stats, child_costs)
        elif node.node_type == NodeType.AGGREGATE:
            return self._estimate_aggregate(node, child_costs)
        elif node.node_type == NodeType.SORT:
            return self._estimate_sort(node, child_costs)
        elif node.node_type == NodeType.LIMIT:
            return self._estimate_limit(node, child_costs)
        else:
            return CostEstimate()
            
    def _estimate_scan(self, 
                       node: PlanNode, 
                       stats: Dict[str, Statistics]) -> CostEstimate:
        """Estimate table scan cost."""
        table_name = node.properties.get('table', '')
        table_stats = stats.get(table_name, Statistics())
        
        rows = table_stats.row_count or 10000  # Default estimate
        row_size = table_stats.avg_row_size or 100
        
        return CostEstimate(
            cpu_cost=rows * 0.01,  # Simple row iteration
            io_cost=rows * row_size / 4096,  # Pages read
            estimated_rows=rows,
            estimated_bytes=rows * row_size
        )
        
    def _estimate_filter(self, 
                        node: PlanNode,
                        stats: Dict[str, Statistics],
                        child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate filter cost."""
        input_rows = child_costs[0].estimated_rows if child_costs else 10000
        
        # Estimate selectivity
        selectivity = self._estimate_selectivity(node.predicates, stats)
        output_rows = int(input_rows * selectivity)
        
        return CostEstimate(
            cpu_cost=input_rows * 0.001,  # Predicate evaluation
            estimated_rows=output_rows,
            estimated_bytes=output_rows * 100
        )
        
    def _estimate_project(self, 
                         node: PlanNode,
                         child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate projection cost."""
        input_rows = child_costs[0].estimated_rows if child_costs else 10000
        num_cols = len(node.columns) if node.columns else 5
        
        # Projection is cheap
        return CostEstimate(
            cpu_cost=input_rows * 0.0001 * num_cols,
            estimated_rows=input_rows,
            estimated_bytes=input_rows * num_cols * 20
        )
        
    def _estimate_join(self, 
                      node: PlanNode,
                      stats: Dict[str, Statistics],
                      child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate join cost."""
        left_rows = child_costs[0].estimated_rows if child_costs else 10000
        right_rows = child_costs[1].estimated_rows if len(child_costs) > 1 else 10000
        
        join_type = node.properties.get('join_type', 'hash')
        
        if join_type == 'hash':
            # Hash join: build + probe
            cpu_cost = left_rows + right_rows * 2  # Build hash + probe
            memory_cost = left_rows * 100  # Hash table
        elif join_type == 'sort_merge':
            # Sort-merge: sort both sides + merge
            cpu_cost = (left_rows * math.log2(max(left_rows, 1)) + 
                       right_rows * math.log2(max(right_rows, 1)))
            memory_cost = (left_rows + right_rows) * 100
        else:
            # Nested loop (worst case)
            cpu_cost = left_rows * right_rows
            memory_cost = 0
            
        # Estimate output size (assume 1:1 match ratio)
        output_rows = max(left_rows, right_rows)
        
        return CostEstimate(
            cpu_cost=cpu_cost * 0.001,
            memory_cost=memory_cost,
            estimated_rows=output_rows,
            estimated_bytes=output_rows * 200
        )
        
    def _estimate_aggregate(self, 
                           node: PlanNode,
                           child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate aggregation cost."""
        input_rows = child_costs[0].estimated_rows if child_costs else 10000
        
        # Estimate group count (assume good cardinality reduction)
        num_groups = node.properties.get('num_groups', int(input_rows ** 0.5))
        
        return CostEstimate(
            cpu_cost=input_rows * 0.005,  # Hashing + aggregation
            memory_cost=num_groups * 100,  # Group state
            estimated_rows=num_groups,
            estimated_bytes=num_groups * 100
        )
        
    def _estimate_sort(self, 
                      node: PlanNode,
                      child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate sort cost."""
        input_rows = child_costs[0].estimated_rows if child_costs else 10000
        
        # O(n log n) sorting
        if input_rows > 0:
            sort_cost = input_rows * math.log2(max(input_rows, 1))
        else:
            sort_cost = 0
            
        return CostEstimate(
            cpu_cost=sort_cost * 0.001,
            memory_cost=input_rows * 100,  # Sort buffer
            estimated_rows=input_rows,
            estimated_bytes=input_rows * 100
        )
        
    def _estimate_limit(self, 
                       node: PlanNode,
                       child_costs: List[CostEstimate]) -> CostEstimate:
        """Estimate limit cost."""
        limit = node.properties.get('limit', 100)
        input_rows = child_costs[0].estimated_rows if child_costs else 10000
        
        output_rows = min(limit, input_rows)
        
        return CostEstimate(
            cpu_cost=output_rows * 0.0001,
            estimated_rows=output_rows,
            estimated_bytes=output_rows * 100
        )
        
    def _estimate_selectivity(self, 
                             predicates: List[Any],
                             stats: Dict[str, Statistics]) -> float:
        """Estimate combined selectivity of predicates."""
        if not predicates:
            return 1.0
            
        # Assume independence (multiply selectivities)
        selectivity = 1.0
        for pred in predicates:
            if isinstance(pred, dict):
                col = pred.get('column', '')
                op = pred.get('operator', '=')
                for table_stats in stats.values():
                    sel = table_stats.selectivity(col, op)
                    selectivity *= sel
            else:
                selectivity *= 0.5  # Default
                
        return max(selectivity, 0.001)  # Minimum 0.1%
        
    def record_execution(self, plan_hash: str, actual_time: float) -> None:
        """
        Record execution time for learning.
        
        This is how the swarm learns - by incorporating
        feedback from actual executions (like waggle dances).
        """
        if plan_hash not in self._history:
            self._history[plan_hash] = []
        self._history[plan_hash].append(actual_time)
        
        # Keep only recent history
        if len(self._history[plan_hash]) > 100:
            self._history[plan_hash] = self._history[plan_hash][-100:]
            
    def get_historical_cost(self, plan_hash: str) -> Optional[float]:
        """Get average historical cost for a plan pattern."""
        if plan_hash in self._history:
            return sum(self._history[plan_hash]) / len(self._history[plan_hash])
        return None
