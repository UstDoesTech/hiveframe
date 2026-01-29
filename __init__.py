"""
HiveFrame: Bee-Inspired Data Processing Framework
================================================

A biomimetic alternative to Apache Spark that uses bee colony
intelligence patterns for distributed data processing.

Key Features:
- Decentralized coordination (no driver bottleneck)
- Adaptive load balancing through waggle dance protocol
- Self-healing through ABC abandonment mechanism
- Quality-weighted task reinforcement
- Pheromone-based backpressure

Quick Start:
    from hiveframe import HiveFrame, HiveDataFrame, col
    
    # RDD-style processing
    hive = HiveFrame(num_workers=8)
    results = hive.map(data, lambda x: x * 2)
    
    # DataFrame API
    df = HiveDataFrame.from_csv('data.csv')
    result = df.filter(col('age') > 21).select('name', 'age')
    result.show()
    
    # Streaming
    stream = HiveStream(num_workers=4)
    stream.start(process_fn)
    stream.submit('key', data)

Biomimicry Concepts:
- Waggle Dance: Workers advertise task quality through dance signals
- Three-Tier Workers: Employed (exploit), Onlooker (reinforce), Scout (explore)
- Stigmergic Coordination: Indirect communication through shared state
- Quorum Consensus: Decisions emerge from local interactions
- Adaptive Allocation: Self-organizing based on local stimuli
"""

__version__ = '0.1.0'
__author__ = 'HiveFrame Contributors'

from .core import (
    HiveFrame,
    Bee,
    BeeRole,
    WaggleDance,
    DanceFloor,
    ColonyState,
    Pheromone,
    FoodSource,
    create_hive,
)

from .dataframe import (
    HiveDataFrame,
    Column,
    col,
    lit,
    Schema,
    DataType,
    GroupedData,
    createDataFrame,
    # Aggregation functions
    sum_agg,
    avg,
    count,
    count_all,
    min_agg,
    max_agg,
    collect_list,
    collect_set,
)

from .streaming import (
    HiveStream,
    AsyncHiveStream,
    StreamRecord,
    StreamPartitioner,
    StreamBuffer,
    StreamBee,
)

__all__ = [
    # Core
    'HiveFrame',
    'Bee',
    'BeeRole',
    'WaggleDance',
    'DanceFloor',
    'ColonyState',
    'Pheromone',
    'FoodSource',
    'create_hive',
    # DataFrame
    'HiveDataFrame',
    'Column',
    'col',
    'lit',
    'Schema',
    'DataType',
    'GroupedData',
    'createDataFrame',
    # Aggregations
    'sum_agg',
    'avg',
    'count',
    'count_all',
    'min_agg',
    'max_agg',
    'collect_list',
    'collect_set',
    # Streaming
    'HiveStream',
    'AsyncHiveStream',
    'StreamRecord',
    'StreamPartitioner',
    'StreamBuffer',
    'StreamBee',
]
