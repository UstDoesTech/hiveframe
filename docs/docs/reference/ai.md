---
sidebar_position: 14
---

# AI Integration

Phase 4 AI integration module provides natural language queries, data preparation, discovery, code generation, and LLM fine-tuning capabilities.

## Overview

The AI integration module brings generative AI capabilities to HiveFrame, enabling natural language interactions, automatic data quality improvements, intelligent schema discovery, and custom model training on lakehouse data.

## Natural Language Queries

Translate plain English questions into SwarmQL queries.

### Classes

#### `NaturalLanguageQuery`

Main interface for natural language to SQL translation.

```python
from hiveframe.ai import NaturalLanguageQuery

nl_query = NaturalLanguageQuery()
```

**Methods:**

##### `query(question: str, context: Dict[str, Any]) -> Dict[str, Any]`

Translates a natural language question into SQL.

```python
result = nl_query.query(
    "Show me all customers who spent more than $1000 last month",
    context={"tables": ["customers", "orders"]}
)
# Returns: {
#   'sql': 'SELECT c.* FROM customers c JOIN orders o ...',
#   'confidence': 0.92,
#   'explanation': 'Filtering orders from last month and summing amounts'
# }
```

##### `record_feedback(question: str, sql: str, was_correct: bool) -> None`

Records feedback to improve future translations.

```python
nl_query.record_feedback(question, generated_sql, was_correct=True)
```

#### `QueryTranslator`

Core translation engine.

```python
from hiveframe.ai import QueryTranslator

translator = QueryTranslator()
intent = translator.parse_question("What are the top 5 products by revenue?")
# Returns structured query intent
```

## AI-Powered Data Preparation

Automatically detect and fix data quality issues.

### Classes

#### `AIDataPrep`

Main orchestrator for AI-powered data preparation.

```python
from hiveframe.ai import AIDataPrep
from hiveframe import HiveDataFrame

data_prep = AIDataPrep()
df = HiveDataFrame.from_csv('raw_data.csv')
```

**Methods:**

##### `analyze_quality(df: HiveDataFrame) -> Dict[str, Any]`

Analyzes data quality and identifies issues.

```python
quality_report = data_prep.analyze_quality(df)
# Returns: {
#   'total_issues': 152,
#   'issues': [
#     {'column': 'email', 'type': 'missing_values', 'count': 45},
#     {'column': 'age', 'type': 'outliers', 'count': 12},
#     ...
#   ]
# }
```

##### `clean(df: HiveDataFrame, quality_report: Dict[str, Any]) -> HiveDataFrame`

Automatically cleans data based on quality report.

```python
cleaned_df = data_prep.clean(df, quality_report)
```

##### `suggest_transformations(df: HiveDataFrame, use_case: str) -> List[Dict[str, Any]]`

Suggests transformations based on use case.

```python
suggestions = data_prep.suggest_transformations(df, use_case="machine_learning")
# Returns: [
#   {
#     'transformation': 'normalize',
#     'column': 'amount',
#     'reason': 'Large range affects model performance',
#     'example': 'MinMaxScaler(0, 1)'
#   },
#   ...
# ]
```

#### `DataCleaner`

Core data cleaning engine.

```python
from hiveframe.ai import DataCleaner

cleaner = DataCleaner()
issues = cleaner.detect_issues(df)
```

## Intelligent Data Discovery

Automatically detect relationships and suggest joins.

### Classes

#### `DataDiscovery`

Main orchestrator for intelligent data discovery.

```python
from hiveframe.ai import DataDiscovery

discovery = DataDiscovery()
```

**Methods:**

##### `register_schema(table_name: str, schema: Dict[str, str]) -> None`

Registers a table schema for discovery.

```python
discovery.register_schema("users", {
    "user_id": "int",
    "name": "string",
    "email": "string"
})
```

##### `detect_relationships() -> List[Dict[str, Any]]`

Detects relationships between registered tables.

```python
relationships = discovery.detect_relationships()
# Returns: [
#   {
#     'from_table': 'users',
#     'from_column': 'user_id',
#     'to_table': 'orders',
#     'to_column': 'user_id',
#     'type': 'one_to_many',
#     'confidence': 0.95
#   },
#   ...
# ]
```

##### `suggest_joins(from_table: str, to_table: str) -> Dict[str, Any]`

Suggests optimal join path between tables using bee-inspired pathfinding.

```python
join_path = discovery.suggest_joins(from_table="users", to_table="products")
# Returns: {
#   'path': ['users', 'orders', 'order_items', 'products'],
#   'estimated_cost': 125.5,
#   'join_conditions': [...]
# }
```

#### `RelationshipDetector`

Core relationship detection engine.

```python
from hiveframe.ai import RelationshipDetector

detector = RelationshipDetector()
relationships = detector.detect(table_schemas)
```

## Code Generation

Generate HiveFrame code from natural language descriptions.

### Classes

#### `HiveFrameCodeGen`

Main orchestrator for code generation.

```python
from hiveframe.ai import HiveFrameCodeGen

codegen = HiveFrameCodeGen()
```

**Methods:**

##### `generate(description: str) -> Dict[str, Any]`

Generates HiveFrame code from a natural language description.

```python
result = codegen.generate(
    "Load CSV file, filter by status='active', and count by region"
)
# Returns: {
#   'code': 'df = HiveDataFrame.from_csv("data.csv")\nfiltered = df.filter(...)',
#   'explanation': 'Loads CSV, applies filter, groups and counts',
#   'imports': ['from hiveframe import HiveDataFrame', ...]
# }
```

##### `generate_pipeline(steps: List[str]) -> Dict[str, Any]`

Generates a multi-step data pipeline.

```python
pipeline = codegen.generate_pipeline([
    "Read parquet files from /data",
    "Filter rows where amount > 100",
    "Group by category and sum amounts",
    "Write results to Delta Lake"
])
```

#### `CodeGenerator`

Core code generation engine.

```python
from hiveframe.ai import CodeGenerator

generator = CodeGenerator()
code = generator.generate_from_template("etl", params)
```

## LLM Fine-tuning Platform

Train custom models on lakehouse data.

### Classes

#### `LLMFineTuner`

Main orchestrator for LLM fine-tuning.

```python
from hiveframe.ai import LLMFineTuner

fine_tuner = LLMFineTuner()
```

**Methods:**

##### `prepare_dataset(data_path: str, format: str = "auto") -> Dict[str, Any]`

Prepares training dataset from lakehouse data.

```python
dataset = fine_tuner.prepare_dataset(
    data_path="/lakehouse/training_data",
    format="jsonl"
)
# Returns: {
#   'train_samples': 10000,
#   'val_samples': 2000,
#   'dataset_path': '/tmp/prepared_dataset'
# }
```

##### `train(dataset_path: str, model_name: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]`

Trains a custom model using ABC-optimized hyperparameters.

```python
result = fine_tuner.train(
    dataset_path="/tmp/prepared_dataset",
    model_name="custom_classifier",
    hyperparams={
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3
    }
)
# Returns: {
#   'model_path': '/models/custom_classifier',
#   'metrics': {'accuracy': 0.94, 'loss': 0.12},
#   'training_time_hours': 2.5
# }
```

##### `optimize_hyperparameters(dataset_path: str, n_trials: int = 20) -> Dict[str, Any]`

Uses swarm intelligence to find optimal hyperparameters.

```python
best_params = fine_tuner.optimize_hyperparameters(
    dataset_path="/tmp/prepared_dataset",
    n_trials=50
)
# Returns: {
#   'best_params': {'learning_rate': 1.8e-5, 'batch_size': 32, ...},
#   'best_score': 0.96,
#   'optimization_time_hours': 8.2
# }
```

## Examples

### Natural Language Query Example

```python
from hiveframe.ai import NaturalLanguageQuery

nl_query = NaturalLanguageQuery()

# Ask questions in plain English
questions = [
    "Show me all customers who spent more than $1000 last month",
    "What are the top 5 products by revenue?",
    "Find users who haven't logged in for 30 days"
]

for question in questions:
    result = nl_query.query(
        question,
        context={"tables": ["customers", "orders", "products", "users"]}
    )
    print(f"Q: {question}")
    print(f"SQL: {result['sql']}")
    print(f"Confidence: {result['confidence']:.2f}\n")
    
    # Provide feedback to improve the system
    # In production, you'd validate the query first
    nl_query.record_feedback(question, result['sql'], was_correct=True)
```

### Data Preparation Example

```python
from hiveframe.ai import AIDataPrep
from hiveframe import HiveDataFrame

# Create AI data prep system
data_prep = AIDataPrep()

# Load raw data
df = HiveDataFrame.from_csv('raw_customer_data.csv')

# Analyze data quality
quality_report = data_prep.analyze_quality(df)
print("Quality Issues Found:")
for issue in quality_report['issues']:
    print(f"  {issue['column']}: {issue['type']} ({issue['count']} instances)")

# Automatically clean data
cleaned_df = data_prep.clean(df, quality_report)
print(f"✓ Cleaned {quality_report['total_issues']} quality issues")

# Get transformation suggestions for ML
suggestions = data_prep.suggest_transformations(
    cleaned_df,
    use_case="machine_learning"
)
for suggestion in suggestions:
    print(f"\n{suggestion['transformation']} on {suggestion['column']}")
    print(f"  Reason: {suggestion['reason']}")
    print(f"  Example: {suggestion['example']}")
```

### Data Discovery Example

```python
from hiveframe.ai import DataDiscovery

# Create discovery system
discovery = DataDiscovery()

# Register table schemas
discovery.register_schema("users", {
    "user_id": "int",
    "name": "string",
    "email": "string"
})
discovery.register_schema("orders", {
    "order_id": "int",
    "user_id": "int",
    "amount": "float"
})
discovery.register_schema("products", {
    "product_id": "int",
    "name": "string",
    "price": "float"
})

# Detect relationships automatically
relationships = discovery.detect_relationships()
print("Detected Relationships:")
for rel in relationships:
    print(f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")
    print(f"    Type: {rel['type']}, Confidence: {rel['confidence']:.2f}")

# Get join suggestions using bee-inspired pathfinding
join_path = discovery.suggest_joins(from_table="users", to_table="products")
print(f"\nOptimal join path: {' -> '.join(join_path['path'])}")
print(f"Estimated cost: {join_path['estimated_cost']}")
```

### Code Generation Example

```python
from hiveframe.ai import HiveFrameCodeGen

codegen = HiveFrameCodeGen()

# Generate code from natural language
descriptions = [
    "Load CSV file, filter by status='active', and count by region",
    "Create a streaming pipeline that processes JSON events and writes to Parquet",
    "Join users and orders, aggregate by month, and save results"
]

for desc in descriptions:
    result = codegen.generate(desc)
    print(f"Description: {desc}")
    print(f"\nGenerated code:")
    print(result['code'])
    print(f"\nExplanation: {result['explanation']}\n")
    print("-" * 60)
```

## See Also

- [Autonomous Operations](./autonomous) - Self-tuning and predictive maintenance
- [Advanced Swarm](./advanced-swarm) - Hybrid swarm algorithms
- [SQL](./sql) - SwarmQL query engine
