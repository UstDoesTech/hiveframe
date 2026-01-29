---
sidebar_position: 2
---

# DataFrame Operations

This tutorial teaches you how to work with HiveFrame's DataFrame API, inspired by Apache Spark but powered by bee colony intelligence.

## Creating DataFrames

There are several ways to create a DataFrame:

### From Records

```python
from hiveframe import HiveDataFrame

records = [
    {'name': 'Alice', 'age': 30, 'city': 'NYC'},
    {'name': 'Bob', 'age': 25, 'city': 'LA'},
    {'name': 'Carol', 'age': 35, 'city': 'NYC'},
]

df = HiveDataFrame(records)
```

### From CSV

```python
df = HiveDataFrame.from_csv('data.csv')
```

### From JSON

```python
df = HiveDataFrame.from_json('data.json')
```

## Basic Operations

### Selecting Columns

```python
from hiveframe import col

# Select specific columns
df_subset = df.select('name', 'age')

# Select with expressions
df_expr = df.select(
    col('name'),
    (col('age') + 5).alias('age_in_5_years')
)
```

### Filtering Rows

```python
# Simple filter
adults = df.filter(col('age') >= 18)

# Multiple conditions
ny_adults = df.filter(
    (col('age') >= 18) & (col('city') == 'NYC')
)

# String operations
df_filtered = df.filter(col('name').contains('Alice'))
```

### Adding Columns

```python
# Add a new column
df_with_category = df.withColumn(
    'age_category',
    col('age') / 10
)
```

### Removing Columns

```python
df_dropped = df.drop('city')
```

## Aggregations

### GroupBy Operations

```python
from hiveframe import avg, count, sum_agg, min_agg, max_agg

# Group by a single column
by_city = df.groupBy('city').agg(
    count(col('name')).alias('count'),
    avg(col('age')).alias('avg_age')
)

by_city.show()
```

### Multiple Aggregations

```python
# Complex aggregation
stats = df.groupBy('city').agg(
    count(col('name')).alias('total_people'),
    min_agg(col('age')).alias('youngest'),
    max_agg(col('age')).alias('oldest'),
    avg(col('age')).alias('average_age')
)
```

## Joins

Combine data from multiple DataFrames:

```python
# Create two DataFrames
employees = HiveDataFrame([
    {'emp_id': 1, 'name': 'Alice', 'dept_id': 10},
    {'emp_id': 2, 'name': 'Bob', 'dept_id': 20},
])

departments = HiveDataFrame([
    {'dept_id': 10, 'dept_name': 'Engineering'},
    {'dept_id': 20, 'dept_name': 'Marketing'},
])

# Inner join
result = employees.join(
    departments,
    on='dept_id',
    how='inner'
)

result.show()
```

### Join Types

- `inner`: Keep only matching records
- `left`: Keep all left records, fill missing with null
- `right`: Keep all right records, fill missing with null
- `outer`: Keep all records from both sides

## Sorting

```python
# Sort ascending
df_sorted = df.orderBy('age')

# Sort descending
df_sorted_desc = df.orderBy('age', ascending=False)

# Sort by multiple columns
df_multi_sort = df.orderBy(['city', 'age'])
```

## Limiting Results

```python
# Get first 10 rows
top_10 = df.limit(10)
```

## Removing Duplicates

```python
# Remove duplicate rows
df_unique = df.distinct()
```

## Column Expressions

HiveFrame provides rich column operations:

```python
from hiveframe import col, lit

# Arithmetic
col('price') * col('quantity')
col('total') / 100

# Comparisons
col('age') > 21
col('status') == 'active'
col('value').isNull()

# String operations
col('name').contains('Smith')
col('email').endswith('@gmail.com')
col('text').startswith('Hello')

# Aliasing
(col('first') + col('last')).alias('full_name')

# Literals
df.withColumn('constant', lit(42))
```

## Collecting Results

```python
# Collect all rows to memory
all_data = df.collect()

# Show first N rows in a table format
df.show(n=20)

# Get count
total = df.count()
```

## Exporting Data

```python
# Save to CSV
df.to_csv('output.csv')

# Save to JSON
df.to_json('output.json')
```

## Complete Example

Here's a complete example that ties everything together:

```python
from hiveframe import HiveDataFrame, col, sum_agg, avg, count

# Load transaction data
transactions = HiveDataFrame.from_csv('transactions.csv')

# Process and analyze
result = (transactions
    # Filter recent high-value transactions
    .filter(col('amount') > 100)
    .filter(col('category') == 'Electronics')
    
    # Add calculated column
    .withColumn('tax', col('amount') * 0.08)
    
    # Group and aggregate
    .groupBy('region')
    .agg(
        count(col('id')).alias('num_transactions'),
        sum_agg(col('amount')).alias('total_sales'),
        avg(col('amount')).alias('avg_transaction'),
        sum_agg(col('tax')).alias('total_tax')
    )
    
    # Sort by sales
    .orderBy('total_sales', ascending=False)
    
    # Top 10 regions
    .limit(10)
)

# Display results
result.show()

# Save to file
result.to_csv('top_regions.csv')
```

## Performance Tips

1. **Filter Early**: Apply filters before other operations to reduce data volume
2. **Use Column References**: Prefer `col('name')` over string literals
3. **Partition Wisely**: The bee colony automatically balances work, but you can hint with data locality
4. **Limit When Exploring**: Use `.limit()` during development to work with smaller datasets

## Next Steps

- **Streaming Basics** - Learn about real-time processing
- **Column Expressions** - Complete column API reference
- **Aggregation Functions** - All aggregation functions
- **Quality-Weighted Processing** - How the bee colony optimizes your queries
