---
sidebar_position: 3
---

# DataFrame Module

Spark-compatible DataFrame API for distributed data manipulation.

```python
import hiveframe as hf
from hiveframe.dataframe import DataFrame, Column, Schema, Field
```

## DataFrame

The primary data structure for distributed data processing.

### Class Definition

```python
class DataFrame:
    """
    Distributed collection of data organized into named columns.
    
    Similar to pandas DataFrames or Spark DataFrames, providing
    a familiar API for data manipulation at scale.
    """
```

### Creating DataFrames

#### From Data

```python
# From dictionary
df = hf.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "Chicago"]
})

# From list of rows
df = hf.DataFrame([
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30}
])

# From pandas
import pandas as pd
pdf = pd.DataFrame({"x": [1, 2, 3]})
df = hf.DataFrame(pdf)
```

#### From Files

```python
# Parquet
df = hf.read.parquet("data.parquet")
df = hf.read.parquet("data/", recursive=True)

# CSV
df = hf.read.csv("data.csv", header=True, inferSchema=True)

# JSON
df = hf.read.json("data.json")
```

### Selection Methods

#### `select()`

```python
def select(self, *cols: Union[str, Column]) -> "DataFrame":
    """
    Select columns from the DataFrame.
    
    Args:
        *cols: Column names or Column expressions
        
    Returns:
        DataFrame with selected columns
        
    Examples:
        # By name
        df.select("name", "age")
        
        # With expressions
        df.select(hf.col("name"), hf.col("age") + 1)
        
        # With aliases
        df.select(
            hf.col("name").alias("user_name"),
            (hf.col("age") * 12).alias("age_months")
        )
    """
```

#### `selectExpr()`

```python
def selectExpr(self, *exprs: str) -> "DataFrame":
    """
    Select with SQL expressions.
    
    Args:
        *exprs: SQL expression strings
        
    Returns:
        DataFrame with computed columns
        
    Example:
        df.selectExpr(
            "name",
            "age * 12 as age_months",
            "UPPER(city) as city_upper"
        )
    """
```

### Filtering Methods

#### `filter()` / `where()`

```python
def filter(self, condition: Union[Column, str]) -> "DataFrame":
    """
    Filter rows based on condition.
    
    Args:
        condition: Column expression or SQL string
        
    Returns:
        Filtered DataFrame
        
    Examples:
        # Column expression
        df.filter(hf.col("age") > 21)
        
        # Multiple conditions
        df.filter(
            (hf.col("age") > 21) & 
            (hf.col("city") == "NYC")
        )
        
        # SQL string
        df.filter("age > 21 AND city = 'NYC'")
    """

# Alias
where = filter
```

#### `distinct()`

```python
def distinct(self) -> "DataFrame":
    """
    Return distinct rows.
    
    Example:
        unique_cities = df.select("city").distinct()
    """
```

#### `dropDuplicates()`

```python
def dropDuplicates(
    self, 
    subset: Optional[List[str]] = None
) -> "DataFrame":
    """
    Remove duplicate rows.
    
    Args:
        subset: Columns to consider for duplicates.
                If None, uses all columns.
                
    Example:
        # Keep first occurrence per name
        df.dropDuplicates(["name"])
    """
```

### Transformation Methods

#### `withColumn()`

```python
def withColumn(
    self, 
    name: str, 
    col: Column
) -> "DataFrame":
    """
    Add or replace a column.
    
    Args:
        name: Column name
        col: Column expression
        
    Returns:
        DataFrame with new/modified column
        
    Example:
        df.withColumn("age_months", hf.col("age") * 12)
    """
```

#### `withColumnRenamed()`

```python
def withColumnRenamed(
    self, 
    existing: str, 
    new: str
) -> "DataFrame":
    """
    Rename a column.
    
    Example:
        df.withColumnRenamed("name", "user_name")
    """
```

#### `drop()`

```python
def drop(self, *cols: str) -> "DataFrame":
    """
    Drop columns.
    
    Example:
        df.drop("temp_column", "debug_info")
    """
```

### Sorting Methods

#### `orderBy()` / `sort()`

```python
def orderBy(
    self, 
    *cols: Union[str, Column],
    ascending: Union[bool, List[bool]] = True
) -> "DataFrame":
    """
    Sort by columns.
    
    Args:
        *cols: Columns to sort by
        ascending: Sort direction(s)
        
    Examples:
        # Ascending
        df.orderBy("age")
        
        # Descending
        df.orderBy(hf.col("age").desc())
        
        # Multiple columns
        df.orderBy("city", hf.col("age").desc())
    """

# Alias
sort = orderBy
```

### Limiting Methods

#### `limit()`

```python
def limit(self, n: int) -> "DataFrame":
    """
    Return first n rows.
    
    Example:
        top_10 = df.orderBy(hf.col("score").desc()).limit(10)
    """
```

#### `head()` / `first()` / `take()`

```python
def head(self, n: int = 1) -> List[Row]:
    """Return first n rows as list."""

def first(self) -> Optional[Row]:
    """Return first row or None."""

def take(self, n: int) -> List[Row]:
    """Return first n rows as list."""
```

### Aggregation Methods

#### `groupBy()`

```python
def groupBy(self, *cols: Union[str, Column]) -> "GroupedData":
    """
    Group by columns for aggregation.
    
    Returns:
        GroupedData for aggregation operations
        
    Example:
        df.groupBy("city").agg(
            hf.avg("age").alias("avg_age"),
            hf.count("*").alias("count")
        )
    """
```

#### `agg()`

```python
def agg(self, *exprs: Column) -> "DataFrame":
    """
    Aggregate entire DataFrame.
    
    Example:
        df.agg(
            hf.min("age"),
            hf.max("age"),
            hf.avg("age")
        )
    """
```

### Join Methods

#### `join()`

```python
def join(
    self,
    other: "DataFrame",
    on: Union[str, List[str], Column] = None,
    how: str = "inner"
) -> "DataFrame":
    """
    Join with another DataFrame.
    
    Args:
        other: DataFrame to join with
        on: Join column(s) or condition
        how: Join type - "inner", "left", "right", 
             "outer", "semi", "anti", "cross"
             
    Examples:
        # Simple join
        df1.join(df2, on="id")
        
        # Multiple columns
        df1.join(df2, on=["id", "date"])
        
        # Condition join
        df1.join(
            df2, 
            on=df1["id"] == df2["user_id"],
            how="left"
        )
    """
```

### Union Methods

#### `union()` / `unionAll()`

```python
def union(self, other: "DataFrame") -> "DataFrame":
    """
    Concatenate DataFrames (removes duplicates).
    
    Example:
        combined = df1.union(df2)
    """

def unionAll(self, other: "DataFrame") -> "DataFrame":
    """
    Concatenate DataFrames (keeps duplicates).
    
    Example:
        combined = df1.unionAll(df2)
    """
```

### Action Methods

#### `collect()`

```python
def collect(self) -> List[Row]:
    """
    Collect all rows to driver.
    
    Warning: May cause OOM for large DataFrames.
    
    Example:
        rows = df.collect()
        for row in rows:
            print(row["name"])
    """
```

#### `toPandas()`

```python
def toPandas(self) -> pd.DataFrame:
    """
    Convert to pandas DataFrame.
    
    Example:
        pdf = df.toPandas()
        pdf.plot()
    """
```

#### `count()`

```python
def count(self) -> int:
    """
    Count rows.
    
    Example:
        total = df.count()
    """
```

#### `show()`

```python
def show(
    self, 
    n: int = 20, 
    truncate: bool = True
) -> None:
    """
    Print rows to console.
    
    Args:
        n: Number of rows to show
        truncate: Truncate long strings
        
    Example:
        df.show(10)
    """
```

### Schema Methods

#### `printSchema()`

```python
def printSchema(self) -> None:
    """
    Print schema to console.
    
    Example:
        df.printSchema()
        # root
        #  |-- name: string
        #  |-- age: integer
        #  |-- city: string
    """
```

#### `schema`

```python
@property
def schema(self) -> Schema:
    """DataFrame schema."""
```

#### `columns`

```python
@property
def columns(self) -> List[str]:
    """List of column names."""
```

#### `dtypes`

```python
@property
def dtypes(self) -> List[Tuple[str, str]]:
    """List of (column_name, data_type) tuples."""
```

### Write Methods

#### `write`

```python
@property
def write(self) -> DataFrameWriter:
    """
    Access the DataFrameWriter.
    
    Examples:
        # Parquet
        df.write.parquet("output.parquet")
        
        # CSV
        df.write.csv("output.csv", header=True)
        
        # Partitioned
        df.write.partitionBy("year", "month").parquet("output/")
        
        # Mode
        df.write.mode("overwrite").parquet("output/")
    """
```

---

## Column

Column expressions for transformations and filters.

### Creating Columns

```python
# From DataFrame
col = df["name"]
col = df.name

# Using hf.col()
col = hf.col("name")

# Literal value
col = hf.lit(42)
```

### Operators

```python
# Arithmetic
hf.col("age") + 1
hf.col("price") * hf.col("quantity")
hf.col("total") / 100

# Comparison
hf.col("age") > 21
hf.col("status") == "active"
hf.col("name").isNull()
hf.col("name").isNotNull()

# Logical
(hf.col("age") > 21) & (hf.col("city") == "NYC")
(hf.col("status") == "A") | (hf.col("status") == "B")
~hf.col("is_deleted")

# String
hf.col("name").contains("John")
hf.col("email").startswith("admin")
hf.col("code").endswith("_v2")
hf.col("name").like("%son")
hf.col("name").rlike(r"^J.*n$")

# Null handling
hf.col("value").isNull()
hf.col("value").isNotNull()
hf.coalesce(hf.col("a"), hf.col("b"), hf.lit(0))
```

### Methods

```python
# Aliasing
hf.col("age").alias("user_age")

# Casting
hf.col("age").cast("string")
hf.col("price").cast(hf.DoubleType())

# Sorting
hf.col("age").asc()
hf.col("age").desc()
hf.col("name").asc_nulls_first()
hf.col("name").desc_nulls_last()

# Conditional
hf.when(hf.col("age") < 18, "minor") \
  .when(hf.col("age") < 65, "adult") \
  .otherwise("senior")
```

---

## Aggregation Functions

```python
# Import
from hiveframe import col, sum, avg, count, min, max

# Or use hf prefix
import hiveframe as hf
```

### Available Functions

| Function | Description | Example |
|----------|-------------|---------|
| `count()` | Count rows/values | `hf.count("*")` |
| `sum()` | Sum values | `hf.sum("amount")` |
| `avg()` | Average | `hf.avg("score")` |
| `mean()` | Mean (alias for avg) | `hf.mean("score")` |
| `min()` | Minimum | `hf.min("date")` |
| `max()` | Maximum | `hf.max("price")` |
| `first()` | First value | `hf.first("name")` |
| `last()` | Last value | `hf.last("name")` |
| `collect_list()` | Collect to list | `hf.collect_list("tag")` |
| `collect_set()` | Collect unique | `hf.collect_set("tag")` |
| `countDistinct()` | Count distinct | `hf.countDistinct("user_id")` |
| `stddev()` | Standard deviation | `hf.stddev("score")` |
| `variance()` | Variance | `hf.variance("score")` |

### Example

```python
df.groupBy("category").agg(
    hf.count("*").alias("total"),
    hf.sum("amount").alias("total_amount"),
    hf.avg("amount").alias("avg_amount"),
    hf.min("amount").alias("min_amount"),
    hf.max("amount").alias("max_amount"),
    hf.countDistinct("user_id").alias("unique_users")
)
```

---

## Schema

Define DataFrame schemas explicitly.

```python
from hiveframe.dataframe import Schema, Field
from hiveframe import DataType

schema = Schema([
    Field("id", DataType.LONG, nullable=False),
    Field("name", DataType.STRING),
    Field("email", DataType.STRING),
    Field("age", DataType.INTEGER),
    Field("created_at", DataType.TIMESTAMP),
    Field("tags", DataType.ARRAY),
    Field("metadata", DataType.MAP)
])

# Use with read
df = hf.read.schema(schema).csv("data.csv")

# Use with DataFrame creation
df = hf.DataFrame(data, schema=schema)
```

---

## Window Functions

```python
from hiveframe.dataframe import Window

# Define window
window = Window.partitionBy("department").orderBy("salary")

# Window functions
df.select(
    "name",
    "salary",
    hf.row_number().over(window).alias("rank"),
    hf.sum("salary").over(window).alias("running_total"),
    hf.lag("salary", 1).over(window).alias("prev_salary"),
    hf.lead("salary", 1).over(window).alias("next_salary")
)
```

### Window Specifications

```python
# Partition and order
Window.partitionBy("dept").orderBy("date")

# Row range
Window.partitionBy("dept").orderBy("date").rowsBetween(-2, 0)

# Value range  
Window.partitionBy("dept").orderBy("amount").rangeBetween(-100, 100)

# Unbounded
Window.partitionBy("dept").orderBy("date").rowsBetween(
    Window.unboundedPreceding, 
    Window.currentRow
)
```

## See Also

- [Core](./core) - Colony and Cell classes
- [SQL](./sql) - SQL queries on DataFrames
- [DataFrame Tutorial](/docs/tutorials/dataframe-basics) - Learn DataFrame basics
