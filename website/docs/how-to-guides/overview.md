---
sidebar_position: 1
---

# How-to Guides Overview

How-to guides are practical, problem-oriented recipes for common tasks. Each guide assumes you already understand the basics and helps you accomplish a specific goal.

## Getting Started

- **Install HiveFrame** - Installation in different environments
- **Configure Your First Hive** - Set up a basic hive
- **Load Data from Files** - Read CSV, JSON, and other formats

## Data Processing

- **Transform Data with Map and Filter** - Basic transformations
- **Aggregate Data by Groups** - GroupBy and aggregations
- **Join Multiple DataFrames** - Combine data from multiple sources
- **Handle Missing Data** - Deal with nulls and missing values
- **Optimize Large Datasets** - Performance tuning

## Streaming

- **Set Up a Stream Processor** - Configure streaming
- **Process Events in Real-Time** - Handle streaming data
- **Implement Windowing** - Time-based windows
- **Handle Stream Backpressure** - Manage overload

## Production Deployment

- **Deploy to Production** - Production deployment guide
- **Monitor HiveFrame Applications** - Metrics and monitoring
- **Handle Errors and Failures** - Error handling strategies
- **Scale Your Application** - Horizontal scaling
- **Secure Your Deployment** - Security best practices

## Integration

- **Connect to Kafka** - Kafka integration
- **Connect to PostgreSQL** - Database integration
- **Expose HTTP Endpoints** - HTTP API integration
- **Export Metrics to Prometheus** - Prometheus monitoring

## Tuning and Optimization

- **Tune Bee Colony Parameters** - Optimize colony behavior
- **Adjust Waggle Dance Thresholds** - Fine-tune coordination
- **Configure Abandonment Limits** - Self-healing settings
- **Optimize Memory Usage** - Reduce memory footprint

## Troubleshooting

- **Debug Slow Performance** - Identify bottlenecks
- **Fix Worker Failures** - Recover from failures
- **Resolve Memory Issues** - Handle OOM errors
- **Investigate Stuck Tasks** - Debug abandonment

## Advanced Topics

- **Custom Quality Metrics** - Define custom quality functions
- **Implement Custom Aggregations** - Create aggregation functions
- **Build Data Pipelines** - Multi-stage processing
- **Integrate with Existing Systems** - Legacy system integration

## Quick Reference

### Common Patterns

**Process CSV data:**
```python
df = HiveDataFrame.from_csv('data.csv')
result = df.filter(col('value') > 100).groupBy('category').agg(sum_agg(col('value')))
result.to_csv('output.csv')
```

**Stream processing:**
```python
stream = HiveStream(num_workers=4)
stream.start(lambda record: process(record))
stream.submit('key', {'data': 'value'})
result = stream.get_result(timeout=1.0)
stream.stop()
```

**Handle errors:**
```python
from hiveframe.resilience import RetryPolicy, CircuitBreaker

policy = RetryPolicy(max_retries=3, base_delay=1.0)
breaker = CircuitBreaker('my-service', failure_threshold=5)
```

**Monitor metrics:**
```python
from hiveframe.monitoring import get_registry, Counter

registry = get_registry()
counter = registry.get_counter('requests_total')
counter.inc()
```

## Navigation Tips

- **Start with basics**: If you're new, start with installation and configuration guides
- **Jump to your problem**: Use the categories above to find relevant guides
- **Combine techniques**: Many guides can be combined for complex tasks
- **Check examples**: Each guide includes working code examples

## Related Sections

- [Tutorials](../tutorials/getting-started.md) - Step-by-step learning
- [Reference](../reference/api-overview.md) - Complete API documentation
- [Explanation](../explanation/bee-colony-metaphor.md) - Understand how it works
