---
sidebar_position: 3
---

# Streaming Basics

Learn how to process streaming data in real-time using HiveFrame's bee-inspired streaming architecture.

## Creating a Stream

```python
from hiveframe import HiveStream

# Create a stream with 4 worker bees
stream = HiveStream(num_workers=4)
```

## Processing Records

### Define a Processing Function

Your processing function receives a record and returns the processed result:

```python
def process_record(record):
    # Simple transformation
    value = record['value']
    return {
        'original': value,
        'doubled': value * 2,
        'squared': value ** 2
    }
```

### Start the Stream

```python
stream.start(process_record)
```

## Submitting Data

Submit records to the stream for processing:

```python
# Submit individual records
stream.submit('key_1', {'value': 10})
stream.submit('key_2', {'value': 20})
stream.submit('key_3', {'value': 30})

# Submit in a loop
for i in range(100):
    stream.submit(f'key_{i}', {'value': i})
```

## Getting Results

Retrieve processed results:

```python
# Get results with timeout
while True:
    result = stream.get_result(timeout=1.0)
    if result is None:
        break  # No more results
    print(result)
```

## Stopping the Stream

Always stop the stream when done:

```python
stream.stop()
```

## Complete Example

Here's a complete streaming application:

```python
from hiveframe import HiveStream
import time

# Create stream
stream = HiveStream(num_workers=6)

# Define processor
def process_sensor_data(record):
    sensor_id = record['sensor_id']
    temperature = record['temperature']
    
    # Convert Celsius to Fahrenheit
    temp_f = (temperature * 9/5) + 32
    
    # Classify
    if temp_f > 100:
        alert = 'HIGH'
    elif temp_f < 32:
        alert = 'LOW'
    else:
        alert = 'NORMAL'
    
    return {
        'sensor_id': sensor_id,
        'temp_c': temperature,
        'temp_f': temp_f,
        'alert': alert,
        'timestamp': time.time()
    }

# Start processing
stream.start(process_sensor_data)

# Simulate sensor data
sensors = ['sensor_1', 'sensor_2', 'sensor_3']
for i in range(50):
    import random
    sensor_id = random.choice(sensors)
    temperature = random.uniform(0, 50)  # Celsius
    
    stream.submit(
        key=sensor_id,
        value={
            'sensor_id': sensor_id,
            'temperature': temperature
        }
    )
    time.sleep(0.1)

# Collect results
results = []
while True:
    result = stream.get_result(timeout=2.0)
    if result is None:
        break
    results.append(result)
    
    # Print alerts
    if result['alert'] != 'NORMAL':
        print(f"⚠️  {result['sensor_id']}: {result['temp_f']:.1f}°F - {result['alert']}")

# Stop stream
stream.stop()

print(f"Processed {len(results)} records")
```

## Advanced Streaming

For production workloads, use the enhanced streaming processor:

```python
from hiveframe.streaming import EnhancedStreamProcessor
from hiveframe.streaming.windows import TumblingWindow
from hiveframe.streaming.watermarks import PeriodicWatermarkGenerator
from hiveframe.streaming.guarantees import DeliveryGuarantee

# Create enhanced processor
stream = EnhancedStreamProcessor(
    num_workers=8,
    window_assigner=TumblingWindow(window_size=60.0),  # 60-second windows
    watermark_generator=PeriodicWatermarkGenerator(interval=10.0),
    delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE
)

# Process with aggregation
def aggregate_fn(records):
    total = sum(r['value'] for r in records)
    count = len(records)
    return {'total': total, 'count': count, 'avg': total / count}

for record in incoming_stream:
    result = stream.process_record(record, aggregate_fn, initial_value=[])
    if result:
        print(f"Window result: {result}")
```

## Monitoring Streams

Get metrics about stream processing:

```python
metrics = stream.get_metrics()
print(f"Records processed: {metrics['records_processed']}")
print(f"Processing rate: {metrics['processing_rate']}/sec")
print(f"Worker utilization: {metrics['worker_utilization']:.1%}")
```

## Bee Colony Streaming

HiveFrame's streaming uses bee colony patterns:

- **Employed Bees**: Process incoming records
- **Onlooker Bees**: Handle high-priority or slow records
- **Scout Bees**: Monitor for new data sources

This provides:
- Automatic load balancing
- Adaptive backpressure
- Self-healing from failures

## Best Practices

1. **Use Appropriate Buffer Sizes**: Balance memory and throughput
2. **Handle Errors Gracefully**: Return error markers instead of raising exceptions
3. **Monitor Metrics**: Track processing rates and worker utilization
4. **Set Reasonable Timeouts**: Match expected processing times
5. **Test Backpressure**: Ensure your system handles burst traffic

## Error Handling

```python
def safe_processor(record):
    try:
        # Your processing logic
        result = process(record)
        return {'success': True, 'data': result}
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'record': record
        }
```

## Next Steps

- **Production Deployment** - Deploy streaming applications
- **Windowing and Watermarks** - Understand streaming concepts
- **Monitoring** - Monitor streaming applications
- **Enhanced Streaming** - Complete streaming API
