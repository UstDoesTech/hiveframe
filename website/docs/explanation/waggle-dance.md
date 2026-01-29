---
sidebar_position: 2
---

# Waggle Dance Protocol

The waggle dance is the heart of HiveFrame's coordination mechanism. This page explains how workers communicate quality information and how the system uses it for optimization.

## What is a Waggle Dance?

In biology, the waggle dance is how bees communicate the location and quality of food sources. In HiveFrame, it's how workers advertise the results of processing tasks.

## The Dance Structure

A waggle dance in HiveFrame contains:

```python
@dataclass
class WaggleDance:
    partition_id: str       # Which partition (like direction in bee dance)
    quality_score: float    # How good the result (0.0 to 1.0)
    processing_time: float  # How long it took (seconds)
    result_size: int        # Output data size (bytes)
    worker_id: str          # Who's reporting
    timestamp: float        # When the dance occurred
```

## Quality Score Calculation

Quality is calculated from multiple factors:

```python
def calculate_quality(result, processing_time, errors):
    # Base quality from result
    base_quality = evaluate_result_quality(result)
    
    # Penalty for slow processing
    time_factor = 1.0 / (1.0 + processing_time / target_time)
    
    # Penalty for errors
    error_factor = 1.0 / (1.0 + error_count)
    
    # Combined quality
    quality = base_quality * time_factor * error_factor
    
    return min(1.0, max(0.0, quality))
```

### Factors Affecting Quality

1. **Result Correctness**: Did the processing succeed?
2. **Processing Speed**: Faster is better
3. **Error Rate**: Fewer errors means higher quality
4. **Resource Efficiency**: Less memory/CPU usage improves quality

## The Dance Floor

The dance floor is a shared data structure where workers post their dances:

```python
class DanceFloor:
    def __init__(self):
        self._dances: Dict[str, List[WaggleDance]] = {}
        self._lock = threading.Lock()
    
    def register_dance(self, dance: WaggleDance):
        with self._lock:
            partition_dances = self._dances.get(dance.partition_id, [])
            partition_dances.append(dance)
            
            # Keep only recent dances
            cutoff = time.time() - DANCE_RETENTION_SECONDS
            self._dances[dance.partition_id] = [
                d for d in partition_dances 
                if d.timestamp > cutoff
            ]
    
    def get_all_dances(self) -> List[WaggleDance]:
        with self._lock:
            all_dances = []
            for dances in self._dances.values():
                all_dances.extend(dances)
            return all_dances
```

## Observer Pattern

Onlooker bees watch the dance floor and select partitions based on quality:

```python
class OnlookerBee:
    def select_partition(self, dance_floor: DanceFloor) -> str:
        dances = dance_floor.get_all_dances()
        
        if not dances:
            return self._random_partition()
        
        # Calculate selection probabilities
        total_quality = sum(d.quality_score for d in dances)
        
        if total_quality == 0:
            return self._random_partition()
        
        # Roulette wheel selection
        probabilities = [
            d.quality_score / total_quality 
            for d in dances
        ]
        
        selected_dance = random.choices(dances, weights=probabilities)[0]
        return selected_dance.partition_id
```

This is called **roulette wheel selection** - partitions with higher quality have higher probability of being selected.

## Fitness Proportional Selection

The mathematical basis for selection:

```
P(partition_i) = quality_i / Σ(quality_j) for all j
```

Example:
- Partition A: quality = 0.8, probability = 0.8/2.0 = 40%
- Partition B: quality = 0.5, probability = 0.5/2.0 = 25%
- Partition C: quality = 0.7, probability = 0.7/2.0 = 35%

High-quality partitions get more workers, accelerating their completion.

## Dance Vigor

The "vigor" of a dance combines quality with throughput:

```python
def calculate_vigor(dance: WaggleDance) -> float:
    # Quality weighted by throughput
    throughput = dance.result_size / dance.processing_time
    
    # Normalize throughput to 0-1 range
    normalized_throughput = throughput / max_throughput
    
    # Vigor is composite metric
    vigor = (
        0.6 * dance.quality_score +
        0.4 * normalized_throughput
    )
    
    return vigor
```

More vigorous dances attract more attention.

## Dance Retention

Dances decay over time to prevent stale information:

```python
DANCE_RETENTION_SECONDS = 30.0  # Keep dances for 30 seconds

def is_dance_valid(dance: WaggleDance) -> bool:
    age = time.time() - dance.timestamp
    return age < DANCE_RETENTION_SECONDS
```

This ensures workers react to current conditions, not historical data.

## Dance Threshold

Workers only dance if quality exceeds a threshold:

```python
DANCE_THRESHOLD = 0.3  # Only advertise if quality > 0.3

if quality_score >= DANCE_THRESHOLD:
    dance_floor.register_dance(WaggleDance(
        partition_id=partition_id,
        quality_score=quality_score,
        processing_time=elapsed_time,
        result_size=len(result),
        worker_id=self.worker_id,
        timestamp=time.time()
    ))
```

Low-quality results don't get advertised, preventing workers from being drawn to poor tasks.

## Communication Flow

```
┌──────────────┐
│ Employed Bee │
│  Processes   │
│  Partition   │
└──────┬───────┘
       │
       ↓
  Quality Score
       │
       ↓ (if score >= threshold)
┌──────────────┐
│ Dance Floor  │
│   (Shared)   │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Onlooker Bee │
│   Observes   │
│   & Selects  │
└──────────────┘
```

## Example: Complete Dance Cycle

```python
# Step 1: Employed bee processes partition
result = employed_bee.process_partition('partition_A')
processing_time = 2.5  # seconds
result_size = 1024 * 1024  # 1 MB

# Step 2: Calculate quality
quality = calculate_quality(
    result=result,
    processing_time=processing_time,
    errors=0
)  # quality = 0.85

# Step 3: Perform waggle dance
if quality >= DANCE_THRESHOLD:
    dance = WaggleDance(
        partition_id='partition_A',
        quality_score=0.85,
        processing_time=2.5,
        result_size=result_size,
        worker_id='bee_1',
        timestamp=time.time()
    )
    dance_floor.register_dance(dance)

# Step 4: Onlooker bee observes
onlooker_bee = OnlookerBee()
selected_partition = onlooker_bee.select_partition(dance_floor)
# High probability of selecting 'partition_A' due to quality 0.85

# Step 5: Onlooker processes selected partition
onlooker_bee.process_partition(selected_partition)
```

## Benefits of the Waggle Dance

### 1. Decentralized Coordination

No central scheduler needed - workers coordinate through shared information.

### 2. Adaptive Load Balancing

Work naturally flows to high-quality partitions, which typically finish faster.

### 3. Fault Tolerance

If a worker fails, its dance disappears. Other workers automatically redistribute.

### 4. Emergent Optimization

The system converges to near-optimal allocation without explicit programming.

## Tuning Parameters

You can tune the waggle dance behavior:

```python
hive = HiveFrame(
    num_workers=8,
    dance_threshold=0.3,      # Minimum quality to advertise
    dance_retention=30.0,     # How long dances are valid
    quality_weight=0.6,       # Weight of quality in vigor calculation
    throughput_weight=0.4     # Weight of throughput in vigor
)
```

### Dance Threshold

- **Lower (0.1-0.3)**: More dances, more coordination overhead
- **Higher (0.5-0.7)**: Fewer dances, only advertise best tasks

### Dance Retention

- **Shorter (10-20s)**: React quickly to changes, higher churn
- **Longer (30-60s)**: More stable, slower to adapt

## Implementation Details

The dance floor is implemented with thread-safe collections:

```python
from collections import defaultdict
from threading import RLock

class DanceFloor:
    def __init__(self):
        self._dances = defaultdict(list)
        self._lock = RLock()
    
    def register_dance(self, dance):
        with self._lock:
            self._dances[dance.partition_id].append(dance)
            self._cleanup_old_dances()
    
    def _cleanup_old_dances(self):
        cutoff = time.time() - DANCE_RETENTION_SECONDS
        for partition_id in list(self._dances.keys()):
            self._dances[partition_id] = [
                d for d in self._dances[partition_id]
                if d.timestamp > cutoff
            ]
            if not self._dances[partition_id]:
                del self._dances[partition_id]
```

## Comparison to Other Coordination Mechanisms

| Mechanism | Overhead | Adaptability | Fault Tolerance |
|-----------|----------|--------------|-----------------|
| Waggle Dance | Medium | High | High |
| Central Scheduler | Low | Low | Low (SPOF) |
| Gossip Protocol | High | Medium | High |
| Consensus (Raft) | High | Low | Medium |

## Next Steps

- **Abandonment Mechanism** - How failed tasks are handled
- **Pheromone Coordination** - Backpressure signaling
- **Colony Architecture** - Overall system design
- **Quality Metrics** - Quality calculation reference
