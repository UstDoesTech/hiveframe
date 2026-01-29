# Core Concepts

HiveFrame is built on bee colony intelligence patterns. This guide explains the key concepts.

## The Bee Colony Metaphor

### Waggle Dance Protocol

In real bee colonies, forager bees communicate the location and quality of food sources through a "waggle dance". In HiveFrame:

- Workers report task quality through dance signals
- Higher quality tasks attract more workers
- The dance floor is a shared data structure for coordination

```python
from hiveframe import WaggleDance, DanceFloor

dance = WaggleDance(
    quality=0.9,      # Task quality score
    source_id='task_1',
    worker_id='bee_1'
)

floor = DanceFloor()
floor.register_dance(dance)
```

### Three-Tier Worker System

HiveFrame uses three types of workers, inspired by the Artificial Bee Colony (ABC) algorithm:

| Role | Behavior | Purpose |
|------|----------|---------|
| **Employed** | Exploit known good sources | Process high-quality tasks |
| **Onlooker** | Reinforce based on dances | Help with popular tasks |
| **Scout** | Explore for new sources | Find new work opportunities |

```python
from hiveframe import BeeRole

# Roles are automatically assigned based on colony needs
# Employed bees: ~50%
# Onlooker bees: ~40%
# Scout bees: ~10%
```

### Pheromone Signaling

Bees communicate through chemical pheromones. In HiveFrame:

- **Throttle pheromone**: Signals backpressure when overloaded
- **Alarm pheromone**: Signals errors or failures
- Pheromones decay over time (stigmergic coordination)

```python
from hiveframe import ColonyState, Pheromone

colony = ColonyState()

# Emit backpressure signal
colony.emit_pheromone(Pheromone(
    signal_type='throttle',
    intensity=0.8,
    source_worker='worker_1'
))

# Sense current throttle level
level = colony.sense_pheromone('throttle')
```

## Self-Healing Mechanism

The ABC algorithm includes an "abandonment" mechanism:

1. If a food source (task) isn't improving, increment a counter
2. When counter exceeds threshold, abandon the source
3. Scout bees will find new sources to replace it

This provides automatic recovery from stuck or failed tasks.

## Colony Temperature

The colony tracks overall "temperature" - a measure of system load:

- Low temperature (< 0.5): Normal operation
- Medium temperature (0.5-0.8): Elevated load
- High temperature (> 0.8): Critical, throttling engaged

```python
colony = ColonyState()
temp = colony.get_colony_temperature()
```

## Food Sources

Tasks in HiveFrame are modeled as "food sources":

```python
from hiveframe import FoodSource

source = FoodSource(
    source_id='batch_1',
    data=[1, 2, 3, 4, 5],
    quality=0.7
)
```

Quality is updated based on processing success, following the waggle dance protocol.
