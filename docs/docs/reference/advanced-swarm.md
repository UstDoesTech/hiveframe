---
sidebar_position: 15
---

# Advanced Swarm Algorithms

Phase 4 advanced swarm algorithms module provides hybrid swarm optimization, quantum-ready algorithms, and federated learning capabilities.

## Overview

The advanced swarm module extends HiveFrame's bee-inspired algorithms with additional swarm intelligence techniques (PSO, ACO, Firefly), quantum-ready optimization, and privacy-preserving federated learning.

## Hybrid Swarm Intelligence

Combine multiple swarm algorithms for advanced optimization.

### Classes

#### `HybridSwarmOptimizer`

Main orchestrator that automatically selects the best algorithm for your problem.

```python
from hiveframe.advanced_swarm import HybridSwarmOptimizer, ProblemType

optimizer = HybridSwarmOptimizer(n_particles=30, max_iterations=100)
```

**Parameters:**
- `n_particles` (int): Number of swarm particles/agents
- `max_iterations` (int): Maximum optimization iterations
- `problem_types` (List[ProblemType], optional): Specific algorithms to use

**Methods:**

##### `optimize(fitness_fn: Callable, dimensions: int, bounds: List[Tuple[float, float]], problem_type: ProblemType) -> Dict[str, Any]`

Optimizes a problem using the most appropriate swarm algorithm.

```python
def fitness_fn(solution):
    return sum((x - 5)**2 for x in solution)

result = optimizer.optimize(
    fitness_fn=fitness_fn,
    dimensions=10,
    bounds=[(-10, 10)] * 10,
    problem_type=ProblemType.CONTINUOUS
)
# Returns: {
#   'best_solution': [5.01, 4.99, 5.02, ...],
#   'best_fitness': 0.0012,
#   'algorithm_used': 'PSO',
#   'iterations': 42
# }
```

**Problem Types:**
- `ProblemType.CONTINUOUS` - Numerical optimization (uses PSO)
- `ProblemType.ROUTING` - Path/routing problems (uses ACO)
- `ProblemType.MULTIMODAL` - Multiple local optima (uses Firefly)
- `ProblemType.AUTO` - Automatically detect best algorithm

#### `ParticleSwarmOptimizer`

Particle Swarm Optimization for continuous numerical problems.

```python
from hiveframe.advanced_swarm import ParticleSwarmOptimizer

pso = ParticleSwarmOptimizer(n_particles=30, max_iterations=100)
```

**Methods:**

##### `optimize(fitness_fn: Callable, dimensions: int, bounds: List[Tuple[float, float]]) -> Dict[str, Any]`

Optimizes using PSO algorithm.

```python
result = pso.optimize(fitness_fn, dimensions=10, bounds=[(-10, 10)] * 10)
```

**PSO Parameters:**
- `w` (float, default=0.7): Inertia weight
- `c1` (float, default=1.5): Cognitive parameter
- `c2` (float, default=1.5): Social parameter

#### `AntColonyOptimizer`

Ant Colony Optimization for routing and graph problems.

```python
from hiveframe.advanced_swarm import AntColonyOptimizer

aco = AntColonyOptimizer(n_ants=30, max_iterations=100)
```

**Methods:**

##### `optimize(fitness_fn: Callable, dimensions: int) -> Dict[str, Any]`

Optimizes using ACO algorithm (for TSP-like problems).

```python
# Define distance matrix for routing problem
distance_matrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

def routing_fitness(path):
    return sum(distance_matrix[path[i]][path[i+1]] for i in range(len(path)-1))

result = aco.optimize(fitness_fn=routing_fitness, dimensions=4)
```

**ACO Parameters:**
- `alpha` (float, default=1.0): Pheromone importance
- `beta` (float, default=2.0): Heuristic importance
- `evaporation` (float, default=0.5): Pheromone evaporation rate

#### `FireflyAlgorithm`

Firefly Algorithm for multimodal optimization problems.

```python
from hiveframe.advanced_swarm import FireflyAlgorithm

firefly = FireflyAlgorithm(n_fireflies=30, max_iterations=100)
```

**Methods:**

##### `optimize(fitness_fn: Callable, dimensions: int, bounds: List[Tuple[float, float]]) -> Dict[str, Any]`

Optimizes using Firefly algorithm.

```python
result = firefly.optimize(fitness_fn, dimensions=10, bounds=[(-10, 10)] * 10)
```

**Firefly Parameters:**
- `alpha` (float, default=0.5): Randomness parameter
- `beta0` (float, default=1.0): Attractiveness at distance 0
- `gamma` (float, default=1.0): Light absorption coefficient

## Quantum-Ready Algorithms

Quantum-inspired optimization for future quantum hardware integration.

### Classes

#### `HybridQuantumClassical`

Combines quantum-inspired exploration with classical optimization.

```python
from hiveframe.advanced_swarm import HybridQuantumClassical

qc_optimizer = HybridQuantumClassical(n_qubits=4, n_classical=10)
```

**Parameters:**
- `n_qubits` (int): Number of quantum bits for exploration
- `n_classical` (int): Number of classical parameters
- `backend` (str, default='simulator'): Quantum backend to use

**Methods:**

##### `optimize(fitness_fn: Callable, max_iterations: int) -> Dict[str, Any]`

Optimizes using hybrid quantum-classical approach.

```python
def fitness_fn(solution):
    quantum_part, classical_part = solution[:4], solution[4:]
    # Evaluate hybrid solution
    quantum_score = sum(q**2 for q in quantum_part)
    classical_score = sum((c - 5)**2 for c in classical_part)
    return quantum_score + classical_score

result = qc_optimizer.optimize(fitness_fn=fitness_fn, max_iterations=50)
# Returns: {
#   'best_solution': [...],
#   'best_fitness': 0.015,
#   'quantum_states': [...],
#   'classical_solution': [...]
# }
```

#### `QuantumGateInterface`

Interface for quantum gate operations.

```python
from hiveframe.advanced_swarm import QuantumGateInterface

gate_interface = QuantumGateInterface(n_qubits=4)
```

**Methods:**

##### `apply_hadamard(qubit: int) -> None`

Applies Hadamard gate to create superposition.

##### `apply_cnot(control: int, target: int) -> None`

Applies controlled-NOT gate for entanglement.

##### `apply_rotation(qubit: int, angle: float, axis: str) -> None`

Applies rotation gate (axis: 'x', 'y', or 'z').

##### `measure() -> List[int]`

Measures quantum state and returns classical bits.

```python
# Example quantum circuit
gate_interface.apply_hadamard(0)
gate_interface.apply_cnot(0, 1)
gate_interface.apply_rotation(2, angle=np.pi/4, axis='y')
result = gate_interface.measure()
```

#### `QuantumInspiredOptimizer`

Quantum-inspired optimization without quantum hardware.

```python
from hiveframe.advanced_swarm import QuantumInspiredOptimizer

qi_optimizer = QuantumInspiredOptimizer(n_particles=30)
```

**Methods:**

##### `optimize(fitness_fn: Callable, dimensions: int, bounds: List[Tuple[float, float]]) -> Dict[str, Any]`

Uses quantum principles (superposition, interference) in classical algorithm.

```python
result = qi_optimizer.optimize(fitness_fn, dimensions=10, bounds=[(-10, 10)] * 10)
```

## Federated Learning Swarm

Privacy-preserving machine learning across organizations.

### Classes

#### `CrossOrgTrainer`

Main orchestrator for federated learning with swarm coordination.

```python
from hiveframe.advanced_swarm import CrossOrgTrainer

fed_trainer = CrossOrgTrainer(n_organizations=5)
```

**Parameters:**
- `n_organizations` (int): Number of participating organizations
- `aggregation_strategy` (str, default='weighted'): How to aggregate models
- `privacy_mode` (str, default='differential'): Privacy preservation method

**Methods:**

##### `submit_local_model(org_id: int, local_model: Dict[str, Any]) -> None`

Submits a locally trained model from an organization.

```python
# Each organization trains locally on private data
local_model = {
    'weights': [0.5, 0.3, ...],  # Model weights
    'n_samples': 1000             # Number of training samples
}
fed_trainer.submit_local_model(org_id=0, local_model=local_model)
```

##### `aggregate_models(privacy_epsilon: float = 1.0, use_secure_aggregation: bool = True) -> Dict[str, Any]`

Aggregates models with privacy guarantees using swarm-weighted averaging.

```python
global_model = fed_trainer.aggregate_models(
    privacy_epsilon=1.0,  # Differential privacy parameter
    use_secure_aggregation=True
)
# Returns: {
#   'weights': [...],  # Global model weights
#   'accuracy': 0.94,
#   'privacy_spent': 1.0,
#   'n_participants': 5
# }
```

**Privacy Methods:**
- `differential` - Differential privacy with noise injection
- `secure` - Secure aggregation (simulated encryption)
- `both` - Both methods combined

#### `PrivacyPreservingML`

Core privacy-preserving machine learning functionality.

```python
from hiveframe.advanced_swarm import PrivacyPreservingML

privacy_ml = PrivacyPreservingML(epsilon=1.0, delta=1e-5)
```

**Methods:**

##### `add_noise(gradients: List[float], sensitivity: float) -> List[float]`

Adds differential privacy noise to gradients.

```python
noisy_gradients = privacy_ml.add_noise(gradients, sensitivity=0.1)
```

##### `secure_aggregate(models: List[Dict], weights: List[float]) -> Dict`

Performs secure aggregation of models.

```python
aggregated = privacy_ml.secure_aggregate(local_models, participant_weights)
```

#### `FederatedSwarm`

Coordinates distributed learning using swarm intelligence.

```python
from hiveframe.advanced_swarm import FederatedSwarm

swarm = FederatedSwarm(n_workers=20)
```

**Methods:**

##### `coordinate_training(dataset_partitions: List[Any], n_rounds: int) -> Dict[str, Any]`

Coordinates federated training across workers.

```python
result = swarm.coordinate_training(
    dataset_partitions=partitioned_data,
    n_rounds=10
)
```

## Examples

### Hybrid Swarm Optimization Example

```python
from hiveframe.advanced_swarm import HybridSwarmOptimizer, ProblemType

# Create optimizer that automatically selects best algorithm
optimizer = HybridSwarmOptimizer(n_particles=30, max_iterations=100)

# Example 1: Numerical optimization (uses PSO)
def numerical_fitness(solution):
    return sum((x - 5)**2 for x in solution)

result = optimizer.optimize(
    fitness_fn=numerical_fitness,
    dimensions=10,
    bounds=[(-10, 10)] * 10,
    problem_type=ProblemType.CONTINUOUS
)
print(f"Best solution: {result['best_solution']}")
print(f"Best fitness: {result['best_fitness']}")
print(f"Algorithm used: {result['algorithm_used']}")

# Example 2: Routing optimization (uses ACO)
distance_matrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

def routing_fitness(path):
    return sum(distance_matrix[path[i]][path[i+1]] for i in range(len(path)-1))

result = optimizer.optimize(
    fitness_fn=routing_fitness,
    dimensions=4,
    problem_type=ProblemType.ROUTING
)
print(f"Best route: {result['best_solution']}")
print(f"Total distance: {result['best_fitness']}")
```

### Quantum-Classical Hybrid Example

```python
from hiveframe.advanced_swarm import HybridQuantumClassical

# Create quantum-classical optimizer
qc_optimizer = HybridQuantumClassical(n_qubits=4, n_classical=10)

# Define fitness function
def fitness_fn(solution):
    quantum_part, classical_part = solution[:4], solution[4:]
    # Quantum part explores complex search space
    quantum_score = sum(q**2 for q in quantum_part)
    # Classical part optimizes parameters
    classical_score = sum((c - 5)**2 for c in classical_part)
    return quantum_score + classical_score

# Optimize
result = qc_optimizer.optimize(fitness_fn=fitness_fn, max_iterations=50)
print(f"Quantum exploration states: {result['quantum_states']}")
print(f"Classical solution: {result['classical_solution']}")
print(f"Best fitness: {result['best_fitness']}")
```

### Federated Learning Example

```python
from hiveframe.advanced_swarm import CrossOrgTrainer

# Create federated learning coordinator
fed_trainer = CrossOrgTrainer(n_organizations=5)

# Each organization trains locally on private data
# Note: Implement these functions for your ML framework
# Example signatures shown:
# def train_model_on_org_data(org_id: int) -> List[float]:
#     """Returns trained model weights"""
# def get_local_dataset_size(org_id: int) -> int:
#     """Returns number of training samples"""

for org_id in range(5):
    local_model = {
        'weights': train_model_on_org_data(org_id),
        'n_samples': get_local_dataset_size(org_id)
    }
    fed_trainer.submit_local_model(org_id, local_model)

# Aggregate models with privacy guarantees
global_model = fed_trainer.aggregate_models(
    privacy_epsilon=1.0,  # Differential privacy parameter
    use_secure_aggregation=True
)

print(f"Global model accuracy: {global_model['accuracy']:.4f}")
print(f"Privacy budget used: {global_model['privacy_spent']:.2f}")
print(f"Participating organizations: {global_model['n_participants']}")
```

## Algorithm Selection Guide

| Problem Type | Recommended Algorithm | Characteristics |
|--------------|----------------------|-----------------|
| Continuous numerical optimization | PSO | Fast convergence, good for convex problems |
| Routing/path finding | ACO | Natural for graph problems, pheromone trails |
| Multiple local optima | Firefly | Attraction mechanism avoids local optima |
| High-dimensional search | Quantum-inspired | Quantum principles for complex spaces |
| Distributed ML | Federated Swarm | Privacy-preserving, swarm coordination |

## Performance Tips

- **PSO**: Adjust `w` (inertia) for exploration/exploitation balance
- **ACO**: Increase `n_ants` for complex routing problems
- **Firefly**: Tune `gamma` based on problem scale
- **Quantum**: Use simulator for testing, real quantum hardware for production
- **Federated**: Balance privacy (`epsilon`) with model accuracy

## See Also

- [Autonomous Operations](./autonomous) - Self-tuning and optimization
- [AI Integration](./ai) - Natural language and data prep
- [Core](./core) - Basic bee colony algorithms
