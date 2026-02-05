"""
Advanced Swarm Algorithms Module

Hybrid swarm intelligence combining multiple nature-inspired algorithms,
quantum-ready implementations, and federated learning.
"""

from .federated_learning import (
    CrossOrgTrainer,
    FederatedSwarm,
    PrivacyPreservingML,
)
from .hybrid_swarm import (
    AntColonyOptimizer,
    FireflyAlgorithm,
    HybridSwarmOptimizer,
    ParticleSwarmOptimizer,
)
from .quantum_ready import (
    HybridQuantumClassical,
    QuantumGateInterface,
    QuantumInspiredOptimizer,
)

__all__ = [
    "ParticleSwarmOptimizer",
    "AntColonyOptimizer",
    "FireflyAlgorithm",
    "HybridSwarmOptimizer",
    "QuantumGateInterface",
    "HybridQuantumClassical",
    "QuantumInspiredOptimizer",
    "FederatedSwarm",
    "PrivacyPreservingML",
    "CrossOrgTrainer",
]
