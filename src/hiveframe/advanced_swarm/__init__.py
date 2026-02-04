"""
Advanced Swarm Algorithms Module

Hybrid swarm intelligence combining multiple nature-inspired algorithms,
quantum-ready implementations, and federated learning.
"""

from .hybrid_swarm import (
    ParticleSwarmOptimizer,
    AntColonyOptimizer,
    FireflyAlgorithm,
    HybridSwarmOptimizer,
)
from .quantum_ready import (
    QuantumGateInterface,
    HybridQuantumClassical,
    QuantumInspiredOptimizer,
)
from .federated_learning import (
    FederatedSwarm,
    PrivacyPreservingML,
    CrossOrgTrainer,
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
