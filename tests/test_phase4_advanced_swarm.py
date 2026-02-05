"""
Tests for Phase 4: Advanced Swarm Algorithms
"""

import math

import pytest

from hiveframe.advanced_swarm.federated_learning import (
    CrossOrgTrainer,
    FederatedSwarm,
    LocalModel,
    PrivacyPreservingML,
)
from hiveframe.advanced_swarm.hybrid_swarm import (
    AntColonyOptimizer,
    FireflyAlgorithm,
    HybridSwarmOptimizer,
    ParticleSwarmOptimizer,
)
from hiveframe.advanced_swarm.quantum_ready import (
    HybridQuantumClassical,
    QuantumGateInterface,
    QuantumInspiredOptimizer,
)


class TestParticleSwarmOptimizer:
    """Test PSO algorithm"""

    def test_initialization(self):
        pso = ParticleSwarmOptimizer(num_particles=10, dimensions=2)
        pso.initialize()

        assert len(pso.particles) == 10
        assert len(pso.particles[0].position) == 2

    def test_optimize_simple(self):
        pso = ParticleSwarmOptimizer(num_particles=10, dimensions=2, bounds=(-10, 10))

        # Optimize sphere function: f(x) = sum(x^2)
        def sphere(x):
            return sum(xi**2 for xi in x)

        best_pos, best_fit = pso.optimize(sphere, max_iterations=50)

        assert best_pos is not None
        assert best_fit < 100  # Should find reasonably good solution


class TestAntColonyOptimizer:
    """Test ACO algorithm"""

    def test_initialization(self):
        aco = AntColonyOptimizer(num_ants=10, num_nodes=5)

        # Create distance matrix
        distance_matrix = [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ]

        aco.initialize(distance_matrix)

        assert len(aco.pheromone) == 5
        assert len(aco.pheromone[0]) == 5

    def test_optimize(self):
        aco = AntColonyOptimizer(num_ants=5, num_nodes=4)

        # Simple distance matrix
        distance_matrix = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]

        path, distance = aco.optimize(distance_matrix, max_iterations=10)

        assert path is not None
        assert len(path) == 4
        assert distance > 0


class TestFireflyAlgorithm:
    """Test Firefly algorithm"""

    def test_initialization(self):
        fa = FireflyAlgorithm(num_fireflies=10, dimensions=2)
        fa.initialize()

        assert len(fa.fireflies) == 10
        assert len(fa.fireflies[0].position) == 2

    def test_optimize(self):
        fa = FireflyAlgorithm(num_fireflies=15, dimensions=2, bounds=(-5, 5))

        # Optimize Rastrigin function (multimodal)
        def rastrigin(x):
            n = len(x)
            return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

        best_pos, best_fit = fa.optimize(rastrigin, max_iterations=30)

        assert best_pos is not None
        assert best_fit < 50  # Should find decent solution


class TestHybridSwarmOptimizer:
    """Test hybrid swarm optimizer"""

    def test_initialization(self):
        optimizer = HybridSwarmOptimizer()

        assert optimizer.pso is not None
        assert optimizer.firefly is not None

    def test_optimize_numerical(self):
        optimizer = HybridSwarmOptimizer()

        def sphere(x):
            return sum(xi**2 for xi in x)

        result = optimizer.optimize(sphere, problem_type="numerical", max_iterations=30)

        assert "algorithm" in result
        assert "best_position" in result
        assert "best_fitness" in result

    def test_optimize_multimodal(self):
        optimizer = HybridSwarmOptimizer()

        def multimodal(x):
            return sum((xi**2 + math.sin(10 * xi)) for xi in x)

        result = optimizer.optimize(multimodal, problem_type="multimodal", max_iterations=30)

        assert result["algorithm"] == "Firefly"


class TestQuantumGateInterface:
    """Test quantum gate interface"""

    def test_hadamard_gate(self):
        interface = QuantumGateInterface(num_qubits=2)
        gate = interface.hadamard(0)

        assert gate["name"] == "H"
        assert gate["target"] == 0
        assert "matrix" in gate

    def test_cnot_gate(self):
        interface = QuantumGateInterface(num_qubits=2)
        gate = interface.cnot(0, 1)

        assert gate["name"] == "CNOT"
        assert gate["control"] == 0
        assert gate["target"] == 1

    def test_rotation_gate(self):
        interface = QuantumGateInterface(num_qubits=2)
        gate = interface.rotation(0, math.pi / 4, "z")

        assert gate["name"] == "RZ"
        assert gate["angle"] == math.pi / 4

    def test_create_superposition(self):
        interface = QuantumGateInterface(num_qubits=4)
        gates = interface.create_superposition([0, 1, 2])

        assert len(gates) == 3
        assert all(g["name"] == "H" for g in gates)


class TestQuantumInspiredOptimizer:
    """Test quantum-inspired optimization"""

    def test_initialization(self):
        optimizer = QuantumInspiredOptimizer(population_size=10, dimensions=2)
        optimizer.initialize_quantum_population()

        assert len(optimizer.quantum_population) == 10
        assert len(optimizer.quantum_population[0]) == 2

    def test_observe(self):
        optimizer = QuantumInspiredOptimizer(population_size=5, dimensions=3)

        individual = [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2)]
        position = optimizer.observe(individual)

        assert len(position) == 3
        assert all(p in [0.0, 1.0] for p in position)

    def test_optimize(self):
        optimizer = QuantumInspiredOptimizer(population_size=10, dimensions=4)

        # Binary optimization problem
        def binary_fitness(x):
            # Maximize number of 1s
            return -sum(x)

        best_sol, best_fit = optimizer.optimize(binary_fitness, max_iterations=20)

        assert best_sol is not None
        assert len(best_sol) == 4


class TestHybridQuantumClassical:
    """Test hybrid quantum-classical computing"""

    def test_initialization(self):
        hybrid = HybridQuantumClassical()

        assert hybrid.quantum_interface is not None
        assert hybrid.quantum_optimizer is not None

    def test_quantum_subroutine_search(self):
        hybrid = HybridQuantumClassical()

        data = [1.0, 2.0, 5.0, 3.0, 2.5]
        result = hybrid.quantum_subroutine(data, operation="search")

        assert len(result) == len(data)

    def test_vqe(self):
        hybrid = HybridQuantumClassical()

        # Simple 2x2 Hamiltonian
        hamiltonian = [
            [1.0, 0.5],
            [0.5, 2.0],
        ]

        initial_params = [0.1, 0.2]
        result = hybrid.variational_quantum_eigensolver(hamiltonian, initial_params)

        assert "ground_state_energy" in result
        assert "optimal_parameters" in result


class TestPrivacyPreservingML:
    """Test privacy-preserving ML"""

    def test_add_noise(self):
        privacy = PrivacyPreservingML(privacy_budget=1.0)

        value = 10.0
        noisy_value = privacy.add_noise(value)

        # Should be different (with high probability)
        assert isinstance(noisy_value, float)

    def test_privatize_gradients(self):
        privacy = PrivacyPreservingML(privacy_budget=1.0)

        gradients = [1.0, 2.0, 3.0, 4.0]
        privatized = privacy.privatize_gradients(gradients, clip_norm=5.0)

        assert len(privatized) == len(gradients)

    def test_secure_aggregation(self):
        privacy = PrivacyPreservingML()

        local_updates = [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [0.8, 1.8, 2.8],
        ]

        aggregated = privacy.secure_aggregation(local_updates, use_encryption=True)

        assert len(aggregated) == 3


class TestFederatedSwarm:
    """Test federated learning swarm"""

    def test_initialization(self):
        swarm = FederatedSwarm(min_participants=2)

        assert swarm.version == 0
        assert swarm.min_participants == 2

    def test_initialize_global_model(self):
        swarm = FederatedSwarm()

        model = swarm.initialize_global_model(num_parameters=5)

        assert model.version == 0
        assert len(model.parameters) == 5

    def test_register_and_aggregate(self):
        swarm = FederatedSwarm(min_participants=2)
        swarm.initialize_global_model(num_parameters=3)

        # Register local updates
        local1 = LocalModel(
            node_id="node1",
            parameters=[1.0, 2.0, 3.0],
            num_samples=100,
            loss=0.5,
        )

        local2 = LocalModel(
            node_id="node2",
            parameters=[1.5, 2.5, 3.5],
            num_samples=150,
            loss=0.4,
        )

        swarm.register_local_update(local1)
        swarm.register_local_update(local2)

        # Aggregate
        global_model = swarm.aggregate_models(privacy_preserving=False)

        assert global_model.version == 1
        assert len(global_model.parameters) == 3


class TestCrossOrgTrainer:
    """Test cross-organization trainer"""

    def test_register_organization(self):
        trainer = CrossOrgTrainer()

        trainer.register_organization(
            org_id="org1",
            data_size=1000,
            privacy_requirements={"epsilon": 1.0},
        )

        assert "org1" in trainer.organizations

    def test_training_round(self):
        trainer = CrossOrgTrainer()

        # Register organizations (need at least 3)
        trainer.register_organization("org1", 1000, {})
        trainer.register_organization("org2", 1500, {})
        trainer.register_organization("org3", 1200, {})

        # Define dummy training function
        def train_local(global_params, org_data):
            # Simulate local training
            local_params = [p + 0.1 for p in global_params]
            loss = 0.5
            return local_params, loss

        # Run training round
        global_model = trainer.train_round(
            training_fn=train_local,
            participants=["org1", "org2", "org3"],
        )

        assert global_model is not None
        assert global_model.version > 0

    def test_get_training_status(self):
        trainer = CrossOrgTrainer()
        trainer.register_organization("org1", 1000, {})

        status = trainer.get_training_status()

        assert "current_version" in status
        assert "num_organizations" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
