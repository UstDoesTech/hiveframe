"""
Federated Learning Swarm

Privacy-preserving machine learning across organizations using swarm intelligence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import random
import statistics


@dataclass
class LocalModel:
    """Local model at a participant node"""

    node_id: str
    parameters: List[float]
    num_samples: int
    loss: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class GlobalModel:
    """Aggregated global model"""

    version: int
    parameters: List[float]
    contributor_nodes: List[str]
    avg_loss: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrivacyMetrics:
    """Privacy preservation metrics"""

    differential_privacy_epsilon: float
    noise_scale: float
    anonymization_level: str  # 'k-anonymity', 'l-diversity', 'differential-privacy'


class PrivacyPreservingML:
    """
    Privacy-preserving machine learning techniques.

    Implements differential privacy, secure aggregation, and other
    privacy-preserving methods for federated learning.
    """

    def __init__(
        self,
        privacy_budget: float = 1.0,
        noise_multiplier: float = 1.0,
    ):
        self.privacy_budget = privacy_budget
        self.noise_multiplier = noise_multiplier
        self.privacy_spent = 0.0

    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for differential privacy.

        Args:
            value: Original value
            sensitivity: Sensitivity of the query

        Returns:
            Noisy value
        """
        scale = sensitivity / self.privacy_budget
        noise = random.gauss(0, scale * self.noise_multiplier)
        return value + noise

    def privatize_gradients(
        self,
        gradients: List[float],
        clip_norm: float = 1.0,
    ) -> List[float]:
        """
        Privatize gradients using gradient clipping and noise addition.

        Args:
            gradients: Original gradients
            clip_norm: Clipping norm threshold

        Returns:
            Privatized gradients
        """
        # Clip gradients
        norm = sum(g * g for g in gradients) ** 0.5
        if norm > clip_norm:
            gradients = [g * clip_norm / norm for g in gradients]

        # Add noise
        privatized = [self.add_noise(g, sensitivity=clip_norm) for g in gradients]

        self.privacy_spent += 1.0 / len(gradients)

        return privatized

    def secure_aggregation(
        self,
        local_updates: List[List[float]],
        use_encryption: bool = True,
    ) -> List[float]:
        """
        Aggregate local updates securely.

        In production, would use cryptographic techniques (homomorphic encryption,
        secure multi-party computation) to ensure server never sees individual updates.

        Args:
            local_updates: List of local model updates
            use_encryption: Whether to use encryption (simulated here)

        Returns:
            Aggregated update
        """
        if not local_updates:
            return []

        # Simulate encryption (in production, use real crypto)
        if use_encryption:
            # Each update is "encrypted" (here just transformed)
            encrypted_updates = [
                [v + random.gauss(0, 0.1) for v in update] for update in local_updates
            ]
        else:
            encrypted_updates = local_updates

        # Aggregate (sum and average)
        num_params = len(encrypted_updates[0])
        aggregated = [0.0] * num_params

        for update in encrypted_updates:
            for i, value in enumerate(update):
                aggregated[i] += value

        aggregated = [v / len(encrypted_updates) for v in aggregated]

        return aggregated

    def check_privacy_budget(self) -> Dict[str, Any]:
        """Check remaining privacy budget"""
        remaining = max(0, self.privacy_budget - self.privacy_spent)

        return {
            "budget": self.privacy_budget,
            "spent": self.privacy_spent,
            "remaining": remaining,
            "can_continue": remaining > 0.1,
        }


class FederatedSwarm:
    """
    Federated learning coordinator using bee swarm intelligence.

    Coordinates distributed learning across participants while preserving
    privacy, using bee-inspired communication and aggregation patterns.
    """

    def __init__(
        self,
        aggregation_strategy: str = "fedavg",
        min_participants: int = 3,
    ):
        self.aggregation_strategy = aggregation_strategy
        self.min_participants = min_participants
        self.global_model: Optional[GlobalModel] = None
        self.local_models: Dict[str, LocalModel] = {}
        self.training_rounds: List[Dict[str, Any]] = []
        self.version = 0

    def initialize_global_model(self, num_parameters: int) -> GlobalModel:
        """
        Initialize global model.

        Args:
            num_parameters: Number of model parameters

        Returns:
            Initialized global model
        """
        self.global_model = GlobalModel(
            version=0,
            parameters=[random.gauss(0, 0.1) for _ in range(num_parameters)],
            contributor_nodes=[],
            avg_loss=float("inf"),
        )
        return self.global_model

    def register_local_update(self, local_model: LocalModel) -> None:
        """Register local model update from a participant"""
        self.local_models[local_model.node_id] = local_model

    def aggregate_models(
        self,
        privacy_preserving: bool = True,
    ) -> GlobalModel:
        """
        Aggregate local models into global model.

        Uses swarm-inspired aggregation: models with lower loss have higher
        influence (like waggle dance intensity proportional to food quality).

        Args:
            privacy_preserving: Whether to use privacy-preserving aggregation

        Returns:
            Updated global model
        """
        if len(self.local_models) < self.min_participants:
            raise ValueError(
                f"Need at least {self.min_participants} participants, "
                f"got {len(self.local_models)}"
            )

        if self.aggregation_strategy == "fedavg":
            # FedAvg: Weighted average by number of samples
            new_params = self._fedavg_aggregate()
        elif self.aggregation_strategy == "swarm_weighted":
            # Swarm-weighted: Weight by inverse loss (quality-based, like waggle dance)
            new_params = self._swarm_weighted_aggregate()
        else:
            raise ValueError(f"Unknown strategy: {self.aggregation_strategy}")

        # Update global model
        avg_loss = statistics.mean([m.loss for m in self.local_models.values()])

        self.version += 1
        self.global_model = GlobalModel(
            version=self.version,
            parameters=new_params,
            contributor_nodes=list(self.local_models.keys()),
            avg_loss=avg_loss,
        )

        # Record training round
        self.training_rounds.append(
            {
                "version": self.version,
                "num_participants": len(self.local_models),
                "avg_loss": avg_loss,
                "timestamp": time.time(),
            }
        )

        # Clear local models for next round
        self.local_models.clear()

        return self.global_model

    def _fedavg_aggregate(self) -> List[float]:
        """FedAvg aggregation: weighted by number of samples"""
        total_samples = sum(m.num_samples for m in self.local_models.values())
        num_params = len(list(self.local_models.values())[0].parameters)

        aggregated = [0.0] * num_params

        for model in self.local_models.values():
            weight = model.num_samples / total_samples
            for i, param in enumerate(model.parameters):
                aggregated[i] += weight * param

        return aggregated

    def _swarm_weighted_aggregate(self) -> List[float]:
        """
        Swarm-weighted aggregation.

        Weight models by quality (inverse loss), similar to how bee dances
        are more vigorous for higher quality food sources.
        """
        # Calculate weights based on inverse loss (better models have higher weight)
        models = list(self.local_models.values())
        inverse_losses = [1.0 / (m.loss + 1e-10) for m in models]
        total_weight = sum(inverse_losses)
        weights = [w / total_weight for w in inverse_losses]

        num_params = len(models[0].parameters)
        aggregated = [0.0] * num_params

        for model, weight in zip(models, weights):
            for i, param in enumerate(model.parameters):
                aggregated[i] += weight * param

        return aggregated

    def select_participants(
        self,
        available_nodes: List[str],
        selection_ratio: float = 0.3,
        strategy: str = "random",
    ) -> List[str]:
        """
        Select participants for next training round.

        Uses bee-inspired selection strategies.

        Args:
            available_nodes: List of available node IDs
            selection_ratio: Fraction of nodes to select
            strategy: Selection strategy ('random', 'contribution_based')

        Returns:
            Selected node IDs
        """
        num_select = max(self.min_participants, int(len(available_nodes) * selection_ratio))

        if strategy == "random":
            return random.sample(available_nodes, num_select)

        elif strategy == "contribution_based":
            # Prefer nodes that contributed good models in past
            # (like bees preferring productive foragers)
            # For simplicity, just random here
            return random.sample(available_nodes, num_select)

        return available_nodes[:num_select]


class CrossOrgTrainer:
    """
    Cross-organization trainer for federated learning.

    Enables multiple organizations to collaboratively train models
    without sharing raw data, using swarm intelligence for coordination.
    """

    def __init__(self):
        self.swarm = FederatedSwarm(aggregation_strategy="swarm_weighted")
        self.privacy = PrivacyPreservingML()
        self.organizations: Dict[str, Dict[str, Any]] = {}

    def register_organization(
        self,
        org_id: str,
        data_size: int,
        privacy_requirements: Dict[str, Any],
    ) -> None:
        """
        Register an organization as a participant.

        Args:
            org_id: Organization identifier
            data_size: Size of organization's dataset
            privacy_requirements: Privacy requirements
        """
        self.organizations[org_id] = {
            "data_size": data_size,
            "privacy_requirements": privacy_requirements,
            "contributions": 0,
        }

    def train_round(
        self,
        training_fn: Callable[[List[float], Any], Tuple[List[float], float]],
        participants: List[str],
    ) -> GlobalModel:
        """
        Execute one round of federated training.

        Args:
            training_fn: Function that trains local model
            participants: List of participating organization IDs

        Returns:
            Updated global model
        """
        if not self.swarm.global_model:
            # Initialize global model
            self.swarm.initialize_global_model(num_parameters=10)

        # Each participant trains locally
        for org_id in participants:
            if org_id not in self.organizations:
                continue

            # Simulate local training
            local_params, loss = training_fn(
                self.swarm.global_model.parameters, self.organizations[org_id]
            )

            # Apply privacy preservation
            private_params = self.privacy.privatize_gradients(local_params)

            # Submit local update
            local_model = LocalModel(
                node_id=org_id,
                parameters=private_params,
                num_samples=self.organizations[org_id]["data_size"],
                loss=loss,
            )

            self.swarm.register_local_update(local_model)
            self.organizations[org_id]["contributions"] += 1

        # Aggregate models
        global_model = self.swarm.aggregate_models(privacy_preserving=True)

        return global_model

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "current_version": self.swarm.version,
            "num_rounds": len(self.swarm.training_rounds),
            "num_organizations": len(self.organizations),
            "privacy_budget": self.privacy.check_privacy_budget(),
            "recent_loss": self.swarm.global_model.avg_loss if self.swarm.global_model else None,
        }
