"""
AutoML Swarm - Hyperparameter Optimization using ABC Algorithm
==============================================================

Applies the Artificial Bee Colony algorithm to hyperparameter tuning,
where each bee explores a different hyperparameter configuration.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class TaskType(Enum):
    """Machine learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


@dataclass
class HyperparameterSpace:
    """
    Defines the search space for hyperparameters.

    Example:
        space = HyperparameterSpace(
            continuous={"learning_rate": (0.001, 0.1)},
            discrete={"max_depth": [3, 5, 7, 9]},
            categorical={"algorithm": ["rf", "gbm", "xgb"]}
        )
    """

    continuous: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    discrete: Dict[str, List[int]] = field(default_factory=dict)
    categorical: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ModelConfiguration:
    """A specific model configuration (hyperparameter set)."""

    params: Dict[str, Any]
    score: float = 0.0
    trials: int = 0
    fitness: float = 0.0


class AutoMLSwarm:
    """
    AutoML using Artificial Bee Colony optimization.

    Uses bee-inspired search to find optimal hyperparameters:
    - Employed bees: Exploit known good configurations
    - Onlooker bees: Reinforce promising configurations
    - Scout bees: Explore new configuration space

    Example:
        # Define hyperparameter space
        space = HyperparameterSpace(
            continuous={"learning_rate": (0.001, 0.1)},
            discrete={"max_depth": [3, 5, 7, 9]},
            categorical={"algorithm": ["rf", "gbm"]}
        )

        # Create AutoML swarm
        automl = AutoMLSwarm(
            n_workers=10,
            space=space,
            task_type=TaskType.CLASSIFICATION
        )

        # Define objective function (model training + evaluation)
        def objective(params):
            model = train_model(X_train, y_train, **params)
            score = evaluate_model(model, X_val, y_val)
            return score

        # Run optimization
        best_params, best_score = automl.optimize(
            objective=objective,
            n_iterations=50,
            maximize=True
        )
    """

    def __init__(
        self,
        n_workers: int = 10,
        space: Optional[HyperparameterSpace] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        limit: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize AutoML swarm.

        Args:
            n_workers: Number of worker bees
            space: Hyperparameter search space
            task_type: Type of ML task
            limit: Abandonment limit for poor configurations
            seed: Random seed for reproducibility
        """
        self.n_workers = n_workers
        self.space = space or HyperparameterSpace()
        self.task_type = task_type
        self.limit = limit

        if seed is not None:
            random.seed(seed)

        self.configurations: List[ModelConfiguration] = []
        self.best_config: Optional[ModelConfiguration] = None
        self.history: List[Tuple[int, float]] = []

    def _sample_configuration(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}

        # Sample continuous parameters
        for param, (low, high) in self.space.continuous.items():
            config[param] = random.uniform(low, high)

        # Sample discrete parameters
        for param, values in self.space.discrete.items():
            config[param] = random.choice(values)

        # Sample categorical parameters
        for param, values in self.space.categorical.items():
            config[param] = random.choice(values)

        return config

    def _mutate_configuration(
        self, base_config: Dict[str, Any], neighbor_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mutate a configuration by combining with a neighbor.

        This is the employed bee phase - exploring nearby configurations.
        """
        new_config = base_config.copy()

        # Mutate continuous parameters
        for param in self.space.continuous.keys():
            if param in base_config and param in neighbor_config:
                phi = random.uniform(-1, 1)
                new_val = base_config[param] + phi * (base_config[param] - neighbor_config[param])
                # Clamp to bounds
                low, high = self.space.continuous[param]
                new_config[param] = max(low, min(high, new_val))

        # For discrete/categorical, randomly choose from base or neighbor
        for param in list(self.space.discrete.keys()) + list(self.space.categorical.keys()):
            if random.random() < 0.5:
                new_config[param] = neighbor_config.get(param, base_config[param])

        return new_config

    def _select_probabilistic(self) -> ModelConfiguration:
        """
        Select a configuration probabilistically based on fitness.

        This is the onlooker bee phase - reinforcing good configurations.
        """
        total_fitness = sum(c.fitness for c in self.configurations)
        if total_fitness == 0:
            return random.choice(self.configurations)

        r = random.uniform(0, total_fitness)
        cumsum = 0
        for config in self.configurations:
            cumsum += config.fitness
            if cumsum >= r:
                return config

        return self.configurations[-1]

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_iterations: int = 50,
        maximize: bool = True,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run bee-inspired hyperparameter optimization.

        Args:
            objective: Function that takes hyperparameters and returns score
            n_iterations: Number of optimization iterations
            maximize: Whether to maximize the objective (True) or minimize (False)
            verbose: Whether to print progress

        Returns:
            Tuple of (best_params, best_score)
        """
        # Initialize configurations
        self.configurations = []
        for _ in range(self.n_workers):
            config = self._sample_configuration()
            score = objective(config)

            model_config = ModelConfiguration(
                params=config, score=score, trials=1, fitness=score if maximize else -score
            )
            self.configurations.append(model_config)

        # Track best
        self.best_config = max(self.configurations, key=lambda c: c.fitness)
        self.history = [(0, self.best_config.score)]

        if verbose:
            print(f"Initial best score: {self.best_config.score:.4f}")

        # Main optimization loop
        for iteration in range(1, n_iterations + 1):
            # Employed bee phase
            for i, config in enumerate(self.configurations):
                # Select random neighbor
                neighbor_idx = random.choice([j for j in range(len(self.configurations)) if j != i])
                neighbor = self.configurations[neighbor_idx]

                # Generate new configuration
                new_params = self._mutate_configuration(config.params, neighbor.params)
                new_score = objective(new_params)
                new_fitness = new_score if maximize else -new_score

                # Greedy selection
                if new_fitness > config.fitness:
                    config.params = new_params
                    config.score = new_score
                    config.fitness = new_fitness
                    config.trials = 0
                else:
                    config.trials += 1

            # Onlooker bee phase
            for _ in range(self.n_workers):
                # Select configuration probabilistically
                selected = self._select_probabilistic()

                # Explore near selected configuration
                neighbor = random.choice(self.configurations)
                new_params = self._mutate_configuration(selected.params, neighbor.params)
                new_score = objective(new_params)
                new_fitness = new_score if maximize else -new_score

                # Update if better
                if new_fitness > selected.fitness:
                    selected.params = new_params
                    selected.score = new_score
                    selected.fitness = new_fitness
                    selected.trials = 0

            # Scout bee phase - abandon poor configurations
            for i, config in enumerate(self.configurations):
                if config.trials >= self.limit:
                    # Replace with random configuration
                    new_params = self._sample_configuration()
                    new_score = objective(new_params)
                    new_fitness = new_score if maximize else -new_score

                    config.params = new_params
                    config.score = new_score
                    config.fitness = new_fitness
                    config.trials = 0

            # Update best
            current_best = max(self.configurations, key=lambda c: c.fitness)
            if current_best.fitness > self.best_config.fitness:
                self.best_config = current_best

            self.history.append((iteration, self.best_config.score))

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Best score = {self.best_config.score:.4f}")

        if verbose:
            print("\nOptimization complete!")
            print(f"Best score: {self.best_config.score:.4f}")
            print(f"Best parameters: {self.best_config.params}")

        return self.best_config.params, self.best_config.score

    def get_best_configuration(self) -> Optional[ModelConfiguration]:
        """Get the best configuration found."""
        return self.best_config

    def get_history(self) -> List[Tuple[int, float]]:
        """Get optimization history as (iteration, best_score) tuples."""
        return self.history
