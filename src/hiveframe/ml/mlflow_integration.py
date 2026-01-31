"""
MLflow Integration - Experiment Tracking and Model Registry
===========================================================

Integration with MLflow for tracking experiments and managing models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class ModelStage(Enum):
    """Model lifecycle stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class Experiment:
    """An MLflow experiment."""

    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str = "active"
    creation_time: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Run:
    """An MLflow run within an experiment."""

    run_id: str
    experiment_id: str
    status: str = "RUNNING"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class RegisteredModel:
    """A registered model in the model registry."""

    name: str
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    latest_versions: Dict[ModelStage, str] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """A version of a registered model."""

    name: str
    version: str
    creation_time: datetime = field(default_factory=datetime.now)
    current_stage: ModelStage = ModelStage.NONE
    description: Optional[str] = None
    run_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MLflowIntegration:
    """
    MLflow integration for experiment tracking and model registry.

    Provides a simplified interface to MLflow-style experiment tracking
    and model management, with bee-inspired organization.

    Example:
        mlflow = MLflowIntegration()

        # Create experiment
        experiment = mlflow.create_experiment("my_experiment")

        # Start run
        run = mlflow.start_run(experiment.experiment_id)

        # Log parameters and metrics
        mlflow.log_param(run.run_id, "learning_rate", 0.01)
        mlflow.log_metric(run.run_id, "accuracy", 0.95)

        # End run
        mlflow.end_run(run.run_id)

        # Register model
        model = mlflow.register_model(
            "my_model",
            run_id=run.run_id,
            description="Production model"
        )

        # Promote to production
        mlflow.transition_model_stage(
            "my_model",
            "1",
            ModelStage.PRODUCTION
        )
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow integration.

        Args:
            tracking_uri: Optional tracking server URI
        """
        self.tracking_uri = tracking_uri or "local"
        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, Run] = {}
        self._models: Dict[str, RegisteredModel] = {}
        self._model_versions: Dict[str, List[ModelVersion]] = {}
        self._next_experiment_id = 1
        self._next_run_id = 1

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            artifact_location: Location to store artifacts
            tags: Optional experiment tags

        Returns:
            Created Experiment
        """
        experiment_id = str(self._next_experiment_id)
        self._next_experiment_id += 1

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location or f"./mlruns/{experiment_id}",
            tags=tags or {},
        )

        self._experiments[experiment_id] = experiment
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Run:
        """
        Start a new run.

        Args:
            experiment_id: Experiment ID
            run_name: Optional run name
            tags: Optional run tags

        Returns:
            Created Run
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        run_id = str(self._next_run_id)
        self._next_run_id += 1

        run_tags = tags or {}
        if run_name:
            run_tags["mlflow.runName"] = run_name

        run = Run(run_id=run_id, experiment_id=experiment_id, tags=run_tags)

        self._runs[run_id] = run
        return run

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """End a run."""
        if run_id in self._runs:
            run = self._runs[run_id]
            run.status = status
            run.end_time = datetime.now()

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter."""
        if run_id in self._runs:
            self._runs[run_id].params[key] = value

    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        if run_id in self._runs:
            self._runs[run_id].metrics[key] = value

    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        """Log an artifact."""
        if run_id in self._runs:
            self._runs[run_id].artifacts.append(artifact_path)

    def register_model(
        self,
        name: str,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> RegisteredModel:
        """
        Register a new model.

        Args:
            name: Model name
            run_id: Optional run ID
            description: Model description
            tags: Model tags

        Returns:
            Registered model
        """
        if name not in self._models:
            model = RegisteredModel(name=name, description=description, tags=tags or {})
            self._models[name] = model
            self._model_versions[name] = []
        else:
            model = self._models[name]

        # Create new version
        version_number = str(len(self._model_versions[name]) + 1)
        version = ModelVersion(
            name=name,
            version=version_number,
            run_id=run_id,
            description=description,
            tags=tags or {},
        )
        self._model_versions[name].append(version)

        return model

    def get_registered_model(self, name: str) -> Optional[RegisteredModel]:
        """Get registered model by name."""
        return self._models.get(name)

    def get_model_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        if name not in self._model_versions:
            return None

        for mv in self._model_versions[name]:
            if mv.version == version:
                return mv

        return None

    def transition_model_stage(self, name: str, version: str, stage: ModelStage) -> bool:
        """
        Transition model version to a new stage.

        Args:
            name: Model name
            version: Model version
            stage: Target stage

        Returns:
            True if successful
        """
        model_version = self.get_model_version(name, version)
        if not model_version:
            return False

        model_version.current_stage = stage

        # Update latest version for this stage
        if name in self._models:
            self._models[name].latest_versions[stage] = version
            self._models[name].last_updated = datetime.now()

        return True

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        return list(self._experiments.values())

    def list_runs(self, experiment_id: str) -> List[Run]:
        """List all runs for an experiment."""
        return [run for run in self._runs.values() if run.experiment_id == experiment_id]
