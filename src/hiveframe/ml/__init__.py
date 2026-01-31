"""
HiveMind ML Platform (Phase 3)
==============================

Machine Learning platform with bee-inspired optimization and distributed training.

This module provides:
- AutoML Swarm: Hyperparameter optimization using ABC algorithm
- Feature Hive: Centralized feature store with automatic feature engineering
- Model Serving: Real-time inference with swarm load balancing
- Distributed Training: Multi-node training orchestrated by the colony
- MLflow Integration: Experiment tracking and model registry

Key Components:
    - AutoMLSwarm: Hyperparameter tuning using artificial bee colony
    - FeatureHive: Feature store with versioning and lineage
    - ModelServer: Production model serving with load balancing
    - DistributedTrainer: Multi-node training coordinator
    - MLflowIntegration: Track experiments and models

Example:
    from hiveframe.ml import AutoMLSwarm, FeatureHive, ModelServer

    # AutoML with bee-inspired optimization
    automl = AutoMLSwarm(n_workers=8)
    best_model = automl.fit(X_train, y_train, task='classification')

    # Feature store
    feature_store = FeatureHive()
    feature_store.register_feature("user_activity_7d", compute_fn)
    features = feature_store.get_features(["user_activity_7d"], user_ids)

    # Model serving
    server = ModelServer(best_model)
    predictions = server.predict(new_data)
"""

__all__ = [
    "AutoMLSwarm",
    "FeatureHive",
    "ModelServer",
    "DistributedTrainer",
    "MLflowIntegration",
    "HyperparameterSpace",
    "TaskType",
    "FeatureType",
    "ModelStage",
]

from .automl import AutoMLSwarm, HyperparameterSpace, TaskType
from .feature_store import FeatureHive, FeatureType
from .model_serving import ModelServer, DistributedTrainer
from .mlflow_integration import MLflowIntegration, ModelStage
