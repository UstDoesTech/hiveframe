"""
Tests for Phase 3 Machine Learning Platform (HiveMind ML)
"""


from hiveframe.ml import (
    AutoMLSwarm,
    DistributedTrainer,
    FeatureHive,
    MLflowIntegration,
    ModelServer,
)
from hiveframe.ml.automl import HyperparameterSpace
from hiveframe.ml.feature_store import FeatureType
from hiveframe.ml.mlflow_integration import ModelStage


class TestAutoMLSwarm:
    """Test AutoML with bee-inspired optimization."""

    def test_initialization(self):
        """Test AutoML swarm initialization."""
        space = HyperparameterSpace(
            continuous={"learning_rate": (0.001, 0.1)},
            discrete={"max_depth": [3, 5, 7]},
            categorical={"algorithm": ["rf", "gbm"]},
        )

        automl = AutoMLSwarm(n_workers=5, space=space)
        assert automl.n_workers == 5
        assert automl.space == space

    def test_optimize(self):
        """Test hyperparameter optimization."""
        space = HyperparameterSpace(
            continuous={"learning_rate": (0.001, 0.1)}, discrete={"max_depth": [3, 5, 7]}
        )

        # Simple objective function (maximize negative squared error)
        def objective(params):
            # Optimal: learning_rate=0.01, max_depth=5
            lr_error = abs(params["learning_rate"] - 0.01)
            depth_error = abs(params["max_depth"] - 5)
            return 1.0 - (lr_error + depth_error * 0.01)

        automl = AutoMLSwarm(n_workers=10, space=space, seed=42)
        best_params, best_score = automl.optimize(
            objective=objective, n_iterations=20, maximize=True, verbose=False
        )

        # Should find good parameters
        assert best_score > 0.8
        assert "learning_rate" in best_params
        assert "max_depth" in best_params

    def test_get_best_configuration(self):
        """Test retrieving best configuration."""
        space = HyperparameterSpace(continuous={"x": (0, 1)})

        def objective(params):
            return params["x"] ** 2

        automl = AutoMLSwarm(n_workers=5, space=space)
        automl.optimize(objective, n_iterations=10, maximize=True, verbose=False)

        best_config = automl.get_best_configuration()
        assert best_config is not None
        assert "x" in best_config.params

    def test_history_tracking(self):
        """Test optimization history tracking."""
        space = HyperparameterSpace(continuous={"x": (0, 1)})

        def objective(params):
            return params["x"]

        automl = AutoMLSwarm(n_workers=5, space=space)
        n_iterations = 15
        automl.optimize(objective, n_iterations=n_iterations, maximize=True, verbose=False)

        history = automl.get_history()
        assert len(history) == n_iterations + 1  # Initial + iterations

        # Best score should improve or stay same
        for i in range(1, len(history)):
            assert history[i][1] >= history[i - 1][1]


class TestFeatureHive:
    """Test Feature Store functionality."""

    def test_register_feature(self):
        """Test registering features."""
        store = FeatureHive()

        metadata = store.register_feature(
            name="user_age",
            feature_type=FeatureType.INT,
            description="User age in years",
            owner="ml_team",
        )

        assert metadata.name == "user_age"
        assert metadata.feature_type == FeatureType.INT
        assert metadata.version == 1

    def test_feature_versioning(self):
        """Test feature versioning."""
        store = FeatureHive()

        # Register v1
        v1 = store.register_feature("feature_a", FeatureType.FLOAT)
        assert v1.version == 1

        # Register v2 (update)
        v2 = store.register_feature("feature_a", FeatureType.FLOAT)
        assert v2.version == 2

    def test_get_features_with_compute(self):
        """Test getting features with computation."""
        store = FeatureHive()

        # Register feature with compute function
        def compute_double(user_id):
            return int(user_id) * 2

        store.register_feature(
            name="user_double", feature_type=FeatureType.INT, compute_fn=compute_double
        )

        # Get features
        results = store.get_features(["user_double"], entity_ids=[1, 2, 3])
        assert results["user_double"] == [2, 4, 6]

    def test_feature_caching(self):
        """Test feature value caching."""
        store = FeatureHive()

        call_count = [0]

        def compute_slow(user_id):
            call_count[0] += 1
            return int(user_id) * 10

        store.register_feature("slow_feature", FeatureType.INT, compute_fn=compute_slow)

        # First call - should compute
        results1 = store.get_features(["slow_feature"], entity_ids=[1, 2])
        assert call_count[0] == 2

        # Second call with cache - should not compute
        results2 = store.get_features(["slow_feature"], entity_ids=[1, 2], use_cache=True)
        assert call_count[0] == 2  # No additional calls
        assert results1 == results2

    def test_set_feature_values(self):
        """Test manually setting feature values."""
        store = FeatureHive()
        store.register_feature("manual_feature", FeatureType.INT)

        # Set values
        store.set_feature_values("manual_feature", {1: 100, 2: 200})

        # Get values
        results = store.get_features(["manual_feature"], entity_ids=[1, 2])
        assert results["manual_feature"] == [100, 200]

    def test_feature_groups(self):
        """Test feature groups."""
        store = FeatureHive()
        store.register_feature("feature_a", FeatureType.INT)
        store.register_feature("feature_b", FeatureType.FLOAT)
        store.register_feature("feature_c", FeatureType.STRING)

        # Create feature group
        group = store.create_feature_group(
            name="user_features", features=["feature_a", "feature_b"], description="User features"
        )

        assert group.name == "user_features"
        assert len(group.features) == 2

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        store = FeatureHive()

        def compute(user_id):
            return int(user_id) * 10

        store.register_feature("cached_feature", FeatureType.INT, compute_fn=compute)

        # Compute and cache
        store.get_features(["cached_feature"], entity_ids=[1, 2])

        # Invalidate specific entity
        store.invalidate_cache("cached_feature", entity_id=1)

        # Check cache state (indirect - would need to check internal state)
        # At least verify it doesn't crash
        store.get_features(["cached_feature"], entity_ids=[1, 2])


class TestModelServer:
    """Test model serving with swarm load balancing."""

    def test_initialization(self):
        """Test model server initialization."""

        # Dummy model
        class DummyModel:
            def predict(self, data):
                return [1, 2, 3]

        model = DummyModel()
        server = ModelServer(model, n_replicas=3)

        assert server.n_replicas == 3
        assert len(server.replicas) == 3

    def test_predict(self):
        """Test model prediction."""

        class DummyModel:
            def predict(self, data):
                return [x * 2 for x in data]

        model = DummyModel()
        server = ModelServer(model, n_replicas=2)

        result = server.predict([1, 2, 3])
        assert result == [2, 4, 6]

    def test_health_status(self):
        """Test health status reporting."""

        class DummyModel:
            def predict(self, data):
                return data

        model = DummyModel()
        server = ModelServer(model, n_replicas=2)

        # Make a prediction
        server.predict([1, 2, 3])

        # Check health
        health = server.get_health_status()
        assert "healthy" in health
        assert "replicas" in health
        assert len(health["replicas"]) == 2

    def test_add_remove_replica(self):
        """Test adding and removing replicas."""

        class DummyModel:
            def predict(self, data):
                return data

        model = DummyModel()
        server = ModelServer(model, n_replicas=2)

        # Add replica
        replica_id = server.add_replica(model)
        assert server.n_replicas == 3
        assert replica_id is not None

        # Remove replica
        result = server.remove_replica(replica_id)
        assert result is True
        assert server.n_replicas == 2


class TestDistributedTrainer:
    """Test distributed training."""

    def test_initialization(self):
        """Test distributed trainer initialization."""
        trainer = DistributedTrainer(n_workers=4)
        assert trainer.n_workers == 4
        assert len(trainer.workers) == 4

    def test_train(self):
        """Test distributed training (simplified)."""
        trainer = DistributedTrainer(n_workers=2)

        def create_model():
            return {"weights": [1.0, 2.0]}

        model = trainer.train(
            training_data=[1, 2, 3], model_fn=create_model, epochs=5, batch_size=32
        )

        assert model is not None
        assert "weights" in model

    def test_training_status(self):
        """Test getting training status."""
        trainer = DistributedTrainer(n_workers=3)

        status = trainer.get_training_status()
        assert status["n_workers"] == 3
        assert "workers" in status


class TestMLflowIntegration:
    """Test MLflow integration."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        mlflow = MLflowIntegration()

        experiment = mlflow.create_experiment("test_experiment")
        assert experiment.name == "test_experiment"
        assert experiment.experiment_id is not None

    def test_start_end_run(self):
        """Test starting and ending runs."""
        mlflow = MLflowIntegration()
        experiment = mlflow.create_experiment("test")

        run = mlflow.start_run(experiment.experiment_id, run_name="run_1")
        assert run.run_id is not None
        assert run.status == "RUNNING"

        mlflow.end_run(run.run_id)
        updated_run = mlflow._runs[run.run_id]
        assert updated_run.status == "FINISHED"

    def test_log_params_metrics(self):
        """Test logging parameters and metrics."""
        mlflow = MLflowIntegration()
        experiment = mlflow.create_experiment("test")
        run = mlflow.start_run(experiment.experiment_id)

        mlflow.log_param(run.run_id, "learning_rate", 0.01)
        mlflow.log_metric(run.run_id, "accuracy", 0.95)

        updated_run = mlflow._runs[run.run_id]
        assert updated_run.params["learning_rate"] == 0.01
        assert updated_run.metrics["accuracy"] == 0.95

    def test_register_model(self):
        """Test registering a model."""
        mlflow = MLflowIntegration()

        model = mlflow.register_model(name="test_model", description="Test model")

        assert model.name == "test_model"
        assert model.description == "Test model"

    def test_model_versioning(self):
        """Test model version management."""
        mlflow = MLflowIntegration()

        # Register v1
        mlflow.register_model("model_a")
        # Register v2
        mlflow.register_model("model_a")

        # Get versions
        v1 = mlflow.get_model_version("model_a", "1")
        v2 = mlflow.get_model_version("model_a", "2")

        assert v1 is not None
        assert v2 is not None
        assert v1.version == "1"
        assert v2.version == "2"

    def test_transition_model_stage(self):
        """Test transitioning model stages."""
        mlflow = MLflowIntegration()
        mlflow.register_model("model_a")

        # Transition to production
        result = mlflow.transition_model_stage("model_a", "1", ModelStage.PRODUCTION)
        assert result is True

        # Verify stage
        version = mlflow.get_model_version("model_a", "1")
        assert version.current_stage == ModelStage.PRODUCTION

    def test_list_experiments(self):
        """Test listing experiments."""
        mlflow = MLflowIntegration()
        mlflow.create_experiment("exp1")
        mlflow.create_experiment("exp2")

        experiments = mlflow.list_experiments()
        assert len(experiments) == 2

    def test_list_runs(self):
        """Test listing runs for an experiment."""
        mlflow = MLflowIntegration()
        experiment = mlflow.create_experiment("test")

        mlflow.start_run(experiment.experiment_id, run_name="run1")
        mlflow.start_run(experiment.experiment_id, run_name="run2")

        runs = mlflow.list_runs(experiment.experiment_id)
        assert len(runs) == 2
