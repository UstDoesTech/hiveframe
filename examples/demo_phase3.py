#!/usr/bin/env python3
"""
HiveFrame Phase 3 Demo - Enterprise Platform Features
=====================================================

Demonstrates the new Phase 3 features:
- Unity Hive Catalog: Unified governance and data discovery
- AutoML Swarm: Hyperparameter optimization with bee-inspired ABC algorithm
- Feature Hive: Centralized feature store
- Model Serving: Production inference with swarm load balancing
- HiveFrame Notebooks: Interactive execution environment

Run with: python examples/demo_phase3.py
"""

import sys
from datetime import datetime


def demo_unity_hive_catalog():
    """Demo Unity Hive Catalog for data governance."""
    print("=" * 70)
    print("Demo 1: Unity Hive Catalog - Data Governance")
    print("=" * 70)

    from hiveframe import (
        UnityHiveCatalog,
        AccessControl,
        LineageTracker,
        PIIDetector,
        PermissionType,
    )

    # Create catalog
    catalog = UnityHiveCatalog()
    print("\n✓ Created Unity Hive Catalog")

    # Register tables
    catalog.register_table(
        name="users",
        schema={"id": "int", "name": "string", "email": "string", "phone": "string", "age": "int"},
        location="/data/users",
        format="parquet",
        owner="data_team",
        tags={"production", "user_data"},
    )
    print("✓ Registered 'users' table")

    catalog.register_table(
        name="orders",
        schema={"order_id": "int", "user_id": "int", "amount": "float"},
        location="/data/orders",
        format="parquet",
        owner="data_team",
        tags={"production", "order_data"},
    )
    print("✓ Registered 'orders' table")

    # List tables
    tables = catalog.list_tables()
    print(f"\n📋 Catalog contains {len(tables)} tables: {', '.join(tables)}")

    # Search by tags
    prod_tables = catalog.search_by_tags({"production"})
    print(f"✓ Found {len(prod_tables)} production tables")

    # Access Control
    acl = AccessControl(catalog)
    acl.grant("analyst@company.com", "users", [PermissionType.SELECT])
    acl.grant("data_engineer@company.com", "users", [PermissionType.ALL])
    print("\n✓ Configured access control for users table")

    has_select = acl.check_permission("analyst@company.com", "users", PermissionType.SELECT)
    has_insert = acl.check_permission("analyst@company.com", "users", PermissionType.INSERT)
    print(f"  - Analyst can SELECT: {has_select}")
    print(f"  - Analyst can INSERT: {has_insert}")

    # Data Lineage
    tracker = LineageTracker(catalog)
    tracker.record_lineage(
        output_table="user_summary", input_tables=["users", "orders"], operation="join"
    )
    tracker.record_lineage(
        output_table="monthly_report", input_tables=["user_summary"], operation="aggregate"
    )
    print("\n✓ Recorded data lineage")

    upstream = tracker.get_upstream("monthly_report", recursive=True)
    print(f"  - Upstream dependencies of 'monthly_report': {sorted(upstream)}")

    path = tracker.get_lineage_path("users", "monthly_report")
    print(f"  - Lineage path: {' → '.join(path)}")

    # PII Detection
    detector = PIIDetector()
    user_schema = catalog.get_table("users").schema
    pii_columns = detector.detect_pii(user_schema)
    print(f"\n✓ Detected {len(pii_columns)} PII columns in 'users': {list(pii_columns.keys())}")


def demo_delta_sharing():
    """Demo Delta Sharing protocol."""
    print("\n" + "=" * 70)
    print("Demo 2: Delta Sharing - Secure Data Sharing")
    print("=" * 70)

    from hiveframe import DeltaSharing, ShareAccessLevel

    sharing = DeltaSharing()
    print("\n✓ Initialized Delta Sharing")

    # Create a share
    share = sharing.create_share(
        name="customer_analytics",
        owner="data_team@company.com",
        tables={"customers", "orders", "products"},
    )
    print(f"✓ Created share '{share.name}' with {len(share.tables)} tables")

    # Add recipient
    recipient = sharing.add_recipient(
        share_id=share.share_id,
        email="partner@external.com",
        access_level=ShareAccessLevel.READ,
        expires_in_days=90,
    )
    print(f"✓ Added recipient: {recipient.email}")
    print(f"  - Access token: {recipient.token[:20]}...")
    print(f"  - Expires: {recipient.expires_at.strftime('%Y-%m-%d')}")

    # List shared tables
    tables = sharing.list_shared_tables(recipient.token)
    print(f"✓ Recipient can access {len(tables)} tables: {', '.join(tables)}")

    # Get share metadata
    metadata = sharing.get_share_metadata(recipient.token)
    print(f"✓ Share metadata: {metadata['name']} ({metadata['num_tables']} tables)")


def demo_automl_swarm():
    """Demo AutoML with bee-inspired optimization."""
    print("\n" + "=" * 70)
    print("Demo 3: AutoML Swarm - Hyperparameter Optimization")
    print("=" * 70)

    from hiveframe import AutoMLSwarm, HyperparameterSpace

    # Define search space
    space = HyperparameterSpace(
        continuous={"learning_rate": (0.001, 0.1), "dropout": (0.0, 0.5)},
        discrete={"max_depth": [3, 5, 7, 9, 11]},
        categorical={"optimizer": ["adam", "sgd", "rmsprop"]},
    )
    print("\n✓ Defined hyperparameter search space")
    print(f"  - Continuous: learning_rate, dropout")
    print(f"  - Discrete: max_depth")
    print(f"  - Categorical: optimizer")

    # Simple objective function (higher is better)
    def mock_objective(params):
        """Mock ML model training - optimize for lr=0.01, depth=7"""
        lr_error = abs(params["learning_rate"] - 0.01)
        depth_error = abs(params["max_depth"] - 7) * 0.02
        dropout_penalty = params["dropout"] * 0.1
        optimizer_bonus = 0.05 if params["optimizer"] == "adam" else 0.0

        score = 1.0 - (lr_error + depth_error + dropout_penalty) + optimizer_bonus
        return max(0, min(1, score))

    # Create AutoML swarm with 15 worker bees
    automl = AutoMLSwarm(n_workers=15, space=space, seed=42)
    print("✓ Initialized AutoML Swarm with 15 worker bees")

    # Run optimization
    print("\n🐝 Running bee-inspired optimization (30 iterations)...")
    best_params, best_score = automl.optimize(
        objective=mock_objective, n_iterations=30, maximize=True, verbose=False
    )

    print(f"\n✓ Optimization complete!")
    print(f"  - Best score: {best_score:.4f}")
    print(f"  - Best parameters:")
    for key, value in best_params.items():
        print(f"    • {key}: {value}")

    # Show convergence
    history = automl.get_history()
    print(f"\n📈 Convergence: Started at {history[0][1]:.4f}, ended at {history[-1][1]:.4f}")


def demo_feature_hive():
    """Demo Feature Store."""
    print("\n" + "=" * 70)
    print("Demo 4: Feature Hive - Centralized Feature Store")
    print("=" * 70)

    from hiveframe import FeatureHive, FeatureType

    store = FeatureHive()
    print("\n✓ Created Feature Hive")

    # Register features with computation functions
    def compute_user_age_bucket(user_id):
        # Mock computation
        age = (int(user_id) * 7) % 80 + 18
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        else:
            return "55+"

    def compute_user_activity_score(user_id):
        # Mock computation
        return (int(user_id) * 13) % 100 / 100.0

    store.register_feature(
        name="user_age_bucket",
        feature_type=FeatureType.STRING,
        compute_fn=compute_user_age_bucket,
        description="User age bucket",
        owner="ml_team",
    )

    store.register_feature(
        name="user_activity_score",
        feature_type=FeatureType.FLOAT,
        compute_fn=compute_user_activity_score,
        description="User activity score (0-1)",
        owner="ml_team",
    )

    store.register_feature(
        name="user_premium",
        feature_type=FeatureType.BOOLEAN,
        description="Premium user flag",
        owner="ml_team",
    )

    print("✓ Registered 3 features")

    # Set some manual values
    store.set_feature_values("user_premium", {123: True, 456: False, 789: True})

    # Create feature group
    group = store.create_feature_group(
        name="user_features",
        features=["user_age_bucket", "user_activity_score", "user_premium"],
        description="Core user features for ML models",
    )
    print(f"✓ Created feature group '{group.name}' with {len(group.features)} features")

    # Get features for entities
    user_ids = [123, 456, 789]
    features = store.get_features(
        ["user_age_bucket", "user_activity_score", "user_premium"], entity_ids=user_ids
    )

    print(f"\n📊 Features for users {user_ids}:")
    for feature_name, values in features.items():
        print(f"  - {feature_name}: {values}")


def demo_model_server():
    """Demo Model Serving with swarm load balancing."""
    print("\n" + "=" * 70)
    print("Demo 5: Model Server - Production Inference")
    print("=" * 70)

    from hiveframe import ModelServer

    # Mock model
    class MockModel:
        def predict(self, data):
            return [x * 2 + 1 for x in data]

    model = MockModel()

    # Create server with 3 replicas
    server = ModelServer(model, n_replicas=3)
    print(f"\n✓ Created Model Server with {server.n_replicas} replicas")

    # Make predictions
    print("\n🔮 Making predictions with swarm load balancing...")
    for i in range(5):
        result = server.predict([1, 2, 3, 4, 5])
        print(f"  Prediction {i+1}: {result[:3]}... (load balanced)")

    # Check health
    health = server.get_health_status()
    print(f"\n✓ Server health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    for replica_status in health["replicas"]:
        print(
            f"  - {replica_status['replica_id']}: "
            f"{replica_status['total_requests']} requests, "
            f"fitness={replica_status['fitness']:.3f}"
        )


def demo_notebooks():
    """Demo HiveFrame Notebooks."""
    print("\n" + "=" * 70)
    print("Demo 6: HiveFrame Notebooks - Interactive Execution")
    print("=" * 70)

    from hiveframe import NotebookSession, KernelLanguage, CellStatus

    session = NotebookSession()
    print("\n✓ Created Notebook Session")

    # Execute Python cells
    print("\n📓 Executing notebook cells...")

    cells = [
        ("x = 10", "Set variable x"),
        ("y = 20", "Set variable y"),
        ("x + y", "Compute sum"),
        ("print(f'Result: {x + y}')", "Print result"),
    ]

    for code, description in cells:
        result = session.execute_cell(code, language=KernelLanguage.PYTHON)
        status_icon = "✅" if result.status == CellStatus.SUCCESS else "❌"
        print(f"  {status_icon} Cell {result.execution_count}: {description}")
        if result.outputs and result.outputs[0].output_type == "stream":
            output = result.outputs[0].data.get("text", "")
            if output:
                print(f"      Output: {output.strip()}")

    print(f"\n✓ Executed {len(session.execution_history)} cells successfully")


def main():
    """Run all Phase 3 demos."""
    print("\n" + "🐝" * 35)
    print("HiveFrame Phase 3: Enterprise Platform Features")
    print("🐝" * 35)

    try:
        demo_unity_hive_catalog()
        demo_delta_sharing()
        demo_automl_swarm()
        demo_feature_hive()
        demo_model_server()
        demo_notebooks()

        print("\n" + "=" * 70)
        print("✅ All Phase 3 demos completed successfully!")
        print("=" * 70)
        print("\nPhase 3 provides enterprise-grade features:")
        print("  • Unity Hive Catalog for data governance")
        print("  • Delta Sharing for secure data exchange")
        print("  • AutoML Swarm for hyperparameter optimization")
        print("  • Feature Hive for centralized feature management")
        print("  • Model Server for production inference")
        print("  • HiveFrame Notebooks for interactive development")
        print("\n🐝 The hive continues to grow stronger!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
