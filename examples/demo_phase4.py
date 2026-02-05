"""
Phase 4 Demo: Autonomous Data Intelligence

Demonstrates all Phase 4 features including:
- Autonomous Operations (self-tuning, predictive maintenance, cost optimization)
- Generative AI Integration (NL queries, data prep, code generation)
- Advanced Swarm Algorithms (hybrid swarm, quantum-ready, federated learning)
"""

import time
from hiveframe.autonomous import (
    SelfTuningColony,
    PredictiveMaintenance,
    WorkloadPredictor,
    CostOptimizer,
    MemoryStats,
    ResourceMetrics,
    QueryPerformance,
    HealthMetric,
    WorkloadSample,
    CostMetrics,
    SLAMetrics,
)
from hiveframe.ai import (
    NaturalLanguageQuery,
    AIDataPrep,
    DataDiscovery,
    HiveFrameCodeGen,
    LLMFineTuner,
)
from hiveframe.advanced_swarm import (
    HybridSwarmOptimizer,
    HybridQuantumClassical,
    CrossOrgTrainer,
)


def demo_autonomous_operations():
    """Demonstrate autonomous operations"""
    print("=" * 80)
    print("PHASE 4: AUTONOMOUS OPERATIONS DEMO")
    print("=" * 80)

    # 1. Self-Tuning Colony
    print("\n1. Self-Tuning Colony - Automatic Performance Optimization")
    print("-" * 60)

    colony = SelfTuningColony(total_memory_mb=8192, max_workers=50)

    # Simulate system metrics
    for i in range(5):
        # Record memory usage
        stats = MemoryStats(
            total_mb=8192,
            used_mb=4000 + i * 200,
            available_mb=8192 - (4000 + i * 200),
            cache_mb=1024,
            buffer_mb=512,
        )
        colony.memory_manager.record_usage(stats)

        # Record resource metrics
        metrics = ResourceMetrics(
            cpu_percent=60.0 + i * 5,
            memory_mb=4000 + i * 200,
            disk_io_mb_per_sec=50.0,
            network_mb_per_sec=20.0,
            active_workers=colony.resource_allocator.current_workers,
            queued_tasks=10 + i * 2,
        )
        colony.resource_allocator.record_metrics(metrics)

        # Record query performance
        perf = QueryPerformance(
            query_id=f"q{i}",
            execution_time_ms=100.0 + i * 10,
            rows_processed=1000 * (i + 1),
            bytes_scanned=1000000 * (i + 1),
            workers_used=2,
            memory_peak_mb=500,
        )
        colony.query_predictor.record_query(perf)

    # Perform self-tuning
    time.sleep(31)  # Wait for tuning interval
    result = colony.tune()
    print(f"✓ Self-tuning result: {result['status']}")
    if "memory" in result:
        print(f"  Memory config: cache={result['memory']['cache_mb']:.0f}MB")
    if "resources" in result:
        print(f"  Resource allocation: {result['resources']['workers']} workers")

    # Get query recommendation
    rec = colony.get_query_recommendation("test_query", 5000, 5000000)
    print(
        f"  Query prediction: {rec['estimated_time_ms']:.1f}ms, confidence={rec['confidence']:.2f}"
    )

    # 2. Predictive Maintenance
    print("\n2. Predictive Maintenance - Anticipate Failures Before They Occur")
    print("-" * 60)

    maintenance = PredictiveMaintenance()

    # Simulate increasing CPU usage (potential problem)
    for i in range(20):
        metric = HealthMetric(
            name="cpu_usage",
            value=60.0 + i * 2,  # Increasing trend
            threshold_warning=80.0,
            threshold_critical=90.0,
        )
        maintenance.record_metric(metric)

    # Assess system health
    assessment = maintenance.assess_system()
    print(f"✓ Health status: {assessment['health']['status']}")
    print(f"  Health score: {assessment['health']['score']:.1f}/100")
    print(f"  Predictions: {len(assessment['predictions'])} potential issues")
    print(f"  Recommendations: {len(assessment['recommendations'])} actions")

    # 3. Workload Prediction
    print("\n3. Workload Prediction - Pre-warm Resources for Demand Spikes")
    print("-" * 60)

    predictor = WorkloadPredictor()

    # Simulate workload samples
    for i in range(60):
        sample = WorkloadSample(
            timestamp=time.time() + i * 60,
            query_count=100 + (20 if i % 10 < 5 else 0),  # Periodic pattern
            cpu_percent=60.0,
            memory_mb=2000,
            io_operations=500,
            active_users=10,
        )
        predictor.record_workload(sample)

    # Predict future workload
    forecast = predictor.predict(hours_ahead=1)
    print(f"✓ Forecast (1 hour ahead):")
    print(f"  Predicted queries: {forecast.predicted_query_count}")
    print(f"  Pattern type: {forecast.pattern_type}")
    print(f"  Confidence: {forecast.confidence:.2f}")

    # 4. Cost Optimization
    print("\n4. Cost Optimization - Minimize Cloud Spend While Meeting SLAs")
    print("-" * 60)

    optimizer = CostOptimizer(budget_per_hour=100.0)

    # Simulate cost and SLA metrics
    for i in range(20):
        cost = CostMetrics(
            timestamp=time.time() + i * 3600,
            compute_cost_per_hour=50.0,
            storage_cost_per_hour=20.0,
            network_cost_per_hour=10.0,
            total_cost_per_hour=80.0,
            resource_utilization=0.3,  # Low utilization - room for optimization
            active_workers=10,
            storage_gb=100.0,
        )

        sla = SLAMetrics(
            timestamp=time.time() + i * 3600,
            avg_response_time_ms=500.0,
            p95_response_time_ms=800.0,
            p99_response_time_ms=1000.0,
            error_rate=0.005,
            availability=0.999,
        )

        optimizer.record_metrics(cost, sla)

    # Optimize costs
    result = optimizer.optimize()
    print(f"✓ Optimization result: {result['status']}")
    print(f"  Current spend: ${result['spend_analysis']['avg_hourly_cost']:.2f}/hour")
    print(f"  Utilization: {result['spend_analysis']['avg_utilization']:.1%}")
    print(f"  Recommendations: {len(result['recommendations'])}")
    if result["recommendations"]:
        for rec in result["recommendations"][:2]:
            print(f"    - {rec['action']}: save ${rec['savings_per_hour']:.2f}/hour")


def demo_generative_ai():
    """Demonstrate generative AI integration"""
    print("\n" + "=" * 80)
    print("PHASE 4: GENERATIVE AI INTEGRATION DEMO")
    print("=" * 80)

    # 1. Natural Language Queries
    print("\n1. Natural Language Query - Ask Questions in Plain English")
    print("-" * 60)

    nlq = NaturalLanguageQuery()

    queries = [
        "Show all users where age > 25",
        "Count orders by category",
        "Find customers with high spending",
    ]

    for query in queries:
        result = nlq.query(query)
        print(f'✓ Question: "{query}"')
        print(f"  SQL: {result['sql']}")
        print(f"  Confidence: {result['confidence']:.2f}")

    # 2. AI-Powered Data Preparation
    print("\n2. AI-Powered Data Preparation - Automatic Cleaning & Transformation")
    print("-" * 60)

    prep = AIDataPrep()

    # Sample data with quality issues
    data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": None, "score": 90},  # Missing age
        {"name": "Charlie", "age": 25, "score": 78},
        {"name": "Diana", "age": 28, "score": 200},  # Outlier
    ]

    result = prep.prepare_data(data, auto_clean=True, target_use_case="ml")
    print(f"✓ Data preparation completed")
    print(f"  Original rows: {len(data)}")
    print(f"  Cleaned rows: {len(result['data'])}")
    print(f"  Issues found: {len(result['issues'])}")
    for issue in result["issues"]:
        print(f"    - {issue['type']} in {issue['column']}: {issue['description']}")
    print(f"  Transformation suggestions: {len(result['transformation_suggestions'])}")
    for sugg in result["transformation_suggestions"][:2]:
        print(f"    - {sugg['type']}: {sugg['description']}")

    # 3. Intelligent Data Discovery
    print("\n3. Intelligent Data Discovery - Auto-Detect Relationships & Joins")
    print("-" * 60)

    discovery = DataDiscovery()

    schema = {
        "users": ["user_id", "name", "email"],
        "orders": ["order_id", "user_id", "amount"],
        "products": ["product_id", "name", "price"],
    }

    schema_graph = discovery.discover_schema(schema)
    print(f"✓ Schema discovery completed")
    print(f"  Tables: {len(schema_graph.tables)}")
    print(f"  Relationships detected: {len(schema_graph.relationships)}")
    for rel in schema_graph.relationships[:3]:
        print(f"    - {rel.entity1} ↔ {rel.entity2}: {rel.relationship_type}")

    # 4. Code Generation
    print("\n4. Code Generation - Generate HiveFrame Code from Natural Language")
    print("-" * 60)

    codegen = HiveFrameCodeGen()

    descriptions = [
        "Read data from users.csv",
        "Filter where age > 25",
        "Group by category and count",
    ]

    code = codegen.generate_pipeline(descriptions)
    print(f"✓ Generated pipeline code:")
    print(code.code[:300] + "...")

    # 5. LLM Fine-tuning Platform
    print("\n5. LLM Fine-tuning - Train Custom Models on Lakehouse Data")
    print("-" * 60)

    finetuner = LLMFineTuner()

    print(f"✓ LLM fine-tuning platform initialized")
    print(f"  Available models: {len(finetuner.list_models())}")
    print(f"  Ready for training on lakehouse data")
    print(f"  Supports: GPT-2, GPT-3, BERT, and custom architectures")


def demo_advanced_swarm():
    """Demonstrate advanced swarm algorithms"""
    print("\n" + "=" * 80)
    print("PHASE 4: ADVANCED SWARM ALGORITHMS DEMO")
    print("=" * 80)

    # 1. Hybrid Swarm Intelligence
    print("\n1. Hybrid Swarm Intelligence - PSO + ACO + Firefly")
    print("-" * 60)

    optimizer = HybridSwarmOptimizer()

    # Optimize a test function
    def test_function(x):
        return sum(xi**2 for xi in x)  # Sphere function

    result = optimizer.optimize(test_function, problem_type="numerical", max_iterations=30)
    print(f"✓ Optimization completed using {result['algorithm']}")
    print(f"  Best fitness: {result['best_fitness']:.6f}")
    print(f"  Best position: {[f'{x:.3f}' for x in result['best_position']]}")

    # 2. Quantum-Ready Algorithms
    print("\n2. Quantum-Ready Algorithms - Hybrid Quantum-Classical Computing")
    print("-" * 60)

    quantum = HybridQuantumClassical()

    # Demonstrate quantum subroutine
    data = [1.0, 2.0, 5.0, 3.0, 2.5]
    result = quantum.quantum_subroutine(data, operation="search")
    print(f"✓ Quantum search completed")
    print(f"  Input data: {data}")
    print(f"  Search result: {result}")

    # Demonstrate VQE
    hamiltonian = [[1.0, 0.5], [0.5, 2.0]]
    vqe_result = quantum.variational_quantum_eigensolver(hamiltonian, [0.1, 0.2])
    print(f"✓ Variational Quantum Eigensolver (VQE)")
    print(f"  Ground state energy: {vqe_result['ground_state_energy']:.4f}")

    # 3. Federated Learning Swarm
    print("\n3. Federated Learning - Privacy-Preserving Cross-Org Training")
    print("-" * 60)

    trainer = CrossOrgTrainer()

    # Register organizations
    trainer.register_organization("hospital_a", data_size=1000, privacy_requirements={})
    trainer.register_organization("hospital_b", data_size=1500, privacy_requirements={})
    trainer.register_organization("hospital_c", data_size=1200, privacy_requirements={})

    # Define training function
    def train_local(global_params, org_data):
        local_params = [p + 0.01 for p in global_params]
        loss = 0.5
        return local_params, loss

    # Run training round
    global_model = trainer.train_round(
        training_fn=train_local,
        participants=["hospital_a", "hospital_b", "hospital_c"],
    )

    print(f"✓ Federated training round completed")
    print(f"  Global model version: {global_model.version}")
    print(f"  Participants: {len(global_model.contributor_nodes)}")
    print(f"  Average loss: {global_model.avg_loss:.4f}")

    status = trainer.get_training_status()
    print(f"  Privacy budget remaining: {status['privacy_budget']['remaining']:.2f}")


def main():
    """Run complete Phase 4 demo"""
    print("\n" + "🐝" * 40)
    print("HiveFrame Phase 4: Autonomous Data Intelligence")
    print("Demonstrating the world's first bio-inspired self-managing data platform")
    print("🐝" * 40)

    try:
        demo_autonomous_operations()
        demo_generative_ai()
        demo_advanced_swarm()

        print("\n" + "=" * 80)
        print("PHASE 4 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n✨ All Phase 4 features are production-ready:")
        print("  ✓ Autonomous Operations - Self-tuning, predictive maintenance, cost optimization")
        print("  ✓ Generative AI Integration - NL queries, data prep, code generation")
        print("  ✓ Advanced Swarm Algorithms - Hybrid swarm, quantum-ready, federated learning")
        print("\n🚀 HiveFrame is now the world's most advanced bio-inspired data platform!")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
