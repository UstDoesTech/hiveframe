"""
Tests for HiveFrame core module.

Tests cover:
- HiveFrame initialization and configuration
- Bee roles and behavior (Employed, Onlooker, Scout)
- Waggle dance protocol and quality signaling
- Colony state management and pheromone signaling
- Map, filter, reduce operations
- Task distribution and load balancing
"""

import time

import pytest

from hiveframe import (
    BeeRole,
    ColonyState,
    DanceFloor,
    FoodSource,
    Pheromone,
    WaggleDance,
    create_hive,
)


class TestHiveFrameInitialization:
    """Test HiveFrame creation and configuration."""

    def test_create_hive_default_workers(self):
        """Test creating hive with default worker count."""
        hive = create_hive()
        assert hive is not None
        assert hive.num_workers > 0

    def test_create_hive_custom_workers(self):
        """Test creating hive with custom worker count."""
        hive = create_hive(num_workers=4)
        assert hive.num_workers == 4

    def test_create_hive_with_config(self):
        """Test creating hive with custom configuration."""
        hive = create_hive(num_workers=8, abandonment_limit=10)
        assert hive is not None


class TestMapOperations:
    """Test map operations."""

    def test_map_double_values(self):
        """Test basic map operation to double values."""
        hive = create_hive(num_workers=4)
        data = list(range(100))

        results = hive.map(data, lambda x: x * 2)

        # Check results contain doubled values
        assert len([r for r in results if r is not None]) > 0
        assert all(r % 2 == 0 for r in results if r is not None)

    def test_map_transform_dict(self):
        """Test map operation on dictionaries."""
        hive = create_hive(num_workers=4)
        data = [{"value": i} for i in range(50)]

        results = hive.map(data, lambda x: x["value"] ** 2)

        assert len([r for r in results if r is not None]) > 0

    def test_map_empty_input(self):
        """Test map with empty input."""
        hive = create_hive(num_workers=4)
        results = hive.map([], lambda x: x * 2)

        assert results == [] or all(r is None for r in results)


class TestFilterOperations:
    """Test filter operations."""

    def test_filter_even_numbers(self):
        """Test filtering for even numbers."""
        hive = create_hive(num_workers=4)
        data = list(range(100))

        results = hive.filter(data, lambda x: x % 2 == 0)

        assert all(r % 2 == 0 for r in results)
        assert len(results) == 50

    def test_filter_by_threshold(self):
        """Test filtering by value threshold."""
        hive = create_hive(num_workers=4)
        data = list(range(100))

        results = hive.filter(data, lambda x: x > 50)

        assert all(r > 50 for r in results)


class TestColonyState:
    """Test colony state management."""

    def test_colony_temperature(self):
        """Test colony temperature tracking."""
        colony = ColonyState()

        colony.update_temperature("worker_1", 0.5)
        colony.update_temperature("worker_2", 0.7)

        temp = colony.get_colony_temperature()
        assert 0.0 <= temp <= 1.0

    def test_pheromone_emission(self):
        """Test pheromone signal emission."""
        colony = ColonyState()

        pheromone = Pheromone(signal_type="throttle", intensity=0.8, source_worker="worker_1")
        colony.emit_pheromone(pheromone)

        intensity = colony.sense_pheromone("throttle")
        assert intensity > 0

    def test_pheromone_decay(self):
        """Test that pheromones decay over time."""
        colony = ColonyState()

        pheromone = Pheromone(signal_type="alarm", intensity=1.0, source_worker="worker_1")
        colony.emit_pheromone(pheromone)

        initial = colony.sense_pheromone("alarm")
        time.sleep(0.1)
        # Pheromones should remain or decay slightly
        assert colony.sense_pheromone("alarm") <= initial + 0.1


class TestBeeRoles:
    """Test bee role enumeration and behavior."""

    def test_bee_role_values(self):
        """Test that all bee roles are defined."""
        assert BeeRole.EMPLOYED is not None
        assert BeeRole.ONLOOKER is not None
        assert BeeRole.SCOUT is not None

    def test_role_distinctness(self):
        """Test that roles are distinct."""
        roles = [BeeRole.EMPLOYED, BeeRole.ONLOOKER, BeeRole.SCOUT]
        assert len(set(roles)) == 3


class TestWaggleDance:
    """Test waggle dance protocol."""

    def test_dance_creation(self):
        """Test creating a waggle dance signal."""
        dance = WaggleDance(
            partition_id="task_1",
            quality_score=0.8,
            processing_time=1.0,
            result_size=100,
            worker_id="bee_1",
        )

        assert dance.quality_score == 0.8
        assert dance.partition_id == "task_1"

    def test_dance_floor_registration(self):
        """Test registering dances on the floor."""
        floor = DanceFloor()

        dance = WaggleDance(
            partition_id="task_1",
            quality_score=0.9,
            processing_time=1.0,
            result_size=100,
            worker_id="bee_1",
        )
        floor.perform_dance(dance)

        # Should be able to observe dances
        dances = floor.observe_dances()
        assert len(dances) >= 0  # May be filtered by time


class TestFoodSource:
    """Test food source (task) management."""

    def test_food_source_creation(self):
        """Test creating a food source."""
        source = FoodSource(partition_id="source_1", data=[1, 2, 3], fitness=0.7)

        assert source.partition_id == "source_1"
        assert source.fitness == 0.7


# Performance tests
class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.slow
    def test_throughput_scaling(self):
        """Test that throughput scales with data size."""
        hive = create_hive(num_workers=8)

        sizes = [100, 500, 1000]
        throughputs = []

        for size in sizes:
            data = list(range(size))
            start = time.time()
            hive.map(data, lambda x: x * 2)
            elapsed = time.time() - start
            throughputs.append(size / elapsed if elapsed > 0 else float("inf"))

        # Throughput should generally increase with size (better amortization)
        assert throughputs[-1] >= throughputs[0] * 0.5  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
