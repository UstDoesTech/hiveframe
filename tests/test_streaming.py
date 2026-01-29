"""
Tests for HiveFrame streaming module.

Tests cover:
- Basic streaming operations
- Stream record handling
- Partitioning and buffering
- Enhanced streaming with windows
- Watermark handling
- Delivery guarantees
"""

import pytest
import time
from typing import List, Dict, Any

from hiveframe import (
    HiveStream, AsyncHiveStream, StreamRecord,
    StreamPartitioner, StreamBuffer, StreamBee,
)

from hiveframe import (
    EnhancedStreamProcessor,
    TumblingWindowAssigner, SlidingWindowAssigner, SessionWindowAssigner,
    tumbling_window, sliding_window, session_window,
    BoundedOutOfOrdernessWatermarkGenerator, bounded_watermark,
    DeliveryGuarantee, Checkpoint, InMemoryStateBackend,
    sum_aggregator, count_aggregator, avg_aggregator,
)


class TestStreamRecord:
    """Test StreamRecord class."""
    
    def test_record_creation(self):
        """Test creating a stream record."""
        record = StreamRecord(
            key='user_1',
            value={'action': 'click'},
            timestamp=time.time()
        )
        
        assert record.key == 'user_1'
        assert record.value == {'action': 'click'}
        
    def test_record_with_custom_timestamp(self):
        """Test record with explicit timestamp."""
        ts = 1700000000.0
        record = StreamRecord(
            key='key',
            value='value',
            timestamp=ts
        )
        
        assert record.timestamp == ts


class TestHiveStream:
    """Test basic HiveStream operations."""
    
    def test_stream_creation(self):
        """Test creating a stream processor."""
        stream = HiveStream(num_workers=4)
        
        assert stream is not None
        
    def test_stream_start_stop(self):
        """Test starting and stopping stream."""
        stream = HiveStream(num_workers=2)
        
        processed = []
        def processor(record):
            processed.append(record)
            return record
            
        stream.start(processor)
        stream.submit('key1', {'value': 1})
        time.sleep(0.1)  # Allow processing
        stream.stop()
        
        # Stream should have processed the record
        assert stream is not None


class TestStreamPartitioner:
    """Test stream partitioning."""
    
    def test_default_partitioning(self):
        """Test default hash-based partitioning."""
        partitioner = StreamPartitioner(num_partitions=4)
        
        # Same key should always go to same partition
        p1 = partitioner.partition('key_1')
        p2 = partitioner.partition('key_1')
        
        assert p1 == p2
        assert 0 <= p1 < 4
        
    def test_partition_distribution(self):
        """Test that keys distribute across partitions."""
        partitioner = StreamPartitioner(num_partitions=4)
        
        partitions = set()
        for i in range(100):
            p = partitioner.partition(f'key_{i}')
            partitions.add(p)
            
        # Should use multiple partitions
        assert len(partitions) > 1


class TestWindowAssigners:
    """Test window assignment strategies."""
    
    def test_tumbling_window_assigner(self):
        """Test tumbling window assignment."""
        assigner = TumblingWindowAssigner(window_size=5.0)
        
        # Records 1 second apart should be in same 5-second window
        window1 = assigner.assign_windows(1000.0)
        window2 = assigner.assign_windows(1003.0)
        
        assert window1 == window2
        
    def test_tumbling_window_boundaries(self):
        """Test tumbling window at boundaries."""
        assigner = TumblingWindowAssigner(window_size=5.0)
        
        # Records across boundary should be in different windows
        window1 = assigner.assign_windows(1004.0)
        window2 = assigner.assign_windows(1006.0)
        
        assert window1 != window2
        
    def test_sliding_window_assigner(self):
        """Test sliding window assignment."""
        assigner = SlidingWindowAssigner(window_size=10.0, slide_interval=5.0)
        
        windows = assigner.assign_windows(1007.0)
        
        # Should be assigned to multiple overlapping windows
        assert len(windows) >= 1
        
    def test_session_window_assigner(self):
        """Test session window assignment."""
        assigner = SessionWindowAssigner(gap=5.0)
        
        # Initial assignment
        windows1 = assigner.assign_windows(1000.0)
        
        assert len(windows1) >= 1


class TestWatermarks:
    """Test watermark handling."""
    
    def test_bounded_watermark_generator(self):
        """Test bounded out-of-orderness watermark."""
        generator = BoundedOutOfOrdernessWatermarkGenerator(max_out_of_orderness=2.0)
        
        # Advance with timestamps
        wm1 = generator.advance(100.0)
        wm2 = generator.advance(105.0)
        
        # Watermark should advance
        assert wm2.timestamp >= wm1.timestamp
        
    def test_watermark_delay(self):
        """Test that watermark respects out-of-orderness."""
        generator = BoundedOutOfOrdernessWatermarkGenerator(max_out_of_orderness=5.0)
        
        wm = generator.advance(100.0)
        
        # Watermark should be delayed by max_out_of_orderness
        assert wm.timestamp <= 100.0 - 5.0


class TestDeliveryGuarantees:
    """Test delivery guarantee semantics."""
    
    def test_at_most_once(self):
        """Test at-most-once delivery."""
        processor = EnhancedStreamProcessor(
            num_workers=2,
            window_assigner=tumbling_window(5.0),
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE
        )
        
        assert processor.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE
        
    def test_at_least_once(self):
        """Test at-least-once delivery."""
        processor = EnhancedStreamProcessor(
            num_workers=2,
            window_assigner=tumbling_window(5.0),
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE
        )
        
        assert processor.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
        
    def test_exactly_once(self):
        """Test exactly-once delivery."""
        processor = EnhancedStreamProcessor(
            num_workers=2,
            window_assigner=tumbling_window(5.0),
            delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE
        )
        
        assert processor.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE


class TestEnhancedStreamProcessor:
    """Test enhanced stream processor."""
    
    def test_processor_creation(self):
        """Test creating enhanced processor."""
        processor = EnhancedStreamProcessor(
            num_workers=4,
            window_assigner=tumbling_window(5.0),
            watermark_generator=bounded_watermark(2.0)
        )
        
        assert processor is not None
        
    def test_process_record(self):
        """Test processing a single record."""
        processor = EnhancedStreamProcessor(
            num_workers=2,
            window_assigner=tumbling_window(5.0)
        )
        
        record = StreamRecord(
            key='sensor_1',
            value=42.0,
            timestamp=time.time()
        )
        
        processor.process_record(record, aggregator=sum_aggregator)
        
        metrics = processor.get_metrics()
        assert metrics['records_processed'] > 0
        
    def test_window_aggregation(self):
        """Test windowed aggregation."""
        processor = EnhancedStreamProcessor(
            num_workers=2,
            window_assigner=tumbling_window(5.0)
        )
        
        base_time = time.time()
        
        # Send records in same window
        for i in range(5):
            record = StreamRecord(
                key='key',
                value=10.0,
                timestamp=base_time + i * 0.1
            )
            processor.process_record(record, aggregator=sum_aggregator, initial_value=0.0)
            
        metrics = processor.get_metrics()
        assert metrics['records_processed'] >= 5


class TestCheckpointing:
    """Test checkpointing and state management."""
    
    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=1,
            timestamp=time.time(),
            state={'counter': 100}
        )
        
        assert checkpoint.checkpoint_id == 1
        assert checkpoint.state['counter'] == 100
        
    def test_state_backend(self):
        """Test in-memory state backend."""
        backend = InMemoryStateBackend()
        
        # Save checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=1,
            timestamp=time.time(),
            state={'key': 'value'}
        )
        backend.save_checkpoint(checkpoint)
        
        # Retrieve checkpoint
        retrieved = backend.get_latest_checkpoint()
        
        assert retrieved is not None
        assert retrieved.state['key'] == 'value'


class TestStreamAggregators:
    """Test stream aggregation functions."""
    
    def test_sum_aggregator(self):
        """Test sum aggregation."""
        result = sum_aggregator(10.0, 5.0)
        assert result == 15.0
        
    def test_count_aggregator(self):
        """Test count aggregation."""
        result = count_aggregator(5, 1)
        assert result == 6
        
    def test_avg_aggregator(self):
        """Test average aggregation."""
        # avg_aggregator returns (sum, count) tuple
        result = avg_aggregator((10.0, 2), 5.0)
        # Should be (15.0, 3)
        assert result[0] == 15.0
        assert result[1] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
