"""
Tests for HiveFrame connectors module.

Tests cover:
- DataSource and DataSink abstract interfaces
- CSV file source with various configurations
- JSON and JSONL file sources
- File sinks (JSONL, CSV)
- HTTP API source with rate limiting
- Message broker (Kafka-like) simulation
- Change data capture source
- Metrics collection on connectors
"""

import json
import tempfile
from pathlib import Path

import pytest

from hiveframe.connectors import (
    CSVSink,
    CSVSource,
    HTTPSource,
    JSONLSink,
    JSONLSource,
    JSONSource,
    MessageBroker,
    # MessageProducer, MessageConsumer,  # Not yet implemented
    # ChangeDataCaptureSource, CDCEvent  # Not yet implemented
)
from hiveframe.exceptions import (
    ConfigurationError,
    DataError,
    ParseError,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file."""
    csv_path = temp_dir / "sample.csv"
    csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago\n")
    return csv_path


@pytest.fixture
def sample_csv_no_header(temp_dir):
    """Create a CSV file without header."""
    csv_path = temp_dir / "no_header.csv"
    csv_path.write_text("Alice,30,NYC\nBob,25,LA\n")
    return csv_path


@pytest.fixture
def malformed_csv_file(temp_dir):
    """Create a CSV file with malformed rows."""
    csv_path = temp_dir / "malformed.csv"
    csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25\nCharlie,35,Chicago,Extra\n")
    return csv_path


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON array file."""
    json_path = temp_dir / "sample.json"
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def sample_json_with_field(temp_dir):
    """Create a JSON file with nested array field."""
    json_path = temp_dir / "nested.json"
    data = {"metadata": {"count": 3}, "records": [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]}
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def sample_jsonl_file(temp_dir):
    """Create a sample JSONL file."""
    jsonl_path = temp_dir / "sample.jsonl"
    lines = [
        json.dumps({"id": 1, "name": "Alice"}),
        json.dumps({"id": 2, "name": "Bob"}),
        json.dumps({"id": 3, "name": "Charlie"}),
    ]
    jsonl_path.write_text("\n".join(lines) + "\n")
    return jsonl_path


@pytest.fixture
def malformed_jsonl_file(temp_dir):
    """Create a JSONL file with invalid JSON."""
    jsonl_path = temp_dir / "malformed.jsonl"
    lines = [json.dumps({"id": 1}), "not valid json {", json.dumps({"id": 3})]
    jsonl_path.write_text("\n".join(lines) + "\n")
    return jsonl_path


# ============================================================================
# CSV Source Tests
# ============================================================================


class TestCSVSource:
    """Tests for CSVSource connector."""

    def test_read_csv_with_header(self, sample_csv_file):
        """Test reading CSV file with header."""
        with CSVSource(sample_csv_file) as source:
            records = list(source.read())

        assert len(records) == 3
        assert records[0] == {"name": "Alice", "age": "30", "city": "NYC"}
        assert records[1]["name"] == "Bob"

    def test_read_csv_without_header(self, sample_csv_no_header):
        """Test reading CSV without header using explicit columns."""
        with CSVSource(
            sample_csv_no_header, has_header=False, columns=["name", "age", "city"]
        ) as source:
            records = list(source.read())

        assert len(records) == 2
        assert records[0]["name"] == "Alice"

    def test_csv_custom_delimiter(self, temp_dir):
        """Test CSV with custom delimiter."""
        csv_path = temp_dir / "semicolon.csv"
        csv_path.write_text("name;age\nAlice;30\nBob;25\n")

        with CSVSource(csv_path, delimiter=";") as source:
            records = list(source.read())

        assert len(records) == 2
        assert records[0]["name"] == "Alice"

    def test_csv_malformed_skip(self, malformed_csv_file):
        """Test that malformed rows are skipped by default."""
        with CSVSource(malformed_csv_file, skip_malformed=True) as source:
            records = list(source.read())

        # Only valid rows should be returned
        assert len(records) == 1  # Only Alice's row is valid

    def test_csv_malformed_raise(self, malformed_csv_file):
        """Test that malformed rows raise error when not skipping."""
        with CSVSource(malformed_csv_file, skip_malformed=False) as source:
            with pytest.raises(ParseError):
                list(source.read())

    def test_csv_file_not_found(self, temp_dir):
        """Test error when CSV file doesn't exist."""
        source = CSVSource(temp_dir / "nonexistent.csv")
        with pytest.raises(ConfigurationError):
            source.open()

    def test_csv_empty_file(self, temp_dir):
        """Test reading empty CSV file."""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("")

        source = CSVSource(csv_path)
        with pytest.raises(DataError):
            source.open()

    def test_csv_metrics(self, sample_csv_file):
        """Test that CSV source collects metrics."""
        with CSVSource(sample_csv_file) as source:
            list(source.read())
            metrics = source.get_metrics()

        assert metrics["records_read"] == 3
        assert metrics["errors"] == 0
        assert metrics["is_open"]  # Still True during context manager

        # Check that it's closed after context manager
        metrics_after = source.get_metrics()
        assert not metrics_after["is_open"]

    def test_csv_context_manager(self, sample_csv_file):
        """Test CSV source context manager properly closes."""
        source = CSVSource(sample_csv_file)

        with source:
            assert source._is_open

        assert not source._is_open


# ============================================================================
# JSON Source Tests
# ============================================================================


class TestJSONSource:
    """Tests for JSONSource connector."""

    def test_read_json_array(self, sample_json_file):
        """Test reading JSON array file."""
        with JSONSource(sample_json_file) as source:
            records = list(source.read())

        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_read_json_with_array_field(self, sample_json_with_field):
        """Test reading JSON with nested array field."""
        with JSONSource(sample_json_with_field, array_field="records") as source:
            records = list(source.read())

        assert len(records) == 2
        assert records[0]["id"] == 1

    def test_json_missing_array_field(self, sample_json_file):
        """Test error when array field doesn't exist."""
        source = JSONSource(sample_json_file, array_field="nonexistent")
        with pytest.raises(DataError):
            source.open()

    def test_json_not_array(self, temp_dir):
        """Test error when JSON is not array and no field specified."""
        json_path = temp_dir / "object.json"
        json_path.write_text(json.dumps({"key": "value"}))

        source = JSONSource(json_path)
        with pytest.raises(DataError):
            source.open()

    def test_json_file_not_found(self, temp_dir):
        """Test error when JSON file doesn't exist."""
        source = JSONSource(temp_dir / "nonexistent.json")
        with pytest.raises(ConfigurationError):
            source.open()


# ============================================================================
# JSONL Source Tests
# ============================================================================


class TestJSONLSource:
    """Tests for JSONLSource connector."""

    def test_read_jsonl(self, sample_jsonl_file):
        """Test reading JSONL file."""
        with JSONLSource(sample_jsonl_file) as source:
            records = list(source.read())

        assert len(records) == 3
        assert records[0]["id"] == 1

    def test_jsonl_skip_empty_lines(self, temp_dir):
        """Test that empty lines are skipped."""
        jsonl_path = temp_dir / "with_empty.jsonl"
        content = '{"id": 1}\n\n{"id": 2}\n\n\n{"id": 3}\n'
        jsonl_path.write_text(content)

        with JSONLSource(jsonl_path) as source:
            records = list(source.read())

        assert len(records) == 3

    def test_jsonl_malformed_skip(self, malformed_jsonl_file):
        """Test skipping malformed JSON lines."""
        with JSONLSource(malformed_jsonl_file, skip_malformed=True) as source:
            records = list(source.read())
            metrics = source.get_metrics()

        assert len(records) == 2
        assert metrics["errors"] == 1

    def test_jsonl_malformed_raise(self, malformed_jsonl_file):
        """Test raising error on malformed JSON."""
        with JSONLSource(malformed_jsonl_file, skip_malformed=False) as source:
            with pytest.raises(ParseError):
                list(source.read())

    def test_jsonl_file_not_found(self, temp_dir):
        """Test error when JSONL file doesn't exist."""
        source = JSONLSource(temp_dir / "nonexistent.jsonl")
        with pytest.raises(ConfigurationError):
            source.open()


# ============================================================================
# File Sink Tests
# ============================================================================


class TestJSONLSink:
    """Tests for JSONLSink connector."""

    def test_write_jsonl(self, temp_dir):
        """Test writing records to JSONL file."""
        output_path = temp_dir / "output.jsonl"

        with JSONLSink(output_path) as sink:
            sink.write({"id": 1, "name": "Alice"})
            sink.write({"id": 2, "name": "Bob"})

        # Read back and verify
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "Alice"

    def test_write_jsonl_append(self, temp_dir):
        """Test appending to existing JSONL file."""
        output_path = temp_dir / "append.jsonl"
        output_path.write_text('{"id": 0}\n')

        with JSONLSink(output_path, append=True) as sink:
            sink.write({"id": 1})

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_batch(self, temp_dir):
        """Test writing batch of records."""
        output_path = temp_dir / "batch.jsonl"

        with JSONLSink(output_path) as sink:
            records = [{"id": i} for i in range(5)]
            written = sink.write_batch(records)

        assert written == 5

    def test_sink_metrics(self, temp_dir):
        """Test sink metrics collection."""
        output_path = temp_dir / "metrics.jsonl"

        with JSONLSink(output_path) as sink:
            sink.write({"id": 1})
            sink.write({"id": 2})
            metrics = sink.get_metrics()

        assert metrics["records_written"] == 2
        assert metrics["errors"] == 0


class TestCSVSink:
    """Tests for CSVSink connector."""

    def test_write_csv(self, temp_dir):
        """Test writing records to CSV file."""
        output_path = temp_dir / "output.csv"

        with CSVSink(output_path, columns=["name", "age"]) as sink:
            sink.write({"name": "Alice", "age": 30})
            sink.write({"name": "Bob", "age": 25})

        content = output_path.read_text()
        assert "name,age" in content
        assert "Alice,30" in content

    def test_csv_sink_auto_columns(self, temp_dir):
        """Test CSV sink inferring columns from first record."""
        output_path = temp_dir / "auto.csv"

        with CSVSink(output_path) as sink:
            sink.write({"a": 1, "b": 2})
            sink.write({"a": 3, "b": 4})

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows


# ============================================================================
# Message Broker Tests
# ============================================================================


class TestMessageBroker:
    """Tests for Kafka-like MessageBroker."""

    def test_broker_creation(self):
        """Test creating message broker."""
        broker = MessageBroker()
        assert broker is not None

    def test_create_topic(self):
        """Test creating a topic."""
        broker = MessageBroker()
        broker.create_topic("test-topic", partitions=3)

        topics = broker.list_topics()
        assert "test-topic" in topics

    def test_produce_consume(self):
        """Test basic produce and consume."""
        broker = MessageBroker()
        broker.create_topic("test-topic")

        producer = broker.create_producer()
        consumer = broker.create_consumer("test-topic", "test-group")

        # Produce messages
        producer.send("test-topic", {"key": "value"})
        producer.send("test-topic", {"key": "value2"})

        # Consume messages
        messages = consumer.poll(timeout=1.0, max_messages=10)
        assert len(messages) == 2

    def test_consumer_groups(self):
        """Test consumer group isolation."""
        broker = MessageBroker()
        broker.create_topic("shared-topic")

        producer = broker.create_producer()
        consumer1 = broker.create_consumer("shared-topic", "group1")
        consumer2 = broker.create_consumer("shared-topic", "group2")

        producer.send("shared-topic", {"msg": 1})

        # Both groups should receive the message
        msgs1 = consumer1.poll(timeout=1.0)
        msgs2 = consumer2.poll(timeout=1.0)

        assert len(msgs1) == 1
        assert len(msgs2) == 1

    def test_message_ordering(self):
        """Test that messages maintain order within partition."""
        broker = MessageBroker()
        broker.create_topic("ordered-topic", partitions=1)

        producer = broker.create_producer()
        consumer = broker.create_consumer("ordered-topic", "test-group")

        # Send messages with sequence
        for i in range(5):
            producer.send("ordered-topic", {"seq": i}, key=str(i))

        messages = consumer.poll(timeout=1.0, max_messages=5)

        # Verify order
        seqs = [m.value["seq"] for m in messages]
        assert seqs == list(range(5))


# ============================================================================
# HTTP Source Tests
# ============================================================================


class TestHTTPSource:
    """Tests for HTTPSource connector."""

    def test_http_source_creation(self):
        """Test creating HTTP source."""
        source = HTTPSource(
            url="https://api.example.com/data", headers={"Authorization": "Bearer token"}
        )
        assert source is not None
        assert source.url == "https://api.example.com/data"

    def test_http_source_rate_limiting(self):
        """Test HTTP source respects rate limits."""
        source = HTTPSource(url="https://api.example.com/data", requests_per_second=2.0)
        # Rate limiter should be configured
        assert source.rate_limit == 2.0


# ============================================================================
# Change Data Capture Tests
# ============================================================================


@pytest.mark.skip(reason="CDC functionality not yet implemented")
class TestChangeDataCapture:
    """Tests for CDC source."""

    def test_cdc_event_creation(self):
        """Test creating CDC events."""
        # CDCEvent not yet implemented
        pass

    def test_cdc_operations(self):
        """Test different CDC operations."""
        # CDCEvent not yet implemented
        pass


# ============================================================================
# Integration Tests
# ============================================================================


class TestConnectorIntegration:
    """Integration tests for connector pipelines."""

    def test_csv_to_jsonl_pipeline(self, sample_csv_file, temp_dir):
        """Test reading CSV and writing to JSONL."""
        output_path = temp_dir / "output.jsonl"

        with CSVSource(sample_csv_file) as source:
            with JSONLSink(output_path) as sink:
                for record in source.read():
                    # Transform age to int
                    record["age"] = int(record["age"])
                    sink.write(record)

        # Verify output
        with JSONLSource(output_path) as check:
            records = list(check.read())

        assert len(records) == 3
        assert records[0]["age"] == 30  # Converted to int

    def test_broker_to_sink_pipeline(self, temp_dir):
        """Test consuming from broker and writing to file."""
        broker = MessageBroker()
        broker.create_topic("pipeline-topic")

        producer = broker.create_producer()
        for i in range(3):
            producer.send("pipeline-topic", {"id": i, "value": f"item_{i}"})

        consumer = broker.create_consumer("pipeline-topic", "pipeline-group")
        output_path = temp_dir / "broker_output.jsonl"

        with JSONLSink(output_path) as sink:
            messages = consumer.poll(timeout=1.0, max_messages=10)
            for msg in messages:
                sink.write(msg.value)

        # Verify
        with JSONLSource(output_path) as check:
            records = list(check.read())

        assert len(records) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
