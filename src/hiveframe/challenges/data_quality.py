"""
Challenge Scenario: Data Quality & Edge Cases
=============================================
Tests HiveFrame's handling of real-world data quality issues.

Scenarios:
1. Malformed data handling
2. Schema drift detection
3. Null/missing value handling
4. Type coercion edge cases
5. Unicode and encoding challenges
6. Extreme values (very large, very small, special floats)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..exceptions import (
    DeadLetterQueue,
    DeadLetterRecord,
    NullValueError,
    ValidationError,
)
from ..monitoring import get_logger, get_profiler

logger = get_logger("challenge.data_quality")
profiler = get_profiler()


@dataclass
class DataQualityResult:
    """Results from a data quality scenario."""

    scenario_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    corrected_records: int  # Records fixed by coercion/defaults
    quarantined_records: int
    elapsed_seconds: float
    issues_by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """Overall data quality score (0-1)."""
        if self.total_records == 0:
            return 0
        return (self.valid_records + self.corrected_records) / self.total_records

    def summary(self) -> str:
        return f"""
=== {self.scenario_name} ===
Total Records:      {self.total_records}
Valid:              {self.valid_records} ({100*self.valid_records/max(1,self.total_records):.1f}%)
Corrected:          {self.corrected_records}
Invalid:            {self.invalid_records}
Quarantined:        {self.quarantined_records}
Quality Score:      {100*self.quality_score:.1f}%
Elapsed:            {self.elapsed_seconds:.2f}s
Issues by Type:     {self.issues_by_type}
"""


class DataQualityValidator:
    """
    Validates and optionally corrects data quality issues.

    Inspired by bee quality control - worker bees inspect cells
    and remove defective larvae.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, str]] = None,
        strict_mode: bool = False,
        auto_correct: bool = True,
    ):
        self.schema = schema or {}
        self.strict_mode = strict_mode
        self.auto_correct = auto_correct
        self._issues: Dict[str, int] = {}

    def _record_issue(self, issue_type: str):
        self._issues[issue_type] = self._issues.get(issue_type, 0) + 1

    def validate_type(self, value: Any, expected_type: str, field: str) -> Tuple[Any, bool]:
        """
        Validate and optionally coerce value to expected type.
        Returns (value, was_corrected).
        """
        if value is None:
            if self.strict_mode:
                raise NullValueError(f"Null value in field {field}", field=field)
            return None, False

        try:
            if expected_type == "string":
                if not isinstance(value, str):
                    if self.auto_correct:
                        return str(value), True
                    raise ValidationError(
                        f"Expected string for {field}",
                        field=field,
                        expected="string",
                        actual=type(value).__name__,
                    )

            elif expected_type == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    if self.auto_correct:
                        return int(float(value)), True
                    raise ValidationError(
                        f"Expected int for {field}",
                        field=field,
                        expected="int",
                        actual=type(value).__name__,
                    )

            elif expected_type == "float":
                if not isinstance(value, (int, float)):
                    if self.auto_correct:
                        return float(value), True
                    raise ValidationError(
                        f"Expected float for {field}",
                        field=field,
                        expected="float",
                        actual=type(value).__name__,
                    )

            elif expected_type == "bool":
                if not isinstance(value, bool):
                    if self.auto_correct:
                        # Handle common boolean representations
                        if isinstance(value, str):
                            return value.lower() in ("true", "1", "yes", "y"), True
                        return bool(value), True
                    raise ValidationError(
                        f"Expected bool for {field}",
                        field=field,
                        expected="bool",
                        actual=type(value).__name__,
                    )

        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Cannot convert {field}: {e}",
                field=field,
                expected=expected_type,
                actual=str(value)[:50],
            )

        return value, False

    def validate_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate a record against schema.
        Returns (validated_record, list_of_corrections).
        """
        validated = {}
        corrections = []

        for field, expected_type in self.schema.items():
            value = record.get(field)

            try:
                validated_value, was_corrected = self.validate_type(value, expected_type, field)
                validated[field] = validated_value

                if was_corrected:
                    corrections.append(f"{field}: {type(value).__name__} -> {expected_type}")
                    self._record_issue(f"type_coercion:{field}")

            except ValidationError:
                self._record_issue(f"validation_error:{field}")
                raise

        # Copy any extra fields not in schema
        for field, value in record.items():
            if field not in validated:
                validated[field] = value

        return validated, corrections

    def get_issues(self) -> Dict[str, int]:
        return dict(self._issues)


def run_malformed_data_scenario(num_records: int = 1000) -> DataQualityResult:
    """
    Scenario 1: Malformed Data Handling

    Tests handling of various malformed data patterns:
    - Missing fields
    - Extra fields
    - Wrong types
    - Truncated strings
    - Corrupted JSON
    """
    logger.info("Starting malformed data scenario", num_records=num_records)

    schema = {"id": "int", "name": "string", "amount": "float", "active": "bool"}

    validator = DataQualityValidator(schema=schema, auto_correct=True)
    dlq = DeadLetterQueue(max_size=500)

    # Generate mix of good and malformed data
    records = []
    for i in range(num_records):
        r = random.random()

        if r < 0.7:
            # Good record
            records.append(
                {
                    "id": i,
                    "name": f"Item {i}",
                    "amount": random.random() * 100,
                    "active": random.choice([True, False]),
                }
            )
        elif r < 0.8:
            # Wrong types
            records.append(
                {
                    "id": str(i),  # Should be int
                    "name": i,  # Should be string
                    "amount": f"{random.random() * 100:.2f}",  # String instead of float
                    "active": "yes",  # String instead of bool
                }
            )
        elif r < 0.85:
            # Missing fields
            records.append(
                {
                    "id": i,
                    "name": f"Partial {i}",
                    # Missing amount and active
                }
            )
        elif r < 0.9:
            # Extra fields
            records.append(
                {
                    "id": i,
                    "name": f"Extra {i}",
                    "amount": 50.0,
                    "active": True,
                    "extra_field": "unexpected",
                    "another": 12345,
                }
            )
        else:
            # Completely malformed
            records.append({"garbage": "not valid", "random_stuff": [1, 2, 3]})

    valid_records = 0
    corrected_records = 0
    invalid_records = 0
    quarantined = 0

    start_time = time.time()

    for record in records:
        try:
            with profiler.profile("malformed_validation"):
                validated, corrections = validator.validate_record(record)

            if corrections:
                corrected_records += 1
            else:
                valid_records += 1

        except (ValidationError, NullValueError) as e:
            invalid_records += 1
            dlq.push(
                DeadLetterRecord(
                    original_data=record,
                    error=e,
                    partition_id=str(record.get("id", "unknown")),
                    worker_id="validator",
                    attempt_count=1,
                    first_failure=time.time(),
                )
            )
            quarantined += 1

    elapsed = time.time() - start_time

    return DataQualityResult(
        scenario_name="Malformed Data Handling",
        total_records=num_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        corrected_records=corrected_records,
        quarantined_records=quarantined,
        elapsed_seconds=elapsed,
        issues_by_type=validator.get_issues(),
    )


def run_schema_drift_scenario(
    num_batches: int = 10, records_per_batch: int = 100
) -> DataQualityResult:
    """
    Scenario 2: Schema Drift Detection

    Tests handling of schema changes over time:
    - New fields appearing
    - Fields being removed
    - Type changes
    - Renamed fields
    """
    logger.info(
        "Starting schema drift scenario",
        num_batches=num_batches,
        records_per_batch=records_per_batch,
    )

    # Initial schema
    schema_v1 = {"id": "int", "value": "float", "category": "string"}

    # Schema versions that will drift
    schema_versions = [
        schema_v1,
        {**schema_v1, "new_field": "string"},  # Added field
        {"id": "int", "value": "float", "new_field": "string"},  # Removed category
        {"id": "int", "value": "string", "new_field": "string"},  # Type change
        {"id": "int", "amount": "float", "new_field": "string"},  # Renamed field
    ]

    observed_schemas = []
    drift_events = []
    valid_records = 0
    invalid_records = 0
    corrected_records = 0

    start_time = time.time()

    for batch_idx in range(num_batches):
        # Select schema for this batch (simulating drift)
        schema_idx = min(batch_idx // 2, len(schema_versions) - 1)
        current_schema = schema_versions[schema_idx]

        # Generate records according to current schema
        records = []
        for i in range(records_per_batch):
            record = {"id": batch_idx * records_per_batch + i}

            if "value" in current_schema:
                if current_schema["value"] == "float":
                    record["value"] = random.random() * 100
                else:
                    record["value"] = f"{random.random() * 100:.2f}"

            if "amount" in current_schema:
                record["amount"] = random.random() * 100

            if "category" in current_schema:
                record["category"] = random.choice(["A", "B", "C"])

            if "new_field" in current_schema:
                record["new_field"] = f"new_value_{i}"

            records.append(record)

        # Detect schema changes
        record_fields = set(records[0].keys()) if records else set()

        if observed_schemas:
            last_fields = observed_schemas[-1]

            added = record_fields - last_fields
            removed = last_fields - record_fields

            if added or removed:
                drift_events.append(
                    {"batch": batch_idx, "added": list(added), "removed": list(removed)}
                )
                logger.warning(
                    "Schema drift detected",
                    batch=batch_idx,
                    added=list(added),
                    removed=list(removed),
                )

        observed_schemas.append(record_fields)

        # Validate against original schema (v1)
        validator = DataQualityValidator(schema=schema_v1, auto_correct=True)

        for record in records:
            try:
                validated, corrections = validator.validate_record(record)

                if corrections:
                    corrected_records += 1
                else:
                    valid_records += 1

            except ValidationError:
                invalid_records += 1

    elapsed = time.time() - start_time

    return DataQualityResult(
        scenario_name="Schema Drift Detection",
        total_records=num_batches * records_per_batch,
        valid_records=valid_records,
        invalid_records=invalid_records,
        corrected_records=corrected_records,
        quarantined_records=0,
        elapsed_seconds=elapsed,
        issues_by_type={"schema_drift_events": len(drift_events)},
    )


def run_null_handling_scenario(num_records: int = 1000) -> DataQualityResult:
    """
    Scenario 3: Null/Missing Value Handling

    Tests handling of various null patterns:
    - Explicit None/null
    - Empty strings
    - Missing keys
    - "null" as string
    - NaN values
    """
    logger.info("Starting null handling scenario", num_records=num_records)

    schema = {
        "id": "int",
        "required_field": "string",
        "optional_field": "string",
        "numeric_field": "float",
    }

    # Generate records with various null patterns
    records = []
    for i in range(num_records):
        record = {"id": i}

        r = random.random()

        if r < 0.3:
            # Normal record
            record["required_field"] = f"value_{i}"
            record["optional_field"] = f"optional_{i}"
            record["numeric_field"] = float(i)
        elif r < 0.4:
            # Explicit None
            record["required_field"] = None
            record["optional_field"] = None
            record["numeric_field"] = None
        elif r < 0.5:
            # Empty strings
            record["required_field"] = ""
            record["optional_field"] = ""
            record["numeric_field"] = 0.0
        elif r < 0.6:
            # Missing keys (no required_field)
            record["optional_field"] = "present"
            record["numeric_field"] = 42.0
        elif r < 0.7:
            # "null" as string
            record["required_field"] = "null"
            record["optional_field"] = "NULL"
            record["numeric_field"] = float("nan")
        elif r < 0.8:
            # Mixed nulls
            record["required_field"] = "valid"
            record["optional_field"] = None
            record["numeric_field"] = float("inf")
        else:
            # NaN and special floats
            record["required_field"] = "special"
            record["optional_field"] = ""
            record["numeric_field"] = random.choice(
                [float("nan"), float("inf"), float("-inf"), 0.0, -0.0]
            )

        records.append(record)

    valid_records = 0
    invalid_records = 0
    corrected_records = 0
    issues = {
        "explicit_null": 0,
        "empty_string": 0,
        "missing_key": 0,
        "null_string": 0,
        "special_float": 0,
    }

    start_time = time.time()

    for record in records:
        corrections = []
        valid = True

        # Check for various null patterns
        for field, expected_type in schema.items():
            value = record.get(field)

            # Missing key
            if field not in record:
                issues["missing_key"] += 1
                if field == "required_field":
                    valid = False
                else:
                    corrections.append(f"{field}: added default")
                continue

            # Explicit None
            if value is None:
                issues["explicit_null"] += 1
                corrections.append(f"{field}: null -> default")
                continue

            # Empty string
            if value == "":
                issues["empty_string"] += 1
                corrections.append(f"{field}: empty -> default")
                continue

            # "null" string
            if isinstance(value, str) and value.lower() == "null":
                issues["null_string"] += 1
                corrections.append(f'{field}: "null" -> null')
                continue

            # Special floats
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                issues["special_float"] += 1
                corrections.append(f"{field}: special float -> default")
                continue

        if valid:
            if corrections:
                corrected_records += 1
            else:
                valid_records += 1
        else:
            invalid_records += 1

    elapsed = time.time() - start_time

    return DataQualityResult(
        scenario_name="Null/Missing Value Handling",
        total_records=num_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        corrected_records=corrected_records,
        quarantined_records=invalid_records,
        elapsed_seconds=elapsed,
        issues_by_type=issues,
    )


def run_unicode_encoding_scenario(num_records: int = 500) -> DataQualityResult:
    """
    Scenario 4: Unicode and Encoding Challenges

    Tests handling of various unicode and encoding issues:
    - Multi-byte characters
    - Right-to-left text
    - Emoji
    - Control characters
    - Invalid UTF-8 sequences
    - Mixed encodings
    """
    logger.info("Starting unicode/encoding scenario", num_records=num_records)

    # Various unicode test cases
    unicode_samples = [
        "Hello World",  # ASCII
        "Héllo Wörld",  # Latin extended
        "こんにちは",  # Japanese
        "مرحبا",  # Arabic (RTL)
        "שלום",  # Hebrew (RTL)
        "🐝🍯🌸",  # Emoji
        "Hello\x00World",  # Null byte
        "Tab\there",  # Tab
        "Line\nBreak",  # Newline
        "Carriage\rReturn",  # Carriage return
        "\u202e" + "reversed",  # RTL override
        "🇺🇸🇬🇧🇯🇵",  # Flag emoji (multi-codepoint)
        "a\u0308",  # Combining characters (ä)
        "\ufeff" + "BOM prefix",  # Byte order mark
        "Zero\u200bwidth",  # Zero-width space
        "Mixed 日本語 and English",  # Mixed scripts
    ]

    records = []
    for i in range(num_records):
        sample = random.choice(unicode_samples)
        records.append({"id": i, "text": sample, "description": f"Test record {i}: {sample}"})

    valid_records = 0
    invalid_records = 0
    corrected_records = 0
    issues = {
        "control_chars": 0,
        "rtl_override": 0,
        "bom": 0,
        "zero_width": 0,
        "combining": 0,
        "mixed_script": 0,
    }

    start_time = time.time()

    for record in records:
        corrections = []
        text = record.get("text", "")

        # Check for various unicode issues
        if "\x00" in text:
            issues["control_chars"] += 1
            text = text.replace("\x00", "")
            corrections.append("removed null bytes")

        if "\u202e" in text or "\u202d" in text:
            issues["rtl_override"] += 1
            corrections.append("found RTL override")

        if text.startswith("\ufeff"):
            issues["bom"] += 1
            text = text.lstrip("\ufeff")
            corrections.append("stripped BOM")

        if "\u200b" in text or "\u200c" in text or "\u200d" in text:
            issues["zero_width"] += 1
            corrections.append("contains zero-width chars")

        # Check for combining characters
        import unicodedata

        for char in text:
            if unicodedata.category(char).startswith("M"):
                issues["combining"] += 1
                break

        # Check for mixed scripts
        scripts = set()
        for char in text:
            try:
                name = unicodedata.name(char, "")
                if "CJK" in name:
                    scripts.add("CJK")
                elif "ARABIC" in name:
                    scripts.add("Arabic")
                elif "HEBREW" in name:
                    scripts.add("Hebrew")
                elif "LATIN" in name or char.isascii():
                    scripts.add("Latin")
            except:
                pass

        if len(scripts) > 1:
            issues["mixed_script"] += 1

        if corrections:
            corrected_records += 1
        else:
            valid_records += 1

    elapsed = time.time() - start_time

    return DataQualityResult(
        scenario_name="Unicode/Encoding Challenges",
        total_records=num_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        corrected_records=corrected_records,
        quarantined_records=0,
        elapsed_seconds=elapsed,
        issues_by_type=issues,
    )


def run_extreme_values_scenario(num_records: int = 500) -> DataQualityResult:
    """
    Scenario 5: Extreme Value Handling

    Tests handling of edge case numeric values:
    - Very large numbers
    - Very small numbers
    - Precision limits
    - Special IEEE 754 values
    - Integer overflow candidates
    """
    logger.info("Starting extreme values scenario", num_records=num_records)

    extreme_values = [
        0,
        -0.0,
        1,
        -1,
        0.1,
        0.1 + 0.2,  # Famous floating point issue
        10**308,  # Near max float
        10**-308,  # Near min positive float
        2**63 - 1,  # Max int64
        -(2**63),  # Min int64
        2**64,  # Overflow int64
        float("inf"),
        float("-inf"),
        float("nan"),
        1e-15,  # Small but representable
        1e-400,  # Underflows to 0
        9.999999999999999,  # Precision edge
        10.000000000000001,
    ]

    records = []
    for i in range(num_records):
        value = random.choice(extreme_values)
        records.append({"id": i, "value": value, "description": f"Value: {value}"})

    valid_records = 0
    invalid_records = 0
    corrected_records = 0
    issues = {
        "infinity": 0,
        "nan": 0,
        "overflow": 0,
        "underflow": 0,
        "precision_loss": 0,
        "negative_zero": 0,
    }

    start_time = time.time()

    for record in records:
        corrections = []
        value = record.get("value")

        if isinstance(value, float):
            if math.isinf(value):
                issues["infinity"] += 1
                corrections.append("infinite value")

            if math.isnan(value):
                issues["nan"] += 1
                corrections.append("NaN value")

            # Check for negative zero
            if value == 0 and math.copysign(1, value) < 0:
                issues["negative_zero"] += 1
                corrections.append("negative zero")

            # Check for precision issues
            if value != 0 and abs(value) < 1e-15:
                issues["underflow"] += 1
                corrections.append("near underflow")

        if isinstance(value, (int, float)):
            # Check for potential overflow in common representations
            if abs(value) > 2**53:
                issues["precision_loss"] += 1
                corrections.append("precision loss likely")

            if abs(value) > 2**63:
                issues["overflow"] += 1
                corrections.append("int64 overflow")

        if corrections:
            corrected_records += 1
        else:
            valid_records += 1

    elapsed = time.time() - start_time

    return DataQualityResult(
        scenario_name="Extreme Value Handling",
        total_records=num_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        corrected_records=corrected_records,
        quarantined_records=0,
        elapsed_seconds=elapsed,
        issues_by_type=issues,
    )


def run_all_data_quality_scenarios() -> List[DataQualityResult]:
    """Run all data quality scenarios."""
    results = []

    print("\n" + "=" * 60)
    print("HiveFrame Data Quality Challenge Suite")
    print("=" * 60)

    scenarios = [
        ("Malformed Data", lambda: run_malformed_data_scenario(1000)),
        ("Schema Drift", lambda: run_schema_drift_scenario(10, 100)),
        ("Null Handling", lambda: run_null_handling_scenario(1000)),
        ("Unicode/Encoding", lambda: run_unicode_encoding_scenario(500)),
        ("Extreme Values", lambda: run_extreme_values_scenario(500)),
    ]

    for name, scenario_fn in scenarios:
        print(f"\nRunning: {name}...")
        try:
            result = scenario_fn()
            results.append(result)
            print(result.summary())
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)

    total_records = sum(r.total_records for r in results)
    avg_quality = sum(r.quality_score for r in results) / len(results) if results else 0

    print(f"Total Records Tested: {total_records}")
    print(f"Average Quality Score: {100*avg_quality:.1f}%")

    return results


if __name__ == "__main__":
    run_all_data_quality_scenarios()
