"""
HiveFrame for Manufacturing - IoT analytics and predictive maintenance

Provides manufacturing analytics including sensor data processing,
predictive maintenance, and quality control.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import random


class SensorType(Enum):
    """Types of industrial sensors"""

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    FLOW = "flow"
    HUMIDITY = "humidity"


class HealthStatus(Enum):
    """Equipment health status"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class SensorReading:
    """Represents a sensor reading"""

    sensor_id: str
    sensor_type: SensorType
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


class SensorDataProcessor:
    """
    Process streaming sensor data from IoT devices.

    Uses swarm intelligence to aggregate and analyze sensor data,
    similar to how bees process environmental information.
    """

    def __init__(self):
        self.sensors: Dict[str, Dict] = {}
        self.readings: Dict[str, List[SensorReading]] = {}
        self.anomalies: List[Dict] = []

    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        location: str,
        normal_range: tuple = (0, 100),
    ) -> bool:
        """Register a new sensor"""
        if sensor_id in self.sensors:
            return False

        self.sensors[sensor_id] = {
            "type": sensor_type,
            "location": location,
            "normal_range": normal_range,
            "reading_count": 0,
        }
        self.readings[sensor_id] = []

        return True

    def process_reading(
        self,
        reading: SensorReading,
    ) -> Dict:
        """
        Process a sensor reading and detect anomalies.

        Returns processing result.
        """
        sensor_id = reading.sensor_id

        if sensor_id not in self.sensors:
            return {"status": "error", "message": "sensor_not_registered"}

        sensor = self.sensors[sensor_id]

        # Store reading
        self.readings[sensor_id].append(reading)
        sensor["reading_count"] += 1

        # Keep only recent readings
        if len(self.readings[sensor_id]) > 1000:
            self.readings[sensor_id] = self.readings[sensor_id][-1000:]

        # Detect anomalies
        normal_min, normal_max = sensor["normal_range"]
        is_anomaly = reading.value < normal_min or reading.value > normal_max

        result = {
            "status": "processed",
            "sensor_id": sensor_id,
            "value": reading.value,
            "is_anomaly": is_anomaly,
        }

        if is_anomaly:
            anomaly = {
                "sensor_id": sensor_id,
                "value": reading.value,
                "expected_range": sensor["normal_range"],
                "timestamp": reading.timestamp,
            }
            self.anomalies.append(anomaly)
            result["severity"] = "high" if abs(reading.value - normal_max) > 50 else "medium"

        return result

    def get_sensor_stats(self) -> Dict:
        """Get sensor processing statistics"""
        total_readings = sum(s["reading_count"] for s in self.sensors.values())

        return {
            "total_sensors": len(self.sensors),
            "total_readings": total_readings,
            "anomalies_detected": len(self.anomalies),
        }


class PredictiveMaintenanceSystem:
    """
    Predict equipment failures before they occur.

    Uses swarm-based pattern recognition to identify failure
    precursors, like how bees detect threats to the hive.
    """

    def __init__(self):
        self.equipment: Dict[str, Dict] = {}
        self.maintenance_schedule: List[Dict] = []
        self.predictions_made = 0

    def register_equipment(
        self,
        equipment_id: str,
        equipment_type: str,
        installed_date: float,
        expected_lifetime_hours: int = 10000,
    ) -> bool:
        """Register equipment for monitoring"""
        if equipment_id in self.equipment:
            return False

        self.equipment[equipment_id] = {
            "type": equipment_type,
            "installed_date": installed_date,
            "expected_lifetime_hours": expected_lifetime_hours,
            "operating_hours": 0,
            "health_status": HealthStatus.HEALTHY,
            "last_maintenance": installed_date,
        }

        return True

    def update_operating_hours(
        self,
        equipment_id: str,
        hours: float,
    ) -> bool:
        """Update equipment operating hours"""
        if equipment_id not in self.equipment:
            return False

        self.equipment[equipment_id]["operating_hours"] += hours
        return True

    def predict_failure(
        self,
        equipment_id: str,
        sensor_data: List[SensorReading],
    ) -> Optional[Dict]:
        """
        Predict equipment failure based on sensor data.

        Returns prediction or None if equipment not found.
        """
        if equipment_id not in self.equipment:
            return None

        equipment = self.equipment[equipment_id]

        # Calculate remaining life percentage
        operating_hours = equipment["operating_hours"]
        expected_lifetime = equipment["expected_lifetime_hours"]
        life_remaining = 1.0 - (operating_hours / expected_lifetime)

        # Analyze sensor data for degradation signs
        degradation_score = 0.0

        for reading in sensor_data:
            if reading.sensor_type == SensorType.VIBRATION and reading.value > 50:
                degradation_score += 0.2
            elif reading.sensor_type == SensorType.TEMPERATURE and reading.value > 80:
                degradation_score += 0.3

        # Combine factors
        failure_risk = (1.0 - life_remaining) * 0.7 + degradation_score * 0.3

        # Determine health status
        if failure_risk < 0.3:
            health_status = HealthStatus.HEALTHY
        elif failure_risk < 0.6:
            health_status = HealthStatus.WARNING
        elif failure_risk < 0.9:
            health_status = HealthStatus.CRITICAL
        else:
            health_status = HealthStatus.FAILED

        equipment["health_status"] = health_status

        # Estimate time to failure
        if failure_risk > 0.5:
            hours_to_failure = int((1.0 - failure_risk) * expected_lifetime)
        else:
            hours_to_failure = None

        prediction = {
            "equipment_id": equipment_id,
            "failure_risk": failure_risk,
            "health_status": health_status.value,
            "hours_to_failure": hours_to_failure,
            "recommended_action": "schedule_maintenance" if failure_risk > 0.6 else "monitor",
            "timestamp": time.time(),
        }

        self.predictions_made += 1

        # Schedule maintenance if needed
        if failure_risk > 0.6 and hours_to_failure:
            self.schedule_maintenance(
                equipment_id,
                time.time() + hours_to_failure * 3600,  # Convert to seconds
                "predictive",
            )

        return prediction

    def schedule_maintenance(
        self,
        equipment_id: str,
        scheduled_time: float,
        maintenance_type: str = "routine",
    ) -> bool:
        """Schedule maintenance for equipment"""
        if equipment_id not in self.equipment:
            return False

        maintenance = {
            "equipment_id": equipment_id,
            "scheduled_time": scheduled_time,
            "type": maintenance_type,
            "status": "scheduled",
        }

        self.maintenance_schedule.append(maintenance)
        return True

    def get_maintenance_stats(self) -> Dict:
        """Get predictive maintenance statistics"""
        scheduled = sum(1 for m in self.maintenance_schedule if m["status"] == "scheduled")

        return {
            "total_equipment": len(self.equipment),
            "predictions_made": self.predictions_made,
            "scheduled_maintenance": scheduled,
        }


class QualityControlAnalytics:
    """
    Analytics for product quality control.

    Uses swarm intelligence to identify quality issues and
    optimize manufacturing processes.
    """

    def __init__(self, defect_threshold: float = 0.02):
        self.defect_threshold = defect_threshold
        self.inspections: List[Dict] = []
        self.defect_patterns: Dict[str, int] = {}

    def record_inspection(
        self,
        product_id: str,
        batch_id: str,
        passed: bool,
        defect_types: Optional[List[str]] = None,
        measurements: Optional[Dict] = None,
    ) -> Dict:
        """
        Record a quality inspection.

        Returns inspection result.
        """
        inspection = {
            "product_id": product_id,
            "batch_id": batch_id,
            "passed": passed,
            "defect_types": defect_types or [],
            "measurements": measurements or {},
            "timestamp": time.time(),
        }

        self.inspections.append(inspection)

        # Track defect patterns
        if not passed and defect_types:
            for defect_type in defect_types:
                self.defect_patterns[defect_type] = self.defect_patterns.get(defect_type, 0) + 1

        return inspection

    def analyze_batch(
        self,
        batch_id: str,
    ) -> Dict:
        """
        Analyze quality for a production batch.

        Returns analysis results.
        """
        batch_inspections = [i for i in self.inspections if i["batch_id"] == batch_id]

        if not batch_inspections:
            return {"batch_id": batch_id, "status": "no_data"}

        total = len(batch_inspections)
        passed = sum(1 for i in batch_inspections if i["passed"])
        defect_rate = (total - passed) / total

        status = "pass" if defect_rate <= self.defect_threshold else "fail"

        return {
            "batch_id": batch_id,
            "total_inspected": total,
            "passed": passed,
            "defect_rate": defect_rate,
            "status": status,
            "threshold": self.defect_threshold,
        }

    def identify_root_causes(self) -> List[Dict]:
        """
        Identify most common defect types (root causes).

        Returns sorted list of defect types.
        """
        sorted_defects = sorted(self.defect_patterns.items(), key=lambda x: x[1], reverse=True)

        return [{"defect_type": defect, "count": count} for defect, count in sorted_defects]

    def get_quality_stats(self) -> Dict:
        """Get quality control statistics"""
        if not self.inspections:
            return {
                "total_inspections": 0,
                "overall_pass_rate": 0,
                "unique_defect_types": 0,
            }

        passed = sum(1 for i in self.inspections if i["passed"])
        pass_rate = passed / len(self.inspections)

        return {
            "total_inspections": len(self.inspections),
            "overall_pass_rate": pass_rate,
            "unique_defect_types": len(self.defect_patterns),
        }
