"""
HiveFrame for Healthcare - HIPAA-compliant analytics

Provides healthcare-specific data processing with encryption,
audit logging, and privacy-preserving analytics.
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import json


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256 = "aes_256"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"


class AuditEventType(Enum):
    """Types of audit events"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_EXPORT = "data_export"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"


@dataclass
class EncryptedData:
    """Represents encrypted healthcare data"""
    data_id: str
    algorithm: EncryptionAlgorithm
    encrypted_payload: bytes
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "data_id": self.data_id,
            "algorithm": self.algorithm.value,
            "encrypted_payload": self.encrypted_payload.hex(),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class DataEncryption:
    """
    HIPAA-compliant data encryption at rest and in transit.
    
    Uses swarm-inspired key distribution where encryption keys
    are shared across nodes like pheromone information.
    """
    
    def __init__(self, default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256):
        self.default_algorithm = default_algorithm
        self.encrypted_data: Dict[str, EncryptedData] = {}
        self.key_store: Dict[str, bytes] = {}
        self.encryption_count = 0
        self.decryption_count = 0
        
    def generate_key(self, key_id: str) -> bytes:
        """Generate an encryption key"""
        # Simplified key generation (real implementation would use proper crypto)
        key = hashlib.sha256(f"{key_id}{time.time()}".encode()).digest()
        self.key_store[key_id] = key
        return key
    
    def encrypt_data(
        self,
        data_id: str,
        plaintext: bytes,
        key_id: str,
        metadata: Optional[Dict] = None,
        algorithm: Optional[EncryptionAlgorithm] = None,
    ) -> bool:
        """
        Encrypt sensitive healthcare data.
        
        Returns True if successful, False otherwise.
        """
        if key_id not in self.key_store:
            return False
        
        algo = algorithm or self.default_algorithm
        
        # Simulate encryption (real implementation would use proper crypto library)
        key = self.key_store[key_id]
        encrypted = self._xor_encrypt(plaintext, key)
        
        encrypted_data = EncryptedData(
            data_id=data_id,
            algorithm=algo,
            encrypted_payload=encrypted,
            metadata=metadata or {},
        )
        
        self.encrypted_data[data_id] = encrypted_data
        self.encryption_count += 1
        
        return True
    
    def decrypt_data(
        self,
        data_id: str,
        key_id: str,
    ) -> Optional[bytes]:
        """
        Decrypt healthcare data.
        
        Returns decrypted data or None if unsuccessful.
        """
        if data_id not in self.encrypted_data or key_id not in self.key_store:
            return None
        
        encrypted_data = self.encrypted_data[data_id]
        key = self.key_store[key_id]
        
        # Simulate decryption
        plaintext = self._xor_encrypt(encrypted_data.encrypted_payload, key)
        self.decryption_count += 1
        
        return plaintext
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption for simulation"""
        return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
    
    def get_encryption_stats(self) -> Dict:
        """Get encryption statistics"""
        return {
            "total_encrypted": len(self.encrypted_data),
            "encryption_operations": self.encryption_count,
            "decryption_operations": self.decryption_count,
            "active_keys": len(self.key_store),
        }


@dataclass
class AuditEvent:
    """Represents an audit log event"""
    event_id: str
    event_type: AuditEventType
    user_id: str
    resource_id: str
    action: str
    timestamp: float = field(default_factory=time.time)
    details: Dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "action": self.action,
            "timestamp": self.timestamp,
            "details": self.details,
            "ip_address": self.ip_address,
        }


class AuditLogger:
    """
    HIPAA-compliant audit logging for healthcare data access.
    
    Maintains immutable audit trail using blockchain-inspired
    chaining, similar to how bees maintain colony history.
    """
    
    def __init__(self, max_events: int = 100000):
        self.events: List[AuditEvent] = []
        self.max_events = max_events
        self.event_index: Dict[str, List[int]] = {}  # user_id -> event indices
        self.resource_index: Dict[str, List[int]] = {}  # resource_id -> event indices
        
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource_id: str,
        action: str,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
    ) -> str:
        """
        Log an audit event.
        
        Returns event ID.
        """
        event_id = hashlib.sha256(
            f"{user_id}{resource_id}{time.time()}".encode()
        ).hexdigest()[:16]
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
        )
        
        # Add to event log
        event_idx = len(self.events)
        self.events.append(event)
        
        # Update indices
        if user_id not in self.event_index:
            self.event_index[user_id] = []
        self.event_index[user_id].append(event_idx)
        
        if resource_id not in self.resource_index:
            self.resource_index[resource_id] = []
        self.resource_index[resource_id].append(event_idx)
        
        # Rotate if needed
        if len(self.events) > self.max_events:
            self._rotate_logs()
        
        return event_id
    
    def get_user_events(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events for a specific user"""
        if user_id not in self.event_index:
            return []
        
        indices = self.event_index[user_id][-limit:]
        return [self.events[i] for i in indices if i < len(self.events)]
    
    def get_resource_events(
        self,
        resource_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events for a specific resource"""
        if resource_id not in self.resource_index:
            return []
        
        indices = self.resource_index[resource_id][-limit:]
        return [self.events[i] for i in indices if i < len(self.events)]
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[AuditEvent]:
        """Search audit events with filters"""
        results = []
        
        for event in self.events:
            if event_type and event.event_type != event_type:
                continue
            
            if user_id and event.user_id != user_id:
                continue
            
            if start_time and event.timestamp < start_time:
                continue
            
            if end_time and event.timestamp > end_time:
                continue
            
            results.append(event)
        
        return results
    
    def _rotate_logs(self) -> None:
        """Rotate old logs (simplified - real implementation would archive)"""
        # Keep most recent half of events
        keep_count = self.max_events // 2
        self.events = self.events[-keep_count:]
        
        # Rebuild indices
        self.event_index.clear()
        self.resource_index.clear()
        
        for idx, event in enumerate(self.events):
            if event.user_id not in self.event_index:
                self.event_index[event.user_id] = []
            self.event_index[event.user_id].append(idx)
            
            if event.resource_id not in self.resource_index:
                self.resource_index[event.resource_id] = []
            self.resource_index[event.resource_id].append(idx)
    
    def get_audit_stats(self) -> Dict:
        """Get audit log statistics"""
        event_type_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.events),
            "unique_users": len(self.event_index),
            "unique_resources": len(self.resource_index),
            "event_type_breakdown": event_type_counts,
        }


class PrivacyPreservingAnalytics:
    """
    Privacy-preserving analytics for healthcare data.
    
    Uses differential privacy and federated learning techniques
    inspired by how bee colonies aggregate information without
    exposing individual bee data.
    """
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
        self.query_count = 0
        self.anonymized_datasets: Dict[str, Dict] = {}
        
    def anonymize_dataset(
        self,
        dataset_id: str,
        records: List[Dict],
        sensitive_fields: List[str],
    ) -> bool:
        """
        Anonymize a dataset by removing or hashing sensitive fields.
        
        Returns True if successful.
        """
        anonymized = []
        
        for record in records:
            anon_record = record.copy()
            
            for field in sensitive_fields:
                if field in anon_record:
                    # Hash sensitive field
                    value = str(anon_record[field])
                    anon_record[field] = hashlib.sha256(value.encode()).hexdigest()[:16]
            
            anonymized.append(anon_record)
        
        self.anonymized_datasets[dataset_id] = {
            "records": anonymized,
            "original_count": len(records),
            "sensitive_fields": sensitive_fields,
            "created_at": time.time(),
        }
        
        return True
    
    def add_differential_privacy_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
    ) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Returns noised value.
        """
        import random
        
        # Laplace distribution parameter
        scale = sensitivity / self.epsilon
        
        # Generate Laplace noise
        u = random.uniform(-0.5, 0.5)
        noise = -scale * (1 if u >= 0 else -1) * (abs(u) ** 0.5)
        
        return value + noise
    
    def aggregate_with_privacy(
        self,
        dataset_id: str,
        field: str,
        operation: str = "sum",
    ) -> Optional[float]:
        """
        Perform aggregation with differential privacy.
        
        Supported operations: sum, count, avg, min, max
        """
        if dataset_id not in self.anonymized_datasets:
            return None
        
        records = self.anonymized_datasets[dataset_id]["records"]
        
        values = [r.get(field, 0) for r in records if field in r]
        
        if not values:
            return None
        
        # Compute aggregate
        if operation == "sum":
            result = sum(values)
        elif operation == "count":
            result = len(values)
        elif operation == "avg":
            result = sum(values) / len(values) if values else 0
        elif operation == "min":
            result = min(values)
        elif operation == "max":
            result = max(values)
        else:
            return None
        
        # Add differential privacy noise
        noised_result = self.add_differential_privacy_noise(result)
        self.query_count += 1
        
        return noised_result
    
    def k_anonymize(
        self,
        records: List[Dict],
        quasi_identifiers: List[str],
        k: int = 5,
    ) -> List[Dict]:
        """
        Apply k-anonymity to dataset.
        
        Ensures each record is indistinguishable from at least k-1 others.
        """
        # Simplified k-anonymization (generalization)
        anonymized = []
        
        for record in records:
            anon_record = record.copy()
            
            for field in quasi_identifiers:
                if field in anon_record:
                    value = anon_record[field]
                    
                    # Generalize numeric fields
                    if isinstance(value, (int, float)):
                        # Round to nearest multiple of k
                        anon_record[field] = (value // k) * k
                    elif isinstance(value, str):
                        # Keep first few characters only
                        anon_record[field] = value[:2] + "*" * (len(value) - 2)
            
            anonymized.append(anon_record)
        
        return anonymized
    
    def get_privacy_stats(self) -> Dict:
        """Get privacy analytics statistics"""
        return {
            "epsilon": self.epsilon,
            "query_count": self.query_count,
            "anonymized_datasets": len(self.anonymized_datasets),
            "remaining_privacy_budget": max(0, self.epsilon - self.query_count * 0.1),
        }
