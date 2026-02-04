"""
Satellite Integration - Support for satellite data links

Handles high-latency, intermittent satellite connections for truly
global data processing, inspired by how migratory bees maintain
colony cohesion over vast distances.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from collections import deque


class LinkQuality(Enum):
    """Satellite link quality levels"""
    EXCELLENT = "excellent"  # <100ms latency
    GOOD = "good"            # 100-500ms latency
    POOR = "poor"            # 500-1000ms latency
    DEGRADED = "degraded"    # >1000ms latency
    OFFLINE = "offline"


@dataclass
class SatelliteLink:
    """Represents a satellite communication link"""
    link_id: str
    satellite_name: str
    ground_station: str
    bandwidth_kbps: int = 1000
    latency_ms: float = 500
    packet_loss: float = 0.01  # 1% default
    quality: LinkQuality = LinkQuality.GOOD
    is_active: bool = True
    last_contact: float = field(default_factory=time.time)
    
    def update_quality(self) -> None:
        """Update link quality based on current metrics"""
        if not self.is_active:
            self.quality = LinkQuality.OFFLINE
        elif self.latency_ms < 100:
            self.quality = LinkQuality.EXCELLENT
        elif self.latency_ms < 500:
            self.quality = LinkQuality.GOOD
        elif self.latency_ms < 1000:
            self.quality = LinkQuality.POOR
        else:
            self.quality = LinkQuality.DEGRADED


class HighLatencyProtocol:
    """
    Network protocol optimized for high-latency satellite links.
    
    Uses techniques inspired by how bees communicate across large
    distances through persistent pheromone trails.
    """
    
    def __init__(self, max_retries: int = 5):
        self.links: Dict[str, SatelliteLink] = {}
        self.max_retries = max_retries
        self.message_queue: deque = deque()
        self.sent_messages: Dict[str, Dict] = {}  # message_id -> message info
        self.ack_timeout = 30.0  # seconds
        
    def register_link(
        self,
        link_id: str,
        satellite_name: str,
        ground_station: str,
        bandwidth_kbps: int = 1000,
    ) -> bool:
        """Register a new satellite link"""
        if link_id in self.links:
            return False
        
        link = SatelliteLink(
            link_id=link_id,
            satellite_name=satellite_name,
            ground_station=ground_station,
            bandwidth_kbps=bandwidth_kbps,
        )
        
        self.links[link_id] = link
        return True
    
    def send_message(
        self,
        link_id: str,
        message: Dict,
        priority: int = 5,
    ) -> Optional[str]:
        """
        Send a message over satellite link with automatic retry.
        
        Returns message ID if queued, None if link not found.
        """
        if link_id not in self.links:
            return None
        
        import hashlib
        message_id = hashlib.sha256(
            f"{link_id}{time.time()}{message}".encode()
        ).hexdigest()[:16]
        
        message_envelope = {
            "message_id": message_id,
            "link_id": link_id,
            "payload": message,
            "priority": priority,
            "timestamp": time.time(),
            "retries": 0,
        }
        
        self.message_queue.append(message_envelope)
        self.sent_messages[message_id] = message_envelope
        
        return message_id
    
    def process_queue(self) -> List[Dict]:
        """
        Process message queue, sending highest priority messages first.
        
        Returns list of messages actually sent.
        """
        if not self.message_queue:
            return []
        
        # Sort by priority (higher first)
        sorted_messages = sorted(
            self.message_queue,
            key=lambda m: m["priority"],
            reverse=True
        )
        
        sent = []
        
        for message in sorted_messages[:10]:  # Process up to 10 at a time
            link_id = message["link_id"]
            link = self.links.get(link_id)
            
            if not link or not link.is_active:
                continue
            
            # Simulate sending (real implementation would transmit)
            link.last_contact = time.time()
            message["sent_time"] = time.time()
            
            sent.append(message)
            self.message_queue.remove(message)
        
        return sent
    
    def acknowledge_message(self, message_id: str) -> bool:
        """Acknowledge receipt of a message"""
        if message_id in self.sent_messages:
            del self.sent_messages[message_id]
            return True
        return False
    
    def retry_failed_messages(self) -> int:
        """
        Retry messages that haven't been acknowledged.
        
        Returns number of messages re-queued.
        """
        current_time = time.time()
        retried = 0
        
        for message_id, message in list(self.sent_messages.items()):
            if "sent_time" not in message:
                continue
            
            time_since_sent = current_time - message["sent_time"]
            
            if time_since_sent > self.ack_timeout:
                message["retries"] += 1
                
                if message["retries"] < self.max_retries:
                    # Re-queue for retry
                    del message["sent_time"]
                    self.message_queue.append(message)
                    retried += 1
                else:
                    # Max retries exceeded, give up
                    del self.sent_messages[message_id]
        
        return retried
    
    def update_link_metrics(
        self,
        link_id: str,
        latency_ms: Optional[float] = None,
        packet_loss: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update satellite link metrics"""
        if link_id not in self.links:
            return False
        
        link = self.links[link_id]
        
        if latency_ms is not None:
            link.latency_ms = latency_ms
        
        if packet_loss is not None:
            link.packet_loss = max(0.0, min(1.0, packet_loss))
        
        if is_active is not None:
            link.is_active = is_active
        
        link.update_quality()
        return True
    
    def get_best_link(self) -> Optional[str]:
        """Get the best available satellite link"""
        active_links = [
            link for link in self.links.values()
            if link.is_active
        ]
        
        if not active_links:
            return None
        
        # Score based on quality and bandwidth
        quality_scores = {
            LinkQuality.EXCELLENT: 4,
            LinkQuality.GOOD: 3,
            LinkQuality.POOR: 2,
            LinkQuality.DEGRADED: 1,
            LinkQuality.OFFLINE: 0,
        }
        
        best_link = max(
            active_links,
            key=lambda l: quality_scores[l.quality] * l.bandwidth_kbps
        )
        
        return best_link.link_id
    
    def get_protocol_stats(self) -> Dict:
        """Get protocol statistics"""
        active_links = sum(1 for l in self.links.values() if l.is_active)
        avg_latency = (
            sum(l.latency_ms for l in self.links.values() if l.is_active)
            / active_links if active_links > 0 else 0
        )
        
        return {
            "total_links": len(self.links),
            "active_links": active_links,
            "queued_messages": len(self.message_queue),
            "pending_acks": len(self.sent_messages),
            "average_latency_ms": avg_latency,
        }


class BandwidthOptimizer:
    """
    Optimizes bandwidth usage for satellite links.
    
    Uses swarm intelligence to prioritize and compress data,
    similar to how bees optimize resource allocation.
    """
    
    def __init__(self, max_bandwidth_kbps: int = 1000):
        self.max_bandwidth_kbps = max_bandwidth_kbps
        self.current_usage_kbps = 0
        self.transfer_queue: List[Dict] = []
        self.compression_enabled = True
        
    def add_transfer(
        self,
        transfer_id: str,
        data_size_kb: int,
        priority: int = 5,
        compressible: bool = True,
    ) -> bool:
        """Add a data transfer to the queue"""
        # Apply compression if enabled
        actual_size = data_size_kb
        if self.compression_enabled and compressible:
            # Simulate compression (typical 30-70% reduction)
            import random
            compression_ratio = random.uniform(0.3, 0.7)
            actual_size = int(data_size_kb * compression_ratio)
        
        transfer = {
            "transfer_id": transfer_id,
            "original_size_kb": data_size_kb,
            "actual_size_kb": actual_size,
            "priority": priority,
            "compressible": compressible,
            "queued_time": time.time(),
        }
        
        self.transfer_queue.append(transfer)
        return True
    
    def schedule_transfers(self) -> List[Dict]:
        """
        Schedule transfers based on available bandwidth.
        
        Uses bee-inspired priority scheduling.
        """
        if not self.transfer_queue:
            return []
        
        # Sort by priority
        self.transfer_queue.sort(key=lambda t: t["priority"], reverse=True)
        
        scheduled = []
        available_bandwidth = self.max_bandwidth_kbps - self.current_usage_kbps
        
        for transfer in self.transfer_queue[:]:
            if available_bandwidth >= transfer["actual_size_kb"]:
                scheduled.append(transfer)
                available_bandwidth -= transfer["actual_size_kb"]
                self.transfer_queue.remove(transfer)
                self.current_usage_kbps += transfer["actual_size_kb"]
        
        return scheduled
    
    def complete_transfer(self, transfer_id: str) -> None:
        """Mark a transfer as complete and free up bandwidth"""
        for transfer in self.transfer_queue:
            if transfer["transfer_id"] == transfer_id:
                self.current_usage_kbps = max(
                    0,
                    self.current_usage_kbps - transfer["actual_size_kb"]
                )
                break
    
    def get_bandwidth_stats(self) -> Dict:
        """Get bandwidth usage statistics"""
        utilization = (
            self.current_usage_kbps / self.max_bandwidth_kbps
            if self.max_bandwidth_kbps > 0 else 0
        )
        
        return {
            "max_bandwidth_kbps": self.max_bandwidth_kbps,
            "current_usage_kbps": self.current_usage_kbps,
            "utilization": utilization,
            "queued_transfers": len(self.transfer_queue),
        }


class DataBufferingStrategy:
    """
    Manages data buffering for intermittent satellite connections.
    
    Buffers data during outages and efficiently transmits when
    connection is restored, like bees storing honey for lean times.
    """
    
    def __init__(self, max_buffer_mb: int = 1000):
        self.max_buffer_mb = max_buffer_mb
        self.current_buffer_mb = 0
        self.buffer: deque = deque()
        self.overflow_count = 0
        
    def buffer_data(
        self,
        data_id: str,
        data_size_mb: float,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Buffer data for later transmission.
        
        Returns True if buffered, False if buffer is full.
        """
        if self.current_buffer_mb + data_size_mb > self.max_buffer_mb:
            self.overflow_count += 1
            return False
        
        item = {
            "data_id": data_id,
            "size_mb": data_size_mb,
            "metadata": metadata or {},
            "buffered_time": time.time(),
        }
        
        self.buffer.append(item)
        self.current_buffer_mb += data_size_mb
        return True
    
    def get_next_batch(self, max_batch_size_mb: float) -> List[Dict]:
        """
        Get next batch of data to transmit.
        
        Returns batch that fits within size limit.
        """
        if not self.buffer:
            return []
        
        batch = []
        batch_size = 0
        
        while self.buffer and batch_size < max_batch_size_mb:
            item = self.buffer[0]
            if batch_size + item["size_mb"] <= max_batch_size_mb:
                batch.append(self.buffer.popleft())
                batch_size += item["size_mb"]
                self.current_buffer_mb -= item["size_mb"]
            else:
                break
        
        return batch
    
    def clear_old_data(self, max_age_seconds: float = 86400) -> int:
        """
        Clear data older than max age.
        
        Returns number of items cleared.
        """
        current_time = time.time()
        cleared = 0
        
        while self.buffer:
            item = self.buffer[0]
            age = current_time - item["buffered_time"]
            
            if age > max_age_seconds:
                self.buffer.popleft()
                self.current_buffer_mb -= item["size_mb"]
                cleared += 1
            else:
                break  # Buffer is ordered by time
        
        return cleared
    
    def get_buffer_stats(self) -> Dict:
        """Get buffer statistics"""
        utilization = (
            self.current_buffer_mb / self.max_buffer_mb
            if self.max_buffer_mb > 0 else 0
        )
        
        return {
            "max_buffer_mb": self.max_buffer_mb,
            "current_buffer_mb": self.current_buffer_mb,
            "utilization": utilization,
            "buffered_items": len(self.buffer),
            "overflow_count": self.overflow_count,
        }
