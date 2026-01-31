"""
Feature Hive - Centralized Feature Store
========================================

Feature store with automatic feature engineering, versioning, and lineage.
Features are organized like honeycomb cells with metadata stored as nectar.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum


class FeatureType(Enum):
    """Feature data types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    STRUCT = "struct"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    feature_type: FeatureType
    description: Optional[str] = None
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    compute_fn: Optional[Callable] = None


@dataclass
class FeatureGroup:
    """A group of related features."""
    name: str
    features: List[str]
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class FeatureHive:
    """
    Feature Hive - Centralized feature store.
    
    Stores, manages, and serves features for machine learning models.
    Features are cached and computed on-demand using a bee-inspired
    caching strategy.
    
    Example:
        feature_store = FeatureHive()
        
        # Register a feature with computation function
        def compute_user_activity(user_id):
            # Compute user activity score
            return calculate_activity(user_id)
        
        feature_store.register_feature(
            name="user_activity_7d",
            feature_type=FeatureType.FLOAT,
            compute_fn=compute_user_activity,
            description="User activity score over 7 days"
        )
        
        # Get feature values
        features = feature_store.get_features(
            ["user_activity_7d"],
            entity_ids=["user_123", "user_456"]
        )
        
        # Create feature group
        feature_store.create_feature_group(
            name="user_features",
            features=["user_activity_7d", "user_age", "user_location"]
        )
    """
    
    def __init__(self):
        self._features: Dict[str, FeatureMetadata] = {}
        self._feature_groups: Dict[str, FeatureGroup] = {}
        self._feature_cache: Dict[str, Dict[Any, Any]] = {}
    
    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        compute_fn: Optional[Callable] = None,
        description: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[List[str]] = None
    ) -> FeatureMetadata:
        """
        Register a new feature in the store.
        
        Args:
            name: Feature name
            feature_type: Feature data type
            compute_fn: Optional function to compute feature value
            description: Human-readable description
            owner: Feature owner identifier
            tags: Tags for categorization
            dependencies: List of dependent feature names
            
        Returns:
            Created FeatureMetadata
        """
        # Check if updating existing feature
        version = 1
        if name in self._features:
            version = self._features[name].version + 1
        
        metadata = FeatureMetadata(
            name=name,
            feature_type=feature_type,
            description=description,
            version=version,
            owner=owner,
            tags=tags or set(),
            dependencies=dependencies or [],
            compute_fn=compute_fn
        )
        
        self._features[name] = metadata
        self._feature_cache[name] = {}
        
        return metadata
    
    def get_feature_metadata(self, name: str) -> Optional[FeatureMetadata]:
        """Get feature metadata by name."""
        return self._features.get(name)
    
    def list_features(self, tags: Optional[Set[str]] = None) -> List[str]:
        """
        List all feature names, optionally filtered by tags.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            List of feature names
        """
        features = list(self._features.keys())
        
        if tags:
            features = [
                name for name in features
                if self._features[name].tags & tags
            ]
        
        return sorted(features)
    
    def get_features(
        self,
        feature_names: List[str],
        entity_ids: List[Any],
        use_cache: bool = True
    ) -> Dict[str, List[Any]]:
        """
        Get feature values for entities.
        
        Args:
            feature_names: List of feature names to retrieve
            entity_ids: List of entity identifiers
            use_cache: Whether to use cached values
            
        Returns:
            Dictionary mapping feature_name -> list of values
        """
        results = {fname: [] for fname in feature_names}
        
        for feature_name in feature_names:
            if feature_name not in self._features:
                raise ValueError(f"Feature '{feature_name}' not found")
            
            metadata = self._features[feature_name]
            cache = self._feature_cache[feature_name]
            
            for entity_id in entity_ids:
                # Check cache first
                if use_cache and entity_id in cache:
                    value = cache[entity_id]
                elif metadata.compute_fn:
                    # Compute feature value
                    value = metadata.compute_fn(entity_id)
                    # Cache the result
                    cache[entity_id] = value
                else:
                    # No compute function and not in cache
                    value = None
                
                results[feature_name].append(value)
        
        return results
    
    def set_feature_values(
        self,
        feature_name: str,
        entity_values: Dict[Any, Any]
    ) -> None:
        """
        Manually set feature values (for materialized features).
        
        Args:
            feature_name: Feature name
            entity_values: Dictionary mapping entity_id -> feature_value
        """
        if feature_name not in self._features:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        cache = self._feature_cache[feature_name]
        cache.update(entity_values)
    
    def create_feature_group(
        self,
        name: str,
        features: List[str],
        description: Optional[str] = None
    ) -> FeatureGroup:
        """
        Create a feature group.
        
        Args:
            name: Group name
            features: List of feature names
            description: Optional description
            
        Returns:
            Created FeatureGroup
        """
        # Validate that all features exist
        for feature_name in features:
            if feature_name not in self._features:
                raise ValueError(f"Feature '{feature_name}' not found")
        
        group = FeatureGroup(
            name=name,
            features=features,
            description=description
        )
        
        self._feature_groups[name] = group
        return group
    
    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Get feature group by name."""
        return self._feature_groups.get(name)
    
    def list_feature_groups(self) -> List[str]:
        """List all feature group names."""
        return sorted(list(self._feature_groups.keys()))
    
    def get_group_features(
        self,
        group_name: str,
        entity_ids: List[Any],
        use_cache: bool = True
    ) -> Dict[str, List[Any]]:
        """
        Get all features in a group for entities.
        
        Args:
            group_name: Feature group name
            entity_ids: List of entity identifiers
            use_cache: Whether to use cached values
            
        Returns:
            Dictionary mapping feature_name -> list of values
        """
        if group_name not in self._feature_groups:
            raise ValueError(f"Feature group '{group_name}' not found")
        
        group = self._feature_groups[group_name]
        return self.get_features(group.features, entity_ids, use_cache)
    
    def invalidate_cache(
        self,
        feature_name: Optional[str] = None,
        entity_id: Optional[Any] = None
    ) -> None:
        """
        Invalidate feature cache.
        
        Args:
            feature_name: Optional feature to invalidate (all if None)
            entity_id: Optional specific entity to invalidate (all if None)
        """
        if feature_name:
            if feature_name in self._feature_cache:
                if entity_id is not None:
                    self._feature_cache[feature_name].pop(entity_id, None)
                else:
                    self._feature_cache[feature_name].clear()
        else:
            # Invalidate all caches
            for cache in self._feature_cache.values():
                if entity_id is not None:
                    cache.pop(entity_id, None)
                else:
                    cache.clear()
    
    def get_feature_lineage(self, feature_name: str) -> List[str]:
        """
        Get dependency lineage for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            List of dependent feature names (transitive)
        """
        if feature_name not in self._features:
            return []
        
        visited = set()
        stack = [feature_name]
        lineage = []
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            lineage.append(current)
            
            # Add dependencies
            if current in self._features:
                deps = self._features[current].dependencies
                stack.extend(d for d in deps if d not in visited)
        
        return lineage[1:]  # Exclude the query feature itself
