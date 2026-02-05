"""
HiveFrame Marketplace - Third-party apps and connectors

Plugin system, app registry, and version management for
building a thriving ecosystem of extensions.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
import hashlib


class PluginType(Enum):
    """Types of plugins"""

    CONNECTOR = "connector"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    VISUALIZER = "visualizer"
    INTEGRATION = "integration"


class PluginStatus(Enum):
    """Plugin lifecycle status"""

    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass
class Plugin:
    """Represents a HiveFrame plugin"""

    plugin_id: str
    name: str
    version: str
    plugin_type: PluginType
    author: str
    description: str = ""
    status: PluginStatus = PluginStatus.INSTALLED
    dependencies: List[str] = field(default_factory=list)
    entry_point: Optional[Callable] = None
    metadata: Dict = field(default_factory=dict)
    installed_at: float = field(default_factory=time.time)


class PluginSystem:
    """
    Plugin system architecture for HiveFrame extensions.

    Uses swarm-based plugin coordination where plugins can
    communicate like bees in a colony.
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
        self.execution_count: Dict[str, int] = {}

    def register_plugin(
        self,
        plugin: Plugin,
    ) -> bool:
        """
        Register a new plugin.

        Returns True if successful, False otherwise.
        """
        if plugin.plugin_id in self.plugins:
            return False

        # Check dependencies
        for dep_id in plugin.dependencies:
            if dep_id not in self.plugins:
                return False

            dep = self.plugins[dep_id]
            if dep.status != PluginStatus.ACTIVE:
                return False

        self.plugins[plugin.plugin_id] = plugin
        self.execution_count[plugin.plugin_id] = 0

        return True

    def activate_plugin(self, plugin_id: str) -> bool:
        """Activate a plugin"""
        if plugin_id not in self.plugins:
            return False

        plugin = self.plugins[plugin_id]

        # Check dependencies are active
        for dep_id in plugin.dependencies:
            if dep_id not in self.plugins:
                return False

            dep = self.plugins[dep_id]
            if dep.status != PluginStatus.ACTIVE:
                return False

        plugin.status = PluginStatus.ACTIVE
        return True

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate a plugin"""
        if plugin_id not in self.plugins:
            return False

        # Check if other plugins depend on this
        for other_plugin in self.plugins.values():
            if plugin_id in other_plugin.dependencies:
                if other_plugin.status == PluginStatus.ACTIVE:
                    return False  # Cannot deactivate - dependency active

        self.plugins[plugin_id].status = PluginStatus.INACTIVE
        return True

    def register_hook(
        self,
        hook_name: str,
        plugin_id: str,
        callback: Callable,
    ) -> bool:
        """Register a plugin hook for event-driven execution"""
        if plugin_id not in self.plugins:
            return False

        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []

        self.plugin_hooks[hook_name].append(callback)
        return True

    def execute_hook(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Execute all callbacks registered for a hook.

        Returns list of results from callbacks.
        """
        if hook_name not in self.plugin_hooks:
            return []

        results = []
        for callback in self.plugin_hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        return results

    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin"""
        if plugin_id not in self.plugins:
            return False

        # Must be inactive first
        plugin = self.plugins[plugin_id]
        if plugin.status == PluginStatus.ACTIVE:
            if not self.deactivate_plugin(plugin_id):
                return False

        del self.plugins[plugin_id]
        del self.execution_count[plugin_id]

        return True

    def get_plugin_stats(self) -> Dict:
        """Get plugin system statistics"""
        active_plugins = sum(1 for p in self.plugins.values() if p.status == PluginStatus.ACTIVE)

        by_type = {}
        for plugin in self.plugins.values():
            plugin_type = plugin.plugin_type.value
            by_type[plugin_type] = by_type.get(plugin_type, 0) + 1

        return {
            "total_plugins": len(self.plugins),
            "active_plugins": active_plugins,
            "registered_hooks": len(self.plugin_hooks),
            "plugins_by_type": by_type,
        }


@dataclass
class App:
    """Represents a marketplace app"""

    app_id: str
    name: str
    version: str
    author: str
    description: str
    category: str
    download_count: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    published_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class AppRegistry:
    """
    Registry for third-party apps and connectors.

    Maintains a swarm-curated marketplace where apps are
    ranked by community engagement like bee waggle dances.
    """

    def __init__(self):
        self.apps: Dict[str, App] = {}
        self.categories: Set[str] = set()
        self.reviews: Dict[str, List[Dict]] = {}  # app_id -> reviews

    def publish_app(
        self,
        app: App,
    ) -> bool:
        """
        Publish an app to the registry.

        Returns True if successful.
        """
        if app.app_id in self.apps:
            return False

        self.apps[app.app_id] = app
        self.categories.add(app.category)
        self.reviews[app.app_id] = []

        return True

    def update_app(
        self,
        app_id: str,
        version: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update an app in the registry"""
        if app_id not in self.apps:
            return False

        app = self.apps[app_id]
        app.version = version

        if metadata:
            app.metadata.update(metadata)

        return True

    def download_app(self, app_id: str) -> Optional[App]:
        """
        Download an app (increment download count).

        Returns app or None if not found.
        """
        if app_id not in self.apps:
            return None

        app = self.apps[app_id]
        app.download_count += 1

        return app

    def add_review(
        self,
        app_id: str,
        user_id: str,
        rating: int,
        comment: str = "",
    ) -> bool:
        """Add a review for an app"""
        if app_id not in self.apps:
            return False

        if rating < 1 or rating > 5:
            return False

        review = {
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time(),
        }

        self.reviews[app_id].append(review)

        # Update app rating
        app = self.apps[app_id]
        reviews = self.reviews[app_id]
        app.rating = sum(r["rating"] for r in reviews) / len(reviews)
        app.reviews_count = len(reviews)

        return True

    def search_apps(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_rating: float = 0.0,
    ) -> List[App]:
        """
        Search for apps with filters.

        Returns list of matching apps.
        """
        results = []

        for app in self.apps.values():
            # Apply filters
            if (
                query
                and query.lower() not in app.name.lower()
                and query.lower() not in app.description.lower()
            ):
                continue

            if category and app.category != category:
                continue

            if app.rating < min_rating:
                continue

            results.append(app)

        # Sort by popularity (downloads * rating)
        results.sort(key=lambda a: a.download_count * (a.rating or 1.0), reverse=True)

        return results

    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        total_downloads = sum(a.download_count for a in self.apps.values())
        avg_rating = (
            sum(a.rating for a in self.apps.values() if a.rating > 0)
            / len([a for a in self.apps.values() if a.rating > 0])
            if any(a.rating > 0 for a in self.apps.values())
            else 0
        )

        return {
            "total_apps": len(self.apps),
            "total_categories": len(self.categories),
            "total_downloads": total_downloads,
            "average_rating": avg_rating,
        }


class VersionManager:
    """
    Manage plugin and app versions with compatibility tracking.

    Uses swarm consensus to determine version compatibility,
    similar to how bee colonies reach agreement.
    """

    def __init__(self):
        self.versions: Dict[str, List[str]] = {}  # plugin_id -> [versions]
        self.compatibility: Dict[str, Dict[str, bool]] = {}  # plugin_id -> {version -> compatible}
        self.deprecations: Dict[str, Set[str]] = {}  # plugin_id -> {deprecated versions}

    def register_version(
        self,
        plugin_id: str,
        version: str,
    ) -> bool:
        """Register a new version of a plugin"""
        if plugin_id not in self.versions:
            self.versions[plugin_id] = []
            self.compatibility[plugin_id] = {}

        if version in self.versions[plugin_id]:
            return False

        self.versions[plugin_id].append(version)
        self.compatibility[plugin_id][version] = True  # Compatible by default

        return True

    def check_compatibility(
        self,
        plugin_id: str,
        version: str,
    ) -> bool:
        """
        Check if a plugin version is compatible.

        Returns True if compatible, False otherwise.
        """
        if plugin_id not in self.compatibility:
            return False

        return self.compatibility[plugin_id].get(version, False)

    def mark_deprecated(
        self,
        plugin_id: str,
        version: str,
    ) -> bool:
        """Mark a version as deprecated"""
        if plugin_id not in self.versions:
            return False

        if version not in self.versions[plugin_id]:
            return False

        if plugin_id not in self.deprecations:
            self.deprecations[plugin_id] = set()

        self.deprecations[plugin_id].add(version)
        return True

    def get_latest_version(self, plugin_id: str) -> Optional[str]:
        """Get the latest compatible version of a plugin"""
        if plugin_id not in self.versions:
            return None

        versions = self.versions[plugin_id]

        # Filter out deprecated
        deprecated = self.deprecations.get(plugin_id, set())
        active_versions = [v for v in versions if v not in deprecated]

        if not active_versions:
            return None

        # Simple version sorting (real implementation would use semver)
        return active_versions[-1]

    def get_version_stats(self) -> Dict:
        """Get version management statistics"""
        total_versions = sum(len(v) for v in self.versions.values())
        total_deprecated = sum(len(d) for d in self.deprecations.values())

        return {
            "total_plugins": len(self.versions),
            "total_versions": total_versions,
            "deprecated_versions": total_deprecated,
        }
