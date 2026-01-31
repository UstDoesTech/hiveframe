"""
Tests for Phase 3 Lakehouse features (Unity Hive Catalog, Delta Sharing)
"""

from hiveframe.lakehouse import (
    AccessControl,
    DeltaSharing,
    LineageTracker,
    PermissionType,
    PIIDetector,
    ShareAccessLevel,
    UnityHiveCatalog,
)


class TestUnityHiveCatalog:
    """Test Unity Hive Catalog functionality."""

    def test_register_table(self):
        """Test registering a table."""
        catalog = UnityHiveCatalog()

        metadata = catalog.register_table(
            name="users",
            schema={"id": "int", "name": "string", "email": "string"},
            location="/data/users",
            format="parquet",
            owner="data_team",
        )

        assert metadata.name == "users"
        assert metadata.location == "/data/users"
        assert metadata.format == "parquet"
        assert metadata.owner == "data_team"
        assert len(metadata.schema) == 3

    def test_get_table(self):
        """Test retrieving table metadata."""
        catalog = UnityHiveCatalog()
        catalog.register_table(name="users", schema={"id": "int"}, location="/data/users")

        table = catalog.get_table("users")
        assert table is not None
        assert table.name == "users"

        # Non-existent table
        table = catalog.get_table("non_existent")
        assert table is None

    def test_list_tables(self):
        """Test listing tables."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")
        catalog.register_table("orders", {"id": "int"}, "/data/orders")
        catalog.register_table("products", {"id": "int"}, "/data/products")

        tables = catalog.list_tables()
        assert len(tables) == 3
        assert "users" in tables
        assert "orders" in tables

        # Pattern filtering
        user_tables = catalog.list_tables(pattern="user.*")
        assert len(user_tables) == 1
        assert "users" in user_tables

    def test_search_by_tags(self):
        """Test searching tables by tags."""
        catalog = UnityHiveCatalog()
        catalog.register_table(
            "users", {"id": "int"}, "/data/users", tags={"production", "user_data"}
        )
        catalog.register_table(
            "orders", {"id": "int"}, "/data/orders", tags={"production", "order_data"}
        )
        catalog.register_table("test_data", {"id": "int"}, "/data/test", tags={"test"})

        # Search for production tables
        prod_tables = catalog.search_by_tags({"production"})
        assert len(prod_tables) == 2

        # Search for user data
        user_tables = catalog.search_by_tags({"user_data"})
        assert len(user_tables) == 1

    def test_update_table(self):
        """Test updating table metadata."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")

        catalog.update_table("users", description="User information table")

        table = catalog.get_table("users")
        assert table.description == "User information table"

    def test_drop_table(self):
        """Test dropping a table."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")

        result = catalog.drop_table("users")
        assert result is True

        table = catalog.get_table("users")
        assert table is None


class TestAccessControl:
    """Test Access Control functionality."""

    def test_grant_permission(self):
        """Test granting permissions."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")

        acl = AccessControl(catalog)
        acl.grant("user@example.com", "users", [PermissionType.SELECT])

        has_permission = acl.check_permission("user@example.com", "users", PermissionType.SELECT)
        assert has_permission is True

    def test_revoke_permission(self):
        """Test revoking permissions."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")

        acl = AccessControl(catalog)
        acl.grant("user@example.com", "users", [PermissionType.SELECT, PermissionType.INSERT])
        acl.revoke("user@example.com", "users", [PermissionType.INSERT])

        has_select = acl.check_permission("user@example.com", "users", PermissionType.SELECT)
        has_insert = acl.check_permission("user@example.com", "users", PermissionType.INSERT)

        assert has_select is True
        assert has_insert is False

    def test_all_permission(self):
        """Test ALL permission grants all access."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")

        acl = AccessControl(catalog)
        acl.grant("admin@example.com", "users", [PermissionType.ALL])

        # Should have all permissions
        assert acl.check_permission("admin@example.com", "users", PermissionType.SELECT)
        assert acl.check_permission("admin@example.com", "users", PermissionType.INSERT)
        assert acl.check_permission("admin@example.com", "users", PermissionType.DELETE)

    def test_list_permissions(self):
        """Test listing user permissions."""
        catalog = UnityHiveCatalog()
        catalog.register_table("users", {"id": "int"}, "/data/users")
        catalog.register_table("orders", {"id": "int"}, "/data/orders")

        acl = AccessControl(catalog)
        acl.grant("user@example.com", "users", [PermissionType.SELECT])
        acl.grant("user@example.com", "orders", [PermissionType.SELECT, PermissionType.INSERT])

        permissions = acl.list_permissions("user@example.com")
        assert len(permissions) == 2
        assert "users" in permissions
        assert "orders" in permissions


class TestLineageTracker:
    """Test data lineage tracking."""

    def test_record_lineage(self):
        """Test recording lineage."""
        catalog = UnityHiveCatalog()
        tracker = LineageTracker(catalog)

        node = tracker.record_lineage(
            output_table="user_summary", input_tables=["users", "orders"], operation="join"
        )

        assert node.table == "user_summary"
        assert node.operation == "join"
        assert len(node.dependencies) == 2

    def test_get_upstream(self):
        """Test getting upstream dependencies."""
        catalog = UnityHiveCatalog()
        tracker = LineageTracker(catalog)

        tracker.record_lineage("summary", ["users", "orders"], "join")
        tracker.record_lineage("report", ["summary", "products"], "join")

        # Direct dependencies
        upstream = tracker.get_upstream("summary", recursive=False)
        assert len(upstream) == 2
        assert "users" in upstream
        assert "orders" in upstream

        # Transitive dependencies
        upstream_recursive = tracker.get_upstream("report", recursive=True)
        assert len(upstream_recursive) == 4  # summary, products, users, orders

    def test_get_downstream(self):
        """Test getting downstream consumers."""
        catalog = UnityHiveCatalog()
        tracker = LineageTracker(catalog)

        tracker.record_lineage("summary", ["users"], "aggregate")
        tracker.record_lineage("report1", ["summary"], "filter")
        tracker.record_lineage("report2", ["summary"], "select")

        downstream = tracker.get_downstream("summary", recursive=False)
        assert len(downstream) == 2
        assert "report1" in downstream
        assert "report2" in downstream

    def test_lineage_path(self):
        """Test finding lineage path between tables."""
        catalog = UnityHiveCatalog()
        tracker = LineageTracker(catalog)

        tracker.record_lineage("b", ["a"], "transform")
        tracker.record_lineage("c", ["b"], "aggregate")
        tracker.record_lineage("d", ["c"], "filter")

        path = tracker.get_lineage_path("a", "d")
        assert path is not None
        assert path == ["a", "b", "c", "d"]

        # No path exists
        path = tracker.get_lineage_path("d", "a")
        assert path is None


class TestPIIDetector:
    """Test PII detection."""

    def test_detect_email(self):
        """Test detecting email columns."""
        detector = PIIDetector()
        schema = {"user_email": "string", "age": "int"}

        pii_columns = detector.detect_pii(schema)
        assert "user_email" in pii_columns
        assert "age" not in pii_columns

    def test_detect_phone(self):
        """Test detecting phone columns."""
        detector = PIIDetector()
        schema = {"phone_number": "string", "name": "string"}

        pii_columns = detector.detect_pii(schema)
        assert "phone_number" in pii_columns

    def test_detect_ssn(self):
        """Test detecting SSN columns."""
        detector = PIIDetector()
        schema = {"ssn": "string", "id": "int"}

        pii_columns = detector.detect_pii(schema)
        assert "ssn" in pii_columns
        assert "id" not in pii_columns

    def test_detect_multiple_pii(self):
        """Test detecting multiple PII columns."""
        detector = PIIDetector()
        schema = {
            "name": "string",
            "email": "string",
            "phone": "string",
            "address": "string",
            "age": "int",
        }

        pii_columns = detector.detect_pii(schema)
        assert len(pii_columns) >= 4
        assert "email" in pii_columns
        assert "phone" in pii_columns
        assert "age" not in pii_columns


class TestDeltaSharing:
    """Test Delta Sharing functionality."""

    def test_create_share(self):
        """Test creating a share."""
        sharing = DeltaSharing()

        share = sharing.create_share(
            name="customer_data", owner="data_team@company.com", tables={"customers", "orders"}
        )

        assert share.name == "customer_data"
        assert share.owner == "data_team@company.com"
        assert len(share.tables) == 2

    def test_add_recipient(self):
        """Test adding a recipient to a share."""
        sharing = DeltaSharing()
        share = sharing.create_share("data", "owner@example.com")

        recipient = sharing.add_recipient(
            share_id=share.share_id,
            email="partner@external.com",
            access_level=ShareAccessLevel.READ,
            expires_in_days=90,
        )

        assert recipient.email == "partner@external.com"
        assert recipient.access_level == ShareAccessLevel.READ
        assert recipient.token is not None
        assert recipient.expires_at is not None

    def test_validate_token(self):
        """Test token validation."""
        sharing = DeltaSharing()
        share = sharing.create_share("data", "owner@example.com")
        recipient = sharing.add_recipient(share.share_id, "user@example.com")

        # Valid token
        is_valid = sharing.validate_token(recipient.token)
        assert is_valid is True

        # Invalid token
        is_valid = sharing.validate_token("invalid_token")
        assert is_valid is False

    def test_revoke_token(self):
        """Test token revocation."""
        sharing = DeltaSharing()
        share = sharing.create_share("data", "owner@example.com")
        recipient = sharing.add_recipient(share.share_id, "user@example.com")

        # Revoke token
        result = sharing.revoke_token(recipient.token)
        assert result is True

        # Token should no longer be valid
        is_valid = sharing.validate_token(recipient.token)
        assert is_valid is False

    def test_list_shared_tables(self):
        """Test listing shared tables."""
        sharing = DeltaSharing()
        share = sharing.create_share(
            "data", "owner@example.com", tables={"customers", "orders", "products"}
        )
        recipient = sharing.add_recipient(share.share_id, "user@example.com")

        tables = sharing.list_shared_tables(recipient.token)
        assert len(tables) == 3
        assert "customers" in tables
        assert "orders" in tables
        assert "products" in tables

    def test_add_remove_table_from_share(self):
        """Test adding and removing tables from a share."""
        sharing = DeltaSharing()
        share = sharing.create_share("data", "owner@example.com", tables={"customers"})

        # Add table
        sharing.add_table_to_share(share.share_id, "orders")
        updated_share = sharing.get_share(share.share_id)
        assert len(updated_share.tables) == 2
        assert "orders" in updated_share.tables

        # Remove table
        sharing.remove_table_from_share(share.share_id, "customers")
        updated_share = sharing.get_share(share.share_id)
        assert len(updated_share.tables) == 1
        assert "customers" not in updated_share.tables

    def test_delete_share(self):
        """Test deleting a share."""
        sharing = DeltaSharing()
        share = sharing.create_share("data", "owner@example.com")
        recipient = sharing.add_recipient(share.share_id, "user@example.com")

        # Delete share
        result = sharing.delete_share(share.share_id)
        assert result is True

        # Share should be gone
        deleted_share = sharing.get_share(share.share_id)
        assert deleted_share is None

        # Token should be revoked
        is_valid = sharing.validate_token(recipient.token)
        assert is_valid is False
