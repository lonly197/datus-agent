# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Schema models for v0 to v1 migration.

This module defines the data models and schemas used for migrating
LanceDB schema metadata from v0 (legacy) to v1 (enhanced) format.

V0 Schema:
    - Basic table metadata (name, type, definition)
    - No enhanced metadata

V1 Schema:
    - Enhanced table metadata (comments, enums, tags)
    - Column statistics
    - Relationship metadata
    - Business tags
    - Metadata version tracking
"""

from typing import Any, Dict, List, Optional
import pyarrow as pa


# V0 Schema Fields (Legacy)
V0_SCHEMA_FIELDS = [
    "identifier",
    "catalog_name",
    "database_name",
    "schema_name",
    "table_name",
    "table_type",
    "definition",
]

# V1 Schema Fields (Enhanced)
V1_SCHEMA_FIELDS = [
    # Core fields (from v0)
    "identifier",
    "catalog_name",
    "database_name",
    "schema_name",
    "table_name",
    "table_type",
    "definition",
    # Enhanced metadata fields (v1 additions)
    "table_comment",
    "column_comments",
    "column_enums",
    "business_tags",
    "row_count",
    "sample_statistics",
    "relationship_metadata",
    "metadata_version",
    "last_updated",
]

# V1 Metadata version constants
METADATA_VERSION_V0 = 0
METADATA_VERSION_V1 = 1


class MigrationRecord:
    """
    Represents a single migration record.

    This class encapsulates the data structure used during migration
    from v0 to v1 format, including both legacy and enhanced metadata.
    """

    def __init__(
        self,
        identifier: str,
        catalog_name: str,
        database_name: str,
        schema_name: str,
        table_name: str,
        table_type: str,
        definition: str,
        table_comment: str = "",
        column_comments: Optional[Dict[str, str]] = None,
        column_enums: Optional[Dict[str, List[Dict[str, str]]]] = None,
        business_tags: Optional[List[str]] = None,
        row_count: int = 0,
        sample_statistics: Optional[Dict[str, Dict[str, Any]]] = None,
        relationship_metadata: Optional[Dict[str, Any]] = None,
        metadata_version: int = METADATA_VERSION_V1,
        last_updated: Optional[int] = None,
    ):
        """Initialize a migration record.

        Args:
            identifier: Unique identifier for the table
            catalog_name: Catalog name
            database_name: Database name
            schema_name: Schema name
            table_name: Table name
            table_type: Table type (table, view, mv)
            definition: DDL definition
            table_comment: Table comment (v1)
            column_comments: Column comments dict (v1)
            column_enums: Column enum values (v1)
            business_tags: Business category tags (v1)
            row_count: Table row count (v1)
            sample_statistics: Column statistics (v1)
            relationship_metadata: Foreign key relationships (v1)
            metadata_version: Schema version (0=v0, 1=v1)
            last_updated: Unix timestamp of last update
        """
        import time

        self.identifier = identifier
        self.catalog_name = catalog_name
        self.database_name = database_name
        self.schema_name = schema_name
        self.table_name = table_name
        self.table_type = table_type
        self.definition = definition
        self.table_comment = table_comment or ""
        self.column_comments = column_comments or {}
        self.column_enums = column_enums or {}
        self.business_tags = business_tags or []
        self.row_count = row_count or 0
        self.sample_statistics = sample_statistics or {}
        self.relationship_metadata = relationship_metadata or {}
        self.metadata_version = metadata_version
        self.last_updated = last_updated or int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for storage.

        Returns:
            Dictionary representation of the record
        """
        import json

        return {
            "identifier": self.identifier,
            "catalog_name": self.catalog_name,
            "database_name": self.database_name,
            "schema_name": self.schema_name,
            "table_name": self.table_name,
            "table_type": self.table_type,
            "definition": self.definition,
            "table_comment": self.table_comment,
            "column_comments": json.dumps(self.column_comments, ensure_ascii=False),
            "column_enums": json.dumps(self.column_enums, ensure_ascii=False),
            "business_tags": self.business_tags,
            "row_count": self.row_count,
            "sample_statistics": json.dumps(self.sample_statistics, ensure_ascii=False),
            "relationship_metadata": json.dumps(self.relationship_metadata, ensure_ascii=False),
            "metadata_version": self.metadata_version,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_v0_record(
        cls,
        v0_record: Dict[str, Any],
        enhanced_metadata: Optional[Dict[str, Any]] = None,
    ) -> "MigrationRecord":
        """Create a v1 record from a v0 record.

        Args:
            v0_record: Legacy v0 record dictionary
            enhanced_metadata: Optional enhanced metadata to add

        Returns:
            New MigrationRecord with v1 data
        """
        import json

        enhanced = enhanced_metadata or {}

        return cls(
            identifier=v0_record.get("identifier", ""),
            catalog_name=v0_record.get("catalog_name", ""),
            database_name=v0_record.get("database_name", ""),
            schema_name=v0_record.get("schema_name", ""),
            table_name=v0_record.get("table_name", ""),
            table_type=v0_record.get("table_type", ""),
            definition=v0_record.get("definition", ""),
            table_comment=enhanced.get("table_comment", ""),
            column_comments=enhanced.get("column_comments", {}),
            column_enums=enhanced.get("column_enums", {}),
            business_tags=enhanced.get("business_tags", []),
            row_count=enhanced.get("row_count", 0),
            sample_statistics=enhanced.get("sample_statistics", {}),
            relationship_metadata=enhanced.get("relationship_metadata", {}),
            metadata_version=METADATA_VERSION_V1,
            last_updated=int(enhanced.get("last_updated", 0)),
        )

    def __repr__(self) -> str:
        """String representation of the record."""
        return (
            f"MigrationRecord("
            f"identifier={self.identifier}, "
            f"table={self.table_name}, "
            f"version=v{self.metadata_version}"
            f")"
        )


class MigrationResult:
    """
    Represents the result of a migration operation.

    Tracks the outcome of migration processes including success/failure
    status and statistics.
    """

    def __init__(
        self,
        success: bool = False,
        schemas_migrated: int = 0,
        values_migrated: int = 0,
        schemas_imported: int = 0,
        verification_passed: bool = False,
        cancelled: bool = False,
        namespace: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Initialize migration result.

        Args:
            success: Overall success status
            schemas_migrated: Number of schema records migrated
            values_migrated: Number of value records migrated
            schemas_imported: Number of schemas imported from database
            verification_passed: Whether verification passed
            cancelled: Whether migration was cancelled
            namespace: Namespace used for migration
            error_message: Error message if failed
        """
        self.success = success
        self.schemas_migrated = schemas_migrated
        self.values_migrated = values_migrated
        self.schemas_imported = schemas_imported
        self.verification_passed = verification_passed
        self.cancelled = cancelled
        self.namespace = namespace
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "success": self.success,
            "schemas_migrated": self.schemas_migrated,
            "values_migrated": self.values_migrated,
            "schemas_imported": self.schemas_imported,
            "verification_passed": self.verification_passed,
            "cancelled": self.cancelled,
            "namespace": self.namespace,
            "error_message": self.error_message,
        }

    def __repr__(self) -> str:
        """String representation of the result."""
        status = "SUCCESS" if self.success else "FAILED"
        if self.cancelled:
            status = "CANCELLED"
        return (
            f"MigrationResult("
            f"status={status}, "
            f"schemas={self.schemas_migrated}, "
            f"namespace={self.namespace}"
            f")"
        )


def get_v1_schema_fields() -> List[str]:
    """Get the list of v1 schema field names.

    Returns:
        List of v1 field names
    """
    return V1_SCHEMA_FIELDS.copy()


def get_v0_schema_fields() -> List[str]:
    """Get the list of v0 schema field names.

    Returns:
        List of v0 field names
    """
    return V0_SCHEMA_FIELDS.copy()


def is_v1_record(record: Dict[str, Any]) -> bool:
    """Check if a record is in v1 format.

    Args:
        record: Record dictionary to check

    Returns:
        True if record has v1 metadata
    """
    return record.get("metadata_version", 0) == METADATA_VERSION_V1


def is_v0_record(record: Dict[str, Any]) -> bool:
    """Check if a record is in v0 format.

    Args:
        record: Record dictionary to check

    Returns:
        True if record has v0 metadata
    """
    return record.get("metadata_version", 0) == METADATA_VERSION_V0
