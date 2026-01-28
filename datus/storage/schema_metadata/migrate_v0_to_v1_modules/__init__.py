# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details

"""
Migration modules for upgrading LanceDB schema from v0 to v1.

This package provides modular components for migrating schema metadata
from the legacy v0 format to the enhanced v1 format.

MODULE STRUCTURE:
    models.py           - Data models and schema definitions
    utils.py            - Helper functions and utilities
    migration_logic.py  - Core migration logic
    batch_processor.py  - Batch processing and reporting

PUBLIC API:
    Main migration functions:
        - migrate_schema_storage()
        - migrate_schema_value_storage()
        - verify_migration()

    Utility functions:
        - select_namespace_interactive()
        - backup_database()
        - report_migration_state()

    CLI functions:
        - import_schema_metadata()
        - print_recovery_suggestions()
        - print_final_migration_report()

USAGE:
    # Programmatic migration
    from datus.storage.schema_metadata.migrate_v0_to_v1_modules import (
        migrate_schema_storage,
        verify_migration,
    )

    # CLI migration
    python -m datus.storage.schema_metadata.migrate_v0_to_v1 \\
        --config=agent.yml \\
        --namespace=my_namespace

For detailed migration information, see the parent module docstring.
"""

# Core migration functions
from .migration_logic import (
    migrate_schema_storage,
    migrate_schema_value_storage,
    verify_migration,
    verify_migration_detailed,
)

# Utility functions
from .utils import (
    setup_signal_handlers,
    shutdown_requested,
    select_namespace_interactive,
    backup_database,
    load_backup_json,
    detect_dialect_from_ddl,
    report_migration_state,
    str_to_bool,
    _normalize_relationship_name,
    _build_table_name_map,
    _infer_relationships_from_names,
)

# Batch processing and reporting
from .batch_processor import (
    import_schema_metadata,
    print_recovery_suggestions,
    print_final_migration_report,
)

# Data models
from .models import (
    MigrationRecord,
    MigrationResult,
    V0_SCHEMA_FIELDS,
    V1_SCHEMA_FIELDS,
    METADATA_VERSION_V0,
    METADATA_VERSION_V1,
    get_v0_schema_fields,
    get_v1_schema_fields,
    is_v0_record,
    is_v1_record,
)

__all__ = [
    # Core migration functions
    "migrate_schema_storage",
    "migrate_schema_value_storage",
    "verify_migration",
    "verify_migration_detailed",
    # Utility functions
    "setup_signal_handlers",
    "shutdown_requested",
    "select_namespace_interactive",
    "backup_database",
    "load_backup_json",
    "detect_dialect_from_ddl",
    "report_migration_state",
    "str_to_bool",
    "_normalize_relationship_name",
    "_build_table_name_map",
    "_infer_relationships_from_names",
    # Batch processing and reporting
    "import_schema_metadata",
    "print_recovery_suggestions",
    "print_final_migration_report",
    # Data models
    "MigrationRecord",
    "MigrationResult",
    "V0_SCHEMA_FIELDS",
    "V1_SCHEMA_FIELDS",
    "METADATA_VERSION_V0",
    "METADATA_VERSION_V1",
    "get_v0_schema_fields",
    "get_v1_schema_fields",
    "is_v0_record",
    "is_v1_record",
]
