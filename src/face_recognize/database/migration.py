"""Migration utilities for converting between database formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..database.json_backend import JsonDatabase
from ..database.sqlite_backend import EncryptedSqliteDatabase


def migrate_json_to_sqlite(
    json_path: Path, sqlite_path: Path, encryption_key: bytes | None = None
) -> None:
    """Migrate data from JSON database to SQLite database.

    Args:
        json_path: Path to source JSON database file.
        sqlite_path: Path to destination SQLite database file.
        encryption_key: Optional encryption key for SQLite database.
    """
    # Load from JSON database
    json_db = JsonDatabase(json_path)
    all_records = json_db.list_all()

    # Create SQLite database
    sqlite_db = EncryptedSqliteDatabase(sqlite_path, encryption_key)

    # Add all records to SQLite database
    for record in all_records:
        sqlite_db.add(record.name, record.embedding)


def migrate_sqlite_to_json(sqlite_path: Path, json_path: Path) -> None:
    """Migrate data from SQLite database to JSON database.

    Args:
        sqlite_path: Path to source SQLite database file.
        json_path: Path to destination JSON database file.
    """
    # Load from SQLite database
    sqlite_db = EncryptedSqliteDatabase(sqlite_path, None)  # No encryption for loading
    all_records = sqlite_db.list_all()

    # Create JSON database
    json_db = JsonDatabase(json_path)

    # Add all records to JSON database
    for record in all_records:
        json_db.add(record.name, record.embedding)


def backup_database(config: Any) -> Path:
    """Create a backup of the current database.

    Args:
        config: Application configuration.

    Returns:
        Path to the backup file.
    """
    import shutil
    from datetime import datetime

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config.database_path.with_suffix(
        f".backup_{timestamp}{config.database_path.suffix}"
    )

    # Copy the database file
    shutil.copy2(config.database_path, backup_path)

    return backup_path  # type: ignore[no-any-return]
