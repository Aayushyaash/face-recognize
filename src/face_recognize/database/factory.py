"""Database factory for creating appropriate database backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import AppConfig

if TYPE_CHECKING:
    from .base import DatabaseBackend


def create_database(config: AppConfig) -> DatabaseBackend:
    """Create a database instance based on configuration.

    Args:
        config: Application configuration with database settings.

    Returns:
        Appropriate database backend instance based on config.
    """
    if config.database_backend == "sqlite":
        from .sqlite_backend import EncryptedSqliteDatabase

        return EncryptedSqliteDatabase(config.database_path)
    else:  # Default to JSON
        from .json_backend import JsonDatabase

        return JsonDatabase(config.database_path)
