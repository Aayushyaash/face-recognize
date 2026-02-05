"""Database backends for face storage."""

from .base import DatabaseBackend, PersonRecord
from .factory import create_database
from .json_backend import JsonDatabase
from .sqlite_backend import EncryptedSqliteDatabase

__all__ = [
    "DatabaseBackend",
    "JsonDatabase",
    "EncryptedSqliteDatabase",
    "PersonRecord",
    "create_database",
]
