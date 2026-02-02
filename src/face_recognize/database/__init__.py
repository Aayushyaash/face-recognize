"""Database backends for face storage."""

from .base import DatabaseBackend, PersonRecord
from .json_backend import JsonDatabase

__all__ = ["DatabaseBackend", "JsonDatabase", "PersonRecord"]
