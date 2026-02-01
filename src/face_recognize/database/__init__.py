"""Database backends for face storage."""

from .json_backend import JsonDatabase, PersonRecord

__all__ = ["JsonDatabase", "PersonRecord"]
