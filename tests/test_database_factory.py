"""Unit tests for database factory."""

import tempfile
from pathlib import Path

from face_recognize.config import AppConfig
from face_recognize.database.factory import create_database


def test_create_json_database() -> None:
    """Test that factory creates JSON database when specified."""
    import json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        # Create an empty but valid JSON database file
        with open(tmp.name, "w") as f:
            json.dump({"version": "1.0", "persons": []}, f)

        config = AppConfig(database_path=Path(tmp.name), database_backend="json")
        db = create_database(config)

        # Should be JsonDatabase instance
        from face_recognize.database.json_backend import JsonDatabase

        assert isinstance(db, JsonDatabase)


def test_create_sqlite_database() -> None:
    """Test that factory creates SQLite database when specified."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        config = AppConfig(database_path=Path(tmp.name), database_backend="sqlite")
        db = create_database(config)

        # Should be EncryptedSqliteDatabase instance
        from face_recognize.database.sqlite_backend import EncryptedSqliteDatabase

        assert isinstance(db, EncryptedSqliteDatabase)


def test_default_to_json_database() -> None:
    """Test that factory defaults to JSON database."""
    import json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        # Create an empty but valid JSON database file
        with open(tmp.name, "w") as f:
            json.dump({"version": "1.0", "persons": []}, f)

        config = AppConfig(database_path=Path(tmp.name))  # No backend specified
        db = create_database(config)

        # Should default to JsonDatabase
        from face_recognize.database.json_backend import JsonDatabase

        assert isinstance(db, JsonDatabase)
