"""Encryption utilities for database fields."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from cryptography.fernet import Fernet


class FieldEncryptor:
    """Utility class for encrypting and decrypting database fields."""

    def __init__(self, encryption_key: bytes | None = None) -> None:
        """Initialize the field encryptor.

        Args:
            encryption_key: Encryption key to use. If None, generates a new key.
        """
        if encryption_key is None:
            encryption_key = Fernet.generate_key()

        self.cipher_suite = Fernet(encryption_key)
        self.encryption_key = encryption_key

    def encrypt_field(self, field: str) -> str:
        """Encrypt a field value.

        Args:
            field: Field value to encrypt.

        Returns:
            Base64-encoded encrypted field.
        """
        field_bytes = field.encode("utf-8")
        encrypted_bytes = self.cipher_suite.encrypt(field_bytes)
        result: str = encrypted_bytes.decode("utf-8")
        return result

    def decrypt_field(self, encrypted_field: str) -> str:
        """Decrypt an encrypted field value.

        Args:
            encrypted_field: Base64-encoded encrypted field.

        Returns:
            Decrypted field value.
        """
        encrypted_bytes = encrypted_field.encode("utf-8")
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        result: str = decrypted_bytes.decode("utf-8")
        return result

    def encrypt_embedding(self, embedding: npt.NDArray[np.float32]) -> str:
        """Encrypt a face embedding.

        Args:
            embedding: Face embedding to encrypt.

        Returns:
            Base64-encoded encrypted embedding as JSON string.
        """
        import json

        embedding_str = json.dumps(embedding.tolist())
        result: str = self.encrypt_field(embedding_str)
        return result

    def decrypt_embedding(self, encrypted_embedding: str) -> npt.NDArray[np.float32]:
        """Decrypt an encrypted face embedding.

        Args:
            encrypted_embedding: Base64-encoded encrypted embedding.

        Returns:
            Decrypted face embedding as numpy array.
        """
        import numpy as np

        decrypted_str = self.decrypt_field(encrypted_embedding)
        import json

        embedding_list = json.loads(decrypted_str)
        return np.array(embedding_list, dtype=np.float32)
