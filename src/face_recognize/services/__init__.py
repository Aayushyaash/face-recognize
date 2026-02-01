"""Service layer for face recognition."""

from .identification import CachedIdentity, IdentificationService, IdentifiedFace

__all__ = ["CachedIdentity", "IdentificationService", "IdentifiedFace"]
