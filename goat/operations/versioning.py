"""Versioning for models, indexes, and lexicons.

Tracks:
  - Embedding model versions (which model/dimension produced each vector)
  - Lexicon versions (which lexicon version was active during indexing)
  - Index generations (version of the full index)

Enables:
  - Safe model upgrades without full re-index
  - Version-aware retrieval (filter by model version)
  - Rollback capability
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Tracks a specific embedding model version."""
    model_id: str
    provider: str
    model_name: str
    dimensions: int
    version_hash: str = ""
    created_at: float = 0.0
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.version_hash:
            content = f"{self.provider}:{self.model_name}:{self.dimensions}"
            self.version_hash = hashlib.sha256(
                content.encode()
            ).hexdigest()[:12]


@dataclass
class LexiconVersion:
    """Tracks a lexicon snapshot version."""
    version_id: str
    entry_count: int
    ticker_count: int
    abbreviation_count: int
    schema_mapping_count: int
    content_hash: str = ""
    created_at: float = 0.0
    is_active: bool = True

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class IndexGeneration:
    """Tracks a full index generation."""
    generation_id: str
    model_version: str
    lexicon_version: str
    chunk_count: int
    created_at: float = 0.0
    status: str = "building"  # building, active, deprecated
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class VersionManager:
    """Manages version tracking for all versioned resources."""

    def __init__(self):
        self._model_versions: dict[str, ModelVersion] = {}
        self._lexicon_versions: dict[str, LexiconVersion] = {}
        self._index_generations: dict[str, IndexGeneration] = {}

    # --- Model versions ---

    def register_model(self, provider: str, model_name: str,
                        dimensions: int,
                        metadata: Optional[dict] = None) -> ModelVersion:
        """Register a new model version."""
        mv = ModelVersion(
            model_id=f"{provider}:{model_name}",
            provider=provider,
            model_name=model_name,
            dimensions=dimensions,
            metadata=metadata or {},
        )
        self._model_versions[mv.model_id] = mv
        logger.info("Registered model: %s (dims=%d, hash=%s)",
                     mv.model_id, mv.dimensions, mv.version_hash)
        return mv

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        for mv in self._model_versions.values():
            if mv.is_active:
                return mv
        return None

    def deactivate_model(self, model_id: str) -> None:
        """Deactivate a model version."""
        if model_id in self._model_versions:
            self._model_versions[model_id].is_active = False

    # --- Lexicon versions ---

    def register_lexicon(self, version_id: str,
                          entry_count: int,
                          ticker_count: int = 0,
                          abbreviation_count: int = 0,
                          schema_mapping_count: int = 0,
                          content_hash: str = "") -> LexiconVersion:
        """Register a new lexicon version."""
        lv = LexiconVersion(
            version_id=version_id,
            entry_count=entry_count,
            ticker_count=ticker_count,
            abbreviation_count=abbreviation_count,
            schema_mapping_count=schema_mapping_count,
            content_hash=content_hash,
        )
        self._lexicon_versions[version_id] = lv
        return lv

    def get_active_lexicon(self) -> Optional[LexiconVersion]:
        """Get the currently active lexicon version."""
        for lv in self._lexicon_versions.values():
            if lv.is_active:
                return lv
        return None

    # --- Index generations ---

    def create_generation(self, generation_id: str,
                           model_version: str,
                           lexicon_version: str,
                           chunk_count: int = 0) -> IndexGeneration:
        """Create a new index generation."""
        ig = IndexGeneration(
            generation_id=generation_id,
            model_version=model_version,
            lexicon_version=lexicon_version,
            chunk_count=chunk_count,
        )
        self._index_generations[generation_id] = ig
        return ig

    def activate_generation(self, generation_id: str) -> None:
        """Activate an index generation and deactivate all others."""
        for gid, ig in self._index_generations.items():
            ig.status = "active" if gid == generation_id else "deprecated"

    def get_active_generation(self) -> Optional[IndexGeneration]:
        """Get the currently active index generation."""
        for ig in self._index_generations.values():
            if ig.status == "active":
                return ig
        return None

    # --- Serialization ---

    def export_state(self) -> dict:
        """Export version state as dict (for persistence)."""
        return {
            "models": {
                k: {
                    "model_id": v.model_id, "provider": v.provider,
                    "model_name": v.model_name, "dimensions": v.dimensions,
                    "version_hash": v.version_hash,
                    "created_at": v.created_at, "is_active": v.is_active,
                }
                for k, v in self._model_versions.items()
            },
            "lexicons": {
                k: {
                    "version_id": v.version_id,
                    "entry_count": v.entry_count,
                    "content_hash": v.content_hash,
                    "is_active": v.is_active,
                }
                for k, v in self._lexicon_versions.items()
            },
            "generations": {
                k: {
                    "generation_id": v.generation_id,
                    "model_version": v.model_version,
                    "lexicon_version": v.lexicon_version,
                    "chunk_count": v.chunk_count,
                    "status": v.status,
                }
                for k, v in self._index_generations.items()
            },
        }

    def import_state(self, data: dict) -> None:
        """Import version state from dict."""
        for k, v in data.get("models", {}).items():
            self._model_versions[k] = ModelVersion(**v)
        for k, v in data.get("lexicons", {}).items():
            self._lexicon_versions[k] = LexiconVersion(**v)
        for k, v in data.get("generations", {}).items():
            self._index_generations[k] = IndexGeneration(**v)
