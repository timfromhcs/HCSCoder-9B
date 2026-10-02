import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    source_id: str
    repo: str
    revision: str
    license: str
    license_confidence: str = "high"
    download_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_hash: str = ""
    commercial_use: bool = True
    record_count: int = 0
    notes: str = ""


class SourceManifest(BaseModel):
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: List[SourceItem] = []

    def save(self, path: Path = Path("artifacts/manifests/source_manifest.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path = Path("artifacts/manifests/source_manifest.json")) -> "SourceManifest":
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return cls.model_validate_json(f.read())
        return cls()
