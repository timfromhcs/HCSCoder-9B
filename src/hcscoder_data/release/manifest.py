import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ReleaseManifestBuilder:
    def __init__(self, root_dir: Path = Path(".")):
        self.root_dir = root_dir

    def generate_checksums(self, file_paths: List[Path], output_file: Path) -> Dict[str, str]:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        checksum_map = {}
        with open(output_file, "w", encoding="utf-8") as out:
            for p in file_paths:
                if p.exists() and p.is_file():
                    sha = calculate_sha256(p)
                    rel_path = p.relative_to(self.root_dir)
                    checksum_map[str(rel_path)] = sha
                    out.write(f"{sha}  {rel_path}\n")
        return checksum_map

    def generate_provenance(
        self,
        base_model_info: Dict[str, Any],
        dataset_info: Dict[str, Any],
        training_info: Dict[str, Any],
        output_file: Path = Path("artifacts/manifests/provenance.json"),
    ) -> Dict[str, Any]:
        provenance = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_info,
            "training_data": dataset_info,
            "training": training_info,
            "reproducibility": {
                "environment": "Windows 11 Pro / Hugging Face Spaces & Cloud Jobs",
                "python_version": "3.12.13",
                "transformers_version": "5.18.0",
                "peft_version": "0.21.2",
                "trl_version": "1.14.1",
            },
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        return provenance
