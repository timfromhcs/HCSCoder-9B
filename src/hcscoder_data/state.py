import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_FILE = Path("state/pipeline.json")

STEPS_ORDER = [
    "INIT",
    "ENV_CHECK",
    "AUTH_CHECK",
    "MODEL_DISCOVERY",
    "MODEL_PINNED",
    "SOURCE_ACQUISITION",
    "NORMALIZATION",
    "LICENSE_FILTER",
    "SECRET_FILTER",
    "DEDUP",
    "CAPABILITY_CLASSIFICATION",
    "QUALITY_FILTER",
    "SYNTHESIS",
    "EXECUTION_VERIFICATION",
    "MIXTURE_BUILD",
    "DATASET_RELEASE_PRIVATE",
    "AUTOTRAIN_SMOKE",
    "AUTOTRAIN_SFT",
    "BASELINE_EVAL",
    "SFT_EVAL",
    "TARGETED_DATA_REPAIR",
    "DPO",
    "DPO_EVAL",
    "MOE_EXPERIMENT",
    "MOE_TRAIN",
    "DENSE_VS_MOE_EVAL",
    "FINAL_MODEL_SELECTION",
    "GGUF_CONVERSION",
    "GGUF_VALIDATION",
    "README_GENERATION",
    "SHA256",
    "HUB_RELEASE",
    "POST_RELEASE_VERIFICATION",
    "DONE",
]


class PipelineState:
    def __init__(self, filepath: Path = STATE_FILE):
        self.filepath = filepath
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "current_step": "INIT",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "steps": {},
        }

    def save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def start_step(self, step: str, inputs: Optional[Dict[str, Any]] = None) -> None:
        self.data["current_step"] = step
        now = datetime.now(timezone.utc).isoformat()
        if step not in self.data["steps"]:
            self.data["steps"][step] = {}
        self.data["steps"][step].update({
            "status": "RUNNING",
            "started_at": now,
            "inputs": inputs or {},
            "outputs": {},
            "logs": [],
            "exit_code": None,
        })
        self.save()

    def complete_step(
        self,
        step: str,
        outputs: Optional[Dict[str, Any]] = None,
        exit_code: int = 0,
        logs: Optional[List[str]] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if step not in self.data["steps"]:
            self.data["steps"][step] = {}
        self.data["steps"][step].update({
            "status": "COMPLETED" if exit_code == 0 else "FAILED",
            "ended_at": now,
            "outputs": outputs or {},
            "exit_code": exit_code,
        })
        if logs:
            self.data["steps"][step].setdefault("logs", []).extend(logs)
        self.save()

    def record_log(self, step: str, message: str) -> None:
        if step in self.data["steps"]:
            self.data["steps"][step].setdefault("logs", []).append(message)
            self.save()

    def is_step_completed(self, step: str) -> bool:
        return self.data.get("steps", {}).get(step, {}).get("status") == "COMPLETED"

    def get_step_output(self, step: str) -> Dict[str, Any]:
        return self.data.get("steps", {}).get(step, {}).get("outputs", {})
