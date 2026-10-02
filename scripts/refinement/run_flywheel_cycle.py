"""Autonomous Frontier Refinement Flywheel Cycle
Ingests benchmark defect patterns, verifies baseline failure and candidate fix with ExecutionHarness,
and synthesizes verified SFT repairs and DPO alignment pairs to continuously improve HCSCoder-9B.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hcscoder_data.synthesis.harness import ExecutionHarness
from hcscoder_data.refinement.repair_engine import TrajectoryRepairEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FlywheelCycle")

DEFECT_CASES = [
    {
        "id": "defect_001_sliding_window_max",
        "description": "Fix sliding window maximum off-by-one error on empty or single-element windows",
        "broken_code": """
def max_sliding_window(nums, k):
    if not nums:
        return []
    res = []
    for i in range(len(nums) - k):  # Bug: off-by-one, ignores last window
        res.append(max(nums[i:i + k]))
    return res
""",
        "fixed_code": """
def max_sliding_window(nums, k):
    if not nums or k <= 0:
        return []
    if k > len(nums):
        return [max(nums)]
    res = []
    for i in range(len(nums) - k + 1):
        res.append(max(nums[i:i + k]))
    return res
""",
        "test_code": """
import pytest
from solution import max_sliding_window

def test_basic_window():
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]

def test_single_element():
    assert max_sliding_window([1], 1) == [1]

def test_k_equals_len():
    assert max_sliding_window([1, -1], 2) == [1]

def test_empty():
    assert max_sliding_window([], 3) == []
""",
    },
    {
        "id": "defect_002_json_parser_type_safety",
        "description": "Handle malformed numeric strings and nested NoneType gracefully in JSON payload normalizer",
        "broken_code": """
import json

def parse_and_normalize(payload_str):
    data = json.loads(payload_str)
    # Bug: raises KeyError / TypeError on None or missing nested fields
    return {
        "user_id": int(data["user"]["id"]),
        "score": float(data["metrics"]["score"]),
        "active": data["status"]["is_active"]
    }
""",
        "fixed_code": """
import json

def parse_and_normalize(payload_str):
    try:
        data = json.loads(payload_str) if isinstance(payload_str, str) else (payload_str or {})
    except (json.JSONDecodeError, TypeError):
        return {"user_id": 0, "score": 0.0, "active": False}

    user = data.get("user") or {}
    metrics = data.get("metrics") or {}
    status = data.get("status") or {}

    try:
        user_id = int(user.get("id", 0))
    except (ValueError, TypeError):
        user_id = 0

    try:
        score = float(metrics.get("score", 0.0))
    except (ValueError, TypeError):
        score = 0.0

    active = bool(status.get("is_active", False))

    return {
        "user_id": user_id,
        "score": score,
        "active": active
    }
""",
        "test_code": """
import pytest
from solution import parse_and_normalize

def test_valid_json():
    payload = '{"user": {"id": "42"}, "metrics": {"score": "98.5"}, "status": {"is_active": true}}'
    assert parse_and_normalize(payload) == {"user_id": 42, "score": 98.5, "active": True}

def test_missing_nested_fields():
    payload = '{"user": null, "metrics": {}}'
    assert parse_and_normalize(payload) == {"user_id": 0, "score": 0.0, "active": False}

def test_invalid_types():
    payload = '{"user": {"id": "invalid"}, "metrics": {"score": "nan_or_bad"}}'
    assert parse_and_normalize(payload) == {"user_id": 0, "score": 0.0, "active": False}

def test_malformed_json():
    assert parse_and_normalize("{broken json") == {"user_id": 0, "score": 0.0, "active": False}
""",
    },
    {
        "id": "defect_003_retry_backoff_jitter",
        "description": "Implement exponential backoff with ceiling cap and full jitter to prevent thundering herd",
        "broken_code": """
import time

def compute_backoff(attempt, base_delay=1.0, max_delay=60.0):
    # Bug: exponential backoff without upper bound cap or proper jitter
    return base_delay * (2 ** attempt)
""",
        "fixed_code": """
import random

def compute_backoff(attempt, base_delay=1.0, max_delay=60.0, seed=None):
    if seed is not None:
        random.seed(seed)
    if attempt < 0:
        return base_delay
    calculated = min(max_delay, base_delay * (2 ** attempt))
    # Full jitter between 0 and calculated delay
    return random.uniform(0.0, calculated)
""",
        "test_code": """
import pytest
from solution import compute_backoff

def test_backoff_respects_ceiling():
    delay = compute_backoff(attempt=20, base_delay=1.0, max_delay=30.0, seed=42)
    assert 0.0 <= delay <= 30.0

def test_backoff_bounds():
    for attempt in range(5):
        val = compute_backoff(attempt, base_delay=0.5, max_delay=10.0, seed=attempt)
        max_possible = min(10.0, 0.5 * (2 ** attempt))
        assert 0.0 <= val <= max_possible

def test_negative_attempt():
    val = compute_backoff(-1, base_delay=1.0, max_delay=10.0)
    assert val == 1.0
""",
    },
    {
        "id": "defect_004_tool_dispatch_sanitization",
        "description": "Strict validation of tool dispatch arguments against dangerous bash injection and path traversal",
        "broken_code": """
import os

def dispatch_file_read(filepath, root_dir="/app/workspace"):
    # Bug: direct join allows directory traversal via ../../etc/passwd
    target = os.path.join(root_dir, filepath)
    with open(target, 'r') as f:
        return f.read()
""",
        "fixed_code": """
import os
from pathlib import Path

def dispatch_file_read(filepath, root_dir="/app/workspace"):
    resolved_root = Path(root_dir).resolve()
    target_path = (resolved_root / filepath).resolve()
    
    # Path traversal security gate
    if not str(target_path).startswith(str(resolved_root)):
        raise PermissionError(f"Access denied: path '{filepath}' escapes sandbox '{root_dir}'")
        
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    return target_path.read_text(encoding="utf-8")
""",
        "test_code": """
import pytest
from pathlib import Path
from solution import dispatch_file_read

def test_path_traversal_blocked(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("classified", encoding="utf-8")
    
    with pytest.raises(PermissionError):
        dispatch_file_read("../secret.txt", root_dir=str(root))

def test_valid_sandboxed_read(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sample = root / "data.txt"
    sample.write_text("hello sandbox", encoding="utf-8")
    
    content = dispatch_file_read("data.txt", root_dir=str(root))
    assert content == "hello sandbox"
""",
    },
    {
        "id": "defect_005_lru_cache_thread_safe",
        "description": "Fix race conditions and stale keys in thread-safe LRU cache with eviction callbacks",
        "broken_code": """
class SimpleLRU:
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key not in self.cache:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        # Bug: fails when updating existing key without evicting duplicate from order
        if len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
""",
        "fixed_code": """
from collections import OrderedDict
from threading import RLock

class SimpleLRU:
    def __init__(self, capacity=3):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
""",
        "test_code": """
import pytest
from solution import SimpleLRU

def test_lru_eviction():
    lru = SimpleLRU(capacity=2)
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") == 1  # accesses 'a', makes 'b' least recently used
    lru.put("c", 3)           # should evict 'b'
    assert lru.get("b") is None
    assert lru.get("a") == 1
    assert lru.get("c") == 3

def test_overwrite_existing_key_eviction_sequence():
    lru = SimpleLRU(capacity=2)
    lru.put("a", 1)
    lru.put("a", 2)
    lru.put("b", 3)
    lru.put("c", 4)
    lru.put("d", 5)
    # At this point, capacity is 2, most recent are 'c' and 'd'
    assert lru.get("d") == 5
    assert lru.get("c") == 4
    assert lru.get("a") is None
    assert lru.get("b") is None
""",
    }
]


def run_flywheel_refinement():
    logger.info("Initializing ExecutionHarness sandbox...")
    harness = ExecutionHarness()
    repair_engine = TrajectoryRepairEngine(harness)

    train_path = Path("data/final/train.jsonl")
    dpo_path = Path("data/final/dpo.jsonl")
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    verified_repairs = []
    verified_dpo = []
    failures = []

    logger.info(f"Processing {len(DEFECT_CASES)} frontier defect repair targets...")

    for case in DEFECT_CASES:
        cid = case["id"]
        logger.info(f"--- Evaluating defect case: {cid} ---")
        repaired_traj, dpo_pair = repair_engine.repair_code_trajectory(
            broken_code=case["broken_code"],
            fixed_code=case["fixed_code"],
            test_code=case["test_code"],
            task_description=case["description"],
        )

        if repaired_traj and dpo_pair:
            logger.info(f"✓ Case {cid} PASS: Baseline failed as expected, candidate fix passed sandbox verification.")
            sft_record = {
                "messages": [m.model_dump(exclude_none=True) for m in repaired_traj.messages],
                "metadata": {
                    "source": "flywheel_defect_repair",
                    "defect_id": cid,
                    "verified_pytest": True,
                },
            }
            verified_repairs.append(sft_record)
            verified_dpo.append(dpo_pair)
        else:
            logger.error(f"✗ Case {cid} FAILED verification harness.")
            failures.append(cid)

    # Append to train.jsonl
    logger.info(f"Appending {len(verified_repairs)} verified SFT repair trajectories to {train_path}...")
    with open(train_path, "a", encoding="utf-8") as f:
        for item in verified_repairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Append to dpo.jsonl
    logger.info(f"Appending {len(verified_dpo)} verified DPO pairs to {dpo_path}...")
    with open(dpo_path, "a", encoding="utf-8") as f:
        for item in verified_dpo:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    telemetry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_targets": len(DEFECT_CASES),
        "verified_sft_generated": len(verified_repairs),
        "verified_dpo_pairs_generated": len(verified_dpo),
        "failed_verifications": failures,
        "train_file": str(train_path),
        "dpo_file": str(dpo_path),
        "status": "SUCCESS" if len(failures) == 0 else "PARTIAL",
    }

    report_file = reports_dir / "flywheel_cycle_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    logger.info(f"Flywheel refinement cycle completed successfully. Telemetry saved to {report_file}")
    return telemetry


if __name__ == "__main__":
    run_flywheel_refinement()
