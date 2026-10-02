import ast
import json
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("SecurityGates")

# RegEx patterns for sensitive credentials
SECRET_PATTERNS = [
    ("GitHub Token (Classic)", re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("GitHub Fine-Grained PAT", re.compile(r"github_pat_[a-zA-Z0-9_]{82}")),
    ("Hugging Face Token", re.compile(r"hf_[a-zA-Z0-9]{34,}")),
    ("AWS Access Key ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS Secret Key", re.compile(r"aws_secret_access_key\s*=\s*[a-zA-Z0-9/+=]{40}")),
    ("Private Key", re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|EC|DSA)?\s*PRIVATE KEY-----")),
    ("OpenAI / LLM API Key", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
]


def calculate_entropy(data: str) -> float:
    if not data:
        return 0.0
    entropy = 0.0
    length = len(data)
    counts = {}
    for char in data:
        counts[char] = counts.get(char, 0) + 1
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


class SecurityGateManager:
    """Automated multi-layered security gates for code, datasets, and model weights."""

    def __init__(self, root_dir: Path = Path(".")):
        self.root_dir = root_dir

    def gate_1_secret_scan(self, target_dir: Optional[Path] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """Layer 1: Scans directory for committed or plain-text secrets and high-entropy strings."""
        scan_dir = target_dir or self.root_dir
        violations = []

        ignore_exts = {".png", ".jpg", ".pyc", ".gguf", ".safetensors", ".lock"}
        ignore_dirs = {".git", ".venv", ".uv", "__pycache__"}

        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                p = Path(root) / file
                if p.suffix in ignore_exts or p.name in {".env", ".gitignore"}:
                    continue

                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                # 1. Regex checks
                for name, pat in SECRET_PATTERNS:
                    matches = pat.findall(content)
                    if matches:
                        violations.append({
                            "type": "regex_secret",
                            "rule": name,
                            "file": str(p.relative_to(self.root_dir)),
                            "count": len(matches),
                        })

                # 2. High-entropy words check (> 32 chars, entropy > 4.6)
                tokens = re.findall(r"\b[A-Za-z0-9_\-]{32,}\b", content)
                for tok in tokens:
                    # Skip common commit hashes or hex representations
                    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", tok):
                        continue
                    if calculate_entropy(tok) > 4.6:
                        violations.append({
                            "type": "high_entropy_token",
                            "file": str(p.relative_to(self.root_dir)),
                            "snippet": tok[:8] + "...",
                            "entropy": round(calculate_entropy(tok), 2),
                        })

        passed = len(violations) == 0
        logger.info(f"Gate 1 (Secret Scan): passed={passed}, violations={len(violations)}")
        return passed, violations

    def gate_2_sast_vulnerability_scan(self, target_path: str = "src") -> Tuple[bool, Dict[str, Any]]:
        """Layer 2: Static Application Security Testing via Bandit on Python codebase."""
        src_path = self.root_dir / target_path
        if not src_path.exists():
            return True, {"skipped": True}

        try:
            cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                str(src_path),
                "-f",
                "json",
                "-ll",  # Medium and High severity only
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            report = {}
            if res.stdout:
                try:
                    report = json.loads(res.stdout)
                except Exception:
                    pass

            high_severity = [
                item for item in report.get("results", []) if item.get("issue_severity") == "HIGH"
            ]
            passed = len(high_severity) == 0
            logger.info(f"Gate 2 (SAST / Bandit): passed={passed}, high_severity={len(high_severity)}")
            return passed, {"high_issues": high_severity, "total_issues": len(report.get("results", []))}
        except Exception as e:
            logger.error(f"Gate 2 execution error: {e}")
            return False, {"error": str(e)}

    def gate_3_weight_deserialization_safety(self, model_dir: Path) -> Tuple[bool, List[str]]:
        """Layer 3: Enforce Safetensors format; strictly reject any pickled checkpoints."""
        unsafe_files = []
        forbidden_extensions = {".bin", ".pt", ".pth", ".pkl", ".pickle"}
        ignore_dirs = {".git", ".venv", ".uv", "__pycache__"}

        for root, dirs, files in os.walk(model_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                p = Path(root) / file
                if p.suffix in forbidden_extensions:
                    unsafe_files.append(str(p.name))

        passed = len(unsafe_files) == 0
        logger.info(f"Gate 3 (Deserialization Safety): passed={passed}, unsafe_files={unsafe_files}")
        return passed, unsafe_files

    def gate_4_model_artifact_size_and_integrity(
        self,
        artifact_path: Path,
        expected_min_bytes: int,
        is_gguf: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Layer 4: Anti-Mock Gate: Verify model weights are real binary tensors, not empty stubs."""
        if not artifact_path.exists():
            return False, {"error": "File does not exist"}

        size = artifact_path.stat().st_size
        passed_size = size >= expected_min_bytes

        details = {
            "file": artifact_path.name,
            "actual_size_bytes": size,
            "expected_min_bytes": expected_min_bytes,
            "passed_size": passed_size,
        }

        # Header validation
        if is_gguf:
            with open(artifact_path, "rb") as f:
                magic = f.read(4)
                details["is_valid_gguf_magic"] = (magic == b"GGUF")
                details["valid_header"] = (magic == b"GGUF")
        else:
            # Safetensors header validation
            try:
                from safetensors import safe_open
                with safe_open(str(artifact_path), framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    details["tensor_count"] = len(keys)
                    details["sample_tensors"] = keys[:5]
                    details["valid_header"] = len(keys) > 0
            except Exception as e:
                details["valid_header"] = False
                details["error"] = str(e)

        passed = passed_size and details.get("valid_header", False)
        logger.info(f"Gate 4 (Artifact Integrity): file={artifact_path.name}, passed={passed}")
        return passed, details
