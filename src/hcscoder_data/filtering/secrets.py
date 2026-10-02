import math
import re
from typing import List, Tuple
from hcscoder_data.normalization.schema import Trajectory

PATTERNS = [
    re.compile(r"(?i)ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"(?i)github_pat_[a-zA-Z0-9_]{82}"),
    re.compile(r"(?i)hf_[a-zA-Z0-9]{34,}"),
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key\s*=\s*[a-zA-Z0-9/+=]{40}"),
    re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|EC|DSA)?\s*PRIVATE KEY-----"),
    re.compile(r"(?i)sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)(?:bearer|token|secret|password|api[_-]?key)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.~]{16,}['\"]"),
]


def shannon_entropy(data: str) -> float:
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


class SecretFilter:
    def __init__(self, entropy_threshold: float = 4.5):
        self.entropy_threshold = entropy_threshold
        self.patterns = PATTERNS

    def scan_text(self, text: str) -> Tuple[bool, List[str]]:
        matches = []
        for pattern in self.patterns:
            found = pattern.findall(text)
            if found:
                matches.extend(found)

        # High entropy token heuristic on long alphanumeric words
        words = re.findall(r"\b[A-Za-z0-9_\-]{32,}\b", text)
        for w in words:
            if shannon_entropy(w) > self.entropy_threshold:
                matches.append(w[:8] + "...")

        return len(matches) > 0, matches

    def filter_trajectory(self, traj: Trajectory) -> Tuple[bool, str]:
        for msg in traj.messages:
            if msg.content:
                has_secret, matches = self.scan_text(msg.content)
                if has_secret:
                    return False, f"Secret detected in message: {matches[:3]}"
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_str = str(tc.arguments)
                    has_secret, matches = self.scan_text(tc_str)
                    if has_secret:
                        return False, f"Secret detected in tool call {tc.name}: {matches[:3]}"
        return True, ""
