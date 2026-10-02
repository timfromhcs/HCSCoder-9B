import json
import time
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class BenchmarkMetrics(BaseModel):
    benchmark_id: str
    total_tasks: int
    resolved_count: int
    success_rate: float
    tool_accuracy: float = 0.0
    recovery_rate: float = 0.0
    verification_rate: float = 0.0
    average_steps: float = 0.0
    tool_errors: int = 0
    repeat_failures: int = 0
    duration_seconds: float = 0.0


class BenchmarkSuite:
    """Deterministic, executable benchmark suite for HCSCoder."""

    def __init__(self, suite_id: str, name: str, tasks: List[Dict[str, Any]]):
        self.suite_id = suite_id
        self.name = name
        self.tasks = tasks

    def evaluate(self, agent_fn) -> BenchmarkMetrics:
        start_time = time.time()
        resolved = 0
        total_steps = 0
        total_tool_calls = 0
        correct_tool_calls = 0
        recovered = 0
        total_errors = 0
        verifications = 0

        for task in self.tasks:
            result = agent_fn(task)
            steps = result.get("steps", 1)
            total_steps += steps

            if result.get("success", False):
                resolved += 1
            if result.get("verified", False):
                verifications += 1
            if result.get("recovered", False):
                recovered += 1
            total_errors += result.get("tool_errors", 0)

            t_calls = result.get("tool_calls", 0)
            c_calls = result.get("correct_tool_calls", 0)
            total_tool_calls += t_calls
            correct_tool_calls += c_calls

        n = len(self.tasks)
        dur = round(time.time() - start_time, 2)
        tool_acc = round(correct_tool_calls / max(1, total_tool_calls), 4) if total_tool_calls else 1.0

        return BenchmarkMetrics(
            benchmark_id=self.suite_id,
            total_tasks=n,
            resolved_count=resolved,
            success_rate=round(resolved / max(1, n), 4),
            tool_accuracy=tool_acc,
            recovery_rate=round(recovered / max(1, n), 4),
            verification_rate=round(verifications / max(1, n), 4),
            average_steps=round(total_steps / max(1, n), 2),
            tool_errors=total_errors,
            repeat_failures=0,
            duration_seconds=dur,
        )


def build_hc_tool_100() -> BenchmarkSuite:
    tasks = []
    for i in range(100):
        tasks.append({
            "id": f"hc_tool_{i:03d}",
            "prompt": f"Call tool terminal.run with command 'echo hello_{i}'",
            "expected_tool": "terminal.run",
            "expected_args": {"command": f"echo hello_{i}"},
        })
    return BenchmarkSuite("hc_tool_100", "HC-Tool-100", tasks)


def build_hc_selfheal_100() -> BenchmarkSuite:
    tasks = []
    for i in range(100):
        tasks.append({
            "id": f"hc_heal_{i:03d}",
            "error_type": "FileNotFoundError" if i % 2 == 0 else "SyntaxError",
            "initial_command": f"python script_{i}.py",
            "fix_strategy": "inspect_directory" if i % 2 == 0 else "fix_syntax",
        })
    return BenchmarkSuite("hc_selfheal_100", "HC-SelfHeal-100", tasks)


def build_hc_verify_100() -> BenchmarkSuite:
    tasks = []
    for i in range(100):
        tasks.append({
            "id": f"hc_verify_{i:03d}",
            "task_type": "code_modification",
            "require_test_execution": True,
        })
    return BenchmarkSuite("hc_verify_100", "HC-Verify-100", tasks)


def build_hc_repo_100() -> BenchmarkSuite:
    tasks = []
    for i in range(100):
        tasks.append({
            "id": f"hc_repo_{i:03d}",
            "repo": f"org/repo_{i % 10}",
            "task": "patch_generation",
        })
    return BenchmarkSuite("hc_repo_100", "HC-Repo-100", tasks)


def build_hc_long_50() -> BenchmarkSuite:
    tasks = []
    for i in range(50):
        tasks.append({
            "id": f"hc_long_{i:03d}",
            "stages": ["plan", "inspect", "modify", "test", "verify"],
            "target_steps": 12,
        })
    return BenchmarkSuite("hc_long_50", "HC-Long-50", tasks)
