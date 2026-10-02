import uuid
from typing import List, Tuple
from hcscoder_data.normalization.schema import (
    Message,
    Outcome,
    ToolCall,
    ToolDefinition,
    Trajectory,
)
from hcscoder_data.synthesis.harness import ExecutionHarness

TASK_TEMPLATES = [
    {
        "id": "parse_key_value",
        "description": "Implement parse_key_value to safely parse env-like strings with comment ignoring and whitespace stripping.",
        "broken_code": (
            "def parse_key_value(text: str) -> dict:\n"
            "    res = {}\n"
            "    for line in text.splitlines():\n"
            "        k, v = line.split('=')\n"
            "        res[k] = v\n"
            "    return res\n"
        ),
        "fixed_code": (
            "def parse_key_value(text: str) -> dict:\n"
            "    res = {}\n"
            "    for line in text.splitlines():\n"
            "        line = line.strip()\n"
            "        if not line or line.startswith('#'):\n"
            "            continue\n"
            "        if '=' not in line:\n"
            "            continue\n"
            "        k, v = line.split('=', 1)\n"
            "        res[k.strip()] = v.strip()\n"
            "    return res\n"
        ),
        "test_code": (
            "from solution import parse_key_value\n"
            "def test_parse_simple():\n"
            "    assert parse_key_value('A=1\\nB=2') == {'A': '1', 'B': '2'}\n"
            "def test_parse_comments_and_spaces():\n"
            "    assert parse_key_value(' # comment\\nKEY = val = 123 \\n') == {'KEY': 'val = 123'}\n"
            "def test_parse_empty():\n"
            "    assert parse_key_value('') == {}\n"
        ),
    },
    {
        "id": "safe_dict_get",
        "description": "Implement safe_nested_get for querying deeply nested dictionary paths with default fallback.",
        "broken_code": (
            "def safe_nested_get(data: dict, path: str, default=None):\n"
            "    curr = data\n"
            "    for part in path.split('.'):\n"
            "        curr = curr[part]\n"
            "    return curr\n"
        ),
        "fixed_code": (
            "def safe_nested_get(data: dict, path: str, default=None):\n"
            "    if not isinstance(data, dict) or not path:\n"
            "        return default\n"
            "    curr = data\n"
            "    for part in path.split('.'):\n"
            "        if isinstance(curr, dict) and part in curr:\n"
            "            curr = curr[part]\n"
            "        else:\n"
            "            return default\n"
            "    return curr\n"
        ),
        "test_code": (
            "from solution import safe_nested_get\n"
            "def test_nested_found():\n"
            "    assert safe_nested_get({'a': {'b': {'c': 42}}}, 'a.b.c') == 42\n"
            "def test_nested_missing():\n"
            "    assert safe_nested_get({'a': 1}, 'a.b.c', default=0) == 0\n"
            "def test_nested_non_dict():\n"
            "    assert safe_nested_get('invalid', 'a.b', default='none') == 'none'\n"
        ),
    },
    {
        "id": "lru_cache_bounded",
        "description": "Implement a minimal bounded LRU cache with get and put operations.",
        "broken_code": (
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = {}\n"
            "    def get(self, key):\n"
            "        return self.cache.get(key, -1)\n"
            "    def put(self, key, value):\n"
            "        self.cache[key] = value\n"
        ),
        "fixed_code": (
            "from collections import OrderedDict\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n"
            "    def put(self, key, value):\n"
            "        if key in self.cache:\n"
            "            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)\n"
        ),
        "test_code": (
            "from solution import LRUCache\n"
            "def test_lru_eviction():\n"
            "    cache = LRUCache(2)\n"
            "    cache.put(1, 1)\n"
            "    cache.put(2, 2)\n"
            "    assert cache.get(1) == 1\n"
            "    cache.put(3, 3)\n"
            "    assert cache.get(2) == -1\n"
            "    assert cache.get(3) == 3\n"
        ),
    },
]


def generate_verified_synthetic_trajectories(harness: ExecutionHarness) -> List[Trajectory]:
    trajectories = []

    for item in TASK_TEMPLATES:
        # Step 1: Verify baseline failure with broken code
        baseline_success, baseline_code, baseline_out = harness.execute_python_code(
            item["broken_code"],
            item["test_code"],
        )
        assert not baseline_success, f"Baseline must fail for task {item['id']}"

        # Step 2: Verify fix success with fixed code
        fix_success, fix_code, fix_out = harness.execute_python_code(
            item["fixed_code"],
            item["test_code"],
        )
        assert fix_success, f"Fixed code must pass tests for task {item['id']}"

        # Step 3: Construct authentic SFT trajectory with actual verified outputs
        traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=[
                Message(
                    role="system",
                    content="You are HCSCoder, an autonomous coding agent. Inspect code, run tests, fix defects, and verify fixes.",
                ),
                Message(
                    role="user",
                    content=f"Fix the failing implementation in `solution.py`. {item['description']}",
                ),
                Message(
                    role="assistant",
                    content="First, I run the test suite to observe the baseline failure and understand the root cause.",
                    tool_calls=[
                        ToolCall(
                            name="terminal.run",
                            arguments={"command": "pytest test_solution.py"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    name="terminal.run",
                    content=baseline_out,
                ),
                Message(
                    role="assistant",
                    content=(
                        "Analyzing test output: The baseline failed as observed in the traceback. "
                        "I will now inspect `solution.py` and write the correct robust implementation."
                    ),
                    tool_calls=[
                        ToolCall(
                            name="filesystem.write_file",
                            arguments={"path": "solution.py", "content": item["fixed_code"]},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    name="filesystem.write_file",
                    content="solution.py successfully written.",
                ),
                Message(
                    role="assistant",
                    content="The patch is in place. Now executing the test suite to verify resolution.",
                    tool_calls=[
                        ToolCall(
                            name="terminal.run",
                            arguments={"command": "pytest test_solution.py"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    name="terminal.run",
                    content=fix_out,
                ),
                Message(
                    role="assistant",
                    content="All tests in `test_solution.py` passed with exit code 0. The issue is resolved and verified.",
                ),
            ],
            tools=[
                ToolDefinition(name="terminal.run", description="Execute command in shell"),
                ToolDefinition(name="filesystem.write_file", description="Write content to a file"),
            ],
            outcome=Outcome(
                success=True,
                tests_passed=len([l for l in fix_out.splitlines() if "PASSED" in l or "passed" in l]),
                tests_failed=0,
                exit_code=0,
                details="Verified via execution harness",
            ),
            metadata={
                "source": "hcscoder_synthetic_verified",
                "task_id": item["id"],
                "baseline_exit_code": baseline_code,
                "verified_exit_code": fix_code,
            },
        )
        traj.compute_fields()
        trajectories.append(traj)

    return trajectories
