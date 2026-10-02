import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ExecutionHarness:
    """Real local execution sandbox for validating synthetic tasks and patches."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "hcscoder_harness"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_command(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int = 30,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str]:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        try:
            res = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Execution timed out"
        except Exception as e:
            return 1, "", str(e)

    def execute_python_code(
        self,
        code: str,
        test_code: str,
    ) -> Tuple[bool, int, str]:
        """Creates a temporary workspace, writes code and test, and executes pytest/python."""
        task_id = tempfile.mkdtemp(prefix="task_", dir=self.base_dir)
        task_dir = Path(task_id)
        try:
            mod_file = task_dir / "solution.py"
            mod_file.write_text(code, encoding="utf-8")

            test_file = task_dir / "test_solution.py"
            test_file.write_text(test_code, encoding="utf-8")

            code_ret, out, err = self.run_command(
                [sys.executable, "-m", "pytest", "-v", str(test_file)],
                cwd=task_dir,
            )
            success = (code_ret == 0)
            combined = f"STDOUT:\n{out}\nSTDERR:\n{err}"
            return success, code_ret, combined
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)
