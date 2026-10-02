import copy
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from hcscoder_data.normalization.schema import Message, Outcome, Trajectory
from hcscoder_data.refinement.diagnostics import FailureDiagnosticEngine
from hcscoder_data.synthesis.harness import ExecutionHarness

logger = logging.getLogger("TrajectoryRepairEngine")


class TrajectoryRepairEngine:
    """Transforms failed agent trajectories into verified SFT repairs and DPO preference pairs."""

    def __init__(self, harness: ExecutionHarness):
        self.harness = harness
        self.diagnostic_engine = FailureDiagnosticEngine()

    def generate_dpo_pair(
        self,
        failed_traj: Trajectory,
        repaired_traj: Trajectory,
    ) -> Dict[str, Any]:
        """Creates chosen vs rejected pair for DPO alignment."""
        user_msgs = [m.model_dump() for m in failed_traj.messages if m.role in {"system", "user"}]
        chosen_msgs = [m.model_dump() for m in repaired_traj.messages if m.role not in {"system", "user"}]
        rejected_msgs = [m.model_dump() for m in failed_traj.messages if m.role not in {"system", "user"}]

        return {
            "id": f"dpo_{failed_traj.id}",
            "prompt": user_msgs,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
            "metadata": {
                "failure_class": failed_traj.metadata.get("failure_class", "unknown"),
            },
        }

    def repair_code_trajectory(
        self,
        broken_code: str,
        fixed_code: str,
        test_code: str,
        task_description: str,
    ) -> Tuple[Optional[Trajectory], Optional[Dict[str, Any]]]:
        """Executes sandbox verification and constructs both clean SFT repair and DPO preference pair."""
        # 1. Verify baseline fails
        base_pass, _, base_out = self.harness.execute_python_code(broken_code, test_code)
        if base_pass:
            logger.warning("Baseline test unexpectedly passed, skipping repair generation.")
            return None, None

        # 2. Verify fix passes
        fix_pass, _, fix_out = self.harness.execute_python_code(fixed_code, test_code)
        if not fix_pass:
            logger.warning("Fix code failed tests, rejecting candidate repair.")
            return None, None

        # 3. Build flawed (rejected) rollout
        failed_traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=[
                Message(role="system", content="You are HCSCoder, an autonomous coding agent."),
                Message(role="user", content=f"Fix the bug in solution.py: {task_description}"),
                Message(
                    role="assistant",
                    content="I believe the implementation is already correct without checking tests. Task complete!",
                ),
            ],
            outcome=Outcome(success=False, tests_passed=0, tests_failed=1),
            metadata={"failure_class": "CLASS_D_PREMATURE_TERMINATION"},
        )
        failed_traj.compute_fields()

        # 4. Build repaired (chosen) rollout
        repaired_traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=[
                Message(role="system", content="You are HCSCoder, an autonomous coding agent."),
                Message(role="user", content=f"Fix the bug in solution.py: {task_description}"),
                Message(role="assistant", content="Running tests to observe the baseline failure and traceback."),
                Message(role="tool", content=base_out),
                Message(role="assistant", content="Diagnosed root cause. Applying minimal verified patch to solution.py."),
                Message(role="tool", content="File written."),
                Message(role="assistant", content="Executing verification suite to confirm resolution."),
                Message(role="tool", content=fix_out),
                Message(role="assistant", content="Verification passed with 0 errors. Issue resolved."),
            ],
            outcome=Outcome(success=True, tests_passed=1, tests_failed=0),
            metadata={"repaired_from": failed_traj.id},
        )
        repaired_traj.compute_fields()

        # 5. Build DPO Pair
        dpo_pair = self.generate_dpo_pair(failed_traj, repaired_traj)

        return repaired_traj, dpo_pair
