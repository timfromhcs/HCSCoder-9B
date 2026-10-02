import json
import logging
from typing import Any, Dict, List, Optional
from hcscoder_data.normalization.schema import Trajectory

logger = logging.getLogger("TraceDiagnostics")

FAILURE_CLASSES = {
    "CLASS_A_TOOL_HALLUCINATION": "Calling a tool not in schema or with invalid argument types",
    "CLASS_B_LOOP_FAILURE": "Repeating the exact same command 3+ times without diagnostic changes",
    "CLASS_C_SYNTAX_IMPORT_ERROR": "Generated patch fails basic AST parse or imports nonexistent symbols",
    "CLASS_D_PREMATURE_TERMINATION": "Claiming success without executing verification tests",
    "CLASS_E_SECONDARY_REGRESSION": "Fixing the target bug but breaking existing test cases",
    "CLASS_F_CONTEXT_OVERFLOW": "Exceeding effective attention window and losing track of objective",
}


class FailureDiagnosticEngine:
    """Diagnoses and clusters agent failure traces into actionable defect categories."""

    def diagnose_trajectory(self, traj: Trajectory) -> Dict[str, Any]:
        diagnosis = {
            "trajectory_id": traj.id,
            "success": traj.outcome.success,
            "failure_class": None,
            "root_cause_turn": None,
            "evidence": "",
        }

        if traj.outcome.success:
            return diagnosis

        # Check for CLASS_B_LOOP_FAILURE
        commands_seen = {}
        for idx, msg in enumerate(traj.messages):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    cmd_str = f"{tc.name}:{str(tc.arguments)}"
                    commands_seen[cmd_str] = commands_seen.get(cmd_str, 0) + 1
                    if commands_seen[cmd_str] >= 3:
                        diagnosis["failure_class"] = "CLASS_B_LOOP_FAILURE"
                        diagnosis["root_cause_turn"] = idx
                        diagnosis["evidence"] = f"Command repeated {commands_seen[cmd_str]} times: {cmd_str[:80]}"
                        return diagnosis

        # Check for CLASS_C_SYNTAX_IMPORT_ERROR
        for idx, msg in enumerate(traj.messages):
            content = (msg.content or "").lower()
            if "syntaxerror" in content or "importerror" in content or "modulenotfounderror" in content:
                diagnosis["failure_class"] = "CLASS_C_SYNTAX_IMPORT_ERROR"
                diagnosis["root_cause_turn"] = idx
                diagnosis["evidence"] = content[:120]
                return diagnosis

        # Check for CLASS_A_TOOL_HALLUCINATION
        valid_tool_names = {t.name for t in traj.tools}
        if valid_tool_names:
            for idx, msg in enumerate(traj.messages):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.name not in valid_tool_names:
                            diagnosis["failure_class"] = "CLASS_A_TOOL_HALLUCINATION"
                            diagnosis["root_cause_turn"] = idx
                            diagnosis["evidence"] = f"Unknown tool '{tc.name}' not in {list(valid_tool_names)}"
                            return diagnosis

        # Check for CLASS_D_PREMATURE_TERMINATION
        if not traj.computed.has_verification:
            diagnosis["failure_class"] = "CLASS_D_PREMATURE_TERMINATION"
            diagnosis["evidence"] = "Assistant claimed completion without running test or verification commands"
            return diagnosis

        # Default to secondary regression or general failure
        diagnosis["failure_class"] = "CLASS_E_SECONDARY_REGRESSION"
        diagnosis["evidence"] = traj.outcome.details or "Tests failed after patch"
        return diagnosis

    def cluster_failures(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        clusters: Dict[str, List[Dict[str, Any]]] = {k: [] for k in FAILURE_CLASSES}
        clusters["UNCLASSIFIED"] = []

        for t in trajectories:
            if not t.outcome.success:
                diag = self.diagnose_trajectory(t)
                fc = diag.get("failure_class")
                if fc in clusters:
                    clusters[fc].append(diag)
                else:
                    clusters["UNCLASSIFIED"].append(diag)

        summary = {
            "total_failed_trajectories": sum(len(v) for v in clusters.values()),
            "cluster_distribution": {k: len(v) for k, v in clusters.items()},
            "details": clusters,
        }
        return summary
