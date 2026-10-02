from typing import List, Set
from hcscoder_data.normalization.schema import Trajectory


class CapabilityClassifier:
    def classify(self, traj: Trajectory) -> List[str]:
        tags: Set[str] = set()
        text_corpus = " ".join([m.content or "" for m in traj.messages]).lower()

        # Tool calling & multi-tool
        if traj.computed.tool_calls > 0:
            tags.add("tool_calling")
            if traj.computed.unique_tools >= 2:
                tags.add("multi_tool")
            tags.add("tool_selection")

        # Long horizon
        if traj.computed.steps >= 10 or traj.computed.long_horizon:
            tags.add("long_horizon")

        # Self-healing and recovery
        if traj.computed.has_recovery or traj.computed.tool_errors > 0:
            tags.add("self_healing")
            tags.add("tool_error_recovery")

        # Planning & reasoning
        if traj.computed.has_planning or "plan:" in text_corpus or "let's think" in text_corpus:
            tags.add("planning")
            tags.add("reasoning")

        # Verification & testing
        if traj.computed.has_verification or "pytest" in text_corpus or "assert" in text_corpus:
            tags.add("verification")
            tags.add("testing")

        # Coding, repository & git
        if "def " in text_corpus or "class " in text_corpus or "return " in text_corpus or "import " in text_corpus:
            tags.add("coding")
            tags.add("code_generation")
        if "git " in text_corpus or "git diff" in text_corpus or "commit" in text_corpus:
            tags.add("git")
        if "patch" in text_corpus or "repo" in text_corpus or traj.metadata.get("repo"):
            tags.add("repository_work")
        if "terminal" in text_corpus or "bash" in text_corpus or "powershell" in text_corpus:
            tags.add("terminal")
        if "bug" in text_corpus or "fix" in text_corpus or "debug" in text_corpus:
            tags.add("debugging")

        # MCP, Web, Computer use
        if "mcp" in text_corpus or "tools/call" in text_corpus:
            tags.add("mcp")
        if "browser" in text_corpus or "http" in text_corpus or "url" in text_corpus or "search" in text_corpus:
            tags.add("web_agent")
        if "mouse" in text_corpus or "screenshot" in text_corpus or "key_press" in text_corpus:
            tags.add("computer_use")

        # HCS identity
        if "hcscoder" in text_corpus or traj.metadata.get("source") == "hcs_identity":
            tags.add("hcs_identity")

        if not tags:
            tags.add("general_instruction")

        result = sorted(list(tags))
        traj.computed.capabilities = result
        return result
