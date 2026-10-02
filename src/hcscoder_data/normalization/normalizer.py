import json
import uuid
from typing import Any, Dict, List, Optional
from hcscoder_data.normalization.schema import (
    Message,
    Outcome,
    ToolCall,
    ToolDefinition,
    Trajectory,
)


class DataNormalizer:
    @staticmethod
    def normalize_toolace(sample: Dict[str, Any]) -> Trajectory:
        messages = []
        if sample.get("system"):
            messages.append(Message(role="system", content=sample["system"]))

        raw_convs = sample.get("conversations", [])
        tools = []
        for turn in raw_convs:
            role = "user" if turn.get("from") == "user" else "assistant"
            content = turn.get("value", "")
            tc_list = []
            if role == "assistant" and "[" in content and "]" in content:
                # Basic tool call detection in ToolACE format
                tc_list.append(ToolCall(name="function_call", arguments={"raw": content}))
            messages.append(Message(role=role, content=content, tool_calls=tc_list or None))

        traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=messages,
            tools=tools,
            outcome=Outcome(success=True),
            metadata={"source": "toolace", "license": "apache-2.0"},
        )
        traj.compute_fields()
        return traj

    @staticmethod
    def normalize_swe_zero(sample: Dict[str, Any]) -> Trajectory:
        messages = []
        raw_msgs = sample.get("trajectory", [])
        if isinstance(raw_msgs, str):
            try:
                raw_msgs = json.loads(raw_msgs)
            except Exception:
                raw_msgs = []

        for m in raw_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            tc = None
            if "tool_calls" in m:
                tc = [
                    ToolCall(
                        name=c.get("function", {}).get("name", "tool"),
                        arguments=c.get("function", {}).get("arguments", {}),
                    )
                    for c in m["tool_calls"]
                ]
            messages.append(Message(role=role, content=content, tool_calls=tc))

        traj = Trajectory(
            id=sample.get("trajectory_id") or str(uuid.uuid4()),
            messages=messages,
            outcome=Outcome(
                success=bool(sample.get("model_patch")),
                tests_passed=1 if sample.get("model_patch") else 0,
            ),
            metadata={
                "source": "swe_zero",
                "instance_id": sample.get("instance_id"),
                "repo": sample.get("repo"),
                "license": sample.get("license", "permissive"),
            },
        )
        traj.compute_fields()
        return traj

    @staticmethod
    def normalize_si2ca(sample: Dict[str, Any]) -> Trajectory:
        messages = []
        raw_msgs = sample.get("messages", [])
        if isinstance(raw_msgs, str):
            try:
                raw_msgs = json.loads(raw_msgs)
            except Exception:
                raw_msgs = []

        for m in raw_msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            messages.append(Message(role=role, content=content))

        tools = []
        raw_tools = sample.get("tools", [])
        if isinstance(raw_tools, list):
            for t in raw_tools:
                if isinstance(t, dict):
                    name = t.get("name") or t.get("function", {}).get("name", "tool")
                    desc = t.get("description") or t.get("function", {}).get("description", "")
                    params = t.get("parameters") or t.get("function", {}).get("parameters", {})
                    tools.append(ToolDefinition(name=name, description=desc, parameters=params))

        resolved = bool(sample.get("resolved"))
        agent_exit = sample.get("agent_exit_code") or 0
        traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=messages,
            tools=tools,
            outcome=Outcome(
                success=resolved,
                exit_code=agent_exit,
                tests_passed=1 if resolved else 0,
            ),
            metadata={
                "source": "si2ca",
                "instance_id": sample.get("instance_id"),
                "repo": sample.get("repo"),
                "license": "mit",
            },
        )
        traj.compute_fields()
        return traj

    @staticmethod
    def normalize_toolgym(sample: Dict[str, Any]) -> Trajectory:
        messages = []
        turns = sample.get("turns", [])
        if isinstance(turns, str):
            try:
                turns = json.loads(turns)
            except Exception:
                turns = []

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            messages.append(Message(role=role, content=content))

        traj = Trajectory(
            id=str(uuid.uuid4()),
            messages=messages,
            outcome=Outcome(success=True),
            metadata={
                "source": "toolgym_long",
                "summary": sample.get("summary", ""),
                "license": "apache-2.0",
            },
        )
        traj.compute_fields()
        return traj
