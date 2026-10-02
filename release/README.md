---
license: apache-2.0
base_model: wangzhang/Qwen3.5-9B-abliterated
tags:
  - code
  - agent
  - tool-calling
  - reasoning
  - hcscoder
pipeline_tag: text-generation
---

# HCSCoder-9B

HCSCoder-9B is an autonomous software-engineering and tool-calling agent model derived from `wangzhang/Qwen3.5-9B-abliterated`.

## Capabilities
- Multi-step agentic tool dispatch and API interaction
- Test-driven self-healing and failure recovery loops
- Repository inspection before editing (minimal safe patches)
- Evidence-grounded verification before claiming success

## Provenance & Training Data
- Public sources: `Team-ACE/ToolACE`, `nvidia/SWE-Zero-openhands-trajectories`, `Self-Improving-Coding-Agents/SI2CA-Training-Trajectories`, `ToolGym/long-horizon-traj`
- Synthetic data: Verified executable tasks with execution harness
- Filtering: Strict multi-stage deduplication, secret removal, and leakage prevention

## Model Checksums
See `checksums.sha256` for SHA-256 integrity verification.
