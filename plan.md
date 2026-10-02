# HCSCoder 9B — Fully Autonomous Training, MoE, Benchmarking & Hugging Face Release Plan

**Version:** 1.0  
**Date:** 2026-10-02  
**Target:** Windows 11 Pro local autonomous multimodal CLI agent + Hugging Face cloud  
**Primary training UI:** `timfromhcs/autotrain-advanced`  
**Primary model target:** Qwen ~9B abliterated → **HCSCoder-9B**  
**Optional architecture experiment:** dense HCSCoder → sparse **HCSCoder-MoE**  
**Final distribution:** Hugging Face Hub + GGUF quantizations + complete model metadata  
**Core principle:** real execution, real verification, no fabricated success, no fake benchmark numbers.

---

## 0. Mission

Build an autonomous training/research pipeline that starts from an empty Windows 11 project directory and independently:

1. checks the local machine and available cloud credentials/credits,
2. discovers and pins the exact Qwen ~9B abliterated base model,
3. downloads and inventories public datasets,
4. records source licenses and provenance,
5. normalizes heterogeneous agent/tool/coding datasets into one common schema,
6. removes secrets, private data, broken records, benchmark leakage, duplicates and low-quality trajectories,
7. creates additional synthetic HCSCoder training data,
8. creates targeted data for:
   - agentic tool calls,
   - long-horizon reasoning,
   - planning,
   - multi-step tool use,
   - coding,
   - repository work,
   - terminal use,
   - test-driven verification,
   - self-healing,
   - failure recovery,
   - MCP-style tool schemas,
   - web research,
   - computer-use style actions,
   - HCSCoder identity and working style,
9. balances the final corpus by capability instead of by raw record count,
10. creates isolated train/validation/test splits with leakage prevention,
11. publishes private intermediate datasets to the user's Hugging Face namespace,
12. trains HCSCoder through the user's private `timfromhcs/autotrain-advanced` Space,
13. retrieves and verifies the resulting adapter/merged checkpoint,
14. runs deterministic local and cloud evaluations,
15. performs a controlled hyperparameter fine-tuning pass,
16. optionally converts the dense model into a sparse MoE using an explicit upcycling experiment,
17. trains the MoE checkpoint in Hugging Face cloud infrastructure,
18. benchmarks dense and MoE variants against the same held-out suites,
19. only promotes the MoE variant if it actually passes the predefined acceptance gates,
20. converts the final validated model to GGUF,
21. creates multiple GGUF quantizations,
22. verifies every GGUF locally with llama.cpp,
23. calculates SHA-256 checksums,
24. creates a complete, honest model card/README,
25. uploads all approved model files and metadata to the user's Hugging Face account,
26. writes a final machine-readable provenance/evaluation manifest,
27. stops when the release is reproducibly complete.

The agent MUST NOT claim a task is complete because code ran without an exception. Every important stage has explicit evidence requirements.

---

# 1. Non-negotiable operating rules

## 1.1 No fake success

The agent MUST NOT:

- invent benchmark results,
- claim a benchmark passed without running it,
- claim a training run completed without a cloud/job/log artifact,
- claim a model was uploaded without verifying the Hub revision,
- claim a GGUF works without loading it,
- claim an MoE conversion succeeded merely because tensors were written,
- silently substitute another model,
- silently substitute another dataset,
- silently change the base model,
- silently skip failed samples,
- silently discard difficult data without recording why.

Every final claim must point to an artifact:

```text
artifact
+ command
+ exit code
+ timestamp
+ git/hash/revision
+ log
```

## 1.2 Provenance is mandatory

Every external dataset, model and code repository receives:

```json
{
  "source": "...",
  "source_revision": "...",
  "license": "...",
  "license_confidence": "...",
  "download_timestamp": "...",
  "source_hash": "...",
  "commercial_use": "...",
  "notes": "..."
}
```

Do not treat a Hugging Face dataset license as automatically proving that every underlying GitHub repository/task is redistributable.

## 1.3 Secrets never enter the dataset

The pipeline must detect and reject/remove:

- Hugging Face tokens
- GitHub tokens
- AWS keys
- API keys
- OAuth tokens
- private SSH keys
- `.env` contents
- database credentials
- passwords
- private URLs containing credentials
- cookies/session tokens
- personal access tokens
- obvious credential-like blobs

Use secret scanners plus regex plus entropy heuristics.

## 1.4 No training on evaluation sets

Known benchmark test sets must be treated as evaluation-only.

Examples:

- BFCL held-out evaluation data
- SWE-bench Verified evaluation instances
- SWE-bench Pro V2 evaluation instances
- Terminal-Bench 2.0 evaluation tasks
- tau3/τ-bench evaluation tasks
- custom HCSCoder held-out tests

If an upstream training dataset contains an evaluation benchmark item, the agent must detect and exclude the item from HCSCoder training.

## 1.5 Human authorization boundaries

The agent may autonomously:

- download,
- preprocess,
- train,
- benchmark,
- upload private datasets/models,
- create branches/repos under the configured namespace,
- start paid Hugging Face Jobs only when a cost ceiling is configured.

The agent MUST stop before:

- spending above the configured budget,
- making an irreversible public release when `AUTO_PUBLIC_RELEASE=false`,
- overwriting an existing production model,
- deleting remote repositories,
- deleting irreplaceable local data,
- publishing private/proprietary source material.

---

# 2. Account and secret contract

The agent is designed to use the user's Hugging Face account, but credentials must be supplied as secrets rather than embedded in code.

Recommended environment variables:

```powershell
$env:HF_TOKEN="<WRITE_TOKEN>"
$env:HF_USERNAME="timfromhcs"
$env:HF_DATASET_NAMESPACE="timfromhcs"
$env:HF_MODEL_NAMESPACE="timfromhcs"
$env:HF_AUTOTRAIN_SPACE="timfromhcs/autotrain-advanced"
$env:HF_AUTO_PUBLIC_RELEASE="false"
$env:HF_MAX_CLOUD_BUDGET_USD="..."
```

Preferred authentication:

```powershell
hf auth login
hf auth whoami
```

For cloud Jobs, pass `HF_TOKEN` through the Jobs secret mechanism rather than writing the token into a script or command line. Hugging Face Jobs explicitly supports secret injection for authenticated model/dataset access and Hub pushes.

For Spaces, follow the Hugging Face recommendation to keep AutoTrain Advanced private and use a write token stored as a Space secret. Do not hard-code a token in the Space or repository.

References:

- Hugging Face Jobs authentication and secrets:
  https://huggingface.co/docs/hub/jobs-configuration
- Hugging Face Spaces secrets:
  https://huggingface.co/docs/hub/spaces-overview
- AutoTrain installation guidance:
  https://huggingface.co/docs/autotrain/getting_started

---

# 3. Exact base-model discovery

## 3.1 Default target

The user intent is a Qwen ~9B abliterated base.

The agent SHOULD currently inspect candidates such as:

```text
wangzhang/Qwen3.5-9B-abliterated
```

but MUST NOT silently assume this is the intended model.

There are multiple Qwen abliterated variants on the Hub, including Qwen3-8B and Qwen3.5-9B variants. The agent must discover and pin the exact repository selected by the configuration or user input.

Example current candidate:

```text
wangzhang/Qwen3.5-9B-abliterated
```

The repository currently describes itself as an abliterated `Qwen/Qwen3.5-9B` checkpoint and documents its immediate base provenance.

Alternative examples exist and MUST be treated as different models, not interchangeable aliases.

## 3.2 Pin a revision

After selecting the model:

```text
BASE_MODEL_REPO
BASE_MODEL_REVISION
BASE_MODEL_SHA256_MANIFEST
BASE_MODEL_LICENSE
BASE_MODEL_ARCHITECTURE
BASE_MODEL_TOKENIZER_REVISION
```

Never train an unpinned moving HEAD for the final reproducible run.

## 3.3 Base-model compatibility checks

Before training:

```text
config.json parses
tokenizer loads
model class loads
generation config loads
chat template exists or is explicitly supplied
dtype loads
parameter count matches expectation
architecture is supported by current transformers
PEFT targets can be identified
AutoTrain compatibility is tested
```

If AutoTrain cannot load the selected architecture, the agent must report the exact error and attempt a compatible training route rather than pretending the run is valid.

---

# 4. Required project structure

Start from an empty directory, e.g.:

```text
HCSCoder-9B/
│
├── GEMINI.md
├── README.md
├── PLAN.md
├── pyproject.toml
├── uv.lock
│
├── config/
│   ├── base.yaml
│   ├── sources.yaml
│   ├── quality.yaml
│   ├── mixture.yaml
│   ├── synth.yaml
│   ├── training.yaml
│   ├── moe.yaml
│   ├── benchmarks.yaml
│   └── release.yaml
│
├── src/
│   └── hcscoder_data/
│       ├── acquisition/
│       ├── normalization/
│       ├── filtering/
│       ├── dedup/
│       ├── classification/
│       ├── synthesis/
│       ├── validation/
│       ├── mixtures/
│       ├── training/
│       ├── moe/
│       ├── evaluation/
│       └── release/
│
├── scripts/
│   ├── bootstrap.ps1
│   ├── download_sources.ps1
│   ├── build_dataset.ps1
│   ├── synthesize.ps1
│   ├── upload_dataset.ps1
│   ├── run_autotrain.ps1
│   ├── run_jobs.ps1
│   ├── run_benchmarks.ps1
│   ├── run_moe.ps1
│   ├── convert_gguf.ps1
│   └── release.ps1
│
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── filtered/
│   ├── verified/
│   ├── final/
│   └── eval/
│
├── artifacts/
│   ├── reports/
│   ├── metrics/
│   ├── manifests/
│   ├── checkpoints/
│   ├── gguf/
│   └── logs/
│
├── external/
│   └── llama.cpp/
│
└── state/
    ├── pipeline.json
    ├── jobs.json
    └── checksums.json
```

---

# 5. Environment bootstrap

## 5.1 Local tools

The agent checks and installs as needed:

```text
Git
Git LFS
Python
uv
Hugging Face Hub CLI
Hugging Face Datasets
Transformers
TRL
PEFT
bitsandbytes
accelerate
datasets
safetensors
sentencepiece
tokenizers
pytest
ruff
mypy
polars
pyarrow
orjson
pydantic
scikit-learn
simhash/minhash implementation
secret scanner
```

Use a project-local virtual environment.

Do not pollute the system Python if avoidable.

## 5.2 Hardware discovery

Collect:

```text
OS version
CPU
RAM
GPU(s)
VRAM
driver versions
CUDA availability
Vulkan availability
disk free space
SSD/HDD type
Python version
Git version
hf CLI version
Transformers version
TRL version
PyTorch version
```

Local hardware is used for:

- preprocessing,
- validation,
- lightweight inference,
- quantization,
- GGUF verification,
- smoke benchmarks.

Cloud GPU is used for the expensive model training/synthesis/MoE stages.

---

# 6. Hugging Face data acquisition

The agent downloads source datasets using `hf`/`huggingface_hub` and stores a source manifest.

Recommended initial sources:

## 6.1 Tool calling

### ToolACE

Source:

```text
Team-ACE/ToolACE
```

Use for:

- tool selection,
- arguments,
- multi-tool interaction,
- API-call formatting.

Do not blindly reproduce all records; normalize, deduplicate and quality-filter them.

### BFCL

Use the public BFCL material primarily for development/evaluation where possible.

BFCL V4 explicitly evaluates agentic function calling, including web search, memory, multi-turn behavior and format sensitivity.

Do NOT train on the held-out benchmark set.

References:

- https://gorilla.cs.berkeley.edu/leaderboard
- https://github.com/OpenBMB/BFCL
- https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information

---

# 7. Coding-agent data

## 7.1 SWE-Zero

Dataset:

```text
nvidia/SWE-Zero-openhands-trajectories
```

Current dataset card:

- ~318k trajectories
- ~118k issues
- OpenHands trajectories
- synthetic trajectories generated by a strong Qwen coding model
- repository/task license metadata
- Apache/MIT/BSD-family source licensing in the described release

Use it as a major SFT source for:

```text
repository inspection
terminal usage
file edits
debugging
patch creation
agent trajectories
```

Do NOT assume every trajectory is equally good.

Reference:

https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories

## 7.2 SWE-Hero

Dataset:

```text
nvidia/SWE-Hero-openhands-trajectories
```

Use for:

```text
execution-oriented coding trajectories
tool interaction
repository tasks
```

Reference:

https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories

## 7.3 SI2CA

Dataset:

```text
Self-Improving-Coding-Agents/SI2CA-Training-Trajectories
```

Current dataset card describes:

- 32,340 coding-agent trajectories
- 10,780 executable Python SWE tasks
- standard sampling
- self-judgement
- recursive self-improvement strategy
- tool calls
- environment outputs
- patches
- test-based reward

This source is particularly relevant for:

```text
self-evaluation
self-judgement
recursive improvement
failure analysis
repair behavior
```

Reference:

https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories

## 7.4 SWE-Gym

Dataset:

```text
SWE-Gym/SWE-Gym
```

Use primarily as:

```text
task seeds
execution environments
synthetic trajectory generation inputs
```

Reference:

https://huggingface.co/datasets/SWE-Gym/SWE-Gym

## 7.5 R2E-Gym

Dataset:

```text
R2E-Gym/R2E-Gym-Subset
```

Use for:

```text
repository-level task diversity
task generation
coding-agent synthesis
```

Reference:

https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset

## 7.6 SWE-rebench V2

Dataset:

```text
ScaleAI/SWE-bench_Pro
nebius/SWE-rebench-V2
```

Use primarily as:

```text
task pool
execution environments
fresh synthetic trajectory generation
held-out evaluation where applicable
```

Never mix official evaluation tasks into training.

SWE-bench Pro V2 currently describes 642 validated tasks in Harbor format.

Reference:

https://huggingface.co/datasets/nebius/SWE-rebench-V2
https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro

---

# 8. Long-horizon agent data

Use a mix of:

```text
ToolGym Long-Horizon
OpenHands trajectories
SWE-Zero
SI2CA
self-generated tasks
custom HCSCoder tasks
```

Dataset:

```text
ToolGym/long-horizon-traj
```

Reference:

https://huggingface.co/datasets/ToolGym/long-horizon-traj

The pipeline should explicitly label trajectory length.

Example:

```text
short       = 2-5 meaningful actions
medium      = 6-10
long        = 11-20
very_long   = 21-40
extreme     = 41+
```

Do not optimize only for the longest trajectories.

---

# 9. Computer-use / browser data

Use public computer-use trajectories only as supplemental data and as seeds for synthesis.

Example source:

```text
markov-ai/computer-use
```

Reference:

https://huggingface.co/datasets/markov-ai/computer-use

Target capabilities:

```text
browser navigation
GUI interaction
VS Code
file selection
document workflows
multi-application workflows
```

Do not over-weight this category if HCSCoder's main role remains a coding/terminal agent.

---

# 10. Unified trajectory schema

Every source is normalized into a common internal schema.

Example:

```json
{
  "id": "uuid",
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "tool_calls": []
    },
    {
      "role": "tool",
      "name": "terminal.run",
      "content": "..."
    }
  ],
  "tools": [
    {
      "name": "terminal.run",
      "description": "...",
      "parameters": {}
    }
  ],
  "outcome": {
    "success": true,
    "tests_passed": 12,
    "tests_failed": 0
  },
  "metadata": {
    "source": "swe-zero",
    "source_id": "...",
    "repo": "...",
    "revision": "...",
    "license": "MIT"
  }
}
```

Add computed metadata:

```json
{
  "steps": 27,
  "tool_calls": 18,
  "unique_tools": 5,
  "tool_errors": 2,
  "retries": 3,
  "files_read": 15,
  "files_modified": 4,
  "tests_run": 9,
  "tests_passed": 9,
  "tests_failed": 0,
  "has_planning": true,
  "has_recovery": true,
  "has_verification": true,
  "long_horizon": true
}
```

---

# 11. Automatic classification

Every trajectory receives capability tags.

Required tags:

```text
general_instruction
coding
coding_explanation
code_generation
debugging
repository_work
terminal
git
tool_calling
multi_tool
parallel_tool_use
tool_selection
tool_error_recovery
planning
reasoning
long_horizon
self_healing
verification
testing
web_agent
computer_use
mcp
state_tracking
memory
self_judgement
hcs_identity
```

Classification should be primarily rule-based and metadata-based, with an optional LLM classifier for ambiguous cases.

Example:

```text
tool_calls > 0
→ tool_calling

unique_tools >= 2
→ multi_tool

steps >= 15
→ long_horizon

error/tool_error followed by changed corrective action
→ self_healing

test/build/lint after modification
→ verification

git commands present
→ git

repository + patch
→ repository_work
```

---

# 12. Data quality system

Each sample receives a quality record.

Suggested score components:

```text
execution evidence       30%
tool correctness         20%
task completion          15%
verification             15%
recovery quality         10%
trajectory coherence      5%
uniqueness                5%
```

Suggested buckets:

```text
GOLD       >= 0.95
HIGH       0.85-0.949
GOOD       0.70-0.849
AUX        0.50-0.699
REJECT     < 0.50
```

These thresholds are tunable; the agent must record the actual thresholds used.

Important:

An LLM judge alone is NOT enough to mark software work as successful.

For executable coding tasks, evidence should preferably include:

```text
exit code
test result
build result
lint result
type-check result
patch applied
expected files changed
```

---

# 13. Secret and private-data removal

Before deduplication and before training:

```text
scan text
scan tool outputs
scan shell transcripts
scan code
scan environment outputs
scan patches
scan logs
```

Redact or reject:

```text
API_KEY
HF_TOKEN
GITHUB_TOKEN
AWS_SECRET_ACCESS_KEY
PRIVATE KEY
password
cookie
session
authorization bearer
database URLs with credentials
```

When in doubt, reject the sample from public release.

Do not publish redacted data if the original source license prohibits redistribution.

---

# 14. Deduplication

Perform at least four levels.

## 14.1 Exact text hash

```text
SHA-256(normalized serialization)
```

## 14.2 Conversation fingerprint

Hash:

```text
user task
tool sequence
final patch
```

## 14.3 Semantic near-duplicate detection

Use MinHash/SimHash/embedding similarity.

## 14.4 Task-level deduplication

Group by:

```text
repository
issue
commit
benchmark item
task hash
```

If the same issue exists in SWE-Zero, SWE-Hero and SI2CA:

```text
keep the best trajectories
```

Do not count duplicates as independent evidence.

---

# 15. Leakage detection

Detect:

```text
same benchmark id
same repo+commit
same problem statement
same patch
near-identical task
same generated trajectory family
```

Then enforce:

```text
repository-disjoint where feasible
task-disjoint
benchmark-disjoint
revision-disjoint
```

Generate:

```text
train.jsonl
validation.jsonl
test.jsonl
```

and a `split_manifest.json`.

The split manifest must explain why every evaluation record is absent from training.

---

# 16. Synthetic data generation

Synthetic data is a core part of HCSCoder.

Do not simply ask a large model:

> "Generate agent examples."

Instead generate **executable tasks with known validation criteria**.

## 16.1 Task generator

Create tasks from templates:

```text
debug this code
fix failing test
add feature
refactor module
upgrade dependency
repair build
add CLI command
add REST endpoint
add configuration
fix type error
fix race condition
write migration
add tests
remove deprecated API
repair CI
implement parser
implement serializer
optimize slow function
repair broken docs build
```

Increase difficulty through:

```text
multiple files
ambiguous issue description
hidden test
dependency conflict
partial existing implementation
multiple interacting bugs
misleading first error
tool failure
test failure after first fix
secondary regression
```

---

# 17. Synthetic agent trajectory generation

For each generated task:

```text
1. create isolated repository
2. inject bug/issue
3. write tests
4. verify baseline failure
5. launch teacher agent
6. expose real tools
7. log every action
8. run tests after final patch
9. record exact outcome
```

Trajectory outputs:

```text
SUCCESS
SUCCESS_AFTER_RECOVERY
PARTIAL
FAILURE
```

Only:

```text
SUCCESS
SUCCESS_AFTER_RECOVERY
```

should normally become GOLD SFT.

Useful failed runs can become negative DPO candidates.

---

# 18. Synthetic failure injection

Explicitly generate self-healing examples.

Inject:

```text
wrong path
missing import
wrong dependency version
syntax error
type error
failing test
timeout
command not found
permission error in sandbox
incorrect argument
stale generated file
build failure
git conflict
unexpected JSON response
tool schema mismatch
network simulation failure
```

Target trajectory:

```text
action
→ error
→ inspect error
→ diagnose
→ corrective action
→ retry
→ verify
```

Reject a "recovery" when the agent simply repeats the same failed action without meaningful change.

---

# 19. Synthetic tool-calling data

Generate tool schemas automatically.

Tool families:

```text
filesystem
terminal
python
git
github
browser
web_search
web_open
http
json
database
test_runner
build_runner
package_manager
model_manager
dataset_manager
mcp
```

Create examples for:

```text
correct tool
wrong tool
multiple possible tools
missing parameters
optional parameters
nested arguments
parallel calls
multi-turn state
tool error
tool result interpretation
tool not needed
```

Critical training objective:

```text
tool result ≠ truth by itself
```

HCSCoder must inspect tool outputs and reason from them.

---

# 20. Synthetic long-horizon data

Generate tasks with explicit state:

```text
task
→ plan
→ inspect
→ action
→ observe
→ revise plan
→ action
→ verify
→ recover
→ continue
→ final verification
```

Target trajectory distributions:

```text
2-5 actions       15%
6-10              20%
11-20             25%
21-40             25%
41-80             12%
80+                3%
```

These are starting targets, not immutable rules.

The agent should update the mixture after benchmarking.

---

# 21. HCSCoder Identity dataset

This is a dedicated dataset.

Do not teach identity with thousands of copies of:

```text
"You are HCSCoder."
```

Instead teach identity through behavior.

Create examples of:

```text
inspect before edit
plan before large change
use tools instead of hallucinating
verify claims
report actual evidence
recover from failure
prefer minimal safe patches
respect repository architecture
ask for clarification only when truly required
do not invent tool results
do not claim success without verification
maintain state across long tasks
summarize changed files
summarize tests
identify remaining uncertainty
```

Example behavior:

```text
USER:
Implement feature X.

HCSCODER:
I will first inspect the repository structure and existing implementation, then identify the smallest compatible change. After editing I will run targeted tests and a broader verification pass. I will only report success after those checks return successfully.
```

Build a dedicated identity set of roughly:

```text
10k-50k
```

high-quality examples depending on diversity.

---

# 22. HCSCoder style specification

Create a machine-readable policy:

```yaml
identity:
  name: HCSCoder
  role: autonomous software engineering agent
  priorities:
    - correctness
    - evidence
    - tool discipline
    - minimal safe changes
    - verification
    - recovery
    - reproducibility

behavior:
  inspect_before_modify: true
  verify_before_claiming_success: true
  tool_result_must_be_read: true
  retry_failed_action_without_diagnosis: false
  destructive_actions_require_authorization: true
  fabricate_evidence: false
  fabricate_benchmarks: false
```

This is NOT a substitute for training; it is also used for synthetic data generation and runtime evaluation.

---

# 23. Final dataset mixture

Suggested starting mixture after filtering:

```text
tool calling                  18%
coding agent / repository     22%
planning + reasoning           8%
long horizon                  10%
self healing                  12%
verification / testing         8%
MCP / tool schemas             5%
web / research agents          4%
computer use                   3%
git workflows                  4%
HCSCoder identity              6%
```

The agent must compute actual token-weighted proportions after tokenization.

Token count matters more than raw record count.

Keep a report:

```text
mixture_before.json
mixture_after.json
token_distribution.json
capability_distribution.json
```

---

# 24. Dataset weighting

Do not just concatenate files.

Assign weights by:

```text
quality
difficulty
capability scarcity
verification strength
trajectory length
source diversity
license confidence
```

Example:

```text
gold executable recovery trajectory = high weight
generic short instruction = lower weight
duplicate trajectory = reject
unverified synthetic answer = reject
benchmark test = evaluation only
```

---

# 25. SFT dataset format for AutoTrain

AutoTrain Advanced currently supports JSONL/CSV for LLM fine-tuning, with SFT/DPO/ORPO modes and chat templates.

Preferred HCSCoder SFT representation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    },
    {
      "role": "tool",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

If the selected AutoTrain setup requires a `text` column, create that derived representation automatically using the base model's exact chat template.

Do not manually add arbitrary control tokens if the tokenizer/chat template already handles them.

---

# 26. DPO dataset

Create:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "chosen": [...],
  "rejected": [...]
}
```

Positive trajectory examples:

```text
inspect → diagnose → patch → test → recover → verify
```

Negative examples:

```text
hallucinated tool result
wrong tool
wrong file
no verification
repeated failed command
premature success claim
unnecessary destructive change
```

Use DPO only after the first strong SFT model exists.

---

# 27. Publish intermediate datasets privately

Create:

```text
timfromhcs/HCSCoder-9B-data-raw
timfromhcs/HCSCoder-9B-data-curated
timfromhcs/HCSCoder-9B-data-sft
timfromhcs/HCSCoder-9B-data-dpo
timfromhcs/HCSCoder-9B-eval
```

Exact names must be checked for availability before creation.

Do not publish raw datasets if their upstream licenses/terms prohibit redistribution.

The final dataset README must contain:

```text
source list
license matrix
filters
removed-data rules
deduplication
synthetic generation
generator models
date
versions
hashes
known limitations
```

---

# 28. Training through the user's AutoTrain Advanced Space

Primary Space:

```text
https://huggingface.co/spaces/timfromhcs/autotrain-advanced
```

The agent must first verify:

```text
Space reachable
Space running
Space belongs to the configured account
Space is private
Space has adequate hardware
AutoTrain version is recent enough
write-token configuration exists
```

The Space was created as a duplicate of the official AutoTrain Advanced Space.

Reference:

https://huggingface.co/spaces/timfromhcs/autotrain-advanced

---

# 29. AutoTrain execution strategy

Do not start the expensive final run immediately.

Run:

## Experiment A — smoke

```text
1-2k examples
short training
small number of steps
```

Verify:

```text
dataset loads
chat template works
loss decreases
checkpoint appears
model loads
Hub push works
```

## Experiment B — tool-heavy

Use a tool-heavy mixture.

Evaluate:

```text
tool accuracy
argument accuracy
format
no-tool decisions
```

## Experiment C — agent-heavy

Use:

```text
coding
planning
long horizon
verification
self healing
```

## Experiment D — balanced final SFT

Use the full curated mixture.

## Experiment E — DPO refinement

Run only after a stable SFT checkpoint exists.

---

# 30. AutoTrain experiment tracking

Every run gets:

```text
run_id
base_model
base_revision
dataset_revision
dataset_hash
config_hash
AutoTrain version
Transformers version
TRL version
PEFT version
hardware
seed
learning_rate
batch size
gradient accumulation
max sequence length
LoRA rank
LoRA alpha
target modules
quantization
steps/epochs
loss
eval loss
training duration
Hub output revision
```

Store as:

```text
artifacts/metrics/<run_id>.json
artifacts/logs/<run_id>.log
```

---

# 31. Initial training strategy

For a ~9B model with limited cloud budget:

1. LoRA/QLoRA SFT first.
2. Do NOT start with full dense pretraining.
3. Keep a BF16/F16 or full-weight reference only when actually needed.
4. Test several small runs before the final run.

Starting sweep:

```text
LoRA rank: 8, 16, 32
alpha:     16, 32, 64
LR:        5e-6, 1e-5, 2e-5
sequence:  8k, 16k, 32k
effective batch: 8-32
```

These are experiment ranges, not claims that a specific setting is optimal.

The best configuration is selected by held-out evaluation, not loss alone.

---

# 32. Long-context strategy

Do not immediately maximize context length.

Run staged experiments:

```text
8k
→ 16k
→ 32k
→ 64k
→ optionally 128k
```

Long context must only be enabled when:

```text
memory stable
loss stable
throughput acceptable
tool-call formatting remains stable
short-context performance does not regress
```

Measure:

```text
token efficiency
trajectory completion
tool-call accuracy
recovery success
context retention
```

---

# 33. Cloud compute orchestration

Use Hugging Face Jobs for autonomous cloud stages:

```text
dataset synthesis
batch inference
execution data generation
MoE conversion experiments
MoE continued training
benchmark runs
quantization if local hardware is insufficient
```

Example conceptual command:

```powershell
hf jobs uv run `
  --flavor a100-large `
  --timeout 6h `
  --secrets HF_TOKEN `
  scripts/cloud_train.py
```

The actual hardware must be selected dynamically according to model size and configured budget.

Hugging Face Jobs are designed for training, batch inference, dataset processing, experiments and benchmarks, and detached Jobs can continue running independently of the terminal session.

References:

- https://huggingface.co/docs/hub/jobs
- https://huggingface.co/docs/hub/jobs-training
- https://huggingface.co/docs/hub/jobs-configuration
- https://huggingface.co/docs/trl/jobs_training

---

# 34. Synthetic teacher selection

The agent may choose a teacher model from an allow-list after checking availability, price and license.

Candidate policy:

```text
1. strongest suitable open model available on HF cloud
2. coding-specialized teacher for SWE tasks
3. general reasoning teacher for planning
4. tool-use capable teacher for function calling
```

Avoid hard-coding one teacher forever.

The agent must record:

```text
teacher repo
revision
generation parameters
prompt version
temperature
sampling
timestamp
```

If a synthetic sample is teacher-generated but never executed, mark:

```text
synthetic_unverified=true
```

and do not mix it into GOLD executable training at the same weight.

---

# 35. Trajectory quality gates

A generated coding trajectory becomes GOLD only if:

```text
task baseline reproducible
agent actions recorded
patch recorded
tests executable
final expected tests pass
no forbidden secret
license allowed
trajectory parseable
tool calls valid
no unexplained hallucinated result
```

A self-healing GOLD trajectory additionally requires:

```text
at least one real failure
diagnostic evidence
meaningfully different corrective action
successful retry
final verification
```

---

# 36. HCSCoder custom agent harness

Build an independent local harness.

Interface:

```text
OpenAI-compatible endpoint
```

Tools:

```text
filesystem
terminal
git
python
web
browser
http
github
mcp
test
build
```

The harness records:

```text
request
system
tool schema
tool call
tool output
timings
tokens
errors
retries
final state
```

This harness becomes the common evaluation interface for:

```text
base model
SFT model
DPO model
MoE model
quantized GGUF models
```

---

# 37. HCSCoder evaluation suite

Do NOT rely on one benchmark.

## 37.1 BFCL V4

Purpose:

```text
function calling
agentic web search
memory
multi-turn
format sensitivity
```

Reference:

https://gorilla.cs.berkeley.edu/leaderboard

The benchmark's official material describes V4 as an agentic function-calling evaluation.

## 37.2 SWE-bench Verified

Use as a primary software-engineering benchmark.

Current official description:

```text
500 human-validated instances
```

Evaluate with a fixed agent harness and record:

```text
resolved
failed
timeout
invalid
patch
tests
tool calls
latency
tokens
```

Reference:

https://www.swebench.com/verified.html

## 37.3 SWE-bench Pro V2

Use for harder/longer software-engineering tasks.

Current public repository information describes:

```text
642 validated V2 tasks
```

Reference:

https://github.com/scaleapi/SWE-bench_Pro-os
https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro

## 37.4 Terminal-Bench 2.0

Use for long-horizon terminal work.

Reference:

https://github.com/harbor-framework/terminal-bench-2

Run through Harbor where practical.

## 37.5 τ-bench / τ3-bench

Use for stateful tool-agent-user interactions.

The current repository includes a 2026 grading update and supports domains such as:

```text airline
retail
telecom
mock
banking_knowledge
```

Reference:

https://github.com/sierra-research/tau2-bench

Pin the benchmark release/version used; do not compare results across incompatible grading revisions.

## 37.6 Custom HCSCoder benchmark

This is mandatory.

Build at least:

```text
HC-Tool-100
HC-Long-50
HC-SelfHeal-100
HC-Repo-100
HC-Verify-100
HC-MCP-50
HC-Web-50
HC-Memory-50
```

Each task has deterministic or test-based scoring where possible.

---

# 38. Metrics

Never report a single number only.

For each benchmark record:

```text
success rate
task completion
tool-call exactness
invalid tool calls
recovery rate
verification rate
average steps
median steps
tool errors
repeat-failure rate
time to success
tokens
cost
context length
```

For coding:

```text
pass@1
resolved %
tests passed
regressions
patch size
retries
time
```

For self-healing:

```text
failure detected %
correct diagnosis %
successful recovery %
verification %
loop rate
```

For tool use:

```text
valid tool %
correct tool %
correct argument %
no-tool precision
multi-tool success
parallel success
```

---

# 39. Baseline-first evaluation

Before any HCS training:

```text
run base Qwen abliterated model
```

Save:

```text
baseline_results.json
baseline_report.md
baseline_trace_samples.jsonl
```

Then compare:

```text
BASE
→ SFT
→ DPO
→ MoE
→ MoE+DPO
→ GGUF variants
```

Same tasks, same seeds, same harness, same tool definitions.

---

# 40. Regression gates

The agent must reject a candidate when it causes a major regression in critical dimensions.

Example configurable gates:

```yaml
tool_calling:
  regression_tolerance: 0.03

coding:
  regression_tolerance: 0.03

self_healing:
  regression_tolerance: 0.05

verification:
  regression_tolerance: 0.03

latency:
  max_relative_regression: 0.30
```

Do not hard-code these values as objective truth; treat them as project acceptance thresholds.

---

# 41. MoE conversion — IMPORTANT ARCHITECTURAL RULE

Do NOT describe the dense-to-MoE transformation as a simple file conversion.

A dense model does not automatically become a useful MoE by merely duplicating tensors.

The pipeline must treat MoE as a **new architecture plus a training stage**.

MoE computation:

```text
hidden state
   ↓
router
   ↓
top-k experts
   ↓
selected FFNs
   ↓
weighted aggregation
```

Hugging Face Transformers provides expert backends and expert-parallel infrastructure for supported MoE architectures.

References:

- https://huggingface.co/docs/transformers/experts_interface
- https://huggingface.co/docs/transformers/expert_parallelism

---

# 42. Recommended HCSCoder MoE experiment

Start small:

```text
experts = 4
top_k = 2
shared attention
expertized FFN/MLP blocks
```

Initialize experts from the validated dense HCSCoder checkpoint.

Two initialization strategies:

## Strategy A — cloned experts

```text
expert_0 = dense_ffn
expert_1 = dense_ffn
expert_2 = dense_ffn
expert_3 = dense_ffn
router ≈ uniform
```

Then continued training breaks symmetry.

## Strategy B — specialized initialization

Cluster representative activation/task types:

```text
coding
planning
tool use
debugging
```

Use this information to initialize expert differences where technically feasible.

The agent must run both only if compute permits.

---

# 43. MoE training objective

The MoE experiment must optimize:

```text
language-model loss
+
optional distillation loss
+
router load-balancing loss
```

Keep:

```text
attention weights
tokenizer
embeddings
LM head
```

from the dense model unless a measured reason justifies changing them.

Train the experts/router primarily.

Use expert parallelism when multiple GPUs are available.

Reference:

https://huggingface.co/docs/transformers/expert_parallelism

Research background on dense-to-MoE upcycling:

- https://arxiv.org/abs/2410.07524
- https://arxiv.org/abs/2604.19835
- https://arxiv.org/abs/2604.13508

These papers are methodological references, not proof that the resulting HCSCoder MoE will outperform the dense model.

---

# 44. MoE specialization experiments

After initial training, measure expert routing.

Collect:

```text
token → expert
task → expert
trajectory → expert
capability → expert
```

Check for:

```text
expert collapse
router collapse
dead experts
single-expert dominance
poor load balance
identical expert behavior
```

Target is not "perfectly equal usage".

The goal is useful specialization without catastrophic routing imbalance.

---

# 45. MoE acceptance criteria

Only promote MoE if:

```text
no catastrophic regression
tool use does not collapse
coding does not collapse
self healing improves or remains within gate
long horizon improves or remains within gate
verification behavior remains intact
latency/memory tradeoff is acceptable
model loads correctly
GGUF export works or a documented non-GGUF limitation is accepted
```

If MoE is worse:

```text
keep dense HCSCoder as production candidate
archive MoE as experiment
write exact failure analysis
do not advertise MoE as an improvement
```

---

# 46. Post-SFT fine-tuning loop

After first benchmark:

```text
collect failure traces
→ cluster failure types
→ synthesize targeted examples
→ execute/verify
→ append high-quality data
→ SFT micro-round
→ evaluate
```

Focus on actual failure patterns:

```text
wrong tool
premature answer
no verification
stuck loop
poor plan
tool result ignored
wrong file
regression introduced
long-context loss
weak recovery
```

Do not add synthetic examples just to increase dataset size.

---

# 47. DPO refinement loop

Create preference pairs from real benchmark traces:

```text
same task
candidate A succeeds
candidate B fails or is inferior
```

Prefer objective rejected candidates:

```text
test failure
invalid patch
wrong tool
invalid arguments
no verification
loop
```

Avoid preference labels based only on stylistic opinion.

Run:

```text
DPO small
→ benchmark
→ DPO larger
→ benchmark
```

Stop when improvements plateau or regressions appear.

---

# 48. GGUF conversion pipeline

Use current `llama.cpp`.

Qwen documentation currently describes:

```text
HF safetensors
→ convert_hf_to_gguf.py
→ F16/BF16 GGUF
→ llama-quantize
```

The current llama.cpp conversion script supports multiple output types and split output options, and current Qwen/llama.cpp materials document GGUF usage.

References:

- https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md
- https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/llama.cpp.md

For Qwen3.5 specifically, the agent MUST pin a recent llama.cpp revision and run a conversion smoke test before committing to the release path. Historical versions have had architecture-support gaps; never assume the converter supports the exact model class.

---

# 49. GGUF variants to produce

Preferred release matrix:

```text
HCSCoder-9B-BF16.gguf
HCSCoder-9B-Q8_0.gguf
HCSCoder-9B-Q6_K.gguf      (if supported/reasonable)
HCSCoder-9B-Q5_K_M.gguf
HCSCoder-9B-Q4_K_M.gguf
```

Optional:

```text
IQ4_XS
IQ3_M
IQ2_XXS
```

only if benchmarked and the resulting quality/compatibility is acceptable.

Do not publish dozens of quantizations merely for quantity.

---

# 50. Quantization verification

For every GGUF:

```text
load with llama.cpp
startup success
tokenizer/chat template works
one-shot generation
multi-turn generation
tool-call generation
long-context smoke test
```

Compare against BF16/F16 reference using a fixed mini-eval:

```text
tool_call_100
coding_50
recovery_50
reasoning_50
long_context_20
```

Record quality delta.

---

# 51. Local inference verification

At minimum:

```powershell
llama-cli -m HCSCoder-9B-Q4_K_M.gguf ...
```

and:

```powershell
llama-server -m HCSCoder-9B-Q4_K_M.gguf ...
```

Verify:

```text
OpenAI-compatible API
streaming
chat template
tool call format
context handling
```

The final report must record the exact llama.cpp commit.

---

# 52. Complete model release contents

Final Hugging Face model repo SHOULD contain:

```text
README.md
LICENSE
NOTICE.md
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.json or embedded equivalent
model.safetensors / shards
model.safetensors.index.json
adapter_config.json        (if adapter is released)
adapter_model.safetensors  (if adapter is released)
*.gguf
checksums.sha256
training_config.yaml
dataset_manifest.json
provenance.json
benchmark_results.json
eval_report.md
quantization_report.md
```

Only include files that are actually generated and necessary.

Do not create placeholder files just to satisfy a checklist.

---

# 53. Repository strategy

Recommended Hub structure:

```text
timfromhcs/HCSCoder-Qwen3.5-9B
timfromhcs/HCSCoder-Qwen3.5-9B-GGUF
timfromhcs/HCSCoder-Qwen3.5-9B-MoE
timfromhcs/HCSCoder-9B-Training-Data
timfromhcs/HCSCoder-9B-Eval
```

Exact names must be checked for collisions.

If the dense and MoE models become separate products, they should have separate README/model cards and separate provenance.

---

# 54. Honest README generation

The README MUST include:

## Model

```text
model name
version
architecture
parameter count
active parameters for MoE if applicable
context length
```

## Provenance

```text
base model
base revision
abliteration provenance
training date
training code commit
```

## Training

```text
SFT
DPO
LoRA/QLoRA details
rank
alpha
learning rate
sequence length
batch
steps
hardware
training duration
```

## Data

```text
datasets
source licenses
filtered counts
final counts
synthetic percentage
executed/verified percentage
```

## Evaluation

Never write:

```text
"state of the art"
"best"
"beats all"
```

unless directly supported by a documented external benchmark result and context.

Instead:

```text
HCSCoder achieved X/Y on our exact run
baseline achieved A/B
```

Include:

```text
benchmark version
commit/revision
environment
agent harness
model revision
temperature
seed
```

## Limitations

Explicitly list:

```text
tool hallucination
long-horizon failure modes
repository-specific brittleness
quantization losses
MoE instability if applicable
known unsupported architectures
```

---

# 55. SHA-256 manifest

Generate:

```text
SHA256  filename
```

for:

```text
model files
GGUFs
configs
dataset manifests
benchmark reports
release archives
```

The model README must link to the checksum manifest.

---

# 56. Hub upload

Use:

```powershell
hf upload <repo_id> <local_folder> .
```

The Hugging Face CLI supports uploading entire folders, including large model artifacts, and resumes interrupted uploads.

Reference:

https://huggingface.co/docs/huggingface_hub/en/guides/cli
https://huggingface.co/docs/huggingface_hub/en/guides/upload

After upload:

```text
hf repo info
```

and API-based validation must verify:

```text
repo exists
revision exists
required files exist
file sizes match
sha256 matches local artifact
README exists
license exists
model config loads
```

---

# 57. Public release policy

Default:

```text
AUTO_PUBLIC_RELEASE=false
```

Intermediate:

```text
private
```

Final model:

```text
private until all acceptance gates pass
```

When explicit auto-release is enabled, the agent can publish only after:

```text
all tests pass
all files verified
README generated
license verified
no secrets detected
benchmark report complete
checksums complete
```

---

# 58. State machine

The entire pipeline is represented as a resumable state machine:

```text
INIT
 ↓
ENV_CHECK
 ↓
AUTH_CHECK
 ↓
MODEL_DISCOVERY
 ↓
MODEL_PINNED
 ↓
SOURCE_ACQUISITION
 ↓
NORMALIZATION
 ↓
LICENSE_FILTER
 ↓
SECRET_FILTER
 ↓
DEDUP
 ↓
CAPABILITY_CLASSIFICATION
 ↓
QUALITY_FILTER
 ↓
SYNTHESIS
 ↓
EXECUTION_VERIFICATION
 ↓
MIXTURE_BUILD
 ↓
DATASET_RELEASE_PRIVATE
 ↓
AUTOTRAIN_SMOKE
 ↓
AUTOTRAIN_SFT
 ↓
BASELINE_EVAL
 ↓
SFT_EVAL
 ↓
TARGETED_DATA_REPAIR
 ↓
DPO
 ↓
DPO_EVAL
 ↓
MOE_EXPERIMENT
 ↓
MOE_TRAIN
 ↓
DENSE_VS_MOE_EVAL
 ↓
FINAL_MODEL_SELECTION
 ↓
GGUF_CONVERSION
 ↓
GGUF_VALIDATION
 ↓
README_GENERATION
 ↓
SHA256
 ↓
HUB_RELEASE
 ↓
POST_RELEASE_VERIFICATION
 ↓
DONE
```

Every state has:

```text
started_at
ended_at
status
inputs
outputs
logs
exit_code
```

---

# 59. Self-healing of the training pipeline

The autonomous agent itself must use the same philosophy as HCSCoder.

Example:

```text
download failed
→ inspect error
→ retry
→ switch mirror/method if permitted
→ verify checksum
```

Example:

```text
dataset parser failed
→ inspect malformed sample
→ isolate bad records
→ update parser
→ rerun
```

Example:

```text
AutoTrain job failed
→ inspect logs
→ classify OOM/format/auth/config issue
→ adjust one parameter
→ rerun
```

Example:

```text
GGUF fails to load
→ inspect architecture support
→ update pinned llama.cpp
→ rerun
→ if unsupported, STOP and report
```

Do not create infinite retries.

Recommended retry ceiling:

```text
3 attempts per failure class
```

Then escalate.

---

# 60. Cost controller

Before every cloud job:

```text
estimate
GPU type
GPU count
duration
estimated cost
budget remaining
```

If:

```text
estimated_cost > remaining_budget
```

do not start.

Record:

```text
budget_before
estimate
actual
budget_after
```

---

# 61. Agent tool policy

The autonomous agent should have tools:

```text
filesystem
shell
python
git
Hugging Face Hub
web search
HTTP
browser
image/video inspection if available
cloud job management
benchmark runner
model loader
tensor inspector
```

For every external result store:

```text
source
timestamp
content hash
```

This is especially important for current Hugging Face APIs and benchmark versions.

---

# 62. Multimodal supervision for the agent itself

The agent can use multimodal inspection when useful:

```text
training plots
loss curves
benchmark dashboards
terminal screenshots
web UI screenshots
error screenshots
model file listings
```

But visual inspection never substitutes for machine-readable validation.

For example:

A screenshot showing "Training complete" is not enough.

Require:

```text
training process exit success
checkpoint exists
checkpoint loads
Hub artifact exists
```

---

# 63. Model selection

At the end, the agent should create:

```text
candidate_matrix.csv
```

with rows:

```text
base
SFT-A
SFT-B
DPO
MoE-A
MoE-B
quantized variants
```

Columns:

```text
tool
coding
long_horizon
self_healing
verification
memory
latency
VRAM
context
GGUF compatibility
failure rate
```

Do not select based on a single aggregate score.

The configured release policy should state what trade-offs are acceptable.

---

# 64. Final HCSCoder definition

The released model should be described as:

```text
HCSCoder is a Qwen-derived software-agent model fine-tuned for
tool-mediated, long-horizon coding work, planning, verification,
failure recovery and self-healing agent loops.

It is intended to operate inside a real tool environment.
The model does not itself create tools; the surrounding agent runtime
provides tool schemas and executes tool calls.
```

Do not claim the model is autonomous outside a runtime that actually provides the required tools.

---

# 65. Expected final artifact tree

```text
release/
│
├── dense/
│   ├── safetensors/
│   ├── config/
│   └── tokenizer/
│
├── moe/
│   ├── safetensors/
│   └── config/
│
├── gguf/
│   ├── HCSCoder-9B-BF16.gguf
│   ├── HCSCoder-9B-Q8_0.gguf
│   ├── HCSCoder-9B-Q6_K.gguf
│   ├── HCSCoder-9B-Q5_K_M.gguf
│   └── HCSCoder-9B-Q4_K_M.gguf
│
├── reports/
│   ├── benchmark_results.json
│   ├── benchmark_report.md
│   ├── training_report.md
│   ├── data_report.md
│   ├── moe_report.md
│   └── quantization_report.md
│
├── manifests/
│   ├── provenance.json
│   ├── dataset_manifest.json
│   ├── split_manifest.json
│   └── checksums.sha256
│
└── README.md
```

---

# 66. Final acceptance checklist

The agent may finish only when all applicable requirements are true:

## Data

```text
[ ] sources downloaded
[ ] licenses recorded
[ ] secrets removed
[ ] malformed data removed
[ ] duplicates removed
[ ] benchmark leakage checked
[ ] capabilities classified
[ ] quality scores computed
[ ] synthetic data generated
[ ] synthetic data execution verified
[ ] final train/validation/eval split created
[ ] dataset manifest written
```

## Training

```text
[ ] AutoTrain smoke run succeeded
[ ] final SFT run succeeded
[ ] checkpoint loads
[ ] Hub revision verified
[ ] DPO run succeeded if enabled
[ ] training metrics saved
```

## MoE

```text
[ ] MoE architecture constructed
[ ] expert initialization recorded
[ ] router initialized
[ ] balancing monitored
[ ] MoE training actually ran
[ ] dense vs MoE benchmark completed
[ ] no unsupported "conversion-only" claim
```

## Evaluation

```text
[ ] BFCL or supported tool benchmark run
[ ] SWE-bench Verified run
[ ] SWE-bench Pro V2 or equivalent harder coding suite run when available
[ ] Terminal-Bench 2.0 run when environment allows
[ ] τ-bench/τ3-bench run when relevant
[ ] custom HCSCoder tests run
[ ] baseline recorded
[ ] candidate comparison recorded
```

## Release

```text
[ ] GGUF conversion succeeded
[ ] every GGUF loads
[ ] tool calls tested on GGUF
[ ] long context tested
[ ] checksums generated
[ ] README generated
[ ] provenance generated
[ ] license verified
[ ] no secrets remain
[ ] Hub repo contains expected files
[ ] uploaded files verified remotely
```

---

# 67. Final report format

Generate:

```text
FINAL_REPORT.md
```

with:

```text
1. Executive summary
2. Exact base model + revision
3. Training data sources
4. Final data composition
5. Synthetic generation methodology
6. Filtering
7. Training settings
8. AutoTrain run IDs
9. Dense results
10. DPO results
11. MoE results
12. Benchmark results
13. GGUF results
14. Known limitations
15. Cost
16. Hardware
17. Reproducibility
18. Release links
19. SHA-256
20. Failed experiments
```

Section 19/20 are mandatory.

A failed experiment is part of the scientific record.

---

# 68. Agent behavior in one sentence

The autonomous agent should behave as:

```text
collect → inspect → normalize → verify → synthesize → execute → filter
→ train → benchmark → diagnose → improve → benchmark again → package
→ verify → publish
```

not:

```text
download → train → claim success
```

---

# 69. Recommended implementation order

Build in this order:

```text
Phase 1
bootstrap + auth + source registry

Phase 2
download + normalize + quality filtering

Phase 3
dedup + leakage + license system

Phase 4
synthetic task generator + execution harness

Phase 5
HCSCoder identity dataset

Phase 6
private dataset release

Phase 7
AutoTrain smoke SFT

Phase 8
full SFT

Phase 9
baseline/SFT benchmark

Phase 10
targeted synthetic repair data

Phase 11
DPO

Phase 12
dense model finalization

Phase 13
MoE upcycling experiment

Phase 14
MoE cloud training

Phase 15
dense vs MoE evaluation

Phase 16
final candidate selection

Phase 17
GGUF

Phase 18
release
```

Do not skip directly to Phase 13.

---

# 70. Technical references

## Hugging Face AutoTrain

- AutoTrain LLM fine-tuning:
  https://huggingface.co/docs/autotrain/tasks/llm_finetuning
- AutoTrain Advanced:
  https://huggingface.co/docs/autotrain/main/tasks/llm_finetuning
- AutoTrain Python:
  https://huggingface.co/docs/autotrain/main/en/quickstart_py
- AutoTrain installation:
  https://huggingface.co/docs/autotrain/getting_started

## Hugging Face Jobs

- Jobs overview:
  https://huggingface.co/docs/hub/jobs
- Training:
  https://huggingface.co/docs/hub/jobs-training
- Configuration/secrets:
  https://huggingface.co/docs/hub/jobs-configuration
- TRL Jobs:
  https://huggingface.co/docs/trl/jobs_training

## Hugging Face Hub

- CLI:
  https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Upload:
  https://huggingface.co/docs/huggingface_hub/en/guides/upload
- Spaces secrets:
  https://huggingface.co/docs/hub/spaces-overview

## Datasets / agent trajectories

- ToolACE:
  https://huggingface.co/datasets/Team-ACE/ToolACE
- SWE-Zero:
  https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
- SWE-Hero:
  https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories
- SI2CA:
  https://huggingface.co/datasets/Self-Improving-Coding-Agents/SI2CA-Training-Trajectories
- SWE-Gym:
  https://huggingface.co/datasets/SWE-Gym/SWE-Gym
- R2E-Gym:
  https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset
- ToolGym long horizon:
  https://huggingface.co/datasets/ToolGym/long-horizon-traj
- Computer Use example:
  https://huggingface.co/datasets/markov-ai/computer-use
- SWE-rebench V2:
  https://huggingface.co/datasets/nebius/SWE-rebench-V2

## Benchmarks

- BFCL:
  https://gorilla.cs.berkeley.edu/leaderboard
- SWE-bench Verified:
  https://www.swebench.com/verified.html
- SWE-bench Pro:
  https://github.com/scaleapi/SWE-bench_Pro-os
- Terminal-Bench 2:
  https://github.com/harbor-framework/terminal-bench-2
- τ-bench:
  https://github.com/sierra-research/tau2-bench

## MoE

- Transformers experts:
  https://huggingface.co/docs/transformers/experts_interface
- Expert parallelism:
  https://huggingface.co/docs/transformers/expert_parallelism
- Dense-to-MoE upcycling research:
  https://arxiv.org/abs/2410.07524
  https://arxiv.org/abs/2604.19835
  https://arxiv.org/abs/2604.13508

## GGUF / llama.cpp

- Conversion:
  https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- Qwen + llama.cpp:
  https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md
- Qwen quantization:
  https://github.com/QwenLM/Qwen3/blob/main/docs/source/quantization/llama.cpp.md

---

# 71. Definition of Done

The project is DONE only when there is a real Hugging Face model revision that:

```text
loads
generates
uses its intended chat template
performs tool calls through the configured runtime
survives long-horizon tasks
recovers from selected failures
passes the project's verification suite
has benchmark evidence
has reproducible provenance
has complete model metadata
has verified GGUF builds
has checksums
has an honest README
```

and the agent can reproduce the release from the saved manifests without relying on hidden state.

The final README must clearly distinguish:

```text
what came from the base model
what came from public datasets
what was synthetically generated
what was actually executed
what was fine-tuned
what was experimentally converted
what was benchmarked
what was not tested
```

Never turn an experiment into a claim of capability.

---

## End of PLAN.md
