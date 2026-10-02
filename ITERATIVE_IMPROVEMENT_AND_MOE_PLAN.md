# HCSCoder 9B — Real Cloud Training, Hard Benchmarks, Automated Security Gates & MoE Upcycling Plan

**Document:** `ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md`  
**Version:** 3.0 (Expanded Production Architecture)  
**Date:** 2026-10-02  
**Target:** Real Cloud Execution (HF Jobs / GitHub Actions) + Strict Automated Security Gates + Hard Benchmarks (SWE-bench Pro, Terminal-Bench 2.0, BFCL V4, τ²-Bench) + Real 9B Dense-to-MoE Upcycling Release  
**Core Principle:** Real weights (~18 GB), real cloud GPU execution, automated security gates in code, zero mock stubs, verified provenance.

---

## 0. Executive Summary & Defect Remediation

### 0.1 Issues Identified & Fixed in V3.0
1. **Mock Artifact Elimination (Gate 4 Enforcement)**:
   Previous local smoke runs wrote minimal byte stubs. The pipeline now enforces automated file size and tensor header verification gates:
   - Dense Checkpoint Safetensors: $\ge 17.5\text{ GB}$ across shards.
   - MoE Checkpoint Safetensors (4 experts): $\ge 34.0\text{ GB}$ across shards.
   - GGUF Quantizations: BF16 $\ge 17.0\text{ GB}$, Q8_0 $\ge 9.5\text{ GB}$, Q4_K_M $\ge 5.2\text{ GB}$.
   Any file failing size or header integrity checks is rejected and blocked from release.
2. **Real Cloud Compute Execution**:
   Because local hardware (AMD Radeon APU with 512 MB VRAM and 10 GB free RAM) cannot fine-tune an 18 GB model in BF16, training is dispatched to verified cloud GPUs via **Hugging Face Jobs** (`hf jobs uv run --flavor a100-large`) or **GitHub Actions GPU Runners**.
   - Authenticated account: `timfromhcs` (`Can pay / billing: True`).
   - Hardware: Nvidia A100-Large (80 GB VRAM, 142 GB RAM) at \$2.50/hr billed per second, or Nvidia A10G-Large (24 GB VRAM, 46 GB RAM) at \$1.50/hr.
3. **Real Dense-to-MoE Upcycling (`mergekit-moe` & Tensor Sharder)**:
   Full parameter expansion across all 32 transformer layers:
   - Duplicates dense FFN layers (`gate_proj`, `up_proj`, `down_proj`) into 4 independent expert blocks.
   - Adds small Gaussian perturbation ($\sigma = 0.015 \cdot \text{std}(W)$) for symmetry breaking.
   - Initializes router projections (`gate.weight`: shape `[4, 4096]`).
   - Updates model configuration to MoE architecture and writes real Safetensors shards.
4. **Automated Security Gates in Code (`src/hcscoder_data/security/gates.py`)**:
   - Gate 1: Shannon Entropy + Regex Secret Scanner (Trufflehog/Gitleaks patterns).
   - Gate 2: SAST Vulnerability Scanner via **Bandit** & **Semgrep** (AST analysis).
   - Gate 3: Weight Deserialization Safety (Strict Safetensors enforcement; zero `.bin`/`.pkl` unpickling).
   - Gate 4: Artifact Size & Tensor Integrity Verification.
5. **Comprehensive, Honest Model Card**:
   Replaces minimal stubs with complete evaluation logs, parameter counts (total vs active), exact dollar costs, hardware logs, and abliteration limitation disclosures.

---

## 1. Automated Security Gates Architecture

All pipeline phases must pass through automated programmatic security gates before proceeding:

```
[ Code / Data / Checkpoint ]
              │
              ▼
   ┌───────────────────────┐
   │ Gate 1: Secret Scan   │ ──(Violations > 0)──> [ HALT & BLOCK ]
   │ (Entropy + Regex)     │
   └──────────┬────────────┘
              │ Passed
              ▼
   ┌───────────────────────┐
   │ Gate 2: SAST (Bandit) │ ──(High Severity > 0)──> [ HALT & BLOCK ]
   │ (AST Vulnerability)   │
   └──────────┬────────────┘
              │ Passed
              ▼
   ┌───────────────────────┐
   │ Gate 3: Weight Safety │ ──(Pickle / .bin Found)──> [ HALT & BLOCK ]
   │ (Safetensors Only)    │
   └──────────┬────────────┘
              │ Passed
              ▼
   ┌───────────────────────┐
   │ Gate 4: Anti-Mock     │ ──(Size < Min Threshold)──> [ REJECT RELEASE ]
   │ (File Size & Header)  │
   └──────────┬────────────┘
              │ Passed
              ▼
    [ PROCEED TO RELEASE ]
```

### Gate Implementation Matrix

| Gate | Target Artifact | Check Mechanism | Pass Criteria |
|---|---|---|---|
| **Gate 1** | Code, configs, datasets, commits | `SecurityGateManager.gate_1_secret_scan()` | 0 plaintext secrets, 0 unhashed tokens with entropy $> 4.6$ |
| **Gate 2** | Python modules (`src/`, `scripts/`) | `SecurityGateManager.gate_2_sast_vulnerability_scan()` | 0 High-severity Bandit issues (`eval`, `exec`, shell injection) |
| **Gate 3** | Downloaded & generated weights | `SecurityGateManager.gate_3_weight_deserialization_safety()` | 100% `.safetensors`, 0 `.bin`/`.pt`/`.pkl` pickle files |
| **Gate 4** | Model weights & GGUFs | `SecurityGateManager.gate_4_model_artifact_size_and_integrity()` | Valid header (GGUF magic / Safetensors dict), file size $\ge$ threshold |

---

## 2. Real Cloud GPU Execution Strategy

### 2.1 Hardware Flavors on Hugging Face Jobs
The Hugging Face account `timfromhcs` is verified and billing-enabled (`canPay: True`). The following cloud instances are available on-demand:

| Flavor | GPU | VRAM | System RAM | Rate / Hour | Best Used For |
|---|---|---|---|---|---|
| `a100-large` | 1x Nvidia A100 | 80 GB | 142 GB | \$2.50 | 9B LoRA SFT & MoE training (fastest throughput) |
| `l40sx1` | 1x Nvidia L40S | 48 GB | 62 GB | \$1.80 | SFT QLoRA & DPO fine-tuning |
| `a10g-large` | 1x Nvidia A10G | 24 GB | 46 GB | \$1.50 | Batch benchmark rollouts & dataset synthesis |
| `cpu-xl` | None (16 vCPU) | N/A | 124 GB | \$0.60 | GGUF conversion & CPU quantization |

### 2.2 Cloud Job Submission Command
Cloud training is executed using authenticated `hf jobs uv run`:

```bash
hf jobs uv run \
  --flavor a100-large \
  --secrets HF_TOKEN \
  --timeout 4h \
  scripts/train_cloud.py \
  --base_model wangzhang/Qwen3.5-9B-abliterated \
  --dataset_repo timfromhcs/HCSCoder-9B-Training-Data \
  --output_repo timfromhcs/HCSCoder-Qwen3.5-9B-SFT \
  --learning_rate 2e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 true
```

---

## 3. Real Dense-to-MoE Upcycling Architecture

### 3.1 Mathematical Specification
The 9B base model has 32 transformer layers. In each layer, the dense MLP consists of:
- `gate_proj`: $\mathbb{R}^{d_{\text{model}} \to d_{\text{ffn}}}$
- `up_proj`: $\mathbb{R}^{d_{\text{model}} \to d_{\text{ffn}}}$
- `down_proj`: $\mathbb{R}^{d_{\text{ffn}} \to d_{\text{model}}}$

For $E = 4$ experts:
1. Clone weights:
   $$W_{\text{expert}_i} = W_{\text{dense}} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}\left(0, (0.015 \cdot \sigma_{\text{weight}})^2\right)$$
2. Router Gate initialization:
   $$W_{\text{gate}} \in \mathbb{R}^{4 \times d_{\text{model}}}, \quad W_{\text{gate}} \sim \mathcal{N}(0, 0.02^2)$$
3. Loss function with Load Balancing:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{aux}} + 0.001 \cdot \mathcal{L}_{\text{z}}$$
   - $\mathcal{L}_{\text{aux}} = 4 \sum_{i=1}^4 f_i \cdot P_i$
   - $\mathcal{L}_{\text{z}} = \frac{1}{B} \sum \log^2 \sum \exp(z_j)$

### 3.2 Parameter Accounting
- Dense Base (9B): 9.0 billion parameters.
- Upcycled MoE (4 experts, Top-2):
  - Shared Attention & Embeddings: ~4.2 billion parameters.
  - 4x Expert MLPs: $4 \times 3.6 = 14.4$ billion parameters.
  - **Total Parameters**: ~18.6 billion parameters.
  - **Active Parameters per Token**: ~9.0 billion parameters.
- Sharded Safetensors Output: 5 shards of ~7.2 GB each (~36 GB total).

---

## 4. Benchmark Battery & Verification Protocol

### 4.1 Benchmark Suites & Execution Environments

| Benchmark | Tasks | Harness / Runner | Evaluation Metric |
|---|---|---|---|
| **SWE-bench Pro** | 642 (or 50 validated subset) | Harbor containerized runner | Resolved instances (AST patch + pytest) |
| **SWE-bench Verified** | 500 | OpenHands / HCS Harness | Resolved instances pass@1 |
| **Terminal-Bench 2.0** | 89+ | Harbor CLI (`terminal-bench@2.0`) | Task completion rate |
| **BFCL V4** | Multi-category | `bfcl-eval` AST evaluator | Web Search, Memory, Format exactness |
| **τ²-Bench** | Multi-domain | `tau2-bench` simulation | Deterministic state transition pass |
| **HC Custom Suites** | 450 total | `src/hcscoder_data/evaluation/` | Tool-100, SelfHeal-100, Verify-100, Repo-100, Long-50 |

---

## 5. Open Questions & User Authorization Boundaries

To proceed safely with cloud GPU runs and large model weights, the following decisions are aligned:

### Question 1: Cloud Budget Ceiling
- An A100-80GB training job runs at \$2.50/hour. A 2-hour SFT run costs ~\$5.00, and a 4-hour MoE training run costs ~\$10.00.
- **Configured Ceiling**: Default max \$25.00 USD. Does the user authorize starting paid cloud GPU jobs up to this ceiling?

### Question 2: Compute Backend Preference
- **Option A (Recommended)**: Hugging Face Jobs (`hf jobs uv run --flavor a100-large`). Already authenticated with `HF_TOKEN`, seamless Hub push.
- **Option B**: GitHub Actions with self-hosted GPU runner.
- **Option C**: Local CPU smoke testing with remote execution deferred to manual trigger.

### Question 3: MoE Capacity Selection
- **Option A (Recommended)**: 4 Experts (Top-2 active, ~18.6B total, ~9B active). Balanced memory footprint for inference.
- **Option B**: 8 Experts (Top-2 active, ~35B total, ~9B active). Higher capacity, requires ~70 GB storage.

### Question 4: Benchmark Scale
- **Option A (Recommended)**: Representative held-out subset (50 SWE-bench Pro instances + 50 Terminal-Bench 2.0 instances + full BFCL V4 sample) to optimize cloud cost and speed.
- **Option B**: Full exhaustive run of all 642 SWE-bench Pro instances (estimated ~12–16 GPU hours).

---

## 6. Execution Roadmap & Scripts

```text
scripts/
├── security/
│   └── run_security_gates.ps1        <- Scans secrets (Gate 1), runs Bandit (Gate 2), verifies safetensors (Gate 3)
├── cloud/
│   ├── submit_hf_training_job.ps1    <- Submits A100 training job via hf jobs CLI
│   └── monitor_hf_job.ps1            <- Streams logs and checks exit status
├── moe/
│   └── run_real_moe_upcycle.ps1      <- Real parameter upcycling across all 32 layers
├── benchmarks/
│   └── run_harbor_evals.ps1          <- Executes Harbor SWE-bench Pro & Terminal-Bench 2.0
└── release/
    └── verify_and_publish_hub.ps1    <- Anti-mock gate (Gate 4) + SHA-256 + Hugging Face upload
```

---
*End of ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md V3.0*
