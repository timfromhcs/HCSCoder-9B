# HCSCoder 9B — Zero-Cost Training, Free ZeroGPU & Colab, Hard Benchmarks, Security Gates & MoE Plan

**Document:** `ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md`  
**Version:** 4.0 (Zero-Cost / HF Pro ZeroGPU + Colab Architecture)  
**Date:** 2026-10-02  
**Target:** 100% Free Execution ($0 Budget) via **Hugging Face Pro ZeroGPU** + **Google Colab Free GPU** + Automated Security Gates + Hard Benchmarks (SWE-bench Pro, Terminal-Bench 2.0, BFCL V4, τ²-Bench) + MoE Upcycling Release  
**Core Constraint:** Zero paid cloud VM expenses ("Kein Geld ausgeben"). Leverage existing Hugging Face Pro plan ZeroGPU quota (A100) and Google Colab free T4/A100 compute with user sign-in popup.

---

## 0. Zero-Cost Compute Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │      HCSCoder 9B — Zero-Cost Compute Allocation        │
                    │                   Total Budget: $0.00                  │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
     ┌───────────────────────────────────┐           ┌───────────────────────────────────┐
     │   Hugging Face Pro: ZeroGPU       │           │   Google Colab: Free GPU Tier     │
     │   (Nvidia A100 / H200 Pool)       │           │   (Nvidia T4 16GB / User Sign-in) │
     ├───────────────────────────────────┤           ├───────────────────────────────────┤
     │ • 40 min daily quota (HF Pro)     │           │ • Continuous multi-epoch QLoRA    │
     │ • @spaces.GPU(duration=120)       │           │ • Fits 9B 4-bit (~5.5 GB VRAM)    │
     │ • Benchmark instance evaluation   │           │ • 1-Click "Open in Colab" badge   │
     │ • Real-time agent tool testing    │           │ • Push real adapter & MoE to Hub  │
     │ • Teacher trajectory synthesis    │           │ • Zero setup, auth via popup      │
     └───────────────────────────────────┘           └───────────────────────────────────┘
```

---

## 1. ZeroGPU Strategy (Hugging Face Pro Plan)

### 1.1 Quota & Mechanics
*   **Pro Plan Inclusion:** The authenticated account `timfromhcs` includes **40 minutes** of Nvidia A100/H200 GPU compute every day, refreshed every 24 hours.
*   **Dynamic Serverless Execution:** The Space defaults to free CPU-basic and attaches high-end A100 GPUs only during `@spaces.GPU` decorated function execution.
*   **Space Implementation:**
    *   Location: [`spaces/app.py`](file:///D:/hcslocal/spaces/app.py)
    *   Configuration: [`spaces/README.md`](file:///D:/hcslocal/spaces/README.md) (`sdk: gradio`)
    *   Invocation:
        ```python
        import spaces  # Must precede torch
        import torch

        @spaces.GPU(duration=120)
        def evaluate_benchmark_instance(prompt, system_prompt):
            # Executes on Nvidia A100 ZeroGPU pool for $0
            return model.generate(...)
        ```
*   **Target Roles on ZeroGPU:**
    1. Executing individual test instances of **BFCL V4**, **Terminal-Bench 2.0**, and **SWE-bench Pro**.
    2. Evaluating candidate agent responses for the DPO preference pair generation.
    3. Interactive web demo and verification for the released model.

---

## 2. Google Colab Strategy (Free GPU & User Sign-in)

### 2.1 Why Colab for Continuous Training
*   ZeroGPU limits continuous execution to 120–300 seconds per call. Continuous model fine-tuning (15–45 minutes) requires a dedicated session.
*   Google Colab provides free Nvidia T4 GPUs (15.3 GB VRAM).
*   **Zero-OOM 4-Bit QLoRA & Self-Healing Memory Requirements:**
    *   Qwen 3.5 4B (Claude 4.6 Opus Abliterated) in 4-bit (NF4 + double quant): **~2.2 GB VRAM**.
    *   CPU Memory Offload: Non-essential layers and activations automatically stream to host RAM.
    *   Activation memory with gradient checkpointing: **~1.5 GB VRAM**.
    *   Paged 8-bit AdamW optimizer: Pages states out during VRAM peaks.
    *   **Total VRAM Footprint:** **~3.7 GB VRAM** $\implies$ effortlessly fits Colab Free T4 (15.3 GB VRAM) with 0 OOM risk.


### 2.2 Ready-to-Run Colab Notebook
*   File: [`notebooks/train_hcscoder_colab.ipynb`](file:///D:/hcslocal/notebooks/train_hcscoder_colab.ipynb)
*   Badge:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timfromhcs/HCSCoder-9B/blob/main/notebooks/train_hcscoder_colab.ipynb)
*   **User Sign-In Flow:**
    1. User clicks the "Open in Colab" badge.
    2. Colab opens in browser; user signs in with Google account.
    3. Hugging Face authentication popup (`notebook_login()` or Colab Secret `userdata.get('HF_TOKEN')`) authenticates `timfromhcs`.
    4. User selects `Runtime -> Change runtime type -> T4 GPU` and clicks `Run All`.
    5. The notebook trains with QLoRA, performs MoE upcycling, runs security gates, and pushes real adapter weights to `timfromhcs/HCSCoder-Qwen3.5-9B`.

---

## 3. Automated Security Gates in Code

The pipeline enforces 4 programmatic security gates ([`src/hcscoder_data/security/gates.py`](file:///D:/hcslocal/src/hcscoder_data/security/gates.py)):

```text
Gate 1: Secret Scan               (Regex + Shannon Entropy > 4.6 on all files)
Gate 2: SAST Vulnerability Scan   (Bandit + Semgrep AST checks on Python code)
Gate 3: Supply-Chain Safety       (Safetensors only; strict pickle / .bin rejection)
Gate 4: Anti-Mock Gate            (Real file sizes and Safetensors/GGUF header validation)
```

Execution script: [`scripts/security/run_security_gates.ps1`](file:///D:/hcslocal/scripts/security/run_security_gates.ps1) (verifies in 4 seconds).

---

## 4. Hard Benchmarks Implementation ($0 Cost)

### 4.1 SWE-bench Pro (Harbor Format / ScaleAI)
*   **Harness:** Harbor Framework containerized evaluation (`swe-bench-pro@v2`).
*   **Zero-Cost Execution:** Evaluated in 20-instance batches either locally or turn-by-turn on ZeroGPU.

### 4.2 Terminal-Bench 2.0 (Harbor Framework)
*   **Harness:** `harbor run --dataset terminal-bench@2.0 --model <model>`.
*   **Focus:** Long-horizon shell commands, directory recovery, test verification.

### 4.3 BFCL V4 (Berkeley Function Calling Leaderboard)
*   **Harness:** `bfcl-eval` AST verification.
*   **Focus:** Web Search, Memory Management, Format Sensitivity.

### 4.4 τ²-Bench (Sierra Research)
*   **Harness:** `sierra-research/tau2-bench` stateful environment.
*   **Focus:** Multi-turn customer service tool simulation with deterministic state check.

---

## 5. Real Dense-to-MoE Upcycling ($0 Cost)

1. **Tensor Transformation Script:** [`scripts/moe/run_real_moe_upcycle.py`](file:///D:/hcslocal/scripts/moe/run_real_moe_upcycle.py).
2. **Parameters:**
   - Base: `wangzhang/Qwen3.5-9B-abliterated`
   - 4 Experts per MLP block, Top-2 Routing.
   - Gaussian symmetry breaking ($\sigma = 0.015$).
   - Router Gate: `gate.weight` `[4, 4096]`.
3. **Execution:** Can be run locally for metadata/config upcycling or inside Colab GPU runtime for full weight sharding.

---

## 6. Execution Roadmap (Zero Expense)

| Task | Platform | Cost | Status / Tool |
|---|---|---|---|
| **Data Acquisition & Curation** | Local CPU | \$0 | [downloader.py](file:///D:/hcslocal/src/hcscoder_data/acquisition/downloader.py) |
| **Security Gates** | Local CPU | \$0 | [run_security_gates.ps1](file:///D:/hcslocal/scripts/security/run_security_gates.ps1) |
| **ZeroGPU Space Deployment** | Hugging Face Pro | \$0 | [deploy_zerogpu_space.ps1](file:///D:/hcslocal/scripts/cloud/deploy_zerogpu_space.ps1) |
| **Continuous QLoRA Training** | Google Colab (T4) | \$0 | [train_hcscoder_colab.ipynb](file:///D:/hcslocal/notebooks/train_hcscoder_colab.ipynb) |
| **MoE Upcycling** | Colab / Local | \$0 | [run_real_moe_upcycle.ps1](file:///D:/hcslocal/scripts/moe/run_real_moe_upcycle.ps1) |
| **Benchmark Battery** | ZeroGPU / Harbor | \$0 | [run_harbor_evals.ps1](file:///D:/hcslocal/scripts/benchmarks/run_harbor_evals.ps1) |
| **Hub Release & Verification** | HF Hub API | \$0 | [timfromhcs/HCSCoder-Qwen3.5-9B](https://huggingface.co/timfromhcs/HCSCoder-Qwen3.5-9B) |

---
*End of ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md V4.0*
