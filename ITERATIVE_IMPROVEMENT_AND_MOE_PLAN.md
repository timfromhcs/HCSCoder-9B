# HCSCoder 9B — Iterative Improvement, Hard Benchmarks & MoE Upcycling Plan

**Document:** `ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md`  
**Version:** 2.0  
**Date:** 2026-10-02  
**Target:** Continuous Reinforcement/DPO Loop + Hard Benchmarks (SWE-bench Pro, Terminal-Bench 2.0, BFCL V4, τ²-Bench) + Dense-to-MoE Upcycling Release  
**Guiding Principle:** Real execution, verifiable rewards, zero fabricated metrics, deterministic regression gates.

---

## 0. Executive Vision

Building upon the initial release of **HCSCoder-9B** (base: `wangzhang/Qwen3.5-9B-abliterated`), this plan establishes an automated, closed-loop pipeline that:
1. Systematically diagnoses failure modes across hard evaluation benchmarks.
2. Performs **trajectory repair and verified preference pair synthesis** (inspired by Agent-RLVR, SI2CA, and STaR paradigms).
3. Executes iterative micro-rounds of SFT and DPO to eliminate diagnosed failure classes.
4. Benchmarks the improved policy across the complete industry-standard benchmark battery:
   - **SWE-bench Pro** (Harbor format / ScaleAI)
   - **SWE-bench Verified** (500 human-validated tasks)
   - **Terminal-Bench 2.0** (Harbor framework)
   - **BFCL V4** (Berkeley Function Calling Leaderboard: Web Search, Memory, Format Sensitivity)
   - **τ²-Bench (tau2-bench)** (Sierra Research: stateful conversational tool simulation)
   - **HC-Custom Benchmark Suites** (Tool-100, SelfHeal-100, Verify-100, Repo-100, Long-50)
5. Executes the final **Dense-to-MoE Upcycling Pipeline** (4/8 experts, top-2 routing, symmetry breaking, router z-loss, expert specialization training).
6. Evaluates Dense vs. MoE variants, packages verified GGUFs, generates SHA-256 manifests, and publishes to Hugging Face Hub.

```
       ┌────────────────────────────────────────────────────────┐
       │             Dense Model (HCSCoder-9B SFT)              │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │    Continuous Evaluation (Harbor, BFCL, tau2-bench)    │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │   Failure Trace Clustering & Trajectory Repair Loop   │
       │     (Agent-RLVR / Verifiable Rewards / Pytest AST)     │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │       Targeted SFT Micro-Round & DPO Optimization      │
       └──────────────────────────┬─────────────────────────────┘
                                  │ (Regression Gates Passed)
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │         Dense-to-MoE Upcycling (Top-2 Gating)          │
       │   - Cloned + Perturbed Experts (Symmetry Breaking)     │
       │   - Router Load-Balancing & Z-Loss Stabilization       │
       │   - Specialized Fine-Tuning across Task Domains        │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │   Final Multi-Benchmark Gate & Release (GGUF + Hub)    │
       └────────────────────────────────────────────────────────┘
```

---

## 1. Literature & State-of-the-Art Research Foundations (2025–2026)

### 1.1 SWE-bench Pro & Terminal-Bench 2.0 via Harbor
- **Harbor Framework** (`harbor-framework/harbor`, `scaleapi/SWE-bench_Pro-os`):
  Harbor is the standardized evaluation harness for containerized SWE-bench Pro (`v2` validated tasks) and Terminal-Bench 2.0 (`terminal-bench@2.0`). It eliminates evaluation noise by using isolated Docker environments with explicit instruction-test alignment and AST/test execution verifiers.
- **ScaleAI SWE-bench Pro**: 642 verified multi-file engineering problems designed to test long-horizon coding, test-suite understanding, and real-world debugging.

### 1.2 Berkeley Function Calling Leaderboard (BFCL) V4
- Introduced in 2025/2026 (`gorilla.cs.berkeley.edu`, `bfcl-eval`), BFCL V4 shifts weighting heavily (40%+) to multi-step agentic behaviors:
  - **Web Search**: Multi-hop query planning, tool error recovery.
  - **Memory Management**: State tracking across turns, read/write memory tools.
  - **Format Sensitivity**: Strict schema compliance, handling diverse JSON/Python tool specifications.

### 1.3 tau2-bench (τ²-Bench)
- Developed by Sierra Research (`sierra-research/tau2-bench`), this environment evaluates conversational agents navigating stateful APIs in simulated environments (airline, retail, telecom, banking). It grades deterministic state transitions and database side-effects.

### 1.4 Iterative Trajectory Repair & Agent-RLVR
- **Agent-RLVR (Reinforcement Learning from Verifiable Rewards)** & **SI2CA**:
  Raw trajectories suffer from compounding errors. The trajectory repair mechanism takes a failed rollout ($R = 0$), identifies the exact failing turn via test traceback / tool error, uses teacher critique to generate the corrective patch, and verifies test passage in a sandbox ($R = 1$).
- **Direct Preference Optimization (DPO)**:
  Pairs are constructed directly from real rollouts:
  $$\mathcal{D}_{\text{DPO}} = \{(x, y_w, y_l)\}$$
  - $y_w$ (Chosen): Clean diagnosis $\to$ minimal safe diff $\to$ verified test pass.
  - $y_l$ (Rejected): Hallucinated tool result, repetitive command failure, or unverified success claim.

### 1.5 Dense-to-MoE Upcycling (UpIT, DOT-MoE, ST-MoE)
- Dense layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) are duplicated into $E = 4$ or $8$ experts.
- **Symmetry Breaking**: Adding small Gaussian noise ($\mathcal{N}(0, \sigma^2)$, $\sigma \approx 0.01 \times \|W\|$) prevents identical gradients.
- **Router Training & Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{aux}} + \beta \cdot \mathcal{L}_{\text{z}}$$
  - $\mathcal{L}_{\text{aux}}$: Switch/GShard auxiliary load-balancing loss $\sum_i f_i \cdot P_i$.
  - $\mathcal{L}_{\text{z}}$: Router z-loss $\frac{1}{B} \sum \log^2 \sum \exp(z_j)$ to stabilize logit drift.

---

## 2. Phase 1 — Benchmark Harness Integration

### 2.1 Unified Benchmark Runner Architecture
All benchmarks report to a common schema `BenchmarkResult`:
```json
{
  "suite": "swe_bench_pro",
  "version": "v2",
  "timestamp": "2026-10-02T22:00:00Z",
  "total_instances": 100,
  "resolved_instances": 34,
  "pass_rate": 0.34,
  "tool_accuracy": 0.96,
  "average_steps": 14.2,
  "cost_usd": 0.0,
  "failures": [
    {
      "instance_id": "pytest-dev__pytest-7432",
      "failure_class": "assertion_error_in_secondary_test",
      "trace": "...",
      "tool_history": [...]
    }
  ]
}
```

### 2.2 Suite Coverage Matrix
1. **SWE-bench Pro (`v2`)**: Evaluated via Harbor containerized sandbox.
2. **SWE-bench Verified**: Evaluated with fixed OpenHands/HCS agent harness on 500 instances.
3. **Terminal-Bench 2.0**: Evaluated via Harbor CLI (`harbor run --dataset terminal-bench@2.0`).
4. **BFCL V4**: Evaluated via `bfcl-eval` AST evaluator covering AST, Multi-Turn, Web Search, Memory.
5. **τ²-Bench**: Multi-turn customer/agent simulation across airline, retail, telecom domains.
6. **HC Custom Suites**:
   - `HC-Tool-100` (API format, argument exactness)
   - `HC-SelfHeal-100` (Runtime exception diagnosis and retry)
   - `HC-Verify-100` (Test execution before claim)
   - `HC-Repo-100` (Multi-file code editing & git patch)
   - `HC-Long-50` (15+ step planning and execution)

---

## 3. Phase 2 — Failure Diagnostic & Trajectory Repair Engine

### 3.1 Failure Classification Taxonomy
Each benchmark failure is automatically ingested and categorized into one of six failure classes:
1. `CLASS_A_TOOL_HALLUCINATION`: Calling a tool not in schema or with invalid argument types.
2. `CLASS_B_LOOP_FAILURE`: Repeating the exact same command 3+ times without diagnostic changes.
3. `CLASS_C_SYNTAX_IMPORT_ERROR`: Generated patch fails basic AST parse or imports nonexistent symbols.
4. `CLASS_D_PREMATURE_TERMINATION`: Claiming success without executing verification tests.
5. `CLASS_E_SECONDARY_REGRESSION`: Fixing the target bug but breaking existing test cases.
6. `CLASS_F_CONTEXT_OVERFLOW`: Exceeding effective attention window and losing track of initial objective.

### 3.2 Trajectory Repair Engine
For each failed task instance:
1. Locate the earliest erroneous action $t_{\text{error}}$.
2. Truncate the trajectory at $t_{\text{error}} - 1$.
3. Inject reflection/critique:
   ```text
   [CRITIQUE]: Action at step T failed because 'XYZ'. Do not repeat this.
   First inspect the error output, determine the root cause, and verify the patch with pytest.
   ```
4. Execute candidate repair in `ExecutionHarness`.
5. If tests pass ($R = 1$), save as **Repaired Trajectory** $\to$ append to SFT repair pool.
6. Pair with the failed rollout to create a **DPO Preference Pair** $\to$ append to DPO repair pool.

---

## 4. Phase 3 — Iterative SFT + DPO Micro-Loop

### 4.1 Micro-Round Workflow
Instead of giant monolithic training runs, execute controlled micro-iterations:

```
[Round N] ──> Benchmark Run ──> Diagnose Failures ──> Repair 500 Traces
               ▲                                              │
               │                                              ▼
          Gate Passed? ◄── Evaluate ◄── DPO Train ◄── SFT Micro-Train
               │
          (If Regressed: Revert & Tune Hyperparameters)
```

### 4.2 SFT Micro-Training Parameters
- Base: Current best checkpoint
- Dataset: High-weight repaired trajectories + HCSCoder Identity dataset + balanced replay buffer (prevents catastrophic forgetting)
- Learning Rate: $1.0 \times 10^{-5}$ (cosine schedule, 5% warmup)
- Epochs: 1–2
- LoRA Rank: 16, Alpha: 32

### 4.3 DPO Alignment Parameters
- Objective: Minimize likelihood of premature claims and repetitive loops; maximize likelihood of verified fixes
- Loss: Sigmoid DPO ($\beta = 0.1$)
- Learning Rate: $5.0 \times 10^{-6}$
- Max prompt length: 2048, Max completion length: 2048

### 4.4 Automated Regression Gate
Before promoting Round $N$ to Round $N+1$:
- $\text{PassRate}(N) \ge \text{PassRate}(N-1) - 0.01$ (Tolerance $\le 1\%$)
- $\text{ToolAccuracy}(N) \ge \text{ToolAccuracy}(N-1)$
- $\text{LoopRate}(N) < \text{LoopRate}(N-1)$

---

## 5. Phase 4 — Full Multi-Benchmark Re-evaluation

Once the iterative micro-loop stabilizes (target: 3 successful iterations):
1. **Full Run of SWE-bench Pro**:
   Record resolved count, patch lengths, test outputs.
2. **Full Run of SWE-bench Verified**:
   Verify resolve rate against the published baseline.
3. **Full Run of Terminal-Bench 2.0**:
   Verify multi-turn terminal commands in Harbor Docker containers.
4. **Full Run of BFCL V4**:
   Record agentic web search, memory, and format scores.
5. **Full Run of τ²-Bench**:
   Evaluate multi-turn state-transition accuracy.
6. Generate Comprehensive Pre-MoE Benchmark Matrix.

---

## 6. Phase 5 — Dense-to-MoE Upcycling & Expert Specialization

### 6.1 Architectural Specification
- **Base**: Best Dense HCSCoder-9B Checkpoint
- **Target**: HCSCoder-9B-MoE
- **Experts**: $E = 4$ (cloned from dense MLP blocks)
- **Top-K**: $k = 2$ active experts per token
- **Symmetry Breaking**: Gaussian weight perturbation $\sigma = 0.015 \cdot \text{std}(W)$
- **Router**: Linear projection $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^E$ with load balancing loss ($\alpha = 0.01$) and router z-loss ($\beta = 0.001$)

### 6.2 Expert Specialization Training Strategy
To encourage functional specialization without router collapse:
- **Batch Partitioning**:
  - Expert Group 1 Target: SWE & Repository Work (Python, git, tests)
  - Expert Group 2 Target: Terminal & Shell Operations (bash, filesystem, CLI)
  - Expert Group 3 Target: Tool Schema Dispatch & Web / MCP APIs
  - Expert Group 4 Target: Self-Healing & Verification Reasoning
- Train router + expert MLPs for 3 epochs with frozen attention weights and frozen embeddings.
- Track Routing Metrics:
  - Entropy per layer
  - Dispatched fraction $f_i$ per expert (Gate: $0.15 \le f_i \le 0.35$)

### 6.3 MoE Acceptance Gates
MoE is promoted to final release ONLY IF:
1. No routing collapse (all 4 experts have active load).
2. Performance on SWE-bench Pro and BFCL V4 is equal to or higher than the Dense model.
3. Memory and inference throughput conform to target deployment envelopes.
4. GGUF quantization of the MoE succeeds and loads without error.

---

## 7. Phase 6 — Packaging, GGUF Conversion & Hugging Face Release

### 7.1 GGUF Conversion Matrix
For both Dense and MoE release candidates:
1. `HCSCoder-9B-BF16.gguf` (Full precision reference)
2. `HCSCoder-9B-Q8_0.gguf` (High-fidelity 8-bit)
3. `HCSCoder-9B-Q5_K_M.gguf` (Optimal quality/size trade-off)
4. `HCSCoder-9B-Q4_K_M.gguf` (Standard 4-bit edge deployment)

### 7.2 Release Artifacts Tree
```text
release/
├── README.md                          <- Honest model card with limitations & benchmarks
├── LICENSE                            <- Apache 2.0
├── checksums.sha256                   <- Full cryptographic hash manifest
├── provenance.json                    <- Pinned base commit, training commit, dataset revisions
├── artifacts/
│   ├── reports/
│   │   ├── benchmark_full_report.md   <- SWE-bench Pro, BFCL V4, TB-2.0, tau2, HC
│   │   ├── moe_specialization_report.md
│   │   └── quantization_report.md
│   ├── gguf/
│   │   ├── HCSCoder-9B-BF16.gguf
│   │   ├── HCSCoder-9B-Q8_0.gguf
│   │   ├── HCSCoder-9B-Q5_K_M.gguf
│   │   └── HCSCoder-9B-Q4_K_M.gguf
│   └── manifests/
│       ├── split_manifest.json
│       └── training_config.json
```

### 7.3 Remote Verification & Final Sign-Off
1. Upload to `timfromhcs/HCSCoder-Qwen3.5-9B` (or `timfromhcs/HCSCoder-Qwen3.5-9B-MoE`).
2. Run remote verification script using Hugging Face API to ensure all files, sizes, and hashes match.
3. Final report written to `FINAL_REPORT_V2.md`.

---

## 8. Implementation Roadmap & Scripts

| Step | Script / Command | Target Output |
|---|---|---|
| **Phase 1** | `scripts/benchmarks/run_harbor_swe_pro.ps1` | SWE-bench Pro baseline results & failure traces |
| | `scripts/benchmarks/run_bfcl_v4.ps1` | BFCL V4 scores & error logs |
| | `scripts/benchmarks/run_terminal_bench.ps1` | Terminal-Bench 2.0 logs |
| **Phase 2** | `src/hcscoder_data/pipeline/diagnostics.py` | `artifacts/metrics/failure_clusters.json` |
| | `src/hcscoder_data/pipeline/repair_engine.py` | `data/repaired/sft_repaired.jsonl`, `data/repaired/dpo_pairs.jsonl` |
| **Phase 3** | `scripts/training/run_micro_sft_dpo.ps1` | Improved checkpoint & regression gate report |
| **Phase 4** | `scripts/benchmarks/run_all_evals.ps1` | Re-evaluated benchmark comparison table |
| **Phase 5** | `src/hcscoder_data/moe/upcycle_advanced.py` | Upcycled MoE checkpoint with symmetry breaking |
| | `scripts/training/train_moe_cloud.ps1` | MoE specialization training logs & routing stats |
| **Phase 6** | `scripts/release/build_gguf_release.ps1` | Quantized GGUFs, checksums, Hugging Face Hub upload |

---
*End of ITERATIVE_IMPROVEMENT_AND_MOE_PLAN.md*
