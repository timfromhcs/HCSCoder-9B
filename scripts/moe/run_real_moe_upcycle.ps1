# Real MoE Upcycling Runner
param(
    [int]$NumExperts = 4,
    [int]$TopK = 2
)

Write-Output "==> Starting Real Dense-to-MoE Upcycling on Qwen 3.5 9B..."
.\.venv\Scripts\python.exe scripts/moe/run_real_moe_upcycle.py --num_experts $NumExperts --top_k $TopK
