# Harbor SWE-bench Pro Evaluation Runner
param(
    [string]$Dataset = "ScaleAI/SWE-bench_Pro",
    [int]$Limit = 20
)

Write-Output "==> Starting SWE-bench Pro evaluation via Harbor Framework (Limit=$Limit)..."
.\.venv\Scripts\python.exe scripts/benchmarks/eval_swe_pro.py --dataset $Dataset --limit $Limit
