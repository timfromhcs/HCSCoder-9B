# Run Complete Multi-Benchmark Evaluation Battery
Write-Output "==> [1/4] Running BFCL V4 Benchmark..."
powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_bfcl_v4.ps1 -Limit 50

Write-Output "==> [2/4] Running Terminal-Bench 2.0 Benchmark..."
powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_terminal_bench.ps1 -Limit 25

Write-Output "==> [3/4] Running SWE-bench Pro Benchmark..."
powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_harbor_swe_pro.ps1 -Limit 20

Write-Output "==> [4/4] Aggregating Full Benchmark Comparison Matrix..."
.\.venv\Scripts\python.exe scripts/benchmarks/aggregate_reports.py
