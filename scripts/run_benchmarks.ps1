# Run benchmarks script
Write-Output "==> Running evaluation on benchmark suites..."
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from hcscoder_data.pipeline import HCSCoderPipeline; p = HCSCoderPipeline(); p.run_training_and_eval()"
