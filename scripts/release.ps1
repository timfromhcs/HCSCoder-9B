# Release script
Write-Output "==> Quantizing GGUFs, verifying manifests and pushing to Hugging Face Hub..."
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from hcscoder_data.pipeline import HCSCoderPipeline; p = HCSCoderPipeline(); p.run_release_pipeline()"
