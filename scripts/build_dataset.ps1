# Build dataset script
Write-Output "==> Downloading sources, synthesizing and building dataset mixture..."
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from hcscoder_data.pipeline import HCSCoderPipeline; p = HCSCoderPipeline(); p.run_data_pipeline()"
