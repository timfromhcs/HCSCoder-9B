# Bootstrap script for HCSCoder
Write-Output "==> Checking Environment & Authenticating..."
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from hcscoder_data.pipeline import HCSCoderPipeline; p = HCSCoderPipeline(); p.run_env_check(); p.run_auth_check(); p.run_model_discovery()"
