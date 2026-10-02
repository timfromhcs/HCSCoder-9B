# Deploy ZeroGPU Space to Hugging Face
param(
    [string]$SpaceId = "timfromhcs/HCSCoder-ZeroGPU"
)

Write-Output "==> Creating and uploading ZeroGPU Space to $SpaceId..."
.\.venv\Scripts\python.exe -c "import os; from huggingface_hub import HfApi; api = HfApi(token=os.environ.get('HF_TOKEN')); api.create_repo(repo_id='$SpaceId', repo_type='space', space_sdk='gradio', private=False, exist_ok=True); api.upload_folder(folder_path='spaces', repo_id='$SpaceId', repo_type='space', commit_message='Deploy HCSCoder ZeroGPU Gradio Space'); print('ZeroGPU Space successfully deployed to https://huggingface.co/spaces/$SpaceId')"
