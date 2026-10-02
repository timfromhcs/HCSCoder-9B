# Submit Real Cloud Training Job to Hugging Face Jobs
param(
    [string]$Flavor = "a100-large",
    [string]$Timeout = "4h"
)

Write-Output "==> Submitting Cloud Training Job on $Flavor..."
hf jobs uv run `
    --flavor $Flavor `
    --timeout $Timeout `
    --secrets HF_TOKEN `
    --with "transformers>=4.48.0" `
    --with "peft>=0.14.0" `
    --with "trl>=0.13.0" `
    --with "datasets>=3.0.0" `
    --with "accelerate>=1.2.0" `
    scripts/cloud/train_cloud.py
