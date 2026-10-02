"""Hugging Face Hub Synchronization Script
Uploads updated dataset artifacts (including DPO pairs and flywheel SFT repairs)
and updated model card, checksums, and MoE architecture config to Hugging Face Hub.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set")

api = HfApi(token=token)

DATASET_REPO = "timfromhcs/HCSCoder-9B-Training-Data"
MODEL_REPO = "timfromhcs/HCSCoder-Qwen3.5-9B"

print(f"==> Authenticated as: {api.whoami()['name']}")

# 1. Upload Dataset Artifacts
print(f"==> Uploading dataset files from data/final to {DATASET_REPO}...")
dataset_files = ["train.jsonl", "validation.jsonl", "test.jsonl", "dpo.jsonl"]
for fn in dataset_files:
    p = Path("data/final") / fn
    if p.exists():
        print(f"    Uploading {fn} ({p.stat().st_size:,} bytes)...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fn,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update {fn} with autonomous flywheel defect repairs",
        )
print("[OK] Dataset synchronization complete.")

# 2. Upload Model Artifacts
print(f"==> Uploading model artifacts from release/ to {MODEL_REPO}...")
model_files = ["README.md", "checksums.sha256", "provenance.json", "config.json"]
for fn in model_files:
    p = Path("release") / fn
    if p.exists():
        print(f"    Uploading {fn} ({p.stat().st_size:,} bytes)...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fn,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Update {fn} with MoE architecture and honest disclosures",
        )
print("[OK] Model repository synchronization complete.")

# 3. Remote Verification
print("==> Verifying remote repository contents:")
remote_dataset = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
remote_model = api.list_repo_files(repo_id=MODEL_REPO, repo_type="model")
print(f"Dataset repo ({DATASET_REPO}): {remote_dataset}")
print(f"Model repo ({MODEL_REPO}): {remote_model}")

print("============================================================")
print("ALL REMOTE REPOSITORIES SUCCESSFULLY SYNCHRONIZED & VERIFIED")
print("============================================================")
