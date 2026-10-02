import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from huggingface_hub import HfApi

logger = logging.getLogger("HubUploader")


class HubUploader:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def ensure_repo(self, repo_id: str, repo_type: str = "model", private: bool = True) -> str:
        logger.info(f"Ensuring {repo_type} repo {repo_id} exists (private={private})...")
        try:
            self.api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            logger.info(f"Repo {repo_id} verified/created.")
            return repo_id
        except Exception as e:
            logger.error(f"Error creating {repo_type} {repo_id}: {e}")
            raise

    def upload_folder(
        self,
        folder_path: Path,
        repo_id: str,
        repo_type: str = "model",
        commit_message: str = "Upload HCSCoder release artifacts",
    ) -> str:
        logger.info(f"Uploading {folder_path} to {repo_id} ({repo_type})...")
        res = self.api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        logger.info(f"Upload complete. Commit: {res}")
        return str(res)

    def verify_remote_repo(
        self,
        repo_id: str,
        expected_files: List[str],
        repo_type: str = "model",
    ) -> Dict[str, Any]:
        logger.info(f"Verifying remote repo {repo_id}...")
        files = list(self.api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        missing = [f for f in expected_files if f not in files]
        is_valid = len(missing) == 0
        info = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "exists": True,
            "total_files": len(files),
            "verified_files": [f for f in expected_files if f in files],
            "missing_files": missing,
            "is_valid": is_valid,
        }
        logger.info(f"Verification result for {repo_id}: is_valid={is_valid}, missing={missing}")
        return info
