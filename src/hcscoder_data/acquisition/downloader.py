import json
import logging
from pathlib import Path
from typing import List
from datasets import load_dataset
from hcscoder_data.acquisition.manifest import SourceItem, SourceManifest
from hcscoder_data.classification.classifier import CapabilityClassifier
from hcscoder_data.dedup.fingerprint import Deduplicator
from hcscoder_data.filtering.quality import QualityScorer
from hcscoder_data.filtering.secrets import SecretFilter
from hcscoder_data.normalization.normalizer import DataNormalizer
from hcscoder_data.normalization.schema import Trajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AcquisitionDownloader")


class AcquisitionPipeline:
    def __init__(
        self,
        raw_dir: Path = Path("data/raw"),
        normalized_dir: Path = Path("data/normalized"),
        filtered_dir: Path = Path("data/filtered"),
    ):
        self.raw_dir = raw_dir
        self.normalized_dir = normalized_dir
        self.filtered_dir = filtered_dir

        self.normalizer = DataNormalizer()
        self.secret_filter = SecretFilter()
        self.quality_scorer = QualityScorer()
        self.deduplicator = Deduplicator()
        self.classifier = CapabilityClassifier()

    def process_source(
        self,
        source_id: str,
        repo_id: str,
        max_samples: int,
        manifest: SourceManifest,
    ) -> List[Trajectory]:
        logger.info(f"Processing source {source_id} from {repo_id} (limit={max_samples})...")
        trajectories: List[Trajectory] = []
        rejected_secrets = 0
        rejected_quality = 0
        rejected_dups = 0

        try:
            ds = load_dataset(repo_id, split="train", streaming=True)
            count = 0
            for raw_sample in ds:
                if count >= max_samples:
                    break

                # 1. Normalize
                if source_id == "toolace":
                    traj = self.normalizer.normalize_toolace(raw_sample)
                elif source_id == "swe_zero":
                    traj = self.normalizer.normalize_swe_zero(raw_sample)
                elif source_id == "si2ca":
                    traj = self.normalizer.normalize_si2ca(raw_sample)
                elif source_id == "toolgym_long":
                    traj = self.normalizer.normalize_toolgym(raw_sample)
                else:
                    continue

                # 2. Secret Scan
                clean, reason = self.secret_filter.filter_trajectory(traj)
                if not clean:
                    rejected_secrets += 1
                    continue

                # 3. Classify
                self.classifier.classify(traj)

                # 4. Quality Scoring
                self.quality_scorer.apply(traj)
                if traj.computed.quality_bucket == "REJECT":
                    rejected_quality += 1
                    continue

                # 5. Deduplicate
                if self.deduplicator.is_duplicate(traj):
                    rejected_dups += 1
                    continue

                trajectories.append(traj)
                count += 1

            manifest.sources.append(
                SourceItem(
                    source_id=source_id,
                    repo=repo_id,
                    revision="main",
                    license="permissive",
                    record_count=len(trajectories),
                    notes=f"accepted={len(trajectories)}, rejected_secrets={rejected_secrets}, rejected_quality={rejected_quality}, rejected_dups={rejected_dups}",
                )
            )

        except Exception as e:
            logger.error(f"Error loading source {source_id}: {e}")

        logger.info(
            f"Source {source_id}: collected {len(trajectories)} clean trajectories. "
            f"(Rejected: secrets={rejected_secrets}, quality={rejected_quality}, duplicates={rejected_dups})"
        )
        return trajectories
