import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from hcscoder_data.normalization.schema import Trajectory

HELD_OUT_BENCHMARK_KEYWORDS = [
    "swe-bench",
    "verified",
    "tau-bench",
    "terminal-bench",
    "bfcl_eval",
    "held_out",
]


class MixtureBuilder:
    def __init__(
        self,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        random.seed(seed)

    def is_leakage(self, traj: Trajectory) -> bool:
        meta_str = json.dumps(traj.metadata).lower()
        for kw in HELD_OUT_BENCHMARK_KEYWORDS:
            if kw in meta_str:
                return True
        return False

    def build_splits(
        self,
        trajectories: List[Trajectory],
    ) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory], Dict[str, Any]]:
        # Filter out leakage
        clean_trajs = []
        excluded_leakage = []
        for t in trajectories:
            if self.is_leakage(t):
                excluded_leakage.append(t.id)
            else:
                clean_trajs.append(t)

        # Disjoint partitioning by repository / task where available
        repo_groups: Dict[str, List[Trajectory]] = {}
        no_repo = []
        for t in clean_trajs:
            repo = t.metadata.get("repo")
            if repo:
                repo_groups.setdefault(repo, []).append(t)
            else:
                no_repo.append(t)

        train_trajs: List[Trajectory] = []
        val_trajs: List[Trajectory] = []
        test_trajs: List[Trajectory] = []

        # Partition repo groups
        repos = list(repo_groups.keys())
        random.shuffle(repos)
        n_train_repos = int(len(repos) * self.train_ratio)
        n_val_repos = int(len(repos) * self.val_ratio)

        for i, repo in enumerate(repos):
            if i < n_train_repos:
                train_trajs.extend(repo_groups[repo])
            elif i < n_train_repos + n_val_repos:
                val_trajs.extend(repo_groups[repo])
            else:
                test_trajs.extend(repo_groups[repo])

        # Partition unassigned
        random.shuffle(no_repo)
        n_train_nr = int(len(no_repo) * self.train_ratio)
        n_val_nr = int(len(no_repo) * self.val_ratio)

        train_trajs.extend(no_repo[:n_train_nr])
        val_trajs.extend(no_repo[n_train_nr : n_train_nr + n_val_nr])
        test_trajs.extend(no_repo[n_train_nr + n_val_nr :])

        # Capability summary
        cap_counts: Dict[str, int] = {}
        for t in train_trajs:
            for cap in t.computed.capabilities:
                cap_counts[cap] = cap_counts.get(cap, 0) + 1

        split_manifest = {
            "seed": self.seed,
            "train_count": len(train_trajs),
            "val_count": len(val_trajs),
            "test_count": len(test_trajs),
            "excluded_leakage_count": len(excluded_leakage),
            "excluded_leakage_ids": excluded_leakage,
            "unique_repos_train": len([r for i, r in enumerate(repos) if i < n_train_repos]),
            "unique_repos_val": len([r for i, r in enumerate(repos) if n_train_repos <= i < n_train_repos + n_val_repos]),
            "unique_repos_test": len([r for i, r in enumerate(repos) if i >= n_train_repos + n_val_repos]),
            "train_capabilities": cap_counts,
        }

        return train_trajs, val_trajs, test_trajs, split_manifest

    def export_autotrain_format(self, trajectories: List[Trajectory], output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for traj in trajectories:
                msgs = []
                for m in traj.messages:
                    entry = {"role": m.role, "content": m.content or ""}
                    msgs.append(entry)
                record = {"messages": msgs}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
