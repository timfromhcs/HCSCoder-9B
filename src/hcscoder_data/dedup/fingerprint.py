import hashlib
from typing import List, Set
from datasketch import MinHash
from hcscoder_data.normalization.schema import Trajectory


class Deduplicator:
    def __init__(self, minhash_threshold: float = 0.90, num_perm: int = 128):
        self.exact_hashes: Set[str] = set()
        self.fingerprints: Set[str] = set()
        self.minhashes: List[MinHash] = []
        self.minhash_threshold = minhash_threshold
        self.num_perm = num_perm

    def exact_hash(self, traj: Trajectory) -> str:
        return traj.content_hash()

    def conversation_fingerprint(self, traj: Trajectory) -> str:
        user_msgs = [m.content or "" for m in traj.messages if m.role == "user"]
        tool_seq = []
        for m in traj.messages:
            if m.tool_calls:
                for tc in m.tool_calls:
                    tool_seq.append(tc.name)
        combined = f"{'|'.join(user_msgs)}::{'->'.join(tool_seq)}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def build_minhash(self, traj: Trajectory) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for msg in traj.messages:
            if msg.content:
                words = set(msg.content.lower().split())
                for w in words:
                    m.update(w.encode("utf-8"))
        return m

    def is_duplicate(self, traj: Trajectory) -> bool:
        # 1. Exact hash
        eh = self.exact_hash(traj)
        if eh in self.exact_hashes:
            return True

        # 2. Conversation fingerprint
        fp = self.conversation_fingerprint(traj)
        if fp in self.fingerprints:
            return True

        # 3. MinHash near duplicate
        mh = self.build_minhash(traj)
        for existing in self.minhashes:
            if mh.jaccard(existing) >= self.minhash_threshold:
                return True

        # Not a duplicate, register it
        self.exact_hashes.add(eh)
        self.fingerprints.add(fp)
        self.minhashes.append(mh)
        return False
