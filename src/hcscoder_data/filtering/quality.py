from hcscoder_data.normalization.schema import Trajectory


class QualityScorer:
    def __init__(
        self,
        gold_thresh: float = 0.95,
        high_thresh: float = 0.85,
        good_thresh: float = 0.70,
        aux_thresh: float = 0.50,
    ):
        self.gold_thresh = gold_thresh
        self.high_thresh = high_thresh
        self.good_thresh = good_thresh
        self.aux_thresh = aux_thresh

    def score(self, traj: Trajectory) -> float:
        # Execution evidence (30%)
        ev_score = 0.0
        if traj.outcome.success:
            ev_score += 0.15
        if traj.outcome.tests_passed > 0 and traj.outcome.tests_failed == 0:
            ev_score += 0.15
        elif traj.outcome.exit_code == 0:
            ev_score += 0.10

        # Tool correctness (20%)
        tool_score = 0.0
        if traj.computed.tool_calls > 0:
            # Ratio of non-error calls
            error_ratio = min(1.0, traj.computed.tool_errors / max(1, traj.computed.tool_calls))
            tool_score = 0.20 * (1.0 - 0.5 * error_ratio)
        else:
            tool_score = 0.15  # text-based reasoning

        # Task completion (15%)
        comp_score = 0.15 if traj.outcome.success else 0.05

        # Verification (15%)
        verif_score = 0.15 if traj.computed.has_verification else 0.05

        # Recovery quality (10%)
        rec_score = 0.10 if (traj.computed.has_recovery or traj.computed.tool_errors == 0) else 0.04

        # Coherence (5%)
        coherence_score = 0.05 if len(traj.messages) >= 2 else 0.01

        # Uniqueness baseline (5%)
        uniq_score = 0.05

        total = ev_score + tool_score + comp_score + verif_score + rec_score + coherence_score + uniq_score
        return round(min(1.0, total), 4)

    def classify_bucket(self, score: float) -> str:
        if score >= self.gold_thresh:
            return "GOLD"
        elif score >= self.high_thresh:
            return "HIGH"
        elif score >= self.good_thresh:
            return "GOOD"
        elif score >= self.aux_thresh:
            return "AUX"
        else:
            return "REJECT"

    def apply(self, traj: Trajectory) -> Trajectory:
        s = self.score(traj)
        traj.computed.quality_score = s
        traj.computed.quality_bucket = self.classify_bucket(s)
        return traj
