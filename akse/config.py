"""Paper parameters and explicit implementation choices."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # No numerical tau_d was supplied in the manuscript.
    novelty_threshold: float | None = None
    salience_threshold: float = 0.6
    trajectories: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 2048
    consistency_threshold: float = 0.8
    beta_semantic: float = 0.4
    beta_temporal: float = 0.6
    temporal_window: int = 3
    assimilation_threshold: float = 0.7
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.0
    # Explicit ablations; disabled modules are recorded in every trace.
    use_detection: bool = True
    use_consistency: bool = True
    use_refinement: bool = True
    use_reasoning_enhancement: bool = True

    def validate(self):
        numeric = ("salience_threshold", "temperature", "top_p", "consistency_threshold",
                   "beta_semantic", "beta_temporal", "assimilation_threshold", "retrieval_min_similarity")
        for key in (*numeric, "novelty_threshold"):
            value = getattr(self, key)
            if key == "novelty_threshold" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.novelty_threshold is not None and not 0 <= self.novelty_threshold <= 2:
            raise ValueError("novelty_threshold must lie in [0, 2]")
        if not math.isclose(self.beta_semantic + self.beta_temporal, 1.0):
            raise ValueError("semantic and temporal weights must sum to 1")
        for name in ("beta_semantic", "beta_temporal", "consistency_threshold", "assimilation_threshold", "top_p"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("trajectories", "temporal_window", "retrieval_top_k", "max_new_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperature < 0 or self.salience_threshold < 0:
            raise ValueError("temperature and salience_threshold must be nonnegative")
        if not -1 <= self.retrieval_min_similarity <= 1:
            raise ValueError("retrieval_min_similarity must lie in [-1, 1]")
        for name in ("use_detection", "use_consistency", "use_refinement", "use_reasoning_enhancement"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
