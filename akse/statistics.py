"""Paired instance bootstrap confidence interval, NOT a multi-seed significance claim."""

import math
import random
from statistics import mean

from .evaluation import read_jsonl


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap(left_path, right_path, metric="rougeL_f1", draws=2000, seed=42):
    if draws < 100:
        raise ValueError("use at least 100 bootstrap draws")
    allowed = {"rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"}
    if metric not in allowed:
        raise ValueError("bootstrap supports per-instance quality metrics only")
    def indexed(path):
        rows = read_jsonl(path)
        result = {r["uid"]: r for r in rows}
        if len(rows) != len(result) or not rows:
            raise ValueError("score file must contain unique UIDs and be nonempty")
        return result
    left, right = indexed(left_path), indexed(right_path)
    if left.keys() != right.keys():
        raise ValueError("paired comparison needs exactly the same UIDs")
    hashes = {r.get("dataset_sha256") for r in [*left.values(), *right.values()]}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("score dataset fingerprints must match")
    differences = []
    for key in sorted(left):
        delta = right[key][metric] - left[key][metric]
        if not math.isfinite(delta):
            raise ValueError("scores must be finite")
        differences.append(delta)
    rng = random.Random(seed)
    estimates = [mean(rng.choices(differences, k=len(differences))) for _ in range(draws)]
    return {"metric": metric, "direction": "right minus left", "n": len(differences),
            "dataset_sha256": next(iter(hashes)), "mean_difference": mean(differences),
            "ci95_percentile": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
            "draws": draws, "seed": seed,
            "warning": "Instance bootstrap only; repeated concepts may violate independence. No p-value or multi-seed claim."}
