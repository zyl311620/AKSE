"""Deterministic scoring primitives; thresholds are STRICT inequalities."""

import math
import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    """NFC, strip Unicode punctuation and whitespace; preserve case/content."""
    return "".join(
        ch for ch in unicodedata.normalize("NFC", text)
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def cosine(a, b) -> float:
    if not a or len(a) != len(b):
        raise ValueError("vectors must be nonempty and have matching dimensions")
    if not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("embedding contains a non-finite value")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def centroid(vectors):
    if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("invalid centroid input")
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]


def consistency_filter(trajectories: list[list[dict]], threshold: float) -> list[dict]:
    """Count each (known entity, normalized hypothesis) once per trajectory."""
    if not trajectories:
        return []
    counts, first = Counter(), {}
    for trajectory in trajectories:
        seen = set()
        for item in trajectory:
            key = (item["known_entity"], normalize(item["text"]))
            if not key[1]:
                continue
            first.setdefault(key, item)
            seen.add(key)
        counts.update(seen)
    result = []
    for key in sorted(counts):
        score = counts[key] / len(trajectories)
        if score > threshold:
            result.append({**first[key], "recurrence": score})
    return result


def mutual_information(joint: list[list[float]]) -> float:
    """Full discrete MI in nats from a supplied normalized joint table.

    This utility is NOT an inferred implementation of the paper's unspecified
    concept/document probability estimator. See docs/METHOD_MAPPING.md.
    """
    if not joint or not joint[0] or any(len(row) != len(joint[0]) for row in joint):
        raise ValueError("joint table must be nonempty and rectangular")
    if any(not math.isfinite(p) or p < 0 for row in joint for p in row):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(sum(map(sum, joint)), 1.0, abs_tol=1e-8):
        raise ValueError("joint probabilities must sum to 1")
    rows = list(map(sum, joint))
    cols = [sum(row[j] for row in joint) for j in range(len(joint[0]))]
    return max(0.0, sum(
        p * math.log(p / (rows[i] * cols[j]))
        for i, row in enumerate(joint) for j, p in enumerate(row) if p > 0
    ))
