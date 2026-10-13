"""Finite, provenance-bearing knowledge store. No gradients or model updates."""

from collections import defaultdict
from dataclasses import asdict, dataclass

from .maths import centroid, cosine, normalize


@dataclass
class Entry:
    concept: str
    text: str
    source: str
    cluster: str = ""
    related_entity: str | None = None
    confidence: float | None = None


class KnowledgeBase:
    def __init__(self, entries, encoder):
        self.encoder = encoder
        self.entries = [Entry(**e) for e in entries]
        for entry in self.entries:
            if any(not isinstance(value, str) or not value.strip() for value in
                   (entry.concept, entry.text, entry.source)):
                raise ValueError("knowledge entries require nonempty concept, text, and source")
        self._vectors = [encoder.encode(e.text) for e in self.entries]

    def cluster_centroids(self):
        groups = defaultdict(list)
        for entry in self.entries:
            groups[entry.cluster or entry.concept].append(self.encoder.encode(entry.concept))
        return [centroid(v) for _, v in sorted(groups.items())]

    def known_entities(self):
        return sorted({e.concept for e in self.entries})

    def add(self, entry):
        key = (entry.concept, entry.related_entity, normalize(entry.text))
        if any((e.concept, e.related_entity, normalize(e.text)) == key for e in self.entries):
            return False
        self.entries.append(entry)
        self._vectors.append(self.encoder.encode(entry.text))
        return True

    def retrieve(self, query, top_k, min_similarity=0.0):
        q = self.encoder.encode(query)
        ranked = sorted(((cosine(q, v), i) for i, v in enumerate(self._vectors)),
                        key=lambda x: (-x[0], x[1]))
        return [{**asdict(self.entries[i]), "retrieval_similarity": score}
                for score, i in ranked[:top_k] if score > min_similarity]

    def export(self):
        return [asdict(e) for e in self.entries]
