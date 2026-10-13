"""A real BGE-M3 adapter and a clearly labeled offline test encoder."""

import hashlib
import math
import re


class HashEncoder:
    """DEMO ONLY. Hashed character/token features, not a semantic model."""
    identifier = "DEMO_ONLY/hash-features-v1"

    def encode(self, text):
        vec = [0.0] * 256
        features = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.lower())
        for token in features:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:2], "big") % len(vec)] += 1.0
        if not any(vec):
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class SemanticEncoder:
    def __init__(self, model="BAAI/bge-m3", revision=None, device=None):
        from sentence_transformers import SentenceTransformer
        self.identifier = f"{model}@{revision or 'UNPINNED'}"
        self.model = SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False)
        self.cache = {}

    def encode(self, text):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, normalize_embeddings=True).tolist()
        return self.cache[text]
