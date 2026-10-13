"""The four AKSE stages corresponding to manuscript Sections 4.1-4.4."""

import json
import math
import time
from dataclasses import asdict
from datetime import datetime

from .knowledge import Entry
from .maths import consistency_filter, cosine
from .prompts import answer_prompt, hypothesis_prompt


def preceding_observations(bundle):
    """Require UTC-offset ISO timestamps and reject current/future observations."""
    if bundle is None:
        return []
    if not isinstance(bundle, dict) or not isinstance(bundle.get("observations"), list):
        raise ValueError("history must contain current_timestamp and an observations list")
    def parse(value):
        if not isinstance(value, str):
            raise ValueError("history timestamps must be ISO-8601 strings")
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("history timestamps must have a UTC offset")
        return date
    current = parse(bundle.get("current_timestamp"))
    observations = bundle["observations"]
    if any(not isinstance(o, dict) or not isinstance(o.get("text"), str)
           or not o["text"].strip() or not isinstance(o.get("source"), str)
           or not o["source"].strip() for o in observations):
        raise ValueError("each historical observation needs text, timestamp, and source")
    timestamps = [parse(o.get("timestamp")) for o in observations]
    if (len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps)
            or any(t >= current for t in timestamps)):
        raise ValueError("observations must be unique, ordered, and strictly before current_timestamp")
    return observations


class AKSE:
    def __init__(self, config, encoder, model, knowledge, salience, history):
        config.validate()
        self.config, self.encoder, self.model, self.knowledge = config, encoder, model, knowledge
        self.salience, self.history = salience, history

    def _generate(self, messages, purpose, seed):
        c = self.config
        return self.model.generate(messages, purpose=purpose, seed=seed,
                                   temperature=c.temperature, top_p=c.top_p,
                                   max_new_tokens=c.max_new_tokens)

    def run(self, query, *, seed=42, method="akse"):
        started = time.perf_counter()
        # query is an allow-listed object without references/distractors.
        if set(query) != {"uid", "question", "new_concept", "context_hint"}:
            raise ValueError("query must contain only uid/question/new_concept/context_hint")
        if method not in {"akse", "baseline", "zero_cot"}:
            raise ValueError("unknown method")
        c, concept = self.config, query["new_concept"]
        call_start = len(self.model.calls)
        trace = {"uid": query["uid"], "method": method, "seed": seed,
                 "config": asdict(c), "stages": {}}
        if method == "akse":
            detected = True
            if c.use_detection:
                if c.novelty_threshold is None:
                    raise ValueError("set novelty_threshold explicitly; the paper does not give tau_d")
                centroids = self.knowledge.cluster_centroids()
                if not centroids:
                    raise ValueError("detection requires a nonempty initial knowledge base")
                novelty = 1 - max(cosine(self.encoder.encode(concept), v) for v in centroids)
                record = self.salience.get(query["uid"])
                if (not isinstance(record, dict) or not isinstance(record.get("source"), str)
                        or not record["source"].strip()):
                    raise ValueError(f"missing provenance-bearing salience for {query['uid']}")
                score = record.get("score")
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or score < 0:
                    raise ValueError("salience score must be a finite nonnegative number")
                detected = novelty > c.novelty_threshold and score > c.salience_threshold
                trace["stages"]["detection"] = {
                    "novelty": novelty, "salience": record, "passed": detected}
            else:
                trace["stages"]["detection"] = {"disabled": True, "passed": True}

            if detected:
                entities = self.knowledge.known_entities()
                messages = hypothesis_prompt(concept, query["context_hint"], entities)
                samples, raw = [], []
                for trajectory in range(c.trajectories):
                    text = self._generate(messages, "hypothesis", seed + trajectory)
                    raw.append(text)
                    try:
                        # Exact JSON contract: malformed generations fail this trajectory.
                        parsed = json.loads(text)
                        items = parsed["hypotheses"]
                        if not isinstance(items, list):
                            raise ValueError("hypotheses is not a list")
                        valid = []
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            if (item.get("known_entity") in entities and
                                    isinstance(item.get("text"), str) and item["text"].strip()):
                                valid.append({"known_entity": item["known_entity"], "text": item["text"]})
                        samples.append(valid)
                    except (ValueError, KeyError, TypeError):
                        samples.append([])
                retained = consistency_filter(samples, c.consistency_threshold if c.use_consistency else -1.0)
                trace["stages"]["generation"] = {"raw": raw, "parsed": samples,
                                                "retained": retained,
                                                "consistency_disabled": not c.use_consistency}
                verification = []
                current_text = concept + "\n" + (query["context_hint"] or "")
                current_vector = self.encoder.encode(current_text)
                observations = preceding_observations(self.history.get(query["uid"]))
                similarities = [cosine(self.encoder.encode(o["text"]), current_vector)
                                for o in observations[-c.temporal_window:]]
                temporal = sum(similarities) / c.temporal_window if len(similarities) == c.temporal_window else None
                for hypothesis in retained:
                    semantic = cosine(self.encoder.encode(concept),
                                      self.encoder.encode(hypothesis["known_entity"]))
                    score = (c.beta_semantic * semantic + c.beta_temporal * temporal
                             if temporal is not None else None)
                    accepted = not c.use_refinement or (score is not None and score > c.assimilation_threshold)
                    added = False
                    if accepted:
                        added = self.knowledge.add(Entry(
                            concept=concept, text=hypothesis["text"],
                            source=f"{query['uid']}:seed={seed}",
                            related_entity=hypothesis["known_entity"], confidence=score))
                    verification.append({
                        **hypothesis, "semantic": semantic, "temporal": temporal,
                        "temporal_similarities": similarities, "history": observations[-c.temporal_window:],
                        "refinement": score, "accepted": accepted, "added": added,
                        "reason": ("refinement_ablation" if not c.use_refinement else
                                   "missing_history" if temporal is None else
                                   "accepted" if accepted else "below_threshold")})
                trace["stages"]["refinement"] = verification
            else:
                trace["stages"]["generation"] = {"skipped": "detection_not_passed"}
                trace["stages"]["refinement"] = []

        retrieved = (self.knowledge.retrieve(query["question"], c.retrieval_top_k,
                                             c.retrieval_min_similarity)
                     if method == "akse" and c.use_reasoning_enhancement else [])
        trace["stages"]["reasoning"] = {"retrieved": retrieved}
        answer = self._generate(answer_prompt(query["question"], query["context_hint"], retrieved, method),
                                "answer", seed + c.trajectories)
        calls = self.model.calls[call_start:]
        usages = [call["usage"] for call in calls]
        token_cost = {}
        for field in ("prompt_tokens", "completion_tokens"):
            token_cost[field] = (sum(usage[field] for usage in usages)
                                 if all(isinstance(usage, dict) and
                                        isinstance(usage.get(field), int) for usage in usages) else None)
        return {**query, "answer": answer, "trace": trace,
                "cost": {"calls": len(calls), "elapsed_seconds": time.perf_counter() - started,
                         "model_elapsed_seconds": sum(x["elapsed_seconds"] for x in calls), **token_cost},
                "model_calls": calls, "knowledge": self.knowledge.export() if self.knowledge else []}
