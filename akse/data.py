"""Read BNeCR without silently repairing source records or leaking labels."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED = {"id", "domain", "question", "new_concept", "context_hint",
            "analogical_possible", "correct_answer", "distractors"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition(record):
    return "NC" if not record["context_hint"] else "CA" if record["analogical_possible"] else "OC"


def load_dataset(path):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError("BNeCR must be a JSON array")
    for index, r in enumerate(records):
        if not isinstance(r, dict) or not REQUIRED <= r.keys():
            raise ValueError(f"row {index}: missing required fields")
        for field in ("id", "domain", "question", "new_concept", "correct_answer"):
            if not isinstance(r[field], str) or not r[field].strip():
                raise ValueError(f"row {index}: {field} must be a nonempty string")
        if r["context_hint"] is not None and not isinstance(r["context_hint"], str):
            raise ValueError(f"row {index}: invalid context_hint")
        if type(r["analogical_possible"]) is not bool:
            raise ValueError(f"row {index}: analogical_possible must be a boolean")
        if not isinstance(r["distractors"], list) or any(not isinstance(x, str) or not x for x in r["distractors"]):
            raise ValueError(f"row {index}: distractors must be a list of nonempty strings")
    return records


def uid(index, record):
    return f"{index:06d}:{record['id']}"


def inference_input(index, record):
    return {"uid": uid(index, record), "question": record["question"],
            "new_concept": record["new_concept"], "context_hint": record["context_hint"]}


def audit(path):
    records = load_dataset(path)
    ids = Counter(r["id"] for r in records)
    domains = defaultdict(Counter)
    for r in records:
        domains[r["domain"]][condition(r)] += 1
    return {"file_sha256": fingerprint(path), "records": len(records),
            "unique_ids": len(ids), "unique_concepts": len({r["new_concept"] for r in records}),
            "conditions": dict(Counter(condition(r) for r in records)),
            "domains": {d: dict(c) for d, c in sorted(domains.items())},
            "duplicate_ids": {key: value for key, value in sorted(ids.items()) if value > 1},
            "no_context_with_analogy": [uid(i, r) for i, r in enumerate(records)
                                       if condition(r) == "NC" and r["analogical_possible"]],
            "empty_distractors": [uid(i, r) for i, r in enumerate(records) if not r["distractors"]],
            "exact_chinese_refusal_references": sum(r["correct_answer"] == "我无法回答。" for r in records)}
