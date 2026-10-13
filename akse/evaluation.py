"""Join on row UID + dataset fingerprint; never join on duplicated source IDs."""

import json
from collections import defaultdict
from pathlib import Path

from .data import condition, dump_json, fingerprint, load_dataset, load_json, uid
from .metrics import TOKENIZER_NAME, corpus_bleu, score_answer, tokenize


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate(dataset_path, predictions_path, output, annotations_path=None,
             bertscore_model=None, bertscore_layers=None, device=None):
    dataset = load_dataset(dataset_path)
    sha = fingerprint(dataset_path)
    by_uid = {uid(i, r): r for i, r in enumerate(dataset)}
    predictions = read_jsonl(predictions_path)
    if not predictions:
        raise ValueError("prediction file is empty")
    seen = set()
    for p in predictions:
        if p.get("uid") not in by_uid or p["uid"] in seen:
            raise ValueError("unknown or duplicate prediction UID")
        if p.get("dataset_sha256") != sha:
            raise ValueError("prediction dataset fingerprint does not match")
        if not isinstance(p.get("answer"), str):
            raise ValueError("answer must be a string")
        seen.add(p["uid"])
    annotations = {}
    if annotations_path:
        bundle = load_json(annotations_path)
        if bundle.get("dataset_sha256") != sha:
            raise ValueError("annotation dataset fingerprint does not match")
        annotations = bundle["labels"]
        if set(annotations) != seen:
            raise ValueError("annotations must cover exactly the evaluated UIDs")
        for label in annotations.values():
            if not label.get("source") or any(type(label.get(f)) is not bool for f in
                                             ("expected_refusal", "refused", "concept_confused")):
                raise ValueError("complete all three Boolean labels and annotation source")
    bert_scores, bert_hash = None, None
    if bertscore_model:
        if bertscore_layers is None or bertscore_layers < 1:
            raise ValueError("BERTScore requires an explicit positive --bertscore-layers")
        from bert_score import score
        _, _, values, bert_hash = score(
            [p["answer"] for p in predictions],
            [by_uid[p["uid"]]["correct_answer"] for p in predictions],
            model_type=bertscore_model, num_layers=bertscore_layers,
            device=device, idf=False, rescale_with_baseline=False, return_hash=True,
            verbose=False)
        bert_scores = values.tolist()

    rows, pairs_by_group, groups = [], defaultdict(list), defaultdict(list)
    for i, p in enumerate(predictions):
        r = by_uid[p["uid"]]
        row = {"uid": p["uid"], "dataset_sha256": sha, "domain": r["domain"],
               "condition": condition(r), **score_answer(p["answer"], r["correct_answer"], r["distractors"]),
               "cost": p.get("cost")}
        if annotations:
            row["labels"] = annotations[p["uid"]]
        if bert_scores is not None:
            row["bertscore_f1"] = bert_scores[i]
        rows.append(row)
        for key in ("overall", "condition:" + row["condition"], "domain:" + row["domain"]):
            groups[key].append(row)
            pairs_by_group[key].append((tokenize(p["answer"]), tokenize(r["correct_answer"])))
    summaries = {}
    mean_fields = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "generated_tokens",
                   "reference_tokens", "generated_characters"]
    if bert_scores is not None:
        mean_fields.append("bertscore_f1")
    for key, group in sorted(groups.items()):
        n = len(group)
        summary = {"n": n, **{f: sum(row[f] for row in group) / n for f in mean_fields},
                   "bleu4_corpus": corpus_bleu(pairs_by_group[key]),
                   "refusal_rate_heuristic": sum(row["refused_heuristic"] for row in group) / n,
                   "distractor_substring_hit_rate": sum(row["distractor_substring_hit"] for row in group) / n}
        for field in ("calls", "elapsed_seconds", "model_elapsed_seconds", "prompt_tokens", "completion_tokens"):
            values = [(row.get("cost") or {}).get(field) for row in group]
            summary["mean_" + field] = (sum(values) / n if all(isinstance(v, (int, float)) for v in values)
                                       else None)
        if annotations:
            expected = sum(row["labels"]["expected_refusal"] for row in group)
            refused = sum(row["labels"]["refused"] for row in group)
            correct = sum(row["labels"]["expected_refusal"] and row["labels"]["refused"] for row in group)
            answerable = n - expected
            inappropriate = sum(not row["labels"]["expected_refusal"] and row["labels"]["refused"] for row in group)
            summary.update(rr=refused / n, crr=correct / expected if expected else None,
                           inappropriate_refusal_rate=inappropriate / answerable if answerable else None,
                           ccfr=sum(row["labels"]["concept_confused"] for row in group) / n,
                           expected_refusal_n=expected, correct_refusal_n=correct)
        summaries[key] = summary
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "per_instance.jsonl", rows)
    result = {"dataset_sha256": sha, "predictions_sha256": fingerprint(predictions_path),
              "evaluated": len(rows), "dataset_total": len(dataset),
              "metric_scale": "0-1, not percentages", "tokenizer": TOKENIZER_NAME,
              "rouge": "macro per-instance F1", "bleu": "corpus BLEU-4; no smoothing; weights=0.25; no effective order",
              "bertscore": {"model": bertscore_model, "layers": bertscore_layers,
                            "idf": False, "rescale_with_baseline": False, "hash": bert_hash},
              "annotation_sha256": fingerprint(annotations_path) if annotations_path else None,
              "groups": summaries}
    dump_json(output / "metrics.json", result)
    return result
