"""Plot measured aggregate metrics, not hard-coded manuscript claims."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="baseline evaluation/metrics.json")
    parser.add_argument("--akse", required=True, help="AKSE evaluation/metrics.json")
    parser.add_argument("--output", default="outputs/metric_comparison.png")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    akse = json.loads(Path(args.akse).read_text(encoding="utf-8"))
    if baseline["dataset_sha256"] != akse["dataset_sha256"]:
        raise ValueError("dataset fingerprints differ")
    # Equal dataset version/count does not prove matching selected rows.
    from akse.evaluation import read_jsonl
    left_rows = read_jsonl(Path(args.baseline).parent / "per_instance.jsonl")
    right_rows = read_jsonl(Path(args.akse).parent / "per_instance.jsonl")
    if {r["uid"] for r in left_rows} != {r["uid"] for r in right_rows}:
        raise ValueError("baseline and AKSE must have exactly the same evaluated UIDs")
    fields = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bleu4_corpus"]
    if all(r["groups"]["overall"].get("bertscore_f1") is not None for r in (baseline, akse)):
        fields.append("bertscore_f1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(fields)))
    ax.bar([i - 0.18 for i in x], [baseline["groups"]["overall"][f] for f in fields],
           width=0.36, label="Baseline")
    ax.bar([i + 0.18 for i in x], [akse["groups"]["overall"][f] for f in fields],
           width=0.36, label="AKSE")
    ax.set_xticks(x, ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BLEU-4"] +
                  (["BERTScore F1"] if len(fields) == 5 else []))
    ax.set_ylabel("Score (0-1)")
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
