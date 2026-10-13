# BNeCR data card: supplied snapshot

## Identity and scope

- Filename: `BNeCR.json`.
- Records: **3,205**, 20 domains.
- Unique source IDs: **3,175**.
- Unique concept strings: **2,883**.
- SHA-256: `6e66a5ad94e3520af3dbe316223a463b0080d32155592154c43b0489fb8512bc`.
- This is a byte-identical copy of the dataset supplied by the user. No rows, IDs, labels, definitions, or distractors were corrected.
- Questions, context hints, and reference answers are predominantly Chinese. The repository does not assert that each hypothetical concept represents a scientifically established entity.

## Fields and model access

| Field | Purpose | Sent to generation |
| --- | --- | --- |
| `id` | Source ID; duplicates exist | Only as part of internal UID metadata |
| `domain` | Domain-level reporting | No |
| `question` | Query | Yes |
| `new_concept` | Given candidate concept | Yes, in hypothesis generation |
| `context_hint` | Optional contextual evidence | Yes |
| `analogical_possible` | Condition grouping | No |
| `correct_answer` | Evaluation reference | No |
| `distractors` | Evaluation confusion analysis | No |

## Conditions and known issues

| Condition | Rule | Count |
| --- | --- | --- |
| CA | Context exists, analogy Boolean true | 2,416 |
| OC | Context exists, analogy Boolean false | 410 |
| NC | Context absent | 379 |

Measured issues:

- IDs `PC0081` through `PC0110` each occur twice (30 duplicated IDs).
- NC records `SS0007`, `SS0008`, `SS0011` have `analogical_possible=true`.
- Records `AS0097`, `AS0098`, `AS0138`, `AS0154` have no distractors.
- Exactly 401 references equal the Chinese string `我无法回答。`; this is a string count, not a validated expected-refusal annotation.

The machine-readable `audit_report.json` includes row UIDs and complete per-domain counts. Regenerate with:

```bash
python -m akse audit --dataset data/BNeCR.json --output outputs/audit.json
```

## Annotation and licensing provenance

The supplied manuscript describes LLM-assisted, human-revised construction. This repository has not independently verified the annotator protocol, model generation records, agreement statistics, adjudication logs, or redistribution license. These details require author confirmation before public release.

The paper mentions the BNeCR GitHub repository, but the public version and its relationship to this exact supplied snapshot were not independently verified in this delivery.

## Evaluation cautions

No splits are fabricated. Every run records its selected UIDs. For future train/validation/test partitioning, split by normalized concept groups rather than by duplicated IDs or arbitrary rows, and keep tuning data separate from evaluation.

Streaming knowledge evolution can pass evidence from one test row to a later row sharing the same concept. The default isolated protocol avoids this cross-row update, but external knowledge and salience files still require an independent leakage audit.

Changes to the dataset should create a separately versioned release and an explicit change log; do not silently overwrite this snapshot and retain its results.
