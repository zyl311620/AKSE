# Research input schemas

Paths below are relative to the repository root. All JSON is UTF-8. `examples/` contains synthetic schema examples only.

## Dataset identity

The original BNeCR file is unchanged. Each record receives `uid = "{zero_based_index:06d}:{id}"`. For example, the first row is `000000:NS0001`. Use this UID for salience, history, predictions, and annotations; source IDs are not unique.

The inference object contains only `uid`, `question`, `new_concept`, and `context_hint`. The metadata condition derives from `context_hint` and `analogical_possible`, but the analogy label itself is not shown to the model. No actual analogy text is synthesized from that Boolean label.

## Seed knowledge: JSON array

```json
[
  {
    "concept": "distributed memory",
    "text": "A definition taken from your independently sourced corpus.",
    "source": "snapshot identifier, document URL/ID, revision, and timestamp",
    "cluster": "optional precomputed cluster identifier"
  }
]
```

`concept`, `text`, and `source` are mandatory nonempty strings. Omit `cluster` to use one group per concept. Do not derive this file from benchmark answers, distractors, or future evaluation contexts. The user must supply independently collected known concepts and their evidence.

The manuscript names Wikipedia but does not supply a snapshot, extraction code, version, chunk sizes, or licenses. This repository does not automatically fetch or redistribute Wikipedia. Obtain a properly licensed snapshot and document your extraction/retrieval process. A bounded initial knowledge list is assumed; passing an entire encyclopedia's entity inventory into a prompt is not supported.

## Salience: JSON object keyed by UID

```json
{
  "000000:NS0001": {
    "score": 0.65,
    "source": "EXAMPLE ONLY: replace with actual estimator, corpus hash, probability model, units and log base"
  }
}
```

The number above is illustrative, not a measured value for NS0001. Scores must be finite and nonnegative. Include estimator details, corpus version, and units in the source or accompanying metadata. Do not use held-out reference answers to estimate salience. The program requires coverage for selected samples and fails early if UID coverage is absent.

The calculation behind Eq. 3 is unresolved; see the MI discussion in [METHOD_MAPPING.md](METHOD_MAPPING.md). The software cannot determine whether your supplied score is valid MI.

## Concept history: JSON object keyed by UID

```json
{
  "000000:NS0001": {
    "current_timestamp": "2026-01-04T00:00:00Z",
    "observations": [
      {"timestamp": "2026-01-01T00:00:00Z", "text": "EXAMPLE concept and prior contextual evidence", "source": "EXAMPLE source 1"},
      {"timestamp": "2026-01-02T00:00:00Z", "text": "EXAMPLE concept and prior contextual evidence", "source": "EXAMPLE source 2"},
      {"timestamp": "2026-01-03T00:00:00Z", "text": "EXAMPLE concept and prior contextual evidence", "source": "EXAMPLE source 3"}
    ]
  }
}
```

These are schema examples, not BNeCR observations. Construct all texts using the same representation rule as the current `concept + newline + context_hint`. Only the final three genuinely preceding observations are scored. Use actual temporal provenance, not repeatedly sampled embeddings of the current context.

An absent UID or fewer than three observations results in no temporal score and no refinement acceptance, recorded as `missing_history`. A completely missing history file in a full AKSE run is rejected. Invalid, duplicated, unordered, or current/future timestamps fail validation.

## Manual evaluation annotations

Create a template with `make-annotations`. It contains the exact dataset hash and one label entry per evaluated UID:

```json
{
  "dataset_sha256": "replace_with_the_exact_dataset_sha256",
  "labels": {
    "000000:NS0001": {
      "expected_refusal": false,
      "refused": false,
      "concept_confused": false,
      "source": "annotator(s), rubric version, adjudication and date"
    }
  }
}
```

`expected_refusal` is the predeclared instance label. `refused` labels the actual model answer. `concept_confused` requires an incorrect conceptual association, not any string mention. Annotation records must cover exactly the evaluated UIDs; null or incomplete labels are rejected.

Do not infer expected refusal from the NC label alone: this file has NC rows with `analogical_possible=true`, and exact refusal references also exist outside the NC condition. Define and record the intended evaluation subset explicitly.
