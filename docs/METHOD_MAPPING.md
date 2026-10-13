# Manuscript-to-code mapping and reproducibility boundaries

Source: the manuscript supplied in the workspace on 2026-10-13, Sections 4.1-4.4 and 5.3-5.4. The companion worked-example `.tex` was also inspected; its hypothetical numbers are not experimental records.

## Implemented equations

| Manuscript | Executable operation | Implementation |
| --- | --- | --- |
| Eq. 1: semantic encoding | BGE-M3 through SentenceTransformer; unit-normalized dense vectors | `akse/embeddings.py` |
| Eq. 2: novelty | `1 - max(cosine(candidate, cluster centroid))`, strict `> tau_d` | `knowledge.py`, `pipeline.py` |
| Eq. 3: contextual salience | strict `score > 0.6`; supplied measured score plus provenance | `pipeline.py` |
| Eqs. 4-5: hypotheses | new structured JSON prompt; five stochastic trajectories | `prompts.py`, `pipeline.py` |
| Eqs. 6-7: self-consistency | normalized textual recurrence, once per trajectory; strict `> 0.8` | `maths.py` |
| Eq. 8: semantic consistency | cosine of candidate concept name and linked known entity name | `pipeline.py` |
| Eq. 9: temporal consistency | mean similarity to the last three preceding observations | `pipeline.py` |
| Eq. 10: refinement | `0.4 * semantic + 0.6 * temporal` | `pipeline.py` |
| Eq. 11: assimilation | append a nonduplicate hypothesis only when refinement `> 0.7` | `knowledge.py`, `pipeline.py` |
| Eq. 12: answering | cosine retrieval of knowledge texts; knowledge-conditioned prompt | `knowledge.py`, `prompts.py` |

All cosine scores retain their signed range [-1, 1]; novelty is in [0, 2]. No undocumented rescaling or backbone gradient updates are performed.

## Reported parameters versus new choices

The manuscript reports `N=5`, hypothesis/final generation temperature 0.7, top-p 0.9, maximum new-token budget 2048, consistency threshold 0.8, MI threshold 0.6, beta weights 0.4/0.6, temporal window 3, and assimilation threshold 0.7. The novelty threshold is not numerically specified and remains `null` until supplied by the user.

The following are new engineering decisions, not recovered original implementation settings:

- Dense BGE-M3 via SentenceTransformer rather than a confirmed original encoder wrapper.
- Cluster assignments supplied in seed data; absent a cluster key, each concept is its own singleton group. No unspecified clustering algorithm is pretended.
- Candidate extraction uses the benchmark's supplied `new_concept`. This is a concept-given evaluation, **not** unsupervised concept discovery from arbitrary documents. The framework does not establish absence from a backbone's pretraining corpus.
- NFC normalization removes whitespace and Unicode punctuation, preserves letter case, and does not merge semantic paraphrases. Hypotheses for different linked entities are kept separate.
- Exact JSON response parsing. A malformed trajectory contributes no accepted hypothesis and still counts in the five-trajectory denominator. Raw outputs are retained.
- Temporal current representation is the encoding of `new_concept + newline + context_hint`; historical texts must be constructed consistently by the user. ISO timestamps must have time-zone offsets, be ordered, unique, and strictly earlier than `current_timestamp`.
- Missing or fewer than three prior observations produce a missing temporal score; refinement cannot accept that hypothesis. The system does not substitute 1.0, repeat the current context, or treat random decoding trajectories as temporal observations.
- Cosine top-5 retrieval, positive-similarity filtering, source-bearing records, default isolated evaluation, and the full prompts are newly specified choices.
- The knowledge store follows the additive update in revised Eq. 11. Independent factual entailment checking, conflict detection, confidence decay, retraction, unsupervised candidate extraction, and live Wikipedia monitoring are **not implemented**. Semantic similarity and recurrence do not certify factual truth.
- Ablations modify this implementation's gates. They do not establish equivalence to the manuscript's original ablation experiments.

## Mutual-information ambiguity

The manuscript does not specify the observation unit, how concept/document joint and marginal probabilities are estimated, logarithm base, smoothing, or corpus snapshot. Consequently the detection stage requires a measured per-UID score and a source describing the calculation. It does not manufacture MI from cosine similarity or randomly assigned scores.

`maths.mutual_information` provides **full** discrete mutual information from a normalized joint probability table, using natural logarithms. It is an explicit utility, not an asserted reconstruction of Eq. 3's unspecified estimator.

There is also a mathematical issue worth resolving before reproduction: if Eq. 3 is interpreted literally as a single concept's contribution to MI under a conventional normalized categorical joint distribution, the contribution is `p(c) * KL(P(D|c) || P(D))`, bounded above by `p(c) * log(1/p(c)) <= 1/e` in nats. That interpretation cannot pass the reported threshold 0.6. This observation does not establish which estimator was used; the author needs to clarify whether the intended score is full MI (e.g. over concept presence/absence), differently normalized, or a different quantity.

For a full binary joint table, maximum MI is log(2) nats, so 0.6 is possible. Do not silently substitute that definition without documenting it and re-running calibration.

## Evaluation mapping

- Reference fields are available only to evaluation, not the inference input whitelist.
- ROUGE-1/2/L are macro per-instance F1. Mixed-v1 tokenization lowercases Latin text, splits CJK characters individually, and removes punctuation.
- BLEU-4 is single-reference, corpus-level, equally weighted, without smoothing or effective order. If any n-gram precision is zero (including a corpus lacking 4-grams), score is zero.
- BERTScore uses an explicitly chosen model and layer count, F1, no IDF, no baseline rescaling, and records the scorer hash. The manuscript does not specify the original checkpoint/layer.
- Annotated RR = refused/all; CRR = expected-and-refused/expected; inappropriate refusal = answerable-and-refused/answerable; CCFR = semantically confused/all. Empty denominators are `null`.
- Without annotations, refusal-prefix and distractor-substring metrics are explicitly named proxies. Substring presence is not evidence that the concept was asserted as the definition; negation and contrast require semantic annotation.
- The paired bootstrap interval resamples per-instance differences. It does not handle multi-seed uncertainty, repeated-concept clustering, multiple comparisons, or causal claims.

## What has actually been verified

Offline unit/integration tests, data audit, deterministic scripted demo, and packaging checks are appropriate software checks. A real GPU-backed Baichuan2/BGE-M3 run, an authenticated external API call, the original Wikipedia evidence, and the manuscript's table values have not been verified by this delivery. See `docs/VERIFICATION.md` for the recorded local checks.
