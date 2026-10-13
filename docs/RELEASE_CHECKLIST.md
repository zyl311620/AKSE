# Author checklist before public release or resubmission

## Rights and identity

- [ ] Choose an explicit code license and replace `LICENSE.md`; no open-source license is granted by the current notice.
- [ ] Confirm rights to redistribute BNeCR and the supplied historical plotting scripts; provide their applicable licenses.
- [ ] Add the actual author list, affiliations, paper URL/DOI or preprint, repository URL, release tag, and citation metadata.
- [ ] Confirm permission to publish any Wikipedia-derived evidence and preserve required attribution.

## Method and data confirmation

- [ ] Resolve Eq. 3's MI estimator and its 0.6 threshold units; publish the actual estimator.
- [ ] Supply the actual novelty threshold and explain calibration without held-out label leakage.
- [ ] Publish the independent seed corpus, checkpoint revisions, true temporal observations, and retrieval configuration.
- [ ] Confirm the intended Baichuan2 checkpoint and chat template; test compatibility on real hardware.
- [ ] Decide on concept-given versus automatically extracted candidates and state that protocol.
- [ ] Decide on isolated versus streaming evaluation; report repeated-concept contamination controls.
- [ ] Resolve duplicated IDs, inconsistent NC analogy flags, and empty distractors by a versioned, documented process.
- [ ] Publish actual annotation and expected-refusal rules, agreement, and adjudication records.

## Experimental evidence

- [ ] Run both conditions on the same selected UIDs and the same fixed dataset snapshot.
- [ ] Run several declared random seeds; record original per-instance outputs, prompts, costs, and complete manifests.
- [ ] Declare the exact metric tokenizer, implementation, package versions, and BERTScore model/layer.
- [ ] Supply full manual labels for real RR/CRR/CCFR; do not rebrand string proxies as semantic correctness.
- [ ] Reproduce paper tables/figures from the saved outputs rather than hard-coded aggregate values.
- [ ] Use appropriate repeated-concept, multi-seed, and multiple-comparison analyses before significance claims.
- [ ] Freeze a tested dependency lock or exact environment report and pin model revisions.

## Repository hygiene

- [ ] Inspect `git diff --cached` before the first push.
- [ ] Do not upload API keys, model weights, confidential prompts, review correspondence, Word lock files, or the parent workspace.
- [ ] Keep `outputs/` excluded unless sanitized research artifacts are deliberately released.
- [ ] Describe this as a paper-guided reference implementation until comparison with the original program and real experimental outputs justifies stronger wording.

These are scientific-release checks, not reasons to delay trying the offline demo or inspecting the code.
