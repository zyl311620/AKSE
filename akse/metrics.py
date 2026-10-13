"""Explicit reference metrics, with a tokenizer suitable for Chinese text."""

import math
import re
from collections import Counter


TOKENIZER_NAME = "mixed-v1: lowercase Latin words/numbers + individual CJK characters; ignore punctuation"


def tokenize(text):
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def f1(overlap, generated, reference):
    return 2 * overlap / (generated + reference) if generated + reference else 0.0


def rouge_n(generated, reference, n):
    g, r = ngrams(generated, n), ngrams(reference, n)
    return f1(sum((g & r).values()), sum(g.values()), sum(r.values()))


def rouge_l(generated, reference):
    previous = [0] * (len(reference) + 1)
    for token in generated:
        current = [0]
        for j, target in enumerate(reference, 1):
            current.append(previous[j - 1] + 1 if token == target else max(previous[j], current[-1]))
        previous = current
    return f1(previous[-1], len(generated), len(reference))


def corpus_bleu(pairs):
    """Single-reference BLEU-4; equal weights, no smoothing/effective order."""
    matched, total = [0] * 4, [0] * 4
    glen = rlen = 0
    for generated, reference in pairs:
        glen += len(generated)
        rlen += len(reference)
        for n in range(1, 5):
            g, r = ngrams(generated, n), ngrams(reference, n)
            matched[n - 1] += sum((g & r).values())
            total[n - 1] += sum(g.values())
    if not glen or any(x == 0 for x in matched) or any(x == 0 for x in total):
        return 0.0
    bp = min(1.0, math.exp(1 - rlen / glen))
    return bp * math.exp(sum(math.log(m / t) for m, t in zip(matched, total)) / 4)


def heuristic_refused(text):
    """Diagnostic prefix rule, not a validated refusal classifier."""
    return bool(re.match(
        r"^\s*(?:我无法回答|无法回答|不能回答|我不能回答|无法确定|"
        r"信息不足|I cannot answer|I can't answer|Insufficient information)", text, re.I))


def score_answer(answer, reference, distractors):
    g, r = tokenize(answer), tokenize(reference)
    return {"rouge1_f1": rouge_n(g, r, 1), "rouge2_f1": rouge_n(g, r, 2),
            "rougeL_f1": rouge_l(g, r), "generated_tokens": len(g),
            "reference_tokens": len(r), "generated_characters": len(answer),
            "refused_heuristic": heuristic_refused(answer),
            # Includes negated mentions; cannot be interpreted as semantic confusion.
            "distractor_substring_hit": any(d in answer for d in distractors if d)}
