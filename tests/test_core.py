import json
import math
import tempfile
import unittest
from pathlib import Path

from akse.config import Config
from akse.data import audit, dump_json, fingerprint, inference_input, uid
from akse.embeddings import HashEncoder
from akse.evaluation import evaluate, write_jsonl
from akse.knowledge import KnowledgeBase
from akse.maths import centroid, consistency_filter, cosine, mutual_information, normalize
from akse.metrics import corpus_bleu, rouge_l, rouge_n, score_answer, tokenize
from akse.models import DemoModel, HTTPModel
from akse.pipeline import AKSE, preceding_observations
from akse.statistics import paired_bootstrap


def record(identifier="TEST"):
    return {"id": identifier, "domain": "测试", "question": "什么是测试概念？",
            "new_concept": "测试概念", "context_hint": "这是测试上下文。",
            "analogical_possible": True, "correct_answer": "测试概念是示例概念。",
            "distractors": ["干扰词"]}


class VectorEncoder:
    identifier = "unit-test"

    def encode(self, text):
        return [0.8, 0.6] if text == "distributed memory" else [1.0, 0.0]


def engine(config=None, with_history=True):
    query = {"uid": "DEMO", "question": "What is an adaptive memory mesh?",
             "new_concept": "adaptive memory mesh", "context_hint": "Memory reallocates."}
    encoder, model = VectorEncoder(), DemoModel()
    kb = KnowledgeBase([{"concept": "distributed memory", "text": "distributed memory",
                         "source": "test fixture"}], encoder)
    history = {"DEMO": {"current_timestamp": "2026-01-04T00:00:00Z", "observations": [
        {"timestamp": f"2026-01-0{i}T00:00:00Z", "text": "observation", "source": "test fixture"}
        for i in range(1, 4)]}} if with_history else {}
    instance = AKSE(config or Config(novelty_threshold=0.1), encoder, model, kb,
                    {"DEMO": {"score": 0.65, "source": "test fixture"}}, history)
    return instance, query


class MathsTests(unittest.TestCase):
    def test_cosine_signed_and_novelty_range(self):
        self.assertEqual(cosine([1, 0], [-1, 0]), -1)
        self.assertEqual(1 - cosine([1, 0], [-1, 0]), 2)
        self.assertEqual(centroid([[1, 0], [0, 1]]), [0.5, 0.5])
        with self.assertRaises(ValueError):
            cosine([0, 0], [1, 0])
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine([float("nan")], [1])

    def test_normalization(self):
        self.assertEqual(normalize(" A， B。\n"), "AB")
        self.assertNotEqual(normalize("A"), normalize("a"))
        self.assertNotEqual(normalize("not true"), normalize("true"))

    def test_strict_consistency(self):
        h = {"known_entity": "k", "text": "hypothesis."}
        self.assertEqual(consistency_filter([[h], [h], [h], [h], []], 0.8), [])
        self.assertEqual(consistency_filter([[h]] * 5, 0.8)[0]["recurrence"], 1)

    def test_duplicate_within_trajectory_is_not_extra_vote(self):
        h = {"known_entity": "k", "text": "x"}
        self.assertEqual(consistency_filter([[h, h, h], [], [], [], []], 0.8), [])

    def test_entities_not_merged(self):
        a = {"known_entity": "a", "text": "x"}
        b = {"known_entity": "b", "text": "x"}
        self.assertEqual(consistency_filter([[a], [a], [a], [a], [b]], 0.8), [])

    def test_mutual_information(self):
        self.assertAlmostEqual(mutual_information([[0.5, 0], [0, 0.5]]), math.log(2))
        self.assertAlmostEqual(mutual_information([[0.25, 0.25], [0.25, 0.25]]), 0)
        with self.assertRaises(ValueError):
            mutual_information([[1, 1]])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            Config(beta_semantic=0.5).validate()
        with self.assertRaises(ValueError):
            Config(trajectories=0).validate()
        with self.assertRaises(ValueError):
            Config(novelty_threshold=float("nan")).validate()


class PipelineTests(unittest.TestCase):
    def test_four_stages(self):
        instance, query = engine()
        result = instance.run(query)
        verification = result["trace"]["stages"]["refinement"][0]
        self.assertAlmostEqual(verification["refinement"], 0.92)
        self.assertTrue(verification["accepted"])
        self.assertEqual(result["cost"]["calls"], 6)
        self.assertEqual(len(result["knowledge"]), 2)

    def test_missing_history_never_defaults_to_perfect_stability(self):
        instance, query = engine(with_history=False)
        result = instance.run(query)
        verification = result["trace"]["stages"]["refinement"][0]
        self.assertIsNone(verification["temporal"])
        self.assertFalse(verification["accepted"])

    def test_strict_assimilation(self):
        instance, query = engine(Config(novelty_threshold=0.1, assimilation_threshold=0.92))
        result = instance.run(query)
        self.assertFalse(result["trace"]["stages"]["refinement"][0]["accepted"])

    def test_strict_detection(self):
        instance, query = engine(Config(novelty_threshold=0.1, salience_threshold=0.65))
        result = instance.run(query)
        self.assertFalse(result["trace"]["stages"]["detection"]["passed"])
        self.assertEqual(result["cost"]["calls"], 1)

    def test_unknown_novelty_requires_explicit_value(self):
        instance, query = engine(Config())
        with self.assertRaises(ValueError):
            instance.run(query)

    def test_refinement_ablation(self):
        instance, query = engine(Config(novelty_threshold=0.1, use_refinement=False), with_history=False)
        self.assertTrue(instance.run(query)["trace"]["stages"]["refinement"][0]["accepted"])

    def test_no_reasoning_enhancement(self):
        instance, query = engine(Config(novelty_threshold=0.1, use_reasoning_enhancement=False))
        result = instance.run(query)
        self.assertEqual(result["trace"]["stages"]["reasoning"]["retrieved"], [])

    def test_baseline_is_one_call(self):
        instance, query = engine()
        result = instance.run(query, method="baseline")
        self.assertEqual(result["cost"]["calls"], 1)
        self.assertEqual(result["trace"]["stages"]["reasoning"]["retrieved"], [])

    def test_reference_fields_are_rejected(self):
        instance, query = engine()
        with self.assertRaises(ValueError):
            instance.run({**query, "correct_answer": "leaked"})

    def test_temporal_order(self):
        bundle = {"current_timestamp": "2026-01-02T00:00:00Z", "observations": [
            {"timestamp": "2026-01-03T00:00:00Z", "text": "future", "source": "test"}]}
        with self.assertRaises(ValueError):
            preceding_observations(bundle)
        with self.assertRaises(ValueError):
            preceding_observations({"current_timestamp": "2026-01-02", "observations": []})

    def test_http_requires_secure_remote_transport(self):
        with self.assertRaises(ValueError):
            HTTPModel("http://example.com/v1", "m")
        HTTPModel("http://127.0.0.1:8000/v1", "m")


class MetricsTests(unittest.TestCase):
    def test_chinese_tokenization(self):
        self.assertEqual(tokenize("量子 AI-2。"), ["量", "子", "ai", "2"])

    def test_exact_match(self):
        tokens = tokenize("一二三四五六")
        self.assertEqual(rouge_n(tokens, tokens, 1), 1)
        self.assertEqual(rouge_n(tokens, tokens, 2), 1)
        self.assertEqual(rouge_l(tokens, tokens), 1)
        self.assertEqual(corpus_bleu([(tokens, tokens)]), 1)

    def test_empty_metrics(self):
        self.assertEqual(rouge_l([], []), 0)
        self.assertEqual(corpus_bleu([([], ["x"])]), 0)

    def test_short_bleu_is_explicitly_zero(self):
        self.assertEqual(corpus_bleu([(["x"], ["x"])]), 0)

    def test_distractor_mention_is_only_proxy(self):
        scores = score_answer("并不是干扰词。", "测试概念", ["干扰词"])
        self.assertTrue(scores["distractor_substring_hit"])
        self.assertNotIn("concept_confused", scores)


class DataEvaluationTests(unittest.TestCase):
    def test_duplicate_ids_keep_distinct_uids(self):
        self.assertNotEqual(uid(0, record()), uid(1, record()))
        self.assertNotIn("correct_answer", inference_input(0, record()))
        self.assertNotIn("distractors", inference_input(0, record()))

    def test_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            dump_json(path, [record(), record()])
            report = audit(path)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["duplicate_ids"], {"TEST": 2})

    def test_evaluation_annotation_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, predictions, annotations = root / "data.json", root / "p.jsonl", root / "a.json"
            r = record()
            dump_json(data, [r])
            sha = fingerprint(data)
            prediction = {"uid": uid(0, r), "answer": r["correct_answer"], "dataset_sha256": sha}
            write_jsonl(predictions, [prediction])
            dump_json(annotations, {"dataset_sha256": sha, "labels": {
                uid(0, r): {"expected_refusal": False, "refused": False,
                            "concept_confused": False, "source": "test labels"}}})
            result = evaluate(data, predictions, root / "evaluation", annotations)
            overall = result["groups"]["overall"]
            self.assertEqual(overall["rouge1_f1"], 1)
            self.assertIsNone(overall["crr"])
            self.assertEqual(overall["ccfr"], 0)
            write_jsonl(predictions, [{**prediction, "dataset_sha256": "wrong"}])
            with self.assertRaises(ValueError):
                evaluate(data, predictions, root / "invalid")

    def test_partial_annotations_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, pred, ann = root / "data.json", root / "p.jsonl", root / "a.json"
            dump_json(data, [record()])
            sha = fingerprint(data)
            write_jsonl(pred, [{"uid": "000000:TEST", "dataset_sha256": sha, "answer": "answer"}])
            dump_json(ann, {"dataset_sha256": sha, "labels": {}})
            with self.assertRaises(ValueError):
                evaluate(data, pred, root / "evaluation", ann)

    def test_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.jsonl", root / "right.jsonl"
            write_jsonl(left, [{"uid": str(i), "dataset_sha256": "sha", "rougeL_f1": 0.2} for i in range(4)])
            write_jsonl(right, [{"uid": str(i), "dataset_sha256": "sha", "rougeL_f1": 0.4} for i in range(4)])
            result = paired_bootstrap(left, right, draws=100)
            self.assertAlmostEqual(result["mean_difference"], 0.2)
            self.assertAlmostEqual(result["ci95_percentile"][0], 0.2)

    def test_hash_encoder_is_reproducible(self):
        encoder = HashEncoder()
        self.assertEqual(encoder.encode("测试 memory"), encoder.encode("测试 memory"))


if __name__ == "__main__":
    unittest.main()
