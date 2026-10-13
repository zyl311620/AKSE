"""Offline end-to-end checks. No internet, model downloads, or paid requests."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from akse.cli import main
from akse.data import dump_json, fingerprint
from akse.evaluation import read_jsonl
from akse.models import DemoModel, HTTPModel
from test_core import VectorEncoder, record


class IntegrationTests(unittest.TestCase):
    def call_cli(self, arguments):
        with patch("sys.argv", ["akse", *map(str, arguments)]), redirect_stdout(io.StringIO()):
            main()

    def test_real_http_adapter_schema_without_network(self):
        model = HTTPModel("http://127.0.0.1:8000/v1", "test-model", api_key_env="AKSE_TEST_UNUSED_KEY")
        payload = {"choices": [{"message": {"content": "Test answer"}}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as transport:
            answer = model.generate([{"role": "user", "content": "test"}], purpose="answer",
                                    seed=7, temperature=0.7, top_p=0.9, max_new_tokens=2048)
        request = transport.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(sent["max_tokens"], 2048)
        self.assertEqual(sent["seed"], 7)
        self.assertEqual(answer, "Test answer")
        self.assertEqual(model.calls[0]["usage"]["completion_tokens"], 2)
        self.assertNotIn("headers", model.calls[0])

    def test_cli_akse_baseline_evaluation_annotations_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.json"
            r = {**record(), "question": "What is an adaptive memory mesh?",
                 "new_concept": "adaptive memory mesh", "context_hint": "Memory reallocates.",
                 "correct_answer": "An adaptive memory mesh adjusts distributed memory allocation as the workload changes."}
            dump_json(dataset, [r])
            config = root / "config.json"
            dump_json(config, {"novelty_threshold": 0.1})
            kb, salience, history = root / "kb.json", root / "salience.json", root / "history.json"
            dump_json(kb, [{"concept": "distributed memory", "text": "distributed memory", "source": "TEST fixture"}])
            dump_json(salience, {"000000:TEST": {"score": 0.65, "source": "TEST fixture"}})
            dump_json(history, {"000000:TEST": {"current_timestamp": "2026-01-04T00:00:00Z",
                                              "observations": [
                                                  {"timestamp": f"2026-01-0{i}T00:00:00Z",
                                                   "text": "observation", "source": "TEST fixture"}
                                                  for i in range(1, 4)]}})
            common = ["run", "--dataset", dataset, "--config", config, "--model", "TEST-fixture",
                      "--seed", "42"]
            with patch("akse.cli.HTTPModel", side_effect=lambda *a, **kw: DemoModel()), \
                    patch("akse.cli.SemanticEncoder", side_effect=lambda *a, **kw: VectorEncoder()):
                self.call_cli([*common, "--knowledge", kb, "--salience", salience,
                               "--history", history, "--output", root / "akse"])
            with patch("akse.cli.HTTPModel", side_effect=lambda *a, **kw: DemoModel()), \
                    patch("akse.cli.SemanticEncoder") as unused_encoder:
                self.call_cli([*common, "--method", "baseline", "--output", root / "baseline"])
                unused_encoder.assert_not_called()
            predictions = root / "akse/predictions.jsonl"
            rows = read_jsonl(predictions)
            self.assertEqual(rows[0]["dataset_sha256"], fingerprint(dataset))
            self.assertEqual(rows[0]["cost"]["calls"], 6)
            self.assertEqual(len(rows[0]["knowledge"]), 2)
            self.assertNotIn("correct_answer", rows[0])
            self.assertNotIn("distractors", rows[0])
            manifest = json.loads((root / "akse/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed"], 1)
            annotations = root / "annotations.json"
            self.call_cli(["make-annotations", "--predictions", predictions, "--output", annotations])
            labels = json.loads(annotations.read_text(encoding="utf-8"))
            labels["labels"]["000000:TEST"].update(expected_refusal=False, refused=False,
                                                  concept_confused=False, source="TEST annotated fixture")
            dump_json(annotations, labels)
            self.call_cli(["evaluate", "--dataset", dataset, "--predictions", predictions,
                           "--annotations", annotations, "--output", root / "eval_akse"])
            self.call_cli(["evaluate", "--dataset", dataset, "--predictions",
                           root / "baseline/predictions.jsonl", "--output", root / "eval_baseline"])
            self.call_cli(["compare", "--left", root / "eval_baseline/per_instance.jsonl",
                           "--right", root / "eval_akse/per_instance.jsonl",
                           "--draws", "100", "--output", root / "interval.json"])
            result = json.loads((root / "interval.json").read_text(encoding="utf-8"))
            self.assertEqual(result["mean_difference"], 0.0)

    def test_supplied_dataset_identity(self):
        path = Path(__file__).resolve().parents[1] / "data/BNeCR.json"
        self.assertEqual(fingerprint(path),
                         "6e66a5ad94e3520af3dbe316223a463b0080d32155592154c43b0489fb8512bc")


if __name__ == "__main__":
    unittest.main()
