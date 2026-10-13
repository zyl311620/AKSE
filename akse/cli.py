"""Portable CLI: python -m akse <command>."""

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Config
from .data import audit, dump_json, fingerprint, inference_input, load_dataset, load_json, uid
from .embeddings import HashEncoder, SemanticEncoder
from .evaluation import evaluate, write_jsonl
from .knowledge import KnowledgeBase
from .models import DemoModel, HTTPModel, TransformersModel
from .pipeline import AKSE, preceding_observations
from .statistics import paired_bootstrap


def environment():
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "accelerate", "bert-score"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "akse_version": __version__,
            "packages": versions}


def demo(output):
    concept = "adaptive memory mesh"
    query = {"uid": "DEMO:0001", "question": "What is an adaptive memory mesh?",
             "new_concept": concept,
             "context_hint": "An adaptive memory mesh reallocates storage across nodes as the workload changes."}
    encoder, model = HashEncoder(), DemoModel()
    kb = KnowledgeBase([{"concept": "distributed memory",
                         "text": "Distributed memory allocates storage across computing nodes.",
                         "source": "synthetic demo fixture"}], encoder)
    history_text = concept + "\n" + query["context_hint"]
    history = {query["uid"]: {"current_timestamp": "2026-01-04T00:00:00Z", "observations": [
        {"timestamp": f"2026-01-0{i}T00:00:00Z", "text": history_text,
         "source": "SYNTHETIC repeated observation; not empirical temporal evidence"}
        for i in range(1, 4)]}}
    engine = AKSE(Config(novelty_threshold=0.1), encoder, model, kb,
                  {query["uid"]: {"score": 0.65, "source": "ASSUMED demo salience, not measured MI"}},
                  history)
    result = engine.run(query)
    result["warning"] = "SCRIPTED OFFLINE DEMO. Scores are toy computations, not paper results."
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "trace.json", result)
    print(result["answer"])
    print(f"Demo trace: {output.resolve() / 'trace.json'}")


def run(args):
    records = load_dataset(args.dataset)
    config = Config(**load_json(args.config))
    if args.novelty_threshold is not None:
        config = replace(config, novelty_threshold=args.novelty_threshold)
    config.validate()
    seed_entries = load_json(args.knowledge) if args.knowledge else []
    salience = load_json(args.salience) if args.salience else {}
    history = load_json(args.history) if args.history else {}
    selected = [(i, r) for i, r in enumerate(records) if args.condition == "all" or
                ("NC" if not r["context_hint"] else "CA" if r["analogical_possible"] else "OC") == args.condition]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no records selected")
    if args.method == "akse":
        if not seed_entries:
            raise ValueError("AKSE requires an external initial --knowledge file")
        if config.use_detection:
            if config.novelty_threshold is None:
                raise ValueError("supply --novelty-threshold or set it in the configuration")
            missing = [uid(i, r) for i, r in selected if uid(i, r) not in salience]
            if missing:
                raise ValueError(f"missing --salience records, beginning with {missing[:3]}")
        if config.use_refinement and not history:
            raise ValueError("provide --history; BNeCR does not contain historical observations")
        for index, record in selected:
            preceding_observations(history.get(uid(index, record)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.backend == "http":
        model = HTTPModel(args.base_url, args.model)
    else:
        model = TransformersModel(args.model, revision=args.model_revision,
                                  trust_remote_code=args.trust_remote_code, chat_format=args.chat_format)
    encoder = SemanticEncoder(args.encoder, args.encoder_revision, args.device) if args.method == "akse" else None
    sha = fingerprint(args.dataset)
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "dataset_sha256": sha, "selected": len(selected),
                "selected_uids": [uid(i, r) for i, r in selected],
                "method": args.method, "mode": args.mode, "seed": args.seed,
                "model": model.identifier, "encoder": encoder.identifier if encoder else None,
                "model_revision": args.model_revision, "encoder_revision": args.encoder_revision,
                "chat_format": args.chat_format, "trust_remote_code": args.trust_remote_code,
                "backend": args.backend, "base_url": args.base_url if args.backend == "http" else None,
                "config": asdict(config), "environment": environment(),
                "input_hashes": {name: fingerprint(path) if path else None for name, path in
                                 (("knowledge", args.knowledge), ("salience", args.salience), ("history", args.history))}}
    dump_json(output / "manifest.json", manifest)
    shared = KnowledgeBase(seed_entries, encoder) if encoder else None
    completed = 0
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for i, r in selected:
                kb = (shared if args.mode == "stream" else KnowledgeBase(seed_entries, encoder)) if encoder else None
                # Baselines don't use knowledge, encoder, salience, or historical evidence.
                engine = AKSE(config, encoder, model, kb, salience, history)
                result = engine.run(inference_input(i, r), seed=args.seed + i * (config.trajectories + 1),
                                    method=args.method)
                result["dataset_sha256"] = sha
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                # Calls are retained in each row, not in RAM over the entire benchmark.
                model.calls.clear()
                completed += 1
                print(f"[{completed}/{len(selected)}] {result['uid']} calls={result['cost']['calls']}", flush=True)
        if args.mode == "stream" and shared:
            dump_json(output / "final_knowledge.json", shared.export())
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["completed"] = completed
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        dump_json(output / "manifest.json", manifest)


def make_annotations(args):
    from .evaluation import read_jsonl
    predictions = read_jsonl(args.predictions)
    if not predictions:
        raise ValueError("empty predictions")
    hashes = {p.get("dataset_sha256") for p in predictions}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("predictions must share a dataset fingerprint")
    if len({p["uid"] for p in predictions}) != len(predictions):
        raise ValueError("duplicate prediction UIDs")
    if Path(args.output).exists():
        raise FileExistsError("annotation output already exists")
    dump_json(args.output, {"dataset_sha256": next(iter(hashes)),
                           "labels": {p["uid"]: {"expected_refusal": None, "refused": None,
                                                "concept_confused": None, "source": ""}
                                      for p in predictions}})
    print(f"Fill the Boolean labels and provenance in {args.output}")


def parser():
    root = argparse.ArgumentParser(description="Paper-guided AKSE reference implementation")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="scripted offline demonstration, not a benchmark")
    d.add_argument("--output", default="outputs/demo")
    a = commands.add_parser("audit", help="validate and report the unmodified BNeCR data")
    a.add_argument("--dataset", default="data/BNeCR.json")
    a.add_argument("--output")
    r = commands.add_parser("run", help="run a real local or served LLM")
    r.add_argument("--dataset", default="data/BNeCR.json")
    r.add_argument("--config", default="configs/paper.json")
    r.add_argument("--method", choices=["akse", "baseline", "zero_cot"], default="akse")
    r.add_argument("--backend", choices=["http", "transformers"], default="http")
    r.add_argument("--model", required=True, help="exact checkpoint path/name or server model ID")
    r.add_argument("--model-revision")
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--chat-format", choices=["template", "baichuan2"], default="template")
    r.add_argument("--trust-remote-code", action="store_true", help="ONLY after reviewing pinned checkpoint code")
    r.add_argument("--encoder", default="BAAI/bge-m3")
    r.add_argument("--encoder-revision")
    r.add_argument("--device")
    r.add_argument("--knowledge", help="external seed KB JSON")
    r.add_argument("--salience", help="per-UID measured salience JSON")
    r.add_argument("--history", help="per-UID preceding concept observations JSON")
    r.add_argument("--novelty-threshold", type=float)
    r.add_argument("--condition", choices=["all", "CA", "OC", "NC"], default="all")
    r.add_argument("--mode", choices=["isolated", "stream"], default="isolated")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--limit", type=int, default=0, help="0 means full dataset")
    r.add_argument("--output", required=True, help="new directory; existing directories are not overwritten")
    e = commands.add_parser("evaluate")
    e.add_argument("--dataset", default="data/BNeCR.json")
    e.add_argument("--predictions", required=True)
    e.add_argument("--annotations")
    e.add_argument("--bertscore-model", help="pinned local model directory preferred")
    e.add_argument("--bertscore-layers", type=int)
    e.add_argument("--device")
    e.add_argument("--output", required=True)
    b = commands.add_parser("compare")
    b.add_argument("--left", required=True, help="baseline evaluation/per_instance.jsonl")
    b.add_argument("--right", required=True, help="AKSE evaluation/per_instance.jsonl")
    b.add_argument("--metric", default="rougeL_f1")
    b.add_argument("--draws", type=int, default=2000)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--output", required=True)
    t = commands.add_parser("make-annotations")
    t.add_argument("--predictions", required=True)
    t.add_argument("--output", required=True)
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "demo":
            demo(args.output)
        elif args.command == "audit":
            result = audit(args.dataset)
            if args.output:
                dump_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "run":
            if args.limit < 0 or args.seed < 0:
                raise ValueError("limit and seed must be nonnegative")
            run(args)
        elif args.command == "evaluate":
            result = evaluate(args.dataset, args.predictions, args.output, args.annotations,
                              args.bertscore_model, args.bertscore_layers, args.device)
            print(json.dumps(result["groups"]["overall"], ensure_ascii=False, indent=2))
        elif args.command == "compare":
            if Path(args.output).exists():
                raise FileExistsError("comparison output already exists")
            result = paired_bootstrap(args.left, args.right, args.metric, args.draws, args.seed)
            dump_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "make-annotations":
            make_annotations(args)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
