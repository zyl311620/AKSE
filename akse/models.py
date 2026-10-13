"""Generation backends. Real backends never receive evaluation reference fields."""

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class DemoModel:
    """Scripted fixture for exercising the pipeline, NOT an evaluated LLM."""
    identifier = "DEMO_ONLY/scripted-memory-mesh"

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, purpose, seed, temperature, top_p, max_new_tokens):
        if purpose == "hypothesis":
            answer = json.dumps({"hypotheses": [{
                "known_entity": "distributed memory",
                "text": "An adaptive memory mesh is an adaptive form of distributed memory."
            }]})
        else:
            answer = "An adaptive memory mesh adjusts distributed memory allocation as the workload changes."
        self.calls.append({"purpose": purpose, "seed": seed, "elapsed_seconds": 0.0,
                           "usage": None, "messages": messages, "output": answer})
        return answer


class HTTPModel:
    """Chat Completions-compatible endpoint; server capabilities must be checked."""
    def __init__(self, base_url, model, api_key_env="AKSE_API_KEY", timeout=120):
        parsed = urlparse(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("base URL must be an HTTP(S) URL without credentials/query/fragment")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("non-local endpoints require HTTPS")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key_env, self.timeout = model, api_key_env, timeout
        self.identifier = f"chat-completions:{model}"
        self.calls = []

    def generate(self, messages, *, purpose, seed, temperature, top_p, max_new_tokens):
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_new_tokens, "seed": seed}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.url, json.dumps(payload).encode("utf-8"), headers)
        started = time.perf_counter()
        # No implicit retries: duplicate paid requests/changed draws are not hidden.
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}; verify model and supported parameters") from None
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("endpoint returned empty or non-text content")
        self.calls.append({"purpose": purpose, "seed": seed,
                           "elapsed_seconds": time.perf_counter() - started,
                           "usage": result.get("usage"), "system_fingerprint": result.get("system_fingerprint"),
                           "messages": messages, "output": text})
        return text


class TransformersModel:
    """Local causal LLM using its own chat template; no remote code by default."""
    def __init__(self, model, revision=None, trust_remote_code=False, chat_format="template"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.chat_format = torch, chat_format
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, trust_remote_code=trust_remote_code,
            torch_dtype="auto", device_map="auto")
        self.model.eval()
        self.identifier = f"transformers:{model}@{revision or 'UNPINNED'}"
        self.calls = []
        if chat_format == "template" and not self.tokenizer.chat_template:
            raise ValueError("checkpoint has no chat template; explicitly provide --chat-format baichuan2 if appropriate")

    def generate(self, messages, *, purpose, seed, temperature, top_p, max_new_tokens):
        from transformers import set_seed
        set_seed(seed)
        if self.chat_format == "baichuan2":
            # Explicit model-specific choice, not a generic fallback template.
            text = "\n".join(m["content"] for m in messages)
            ids = [195] + self.tokenizer.encode(text, add_special_tokens=False) + [196]
            input_ids = self.torch.tensor([ids], device=self.model.device)
            inputs = {"input_ids": input_ids, "attention_mask": self.torch.ones_like(input_ids)}
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                   "pad_token_id": (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                    else self.tokenizer.eos_token_id)}
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, **options)
        answer_ids = result[0, input_length:]
        text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        self.calls.append({"purpose": purpose, "seed": seed,
                           "elapsed_seconds": time.perf_counter() - started,
                           "usage": {"prompt_tokens": input_length, "completion_tokens": len(answer_ids)},
                           "messages": messages, "output": text})
        return text
