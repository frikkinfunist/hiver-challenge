"""
Generate an email reply for one EmailExample using:

  retrieval (SBERT top-k similar threads)  +  persona-conditioned prompt
      -> LLM (local HF model / OpenAI API / LoRA fine-tuned model)

This is the "mixed" strategy described in the README: retrieval supplies
grounding + few-shot style examples, the persona field supplies explicit
style instructions, and the base/fine-tuned LLM does the actual drafting.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .data_utils import EmailExample
from .prompt_builder import build_prompt
from .persona_variants import build_reply_personas
from .retrieval import SBERTRetriever


class ReplyGenerator:
    def __init__(self, config: dict, retriever: SBERTRetriever | None = None):
        self.config = config
        self.backend = config["generation"]["backend"]
        self.retriever = retriever
        self._llm = None
        self._tokenizer = None

        if self.backend in ("hf", "lora"):
            self._load_hf_backend()
        elif self.backend == "openai":
            from openai import OpenAI
            from .secrets_utils import load_api_key

            api_key = load_api_key("OPENAI_API_KEY", "openai_api_key.txt")
            if not api_key:
                raise RuntimeError(
                    "No OpenAI key found. Set OPENAI_API_KEY, or paste your "
                    "key into openai_api_key.txt at the repo root."
                )
            self._openai_client = OpenAI(api_key=api_key)
        elif self.backend == "gemini":
            self._load_gemini_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------ #
    def _load_hf_backend(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gen_cfg = self.config["generation"]
        base_name = gen_cfg["hf_model_name"]
        self._tokenizer = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        if self.backend == "lora":
            from peft import PeftModel

            adapter_path = gen_cfg["lora_adapter_path"]
            model = PeftModel.from_pretrained(model, adapter_path)

        self._llm = model

    def _load_gemini_backend(self):
        from google import genai
        from .secrets_utils import load_api_key

        api_key = load_api_key("GEMINI_API_KEY", "gemini_api_key.txt")
        if not api_key:
            raise RuntimeError(
                "No Gemini key found. Set GEMINI_API_KEY, or paste your "
                "key into gemini_api_key.txt at the repo root."
            )
        self._gemini_client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------ #
    def _get_fewshot(self, example: EmailExample) -> list[dict]:
        if self.retriever is None:
            return []
        top_k = self.config["retrieval"]["top_k"]
        scope = self.config["retrieval"]["scope"]
        return self.retriever.retrieve(
            query_thread_text=example.thread_as_text(),
            user_id=example.user_id,
            top_k=top_k,
            scope=scope,
            exclude_post_id=example.post_id,
        )

    # ------------------------------------------------------------------ #
    def generate(self, example: EmailExample) -> str:
        fewshot = self._get_fewshot(example)
        messages = build_prompt(example.persona, example.thread_as_text(), fewshot)
        return self._generate_from_messages(messages)

    def _generate_from_messages(self, messages: list[dict]) -> str:
        if self.backend == "openai":
            return self._generate_openai(messages)
        if self.backend == "gemini":
            return self._generate_gemini(messages)
        return self._generate_hf(messages)

    # ------------------------------------------------------------------ #
    def suggest_replies(
        self,
        example: EmailExample,
        n: int = 4,
        dedup_threshold: float = 0.93,
        max_resample_attempts: int = 2,
    ) -> list[dict]:
        """Generate n≈3-5 distinct reply suggestions for the same email.

        Each suggestion is grounded in the SAME retrieved few-shot examples
        (so they're all still "this person's" replies), but drafted under a
        different style variant (concise, warm, formal, action-oriented,
        ...) from persona_variants.py -- so the options differ in a
        meaningful, controllable way rather than being random resamples of
        one prompt.

        SBERT is used a second time here, as a dedup check: if two
        suggestions end up saying almost the same thing (cosine similarity
        above `dedup_threshold`), we resample the later one under a fresh
        style variant instead of returning near-duplicates.
        """
        fewshot = self._get_fewshot(example)
        persona_variants = build_reply_personas(example.persona, n)

        sbert = self._get_dedup_model()
        accepted: list[dict] = []
        accepted_embs: list[np.ndarray] = []

        # spare variants to draw from if we need to resample a duplicate
        spare_pool = build_reply_personas(example.persona, len(persona_variants) + max_resample_attempts * n)
        spare_iter = iter(spare_pool[len(persona_variants):])

        candidates = list(persona_variants)
        while candidates:
            variant_name, merged_persona = candidates.pop(0)
            messages = build_prompt(merged_persona, example.thread_as_text(), fewshot)
            reply_text = self._generate_from_messages(messages)

            emb = sbert.encode([reply_text], normalize_embeddings=True)[0]
            is_dup = any(float(np.dot(emb, e)) >= dedup_threshold for e in accepted_embs)

            if is_dup:
                # try again once under a different, unused style variant
                try:
                    replacement = next(spare_iter)
                    candidates.insert(0, replacement)
                    continue
                except StopIteration:
                    pass  # no spares left -- keep it anyway, flagged below

            accepted.append(
                {
                    "style": variant_name,
                    "reply": reply_text,
                    "possible_duplicate": is_dup,
                }
            )
            accepted_embs.append(emb)

        return accepted

    def _get_dedup_model(self):
        """Reuse the retriever's SBERT model for dedup checks if we have
        one loaded already; otherwise lazily load one just for this."""
        if self.retriever is not None:
            return self.retriever.model
        if not hasattr(self, "_dedup_sbert") or self._dedup_sbert is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.config["sbert"]["model_name"]
            device = self.config["sbert"]["device"]
            self._dedup_sbert = SentenceTransformer(
                model_name, device=None if device == "auto" else device
            )
        return self._dedup_sbert

    # ------------------------------------------------------------------ #
    def _generate_openai(self, messages: list[dict]) -> str:
        gen_cfg = self.config["generation"]
        resp = self._openai_client.chat.completions.create(
            model=gen_cfg["openai_model_name"],
            messages=messages,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_new_tokens"],
        )
        return resp.choices[0].message.content.strip()

    def _generate_gemini(self, messages: list[dict]) -> str:
        """messages is always [{"role": "system", ...}, {"role": "user", ...}]
        as produced by prompt_builder.build_prompt -- Gemini takes the
        system message as a separate `system_instruction` rather than as
        part of the chat turns."""
        from google.genai import types

        gen_cfg = self.config["generation"]
        system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_text = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

        resp = self._gemini_client.models.generate_content(
            model=gen_cfg["gemini_model_name"],
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                max_output_tokens=gen_cfg["max_new_tokens"],
            ),
        )
        return (resp.text or "").strip()

    def _generate_hf(self, messages: list[dict]) -> str:
        gen_cfg = self.config["generation"]
        prompt_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._llm.device)

        output_ids = self._llm.generate(
            prompt_ids,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][prompt_ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_config(path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import argparse

    from .data_utils import load_humanual_email, iter_examples

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n", type=int, default=3, help="how many test examples to run")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--suggestions",
        type=int,
        default=1,
        help="replies to generate PER email (1 = single reply, 3-5 = multi-persona suggestions)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    retriever = None
    if not args.no_retrieval:
        retriever = SBERTRetriever(cfg["sbert"]["model_name"], cfg["sbert"]["device"])
        retriever.load(cfg["retrieval"]["index_path"], cfg["retrieval"]["meta_path"])

    generator = ReplyGenerator(cfg, retriever)

    ds = load_humanual_email(cfg)
    test_examples = iter_examples(ds[cfg["dataset"]["test_split"]])[: args.n]

    for ex in test_examples:
        print("=" * 80)
        print("PERSONA:", ex.persona[:200])
        print("-" * 80)

        if args.suggestions <= 1:
            reply = generator.generate(ex)
            print("GENERATED REPLY:\n", reply)
        else:
            suggestions = generator.suggest_replies(ex, n=args.suggestions)
            for i, s in enumerate(suggestions, 1):
                flag = "  [similar to an earlier suggestion]" if s["possible_duplicate"] else ""
                print(f"\n--- Suggestion {i} [{s['style']}]{flag} ---\n{s['reply']}")

        print("-" * 80)
        print("GROUND TRUTH:\n", ex.completion)
