"""
Optional: LoRA fine-tune a small open-weight causal LM on the train split so
it internalizes general "how Enron employees reply to emails" patterns,
instead of relying purely on in-context few-shot examples at inference time.

This is meant to be combined with (not replace) the retrieval + prompting
pipeline: at inference you can point `generation.backend: lora` in the
config at the resulting adapter and it will still receive persona + RAG
few-shot context on top of the fine-tuned weights.

Usage:
    python -m src.finetune_lora --config configs/config.yaml
"""

from __future__ import annotations

import argparse

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .data_utils import load_humanual_email, iter_examples
from .prompt_builder import build_prompt


def build_training_examples(examples, tokenizer, max_seq_len: int):
    """Turn each EmailExample into a single supervised (prompt + target) string."""
    records = []
    for ex in examples:
        messages = build_prompt(ex.persona, ex.thread_as_text(), fewshot_examples=None)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + ex.completion + tokenizer.eos_token
        records.append({"text": full_text})
    ds = Dataset.from_list(records)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_seq_len, padding="max_length"
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ft_cfg = cfg["finetune"]

    tokenizer = AutoTokenizer.from_pretrained(ft_cfg["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ft_cfg["base_model_name"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        r=ft_cfg["lora_r"],
        lora_alpha=ft_cfg["lora_alpha"],
        lora_dropout=ft_cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_humanual_email(cfg)
    train_examples = iter_examples(ds[cfg["dataset"]["train_split"]])
    train_ds = build_training_examples(train_examples, tokenizer, ft_cfg["max_seq_len"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=ft_cfg["output_dir"],
        num_train_epochs=ft_cfg["epochs"],
        per_device_train_batch_size=ft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=ft_cfg["gradient_accumulation_steps"],
        learning_rate=ft_cfg["learning_rate"],
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(ft_cfg["output_dir"])
    tokenizer.save_pretrained(ft_cfg["output_dir"])
    print(f"LoRA adapter saved to {ft_cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
