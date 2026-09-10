"""
Evaluate generated replies against ground-truth completions.

Primary metric: SBERT cosine similarity.
    We embed the generated reply and the ground-truth reply with the same
    SBERT model used for retrieval (all-mpnet-base-v2 by default) and take
    the cosine similarity of the two embeddings. Because SBERT sentence
    embeddings capture *meaning* rather than exact wording, two replies
    that say "the same thing" in different words still score highly --
    which is what we want for an open-ended generation task like this,
    where there's no single "correct" reply string.

Secondary metric: ROUGE-L, for a lexical-overlap sanity check alongside the
semantic score.

Usage:
    python -m src.evaluate --config configs/config.yaml --n 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data_utils import load_humanual_email, iter_examples
from .generate import ReplyGenerator
from .retrieval import SBERTRetriever


def sbert_cosine_similarity(sbert_model: SentenceTransformer, a: str, b: str) -> float:
    embs = sbert_model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def evaluate(config: dict, n: int | None, use_retrieval: bool = True) -> dict:
    sbert_model = SentenceTransformer(
        config["sbert"]["model_name"], device=None if config["sbert"]["device"] == "auto" else config["sbert"]["device"]
    )
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    retriever = None
    if use_retrieval:
        retriever = SBERTRetriever(config["sbert"]["model_name"], config["sbert"]["device"])
        retriever.load(config["retrieval"]["index_path"], config["retrieval"]["meta_path"])

    generator = ReplyGenerator(config, retriever)

    ds = load_humanual_email(config)
    test_examples = iter_examples(ds[config["dataset"]["test_split"]])
    if n:
        test_examples = test_examples[:n]

    per_example = []
    sbert_scores, rouge_scores = [], []

    for ex in tqdm(test_examples, desc="Evaluating"):
        generated = generator.generate(ex)

        sim = sbert_cosine_similarity(sbert_model, generated, ex.completion)
        rl = rouge.score(ex.completion, generated)["rougeL"].fmeasure

        sbert_scores.append(sim)
        rouge_scores.append(rl)

        per_example.append(
            {
                "post_id": ex.post_id,
                "generated": generated,
                "ground_truth": ex.completion,
                "sbert_cosine": sim,
                "rouge_l": rl,
            }
        )

    results = {
        "n_examples": len(test_examples),
        "mean_sbert_cosine": float(np.mean(sbert_scores)) if sbert_scores else None,
        "median_sbert_cosine": float(np.median(sbert_scores)) if sbert_scores else None,
        "mean_rouge_l": float(np.mean(rouge_scores)) if rouge_scores else None,
        "per_example": per_example,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n", type=int, default=100, help="number of test examples to score")
    parser.add_argument("--no-retrieval", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = evaluate(cfg, args.n, use_retrieval=not args.no_retrieval)

    out_path = Path(cfg["evaluation"]["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Mean SBERT cosine similarity: {results['mean_sbert_cosine']:.4f}")
    print(f"Mean ROUGE-L: {results['mean_rouge_l']:.4f}")
    print(f"Full results written to {out_path}")
