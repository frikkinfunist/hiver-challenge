"""
Retrieval-augmented generation (RAG) component.

We embed every (thread-context -> completion) pair from the TRAIN split with
an SBERT model and index it with FAISS. At inference time, given a new
email thread, we embed the incoming thread and pull back the top-k most
*semantically* similar past threads. Their (context, reply) pairs are used
as few-shot demonstrations in the generation prompt, and -- because we bias
retrieval toward the same persona/user first -- they also inject that
person's real phrasing, tone and sign-off habits.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .data_utils import EmailExample, iter_examples


class SBERTRetriever:
    def __init__(self, model_name: str, device: str = "auto"):
        if device == "auto":
            device = None  # let sentence-transformers pick cuda/mps/cpu
        self.model = SentenceTransformer(model_name, device=device)
        self.index: faiss.Index | None = None
        self.meta: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Index building
    # ------------------------------------------------------------------ #
    def build_index(self, examples: list[EmailExample]) -> None:
        texts = [ex.thread_as_text() for ex in examples]
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
        index.add(embeddings)
        self.index = index

        self.meta = pd.DataFrame(
            [
                {
                    "post_id": ex.post_id,
                    "user_id": ex.user_id,
                    "persona": ex.persona,
                    "thread_text": ex.thread_as_text(),
                    "completion": ex.completion,
                }
                for ex in examples
            ]
        )

    def save(self, index_path: str, meta_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        self.meta.to_parquet(meta_path)

    def load(self, index_path: str, meta_path: str) -> None:
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_parquet(meta_path)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query_thread_text: str,
        user_id: str | None = None,
        top_k: int = 3,
        scope: str = "user_first",
        exclude_post_id: str | None = None,
    ) -> list[dict]:
        """Return the top_k most similar (thread, completion) pairs."""
        q_emb = self.model.encode(
            [query_thread_text], normalize_embeddings=True
        ).astype("float32")

        # Over-fetch, then filter, so user-scoping / self-exclusion doesn't
        # leave us short of top_k results.
        fetch_k = max(top_k * 8, 32)
        scores, idxs = self.index.search(q_emb, fetch_k)
        scores, idxs = scores[0], idxs[0]

        candidates = self.meta.iloc[idxs].copy()
        candidates["score"] = scores

        if exclude_post_id is not None:
            candidates = candidates[candidates["post_id"] != exclude_post_id]

        results: list[dict]
        if scope == "user_first" and user_id is not None:
            same_user = candidates[candidates["user_id"] == user_id]
            if len(same_user) >= top_k:
                results = same_user.head(top_k).to_dict("records")
            else:
                # top up with the best global matches not already included
                remaining = top_k - len(same_user)
                other = candidates[candidates["user_id"] != user_id].head(remaining)
                results = pd.concat([same_user, other]).to_dict("records")
        else:
            results = candidates.head(top_k).to_dict("records")

        return results


def build_and_save_index(config: dict) -> SBERTRetriever:
    from .data_utils import load_humanual_email

    ds = load_humanual_email(config)
    train_examples = iter_examples(ds[config["dataset"]["train_split"]])

    retriever = SBERTRetriever(
        config["sbert"]["model_name"], device=config["sbert"]["device"]
    )
    retriever.build_index(train_examples)
    retriever.save(
        config["retrieval"]["index_path"], config["retrieval"]["meta_path"]
    )
    return retriever


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    build_and_save_index(cfg)
    print("Index built and saved.")
