"""
Load and normalize the snap-stanford/humanual-email dataset.

Schema (per row):
    completion : str            -> ground-truth reply to generate
    post_id    : str             -> thread id
    user_id    : str             -> sha256-hashed sender address (the replier)
    timestamp  : int              -> unix timestamp of the reply
    turn_id    : int              -> position of this reply in the thread
    persona    : str              -> free-text description of the replier's
                                      communication style / role
    prompt     : list[{"role","content"}] -> the thread context the replier saw
    metadata   : str (JSON)        -> {"from","to","subject","cc","timestamp"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset, DatasetDict


@dataclass
class EmailExample:
    post_id: str
    user_id: str
    turn_id: int
    persona: str
    thread: list[dict]         # [{"role": ..., "content": ...}, ...]
    metadata: dict
    completion: str
    subject: str = ""

    def thread_as_text(self) -> str:
        """Flatten the prior thread into a readable block for prompting."""
        lines = []
        for msg in self.thread:
            sender = msg.get("role") or msg.get("from") or "sender"
            content = msg.get("content", "").strip()
            lines.append(f"From: {sender}\n{content}")
        return "\n\n---\n\n".join(lines)


def _parse_metadata(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def load_humanual_email(config: dict) -> DatasetDict:
    """Download (or read from HF cache) all three splits."""
    ds = load_dataset(config["dataset"]["name"])
    return ds


def row_to_example(row: dict[str, Any]) -> EmailExample:
    meta = _parse_metadata(row.get("metadata", "{}"))
    return EmailExample(
        post_id=row["post_id"],
        user_id=row["user_id"],
        turn_id=row["turn_id"],
        persona=row.get("persona", ""),
        thread=row.get("prompt", []),
        metadata=meta,
        completion=row["completion"],
        subject=meta.get("subject", ""),
    )


def iter_examples(split) -> list[EmailExample]:
    return [row_to_example(r) for r in split]
