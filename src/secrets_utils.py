"""
API key loading, in order of preference:

  1. An already-set environment variable (e.g. GEMINI_API_KEY).
  2. A plaintext file at the repo root containing just the key
     (e.g. gemini_api_key.txt).

The plaintext-file option exists purely for local convenience -- paste your
key into the file once, and every script picks it up automatically. These
files are listed in .gitignore, so `git add -A` / `git push` will NOT pick
them up as long as you don't force-add them. Still, treat them like any
other secret: don't screenshot them, don't paste their contents into a
chat, and don't zip/share the repo folder without deleting them first.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_api_key(env_var: str, filename: str) -> str | None:
    """Return the API key, or None if it isn't set anywhere."""
    value = os.environ.get(env_var)
    if value:
        return value.strip()

    key_path = REPO_ROOT / filename
    if key_path.exists():
        value = key_path.read_text(encoding="utf-8").strip()
        if value and not value.startswith("#"):
            os.environ[env_var] = value  # cache for this process
            return value

    return None
