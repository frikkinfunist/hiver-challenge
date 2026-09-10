# Humanual-Email Reply Generator

An LLM system that drafts email replies in a specific person's voice, built
on the [`snap-stanford/humanual-email`](https://huggingface.co/datasets/snap-stanford/humanual-email)
dataset (7,043 real Enron-corpus replies from 399 users, each with a
persona description and full thread context).

## How replies are generated (the actual strategy)

This is a **mix of retrieval-augmented few-shot prompting, with optional
fine-tuning** — not a single technique. Each piece pulls its own weight:

| Component | Role |
|---|---|
| **Persona conditioning** | Every row in the dataset ships a `persona` field describing that user's role and communication style. We inject it verbatim as a system instruction so the model knows *whose* voice to write in. |
| **Retrieval (RAG) via SBERT** | We embed every `(thread → reply)` pair in the **train** split with an SBERT model (`all-mpnet-base-v2`) and index it with FAISS. At inference time we embed the *new* thread and retrieve the top-k most semantically similar past threads — preferring ones from the **same user** first, falling back to the global pool. This grounds the model in real phrasing/sign-offs instead of a generic "Best regards" reply. |
| **Few-shot prompting** | The retrieved `(thread, reply)` pairs are inserted into the prompt as worked examples, followed by the new thread to reply to. This is the default, zero-training path — swap in any instruction-tuned LLM and it works out of the box. |
| **Fine-tuning (optional, `src/finetune_lora.py`)** | For a stronger, cheaper-at-inference baseline, we also provide a LoRA script that fine-tunes a small open LLM directly on `(persona + thread) → completion` pairs from the train split. The fine-tuned adapter can still be combined with the retrieval/few-shot prompt at inference (`generation.backend: lora`) — fine-tuning and RAG are complementary here, not either/or. |

So, concretely, generating one reply looks like:

```
new thread  ─┬─> SBERT embed ─> FAISS search (same-user first) ─> top-k similar (thread, reply) pairs
             │
persona     ─┴─────────────────────────────────────────────────┐
                                                                  v
                                              prompt = persona + few-shot examples + new thread
                                                                  |
                                                                  v
                                    LLM (local HF model / OpenAI API / LoRA fine-tuned model)
                                                                  |
                                                                  v
                                                          generated reply
```

## Multiple suggested replies (3-5 per email)

Instead of one reply, `generator.suggest_replies(example, n=4)` returns
several distinct options — the same way Gmail's "Smart Reply" gives you a
few chips to pick from, but each one is deliberately written from a
different angle instead of being a random resample of the same prompt:

1. Each suggestion pairs the dataset's `persona` field with a **style
   variant** from `src/persona_variants.py` — e.g. *as-is* (their normal
   default), *concise & direct*, *warm & relational*, *formal & detailed*,
   *action-oriented*. All suggestions still use the same SBERT-retrieved
   few-shot examples, so they stay grounded in how this person actually
   writes — they just lean the tone/length differently per suggestion.
2. After each suggestion is generated, it's embedded with SBERT and
   compared against the ones already accepted. If cosine similarity is
   above `multi_reply.dedup_threshold` (default `0.93`), it's treated as a
   near-duplicate and one more style variant is drawn to replace it, so you
   don't end up with 4 suggestions that all say the same thing.

```bash
python -m src.generate --config configs/config.yaml --n 2 --suggestions 4
```

```python
from src.generate import ReplyGenerator, load_config
from src.retrieval import SBERTRetriever

cfg = load_config()
retriever = SBERTRetriever(cfg["sbert"]["model_name"], cfg["sbert"]["device"])
retriever.load(cfg["retrieval"]["index_path"], cfg["retrieval"]["meta_path"])

generator = ReplyGenerator(cfg, retriever)
suggestions = generator.suggest_replies(example, n=4)
# -> [{"style": "as_is", "reply": "...", "possible_duplicate": False}, ...]
```

Number of suggestions and the dedup threshold are configurable under
`multi_reply` in `configs/config.yaml`.

## How accuracy is measured

There's no single "correct" reply for an open-ended email — two replies
can say the same thing in completely different words. So instead of exact
match / BLEU as the headline metric, we score **semantic similarity with
SBERT**:

1. Embed the generated reply and the ground-truth `completion` with the
   same SBERT model used for retrieval.
2. Compute cosine similarity between the two embeddings.
3. Average over the test set → **mean SBERT cosine similarity**, reported
   as the primary "accuracy" number (`src/evaluate.py`).

ROUGE-L is also reported alongside it as a secondary, lexical-overlap
sanity check, but SBERT cosine similarity is the metric that actually
reflects "did the model understand what to say," which is the goal here.

## Repo layout

```
configs/config.yaml       # all knobs: models, retrieval top_k, generation backend, etc.
src/data_utils.py         # load + normalize the HF dataset into EmailExample objects
src/retrieval.py          # build/query the SBERT + FAISS retrieval index (RAG)
src/prompt_builder.py     # assembles persona + few-shot + thread into the final prompt
src/generate.py           # runs the pipeline end-to-end; single reply or multi-suggestion
src/persona_variants.py   # style variants used to generate 3-5 distinct suggestions
src/secrets_utils.py      # loads API keys from env var, falling back to a local .txt file
src/finetune_lora.py      # optional LoRA fine-tuning on the train split
src/evaluate.py           # SBERT-cosine + ROUGE-L evaluation on the test split
scripts/run_pipeline.sh   # build index + generate a few sample replies
scripts/run_eval.sh       # run evaluation over N test examples
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**API key** (for `generation.backend: gemini`, the default, or `openai`) —
pick whichever of these two is more convenient:

- **Environment variable:**
  ```bash
  export GEMINI_API_KEY=your-key-here      # or OPENAI_API_KEY
  ```
- **Plaintext file** (simpler for local/personal use): copy
  `gemini_api_key.txt.example` → `gemini_api_key.txt` (or
  `openai_api_key.txt.example` → `openai_api_key.txt`) and paste your key
  inside. These filenames are already in `.gitignore`, so a normal
  `git add -A && git commit` will never pick them up.
  > These files hold a live secret in plaintext on disk. Fine for a local
  > dev machine; don't use this method on a shared/public machine, and
  > don't zip up the repo folder to send to someone without deleting the
  > file first.

Either way works — the code checks the environment variable first, then
falls back to the matching `.txt` file.

## Usage

1. **Build the retrieval index** (embeds the train split once, caches to disk):
   ```bash
   python -m src.retrieval --config configs/config.yaml
   ```

2. **Generate replies** for a few test examples:
   ```bash
   python -m src.generate --config configs/config.yaml --n 5
   # add --no-retrieval to test pure zero-shot prompting for comparison
   ```

3. **(Optional) Fine-tune a LoRA adapter** on the train split:
   ```bash
   python -m src.finetune_lora --config configs/config.yaml
   # then set generation.backend: lora in configs/config.yaml
   ```

4. **Evaluate**:
   ```bash
   python -m src.evaluate --config configs/config.yaml --n 100
   ```
   Writes `outputs/eval_results.json` with per-example generations, the
   mean/median SBERT cosine similarity, and mean ROUGE-L.

## Switching the generation backend

Set `generation.backend` in `configs/config.yaml`:

- `gemini` (default) — Google's Gemini API (`generation.gemini_model_name`, needs `GEMINI_API_KEY`) — easiest to get running, no gating, has a free tier
- `hf` — any local Hugging Face instruction-tuned model (`generation.hf_model_name`)
- `openai` — an OpenAI chat model via API (`generation.openai_model_name`, needs `OPENAI_API_KEY`)
- `lora` — the base HF model + your fine-tuned LoRA adapter (`generation.lora_adapter_path`)

Retrieval, persona-conditioning, and evaluation are identical across all
four — only the drafting model changes.

## Notes / limitations

- The dataset is derived from the Enron email corpus (business-domain,
  English, released `CC BY-NC 4.0`) — style and evaluation numbers won't
  necessarily transfer to, say, casual personal email.
- `user_id` is a SHA-256 hash of the original address; there's no PII to
  handle beyond what's already in the public dataset.
- Retrieval is scoped to the **train** split only, so evaluation on
  val/test never leaks the ground-truth reply being scored.
