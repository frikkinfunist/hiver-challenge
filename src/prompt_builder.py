"""
Builds the final generation prompt from three ingredients:

  1. The target user's PERSONA (role, communication style) pulled straight
     from the dataset.
  2. The current email THREAD the person needs to reply to.
  3. FEW-SHOT examples of that same style of thread -> reply, retrieved via
     SBERT (see retrieval.py). These act as lightweight, retrieval-grounded
     "style transfer" examples so the model imitates real phrasing instead
     of generating a generic corporate reply.
"""

from __future__ import annotations

SYSTEM_TEMPLATE = """You are drafting an email reply on behalf of a specific person. \
Write ONLY the body of their reply -- no subject line, no explanations, \
no meta-commentary. Match the person's tone, formality, sentence length and \
sign-off style as closely as possible.

Person's profile / communication style:
{persona}
"""

FEWSHOT_BLOCK_TEMPLATE = """--- Example {i}: a similar thread this person (or someone with a similar style) replied to ---
Thread:
{thread_text}

Their reply:
{completion}
"""

USER_TEMPLATE = """Here is the new email thread that needs a reply:

{thread_text}

Write this person's reply now (body text only):"""


def build_prompt(
    persona: str,
    thread_text: str,
    fewshot_examples: list[dict] | None = None,
) -> list[dict]:
    """Returns a chat-style message list: [{"role", "content"}, ...]."""
    system_msg = SYSTEM_TEMPLATE.format(persona=persona or "No persona info available.")

    fewshot_text = ""
    if fewshot_examples:
        blocks = [
            FEWSHOT_BLOCK_TEMPLATE.format(
                i=i + 1,
                thread_text=ex["thread_text"],
                completion=ex["completion"],
            )
            for i, ex in enumerate(fewshot_examples)
        ]
        fewshot_text = "\n".join(blocks) + "\n"

    user_msg = fewshot_text + USER_TEMPLATE.format(thread_text=thread_text)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
