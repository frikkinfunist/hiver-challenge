"""
Style "persona" variants used to generate several distinct reply
suggestions for the same email, instead of just one.

The dataset's own `persona` field describes *who* is replying (their role,
their usual tone). We keep that as the foundation, but layer a second,
smaller "how to lean for this particular draft" instruction on top of it.
Each variant nudges the same base voice in a different direction — concise
vs. detailed, warm vs. brisk, etc. — so the user gets a genuinely useful
set of options rather than 5 near-identical resamples.

`build_reply_personas()` merges the two into a list of ready-to-use
persona strings, one per suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StyleVariant:
    name: str
    instruction: str


# Order matters: if num_suggestions < len(DEFAULT_STYLE_VARIANTS), we take
# the first N. "as_is" is always first so the top suggestion is always the
# closest match to how this person actually writes.
DEFAULT_STYLE_VARIANTS: list[StyleVariant] = [
    StyleVariant(
        name="as_is",
        instruction="Write this the way you normally would -- no extra "
        "stylistic lean, just your default tone and length.",
    ),
    StyleVariant(
        name="concise_direct",
        instruction="Lean concise and direct for this draft: short "
        "sentences, no pleasantries or hedging, get straight to the point.",
    ),
    StyleVariant(
        name="warm_relational",
        instruction="Lean warmer and more relationship-focused for this "
        "draft: briefly acknowledge the other person/their situation before "
        "getting to the substance, softer phrasing.",
    ),
    StyleVariant(
        name="formal_detailed",
        instruction="Lean more formal and thorough for this draft: fuller "
        "sentences, spell out reasoning and next steps explicitly, "
        "professional register.",
    ),
    StyleVariant(
        name="action_oriented",
        instruction="Lean action-oriented for this draft: lead with the "
        "decision/ask/next step in the first sentence, keep supporting "
        "detail brief and below it.",
    ),
]


def build_reply_personas(
    base_persona: str,
    n: int,
    variants: list[StyleVariant] | None = None,
) -> list[tuple[str, str]]:
    """Return up to n (variant_name, merged_persona_text) pairs.

    merged_persona_text = base persona + this variant's stylistic nudge,
    so the model still knows *whose* voice it's writing in, plus how to
    lean for that particular suggestion.
    """
    pool = variants or DEFAULT_STYLE_VARIANTS
    n = max(1, min(n, len(pool)))
    chosen = pool[:n]

    merged = []
    for v in chosen:
        text = (
            f"{base_persona or 'No persona info available.'}\n\n"
            f"Style guidance for this specific draft: {v.instruction}"
        )
        merged.append((v.name, text))
    return merged
