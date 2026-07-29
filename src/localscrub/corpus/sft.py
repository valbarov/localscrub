"""SFT dataset builder for the stage-2 LoRA fine-tune recipe.

Each example pairs the *exact* prompt stage 2 sends at inference — built by
running stage 1 on the note, escalation hints included — with the gold
answer serialized in the extractor's reply format. The tuned model therefore
trains on the precise input distribution it will serve, and its outputs
parse with the same salvage-tolerant parser.

Training data is synthetic-only (ROADMAP non-goal: no real PHI, ever) and
should use seeds disjoint from evaluation seeds. Template overlap makes
template-corpus "after" numbers optimistic; the MTSamples benchmark, whose
prose the model never saw, is the honest generalization test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from localscrub.analyzer import Analyzer
from localscrub.config import ScrubConfig
from localscrub.corpus.generator import SyntheticNote
from localscrub.stage2 import extraction_prompt


def build_sft_examples(
    notes: Sequence[SyntheticNote], config: ScrubConfig | None = None
) -> list[dict[str, str]]:
    """``{"id", "prompt", "completion"}`` rows, one per note."""
    analyzer = Analyzer(config=config)
    examples: list[dict[str, str]] = []
    for note in notes:
        result = analyzer.analyze(note.text)
        seen: set[tuple[str, str]] = set()
        entities: list[dict[str, str]] = []
        for span in note.spans:
            key = (span.type.value, span.text)
            if key not in seen:
                seen.add(key)
                entities.append({"type": span.type.value, "text": span.text})
        examples.append(
            {
                "id": note.id,
                "prompt": extraction_prompt(note.text, result.escalations),
                "completion": json.dumps({"entities": entities}, ensure_ascii=False),
            }
        )
    return examples
