"""NeuroLM prompt and answer handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NeuroLMTaskSpec:
    """Description of one numeric-to-text downstream task.

    ``answers`` is ordered by the numeric class index used by NeuralBench.
    The official NeuroLM checkpoints are trained to answer natural-language
    prompts, so this mapping is part of the evaluation protocol rather than a
    model-head configuration.
    """

    name: str
    prompt: str
    answers: tuple[str, ...]
    max_new_tokens: int = 5
    background_label: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NeuroLMTaskSpec":
        """Load a task specification without importing the NeuroLM source."""
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        config = raw.get("neurolm", raw)
        answers = config.get("answers", config.get("labels"))
        if isinstance(answers, dict):
            answers = [answers[key] for key in sorted(answers, key=lambda value: int(value))]
        if not isinstance(answers, list) or not all(isinstance(x, str) for x in answers):
            raise ValueError(f"{path} must define a list of string answers.")
        return cls(
            name=str(config.get("name", Path(path).stem)),
            prompt=str(config["prompt"]),
            answers=tuple(answers),
            max_new_tokens=int(config.get("max_new_tokens", 5)),
            background_label=config.get("background_label"),
        )

    def answer_for_label(self, label: int) -> str:
        if not 0 <= label < len(self.answers):
            raise ValueError(f"Label {label} is outside task {self.name!r}.")
        return self.answers[label]

    def answer_continuation(self, label: int) -> str:
        """Return the answer suffix exactly as official NeuroLM trains it.

        Official multiple-choice prompts already contain the opening
        parenthesis (``"Answer: ("``), so their target continuation is
        ``"A)"`` rather than a second ``"(A)"``.  Natural-language answers
        such as ``Yes`` and ``No`` follow ``"Answer:"`` with a leading space.
        """
        answer = self.answer_for_label(label)
        if self.prompt.rstrip().endswith("(") and answer.startswith("("):
            return answer[1:]
        return f" {answer}"

    def parse_answer(self, generated_text: str) -> int | None:
        """Parse a generated choice such as ``(A)`` into a class index."""
        candidate = generated_text
        if self.prompt.rstrip().endswith("("):
            candidate = "(" + generated_text.lstrip()
        match = re.search(r"\(([A-Za-z])\)", candidate)
        if match is not None:
            # ``answers`` follows the prepared NeuralBench label order, which
            # need not be alphabetical.  Return that label index rather than
            # the ordinal of the generated option letter.
            choice = f"({match.group(1).upper()})"
            try:
                return self.answers.index(choice)
            except ValueError:
                pass

        normalized = candidate.strip().lower()
        for index, answer in enumerate(self.answers):
            if re.search(
                rf"(?<!\w){re.escape(answer.lower())}(?!\w)",
                normalized,
            ):
                return index
        return None
