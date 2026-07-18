from __future__ import annotations
from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    family: str # class attr => dispatch tag ("gemma2", "gpt2")

    @abstractmethod
    def build_prompt(self, instructions: str) -> str:
        """
        Wrap a bare instruction in the family's chat template.
        """

    @abstractmethod
    def post_instruction_positions(self) -> list[int]:
        """
        Negative-indexed token positions whers the refusal direction is predominantly computed 
        (after the instructions, at the model-turn header or special-token). Negative so they're
        counted from the END. This makes it robust to prompts of different lengths.
        """

    @abstractmethod
    def refusal_token_ids(self, tokenizer) -> list[int]:
        """
        The model family's refusal-token set R (used by a cheap logit refusal_metric).
        Empty for non-safety-tuned models.
        """

    @abstractmethod
    def attn_implementation(self) -> str:
        return "sdpa" # a trustworthy default
        # families with softcap override + forbid FA2 (since it is hardware specific to CUDA)
    