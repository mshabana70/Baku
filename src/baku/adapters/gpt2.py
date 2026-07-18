from .base import ModelAdapter
from .registry import register

@register
class GPT2Adapter(ModelAdapter):
    family = "gpt2"

    def build_prompt(self, instructions: str) -> str:
        return f"User: {instructions}\nAssistant:" # just a synthetic template irght now
    
    def post_instruction_positions(self) -> list[int]:
        return [-1]
    
    def refusal_token_ids(self, tokenizer) -> list[int]:
        return [] # GPT-2 isn't safety-tuned so no meaningful R
    
