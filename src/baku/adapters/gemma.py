from .base import ModelAdapter
from .registry import register

@register
class GemmaAdapter(ModelAdapter):
    family = "gemma2"

    def build_prompt(self, instructions: str) -> str:
        # trailing model header, refusal usually decides here
        return ("<start_of_turn>user\n"
                f"{instructions}<end_of_turn>\n"
                "<start_of_turn>model\n")
    
    def post_instruction_positions(self) -> list[int]:
        return [-5, -4, -3, -2, -1] # last few tokens for model-turn header
    
    def refusal_token_ids(self, tokenizer) -> list[int]:
        words = ["I", "I'm", "Sorry", "As", "cannot", "unable", "struggling"]
        return [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in words]
    
    @property
    def attn_implementation(self) -> str:
        return "eager" # gemma-2 softcap: eager/sdpa only