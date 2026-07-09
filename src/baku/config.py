from __future__ import annotations
from pydantic import BaseModel, Field
import yaml, torch

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16
}

class ModelConfig(BaseModel):
    name: str = "gpt2"
    family: str = "gpt2"
    backend: str = "hooked_transformer"
    dtype: str = "float32"
    revision: str = "main"
    no_processing: bool = True
    attn_implementation: str = "eager" # don't want to use flash_attention_2 for Gemma-2

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

class RunConfig(BaseModel):
    tier: str = "local" # local or cloud TODO: might add NDIF here.
    device: str = "cuda"
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    artifacts_dir: str = "runs"
    # search / fitness / judge sub-configs get added in their pieces

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
    
    
