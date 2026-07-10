from dataclasses import dataclass
from typing import Literal

Site = tuple[str, int]      # ("resid_post", 0)

@dataclass(frozen=True)
class Convention:
    backend: str
    processing: Literal["raw", "folded_centered"]     # "raw" | "folded_centered", important for residual stream scaling
    model_revision: str