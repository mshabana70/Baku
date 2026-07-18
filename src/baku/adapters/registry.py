from .base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}

def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    _REGISTRY[cls.family] = cls
    return cls # returns the class unchanged

def get_adapter(family: str) -> ModelAdapter:
    try:
        return _REGISTRY[family]()
    except KeyError:
        raise KeyError(
            f"No adapter for family={family!r}. Known: {sorted(_REGISTRY)}"
        ) # unknown families fail loud here

