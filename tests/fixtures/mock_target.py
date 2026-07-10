import torch

class MockProbe:
    """
    Returns deterministic activations: harmful prompts get +direction, harmless get -direction.
    """

    def __init__(self, d_model=16, direction=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.direction = direction if direction is not None else torch.randn(d_model, generator=g)
        self.convention = ("mock", "raw", "text")

    def capture(self, prompts, sites):
        # encode a known label in the prompt (prefix "H:" harmful / "B:" benign)
        signs = torch.tensor([1.0 if p.startswith("H:") else -1.0 for p in prompts])
        base = torch.randn(len(prompts), 4, self.direction.numel())
        base += signs[:, None, None] * self.direction   # inject separable signal
        return {sites[0]: base}