import torch
from transformer_lens import HookedTransformer
from .base import Site, Convention

class TLProbeBackend:
    """
    Phase-0 Capability: load a TransformerLens model and capture ONE site,
    memory-disciplined.
    """
    def __init__(self, model_cfg, device: str = "cuda"):
        load = HookedTransformer.from_pretrained_no_processing if model_cfg.no_processing else HookedTransformer.from_pretrained

        self.model = load(model_cfg.name, dtype=model_cfg.torch_dtype)

        # honor config, but degrade instead of craching CI boxes
        resolved = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARNING] falling back to CPU since CUDA is not available...")
            resolved = "cpu"    # TODO: add log warning here so the Fallback is visible
        self.model.to(resolved)
        self.model.eval()
        self.convention = Convention(
            backend=model_cfg.backend,
            processing="raw" if model_cfg.no_processing else "folded_centered",
            model_revision=model_cfg.revision,
        )

    def site_to_hook(self, site: Site) -> str:
        kind, layer = site
        return {
            "resid_pre": f"blocks.{layer}.hook_resid_pre",
            "resid_mid": f"blocks.{layer}.hook_resid_mid",
            "resid_post": f"blocks.{layer}.hook_resid_post"
        }[kind]

    @torch.inference_mode()
    def capture(self, text: str | list[str], sites: list[Site]) -> dict[Site, torch.Tensor]:
        tokens = self.model.to_tokens(text)     # [batch, seq]
        names = {self.site_to_hook(s): s for s in sites}
        max_layer = max(layer for _, layer in sites)

        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda n: n in names,  # capture ONLY requested sites
            stop_at_layer=max_layer + 1,    # stop computing after the last needed layer
        )

        # will detach and move off-GPU so we don't end up pinning VRAM across the loop
        return {names[n]: cache[n].detach().to("cpu") for n in names}