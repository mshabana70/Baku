from ..probes.tl_backend import TLProbeBackend

class TLTarget:
    """
    White-box-capable target: owns the backend (and later a ModelAdapter).
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.probe = TLProbeBackend(cfg.model, device=cfg.device)

    @property
    def model(self):
        return self.probe.model

    def generate(self, text, **gen):
        return self.model.generate(text, **gen) # this is going to be used later