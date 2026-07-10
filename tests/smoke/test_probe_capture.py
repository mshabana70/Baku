import pytest, torch
from baku.config import RunConfig
from baku.targets.tl_target import TLTarget

@pytest.mark.smoke
def test_capture_returns_one_site_shape():
    cfg = RunConfig(model={
        "name": "gpt2",
        "backend": "hooked_transformer",
        "no_processing": True
    })
    t = TLTarget(cfg)
    out = t.probe.capture(["hello world", "a b c d"], sites=[("resid_post", 0)])
    act = out[("resid_post", 0)]
    assert act.shape[0] == 2 and act.shape[-1] == t.model.cfg.d_model
    assert act.device.type == "cpu"  # memory discipline test

@pytest.mark.smoke
def test_capture_only_requested_site():
    # requesting layer 0 must not populate later-layer caches (names_filter + stop_at_layer)
    cfg = RunConfig(model={"name": "gpt2"})
    t = TLTarget(cfg)
    out = t.probe.capture(["hello world"], sites=[("resid_post", 0)])
    assert set(out.keys()) == {("resid_post", 0)} # names_filter kept ONLY the requested site

def test_seed_determinism():
    from baku.seeding import seed_everything
    seed_everything(0); a = torch.randn(3)
    seed_everything(0); b = torch.randn(3)
    assert torch.equal(a, b)