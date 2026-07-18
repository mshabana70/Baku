import pytest
from baku.adapters.registry import get_adapter
from baku.adapters.gemma import GemmaAdapter
from baku.adapters.gpt2 import GPT2Adapter

def test_registry_dispatches_on_family_not_name():
    # the registry is keyed on the stable family tag (config.model.family), NOT the HF id (config.model.name).
    assert isinstance(get_adapter("gemma2"), GemmaAdapter)
    # distinct HF snapshots (different `name`) sharing a family resolve to the same adapter type
    assert type(get_adapter("gemma2")) is type(get_adapter("gemma2"))
    # an unknown family fails loud (never silently mis-dispatches a new target)
    with pytest.raises(KeyError):
        get_adapter("not-a-registered-family")

def test_post_instruction_positions_are_negative():
    pos = GemmaAdapter().post_instruction_positions()
    assert all(p < 0 for p in pos)                      # negative-indexed from end

@pytest.mark.parametrize("adapter", [GPT2Adapter(), GemmaAdapter()])
def test_synthetic_template_runs_on_gpt2(adapter):
    out = adapter.build_prompt("PLACEHOLDER_INSTRUCTION")
    assert isinstance(out, str)
    assert "PLACEHOLDER_INSTRUCTION" in out

def test_adapter_forbids_fa2_on_gemma():
    with pytest.raises(ValueError):
        GemmaAdapter().assert_attn_compatible("flash_attention_2")

def test_adapter_accepts_valid_atten_on_gemma():
    assert GemmaAdapter().assert_attn_compatible("sdpa") is not ValueError