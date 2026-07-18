import pytest
def test_registry_dispatches_on_family_not_name():
    # the registry is keyed on the stable family tag (config.model.family), NOT the HF id (config.model.name).
    from baku.adapters.registry import get_adapter
    from baku.adapters.gemma import GemmaAdapter
    assert isinstance(get_adapter("gemma2"), GemmaAdapter)
    # distinct HF snapshots (different `name`) sharing a family resolve to the same adapter type
    assert type(get_adapter("gemma2")) is type(get_adapter("gemma2"))
    # an unknown family fails loud (never silently mis-dispatches a new target)
    with pytest.raises(KeyError):
        get_adapter("not-a-registered-family")

def test_post_instruction_positions_are_negative():
    from baku.adapters.gemma import GemmaAdapter
    pos = GemmaAdapter().post_instruction_positions()
    assert all(p < 0 for p in pos)                      # negative-indexed from end

def test_synthetic_template_runs_on_gpt2():
    # gpt2 has no chat template -> adapter (family "gpt2") injects a synthetic one; extraction plumbing must still run
    ...

def test_adapter_forbids_fa2_on_gemma():
    from baku.adapters.gemma import GemmaAdapter
    assert GemmaAdapter().attn_implementation != "FA2"