import pytest

@pytest.mark.slow
@pytest.mark.gpu
def test_tl_logits_match_hf(per_backend_tol):
    # backend correctness: TL logits ≈ raw-HF logits on N prompts.
    # legacy HookedTransformer reimplements the forward pass -> LOOSER tol than TransformerBridge.
    assert (tl_logits - hf_logits).abs().max() < per_backend_tol     # e.g. 1e-2 legacy, 1e-4 bridge

@pytest.mark.slow
@pytest.mark.gpu
def test_flash_attention2_diverges_on_gemma2():
    # FA2 drops Gemma-2 soft-cap -> activations diverge. Guards Decision 4.
    # load with attn_implementation="flash_attention_2" and assert it does NOT match eager
    ...

def test_adapter_forbids_fa2_on_gemma():
    from baku.adapters.gemma import GemmaAdapter
    assert GemmaAdapter().attn_implementation in ("eager", "sdpa")