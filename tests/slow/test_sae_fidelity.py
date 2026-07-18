# tests/slow/test_sae_fidelity.py — reconstruction must match published metrics, else SAE untrusted
@pytest.mark.slow
@pytest.mark.gpu
def test_sae_reconstruction_matches_published():
    # variance_explained / L0 / delta_CE within tolerance of the SAE card's numbers on target+backend
    ...