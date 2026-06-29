# Baku — Testing Guide

Test-driven: for each piece, write the failing test(s) **first**, watch them fail for the right reason, then implement to
green. See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the build order and [`ARCHITECTURE.md`](ARCHITECTURE.md) for contracts.

> You write the tests by hand too. Reference snippets here are to study; adapt them.

---

## 1. Strategy — a fast pyramid with three speeds

| Tier | Marker | Target | Runs | Purpose |
|---|---|---|---|---|
| **Unit** | (none) | `MockTarget` / tiny tensors | every save, < 1s each | logic: shapes, math, aggregation, stats, registry |
| **Smoke** | `@pytest.mark.smoke` | **GPT-2** (`no_processing`) | seconds, CPU/GPU | plumbing: hooks fire, capture shapes, synthetic-template extraction |
| **Slow / GPU** | `@pytest.mark.slow`, `@pytest.mark.gpu` | **Gemma-2-2B-IT** | minutes, GPU | real signals: bimodal projection, ablation bypasses refusal, end-to-end attack |

CI default runs unit + smoke (`pytest -m "not slow"`). The slow/GPU suite runs on the dev box / pod before a milestone.

**The mock target** is the workhorse — deterministic activations let you test every downstream component without a model:

```python
# tests/fixtures/mock_target.py
import torch
class MockProbe:
    """Returns deterministic activations: harmful prompts get +direction, harmless get -direction."""
    def __init__(self, d_model=16, direction=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.direction = direction if direction is not None else torch.randn(d_model, generator=g)
        self.convention = ("mock", "raw", "test")
    def capture(self, prompts, sites):
        # encode a known label in the prompt (e.g. prefix "H:" harmful / "B:" benign)
        signs = torch.tensor([1.0 if p.startswith("H:") else -1.0 for p in prompts])
        base = torch.randn(len(prompts), 4, self.direction.numel())
        base += signs[:, None, None] * self.direction            # inject separable signal
        return {sites[0]: base}
```

---

## 2. TDD path — first failing tests, in order

### Piece 0 — capture plumbing & config
```python
# tests/smoke/test_probe_capture.py
import pytest, torch
from baku.config import RunConfig
from baku.targets.tl_target import TLTarget

@pytest.mark.smoke
def test_capture_returns_one_site_shape():
    cfg = RunConfig(model={"name": "gpt2", "backend": "hooked_transformer", "no_processing": True})
    t = TLTarget(cfg)
    out = t.probe.capture(["hello world", "a b c d"], sites=[("resid_post", 0)])
    act = out[("resid_post", 0)]
    assert act.shape[0] == 2 and act.shape[-1] == t.model.cfg.d_model
    assert act.device.type == "cpu"                      # detached off-GPU (memory discipline)

@pytest.mark.smoke
def test_capture_only_requested_site():
    # requesting layer 0 must not populate later-layer caches (names_filter + stop_at_layer)
    ...

def test_seed_determinism():
    from baku.seeding import seed_everything
    seed_everything(0); a = torch.randn(3)
    seed_everything(0); b = torch.randn(3)
    assert torch.equal(a, b)
```
Assert: shape `[batch, seq, d_model]`, CPU device, only the requested site present, determinism.

### Piece 0/1 — the TWO fidelity tests
```python
# tests/slow/test_backend_fidelity.py
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
```

The **SAE-activation correctness** test (a hard gate; harness now, active when SAEs are added — Piece 12):
```python
# tests/slow/test_sae_fidelity.py — reconstruction must match published metrics, else SAE untrusted
@pytest.mark.slow
@pytest.mark.gpu
def test_sae_reconstruction_matches_published():
    # variance_explained / L0 / delta_CE within tolerance of the SAE card's numbers on target+backend
    ...
```

### Piece 1 — ModelAdapter
```python
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
    pos = GemmaAdapter().post_instruction_positions()
    assert all(p < 0 for p in pos)                      # negative-indexed from end
def test_synthetic_template_runs_on_gpt2():
    # gpt2 has no chat template -> adapter (family "gpt2") injects a synthetic one; extraction plumbing must still run
    ...
```

### Piece 2 — DiffInMeans (+ the interp sanity check)
```python
# tests/unit/test_diff_in_means.py  — math on the mock
def test_diff_of_means_recovers_known_direction():
    from baku.directions.diff_in_means import diff_in_means
    # with MockProbe injecting +d for harmful, -d for harmless, recovered r should align with d
    r = diff_in_means(...)[(0, -1)]
    assert torch.nn.functional.cosine_similarity(r, mock_direction, dim=0) > 0.9
```
```python
# tests/slow/test_projection_signal.py  — THE interpretability validation the brief asks for
@pytest.mark.slow
@pytest.mark.gpu
def test_projection_separates_refuse_from_comply():
    # on held-out Gemma prompts: projection r̂·x is HIGHER for harmful (refused) than harmless (complied),
    # and the histogram is BIMODAL. This is "does the signal behave correctly?"
    assert harmful_proj.mean() > harmless_proj.mean()
    assert bimodality_coefficient(all_proj) > 0.55
```

### Piece 3 — interventions (causal sanity)
```python
@pytest.mark.slow
@pytest.mark.gpu
def test_ablation_bypasses_refusal_and_keeps_coherence():
    assert refusal_metric_after_ablation < refusal_metric_before     # bypass works
    assert kl_on_harmless < 0.1                                      # coherence guard holds
def test_addition_induces_refusal_on_harmless():
    assert refusal_metric_after_addition > refusal_metric_before
```

### Piece 6 — Statistics (against known distributions)
```python
# tests/unit/test_statistics.py
import numpy as np
def test_bca_ci_brackets_true_mean():
    from baku.scorers.statistics import asr_bootstrap_ci
    rng = np.random.default_rng(0)
    succ = (rng.random(200) < 0.3).astype(int)          # true ASR ≈ 0.3
    mean, (lo, hi) = asr_bootstrap_ci(succ)
    assert lo < 0.3 < hi and lo < mean < hi
def test_paired_delta_detects_real_improvement():
    # arm A strictly dominates arm B per-behavior -> delta CI excludes 0
    ...
def test_kappa_reports_base_rate():
    # at an extreme refusal base rate, surface kappa AND base rate (kappa paradox)
    ...
```

### Piece 7 — Calibration
```python
def test_spearman_and_precision_at_k_on_synthetic_pairs():
    # construct fitness/judge pairs with a known monotone relationship -> Spearman ~ 1, precision@K high;
    # construct an anti-correlated set -> Spearman < 0 (the bridge must catch a bad layer/position)
    ...
```

### Piece 8 — integration (fast + slow)
```python
@pytest.mark.smoke
def test_genetic_loop_runs_and_improves_on_mock():
    # GeneticSearch over MockProbe fitness: best-fitness at last gen >= first gen; returns AttackResult
    ...
@pytest.mark.slow
@pytest.mark.gpu
def test_end_to_end_attack_produces_report():
    # Gemma-2-2B-IT: loop -> top-K -> EnsembleScorer -> AttackResult with BCa CI + a MechanisticReport.
    # Phase-1 report = projection trajectory + DLA only. Causal confirmation is Piece 11 / Phase 1.5,
    # and must NOT be required for Phase-1 done.
    assert result.asr_ci is not None
    assert report.projection_trajectory is not None and report.dla is not None
    assert report.causal_confirmation is None        # Phase 1.5 — explicitly not yet
```

### Piece 9 — the headline ablation
```python
# tests/slow/test_ablation_harness.py
@pytest.mark.slow
@pytest.mark.gpu
def test_ablation_harness_returns_delta_ci():
    # paired bootstrap returns a delta-ASR CI across seeds (sign is the result, not asserted)
    assert result.delta_asr_ci is not None

def test_fitness_and_success_judges_are_disjoint():
    # the black-box arm's fitness judge must NOT be the success judge (no training on the eval)
    assert cfg.arms["blackbox"]["fitness_judge"] != cfg.success_judge

def test_both_budget_modes_present():
    assert {"equal_iterations", "equal_compute"} <= set(cfg.budget_modes)

def test_cost_counters_recorded():
    # each arm logs target forward-passes and wall-clock; judge-calls tracked separately
    for arm in ("whitebox", "blackbox"):
        assert result.cost[arm].target_forward_passes is not None
        assert result.cost[arm].wallclock_s is not None
        assert result.cost[arm].judge_calls is not None
    assert result.cost["whitebox"].judge_calls == 0      # white-box never judges during search
```

---

## 3. Validating the interpretability signals themselves

These are the tests that protect against *plausible-but-wrong*:

1. **Projection sanity** (Piece 2): `r̂·x` higher for refuse than comply, bimodal histogram on held-out prompts.
2. **Causal sufficiency** (Piece 3): ablation bypasses refusal (drops `refusal_metric`); addition induces it.
3. **Coherence guard** (Piece 3/5): KL-on-harmless `< 0.1`; the fitness coherence penalty fires on gibberish.
4. **Calibration gate** (Piece 7): Spearman/precision@K between fitness and judge is adequate *before* trusting
   internal-signal-gated ASR — and a bad `(layer, position)` produces low/negative Spearman (the bridge must catch it).
5. **Convention invariant** (cross-cutting): loading a `RefusalSubspace` whose `Convention` differs from the run **raises**.
6. **Causal confirmation** (Piece 11, *Phase 1.5* — **not** part of the Phase-1 gate; items 1–5 are): random/benign-direction
   controls — a "why it worked" claim must show the *refusal* direction specifically is responsible, since random
   directions also break safety.

---

## 4. Conventions

- One assertion-theme per test; name tests for the behavior, not the function.
- Mark anything that loads a real model `slow`/`gpu`; keep the default suite < ~30s.
- Seed every stochastic test (`seed_everything(0)`); bootstrap tests use a fixed `np.random.default_rng(seed)`.
- Fixtures for the mock target and tiny SAE params live in `tests/fixtures/`.
