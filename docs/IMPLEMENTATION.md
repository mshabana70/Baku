# Baku — Implementation Guide

How to build Baku one component at a time. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first for the *why*; this doc is the
*how* and the order. Test strategy lives in [`TESTING.md`](TESTING.md); the teaching-dial state lives in
[`CONCEPTS_COVERED.md`](CONCEPTS_COVERED.md).

> **You write all `src/baku/`, `tests/`, and `configs/` code by hand.** The code blocks here are *reference to study and
> reimplement*. Per the teaching dial: **NEW** concepts get full reference code; **REPEAT** concepts get interface + hints
> only, and you attempt the body cold before review. The marker on each piece tells you which.

---

## Phasing

| Phase | Goal | Pieces |
|---|---|---|
| **Phase 1** (minimal, demoable) | A working white-box refusal-direction-steered genetic attack on Gemma-2-2B-IT with a behavioral ASR (BCa CI), a basic mechanistic report, and the white-box-vs-black-box ablation. | 0–10 |
| **Phase 1.5** | Heavy attribution in the report. | 11 |
| **Phase 2 / cloud** | Subspace mode, gradient attacks, SAE fitness, QD, scaling backends, defense-aware. | 12 |

---

## Environment & reproducibility (do this with Piece 0)

- **uv + `pyproject.toml` + `uv.lock`.** Pin everything; commit the lock. Reproduce on a pod with `uv sync --frozen`.
- **Pin the beta/churn-prone deps explicitly:** `transformer-lens` (legacy + bridge), `sae-lens`, `nnsight`, `torch`/CUDA.
- **`Dockerfile`** on a CUDA base for RunPod; treat pods as ephemeral → persist `RunRecord`s + artifacts to a network volume.
- Reference `pyproject` dependency core (versions are illustrative — resolve current ones with `uv`):

```toml
[project]
name = "baku"
requires-python = ">=3.11"
dependencies = [
  "torch",                    # CUDA build via the appropriate index
  "transformer-lens>=2",      # add >=3.0.0b0 only when you need the bridge / Gemma-3
  "sae-lens",                 # provides HookedSAETransformer + SAE.from_pretrained
  "transformers", "accelerate", "datasets",
  "pydantic>=2", "pyyaml",
  "numpy", "scipy",           # scipy.stats.bootstrap for BCa CIs
  "nltk",                     # BLEU novelty (or sacrebleu)
]
[project.optional-dependencies]
dev = ["pytest", "pytest-xdist", "ruff", "mypy"]
```

---

## Ordered build checklist (the guided-build loop)

| # | Piece | Dial | Phase |
|---|---|---|---|
| 0 | Scaffold + config + smoke `Target`/`Probe` | **NEW → full** | 1 |
| 1 | `ModelAdapter` / family registry | **NEW → full** | 1 |
| 2 | `DiffInMeans` + `RefusalSubspace` | **NEW → full** | 1 |
| 3 | Three interventions (ablation / addition / orthogonalization) | **NEW → full** | 1 |
| 4 | Direction selection + calibration cache | NEW (selection) / REPEAT (projection) | 1 |
| 5 | `FitnessFn` protocol (`RefusalProjectionFitness` + coherence) | **REPEAT → interface+hints** | 1 |
| 6 | `Scorer` / `EnsembleScorer` + `Statistics` | **NEW → full** | 1 |
| 7 | `Calibration` bridge (Spearman + precision@K) | NEW | 1 |
| 8 | `GeneticSearch` Orchestrator | **NEW → full** | 1 |
| 9 | Headline ablation harness (white-box vs black-box fitness) | REPEAT → interface+hints | 1 |
| 10 | `Reporter` (basic): projection trajectory + DLA + redaction | NEW | 1 |
| 11 | `Reporter` (heavy attribution) | NEW | 1.5 |
| 12 | Cloud/later (subspace, GCG, SAE fitness, QD, scaling) | later | 2 |

---

## Piece 0 — Scaffold + config + smoke `Target`/`Probe`  ·  *NEW → full reference*

**Goal:** prove the toolchain and the two non-negotiables — *config-driven device/dtype/backend* and *memory-disciplined
hooks* — before any interpretability math. Load **GPT-2** (it doesn't refuse; that's fine here — we're testing plumbing),
capture **one** residual site at **one** layer via a single hook, and assert the captured tensor's shape.

**Why these choices:**
- **`from_pretrained_no_processing`** loads raw weights (no LN folding / weight centering). We adopt the SAE-correct
  convention from the very first line so later `RefusalSubspace` artifacts live in the right activation space.
- **`names_filter` + `stop_at_layer`** make `run_with_cache` capture *only* the site you asked for and *stop computing*
  after that layer — this is the difference between an 8GB peak and an OOM on the 16GB tier.
- **detach → CPU** in the capture path so activations don't pin VRAM across the search loop.

### 0a. Config (tier-agnostic)

```python
# src/baku/config.py
from __future__ import annotations
from pydantic import BaseModel, Field
import yaml, torch

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

class ModelConfig(BaseModel):
    name: str = "gpt2"                      # HF id understood by the backend
    backend: str = "hooked_transformer"    # see Backend & Convention Policy
    dtype: str = "float32"                  # bf16 for Gemma; gpt2 smoke is fine in fp32
    revision: str = "main"                  # pins reproducibility (goes into the Convention tag)
    no_processing: bool = True             # raw-activation convention
    attn_implementation: str = "eager"    # never flash_attention_2 for Gemma-2

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

class RunConfig(BaseModel):
    tier: str = "local"                     # "local" | "cloud" — never branch on this in algorithm code
    device: str = "cuda"
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    artifacts_dir: str = "runs"
    # search / fitness / judge sub-configs get added in their pieces

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
```

```python
# src/baku/seeding.py
import random, numpy as np, torch
def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

### 0b. Smoke Target + a minimal ProbeBackend (capture only)

```python
# src/baku/probes/base.py    (interfaces — see ARCHITECTURE §3.1 for the full ABC)
from dataclasses import dataclass
Site = tuple[str, int]                       # ("resid_post", 0)

@dataclass(frozen=True)
class Convention:
    backend: str
    processing: str          # "raw" | "folded_centered"
    model_revision: str
```

```python
# src/baku/probes/tl_backend.py
import torch
from transformer_lens import HookedTransformer
from .base import Site, Convention

class TLProbeBackend:
    """Phase-0 capability: load a TransformerLens model and capture ONE site, memory-disciplined."""
    def __init__(self, model_cfg):
        load = HookedTransformer.from_pretrained_no_processing if model_cfg.no_processing \
            else HookedTransformer.from_pretrained
        self.model = load(model_cfg.name, dtype=model_cfg.torch_dtype)   # device handled below
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.convention = Convention(
            backend="hooked_transformer",
            processing="raw" if model_cfg.no_processing else "folded_centered",
            model_revision=model_cfg.revision,
        )

    def site_to_hook(self, site: Site) -> str:
        kind, layer = site
        return {"resid_pre": f"blocks.{layer}.hook_resid_pre",
                "resid_mid": f"blocks.{layer}.hook_resid_mid",
                "resid_post": f"blocks.{layer}.hook_resid_post"}[kind]

    @torch.inference_mode()
    def capture(self, text: str | list[str], sites: list[Site]) -> dict[Site, torch.Tensor]:
        tokens = self.model.to_tokens(text)                         # [batch, seq]
        names = {self.site_to_hook(s): s for s in sites}
        max_layer = max(layer for _, layer in sites)
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda n: n in names,                      # capture ONLY requested sites
            stop_at_layer=max_layer + 1,                            # stop computing after the last needed layer
        )
        # detach + move off-GPU so we don't pin VRAM across the loop
        return {names[n]: cache[n].detach().to("cpu") for n in names}
```

```python
# src/baku/targets/tl_target.py
from ..probes.tl_backend import TLProbeBackend

class TLTarget:
    """White-box-capable target: owns the backend (and later a ModelAdapter)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.probe = TLProbeBackend(cfg.model)

    @property
    def model(self):
        return self.probe.model

    def generate(self, text, **gen):                # used later; fine to stub for piece 0
        return self.model.generate(text, **gen)
```

**What to verify (see TESTING.md):** `capture(("resid_post", 0))` on a batch of 2 prompts returns shape
`[2, seq, d_model]` on CPU; requesting layer 0 does *not* run later layers (spot-check with a timing or a
`stop_at_layer` assertion); seeding makes two runs identical.

---

## Piece 1 — `ModelAdapter` / family registry  ·  *NEW → full reference*

**Goal:** isolate *all* family-specific knowledge so the rest of the system is model-agnostic.

**Responsibilities & interface:** see ARCHITECTURE §3.1. Key methods: `apply_chat_template`, `post_instruction_positions`
(negative-indexed), `site_to_hook`, `refusal_token_set` (`R`), `orthogonalization_matrices`, plus `attn_implementation`.

**Library notes / gotchas:**
- **Chat templates (post-instruction region).** Gemma-2-IT: `<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n`.
  GPT-2 has **no** template → inject a *synthetic* one so the extraction plumbing still runs (you'll validate attack
  success only on Gemma).
- **Positions are negative-indexed** from the end (prompts have variable length). Arditi's selected positions are in
  `{-1, -2, -5}`; build candidates over the last ~5 post-template tokens.
- **`refusal_token_set` `R`** is the family's set of refusal-initiating first tokens (e.g. the id of "I") — used by the
  cheap logit `refusal_metric` in Piece 4.
- **`attn_implementation="eager"` for Gemma-2** on every HF-backed path; assert it (FA2 silently drops the soft-cap).

```python
# src/baku/adapters/base.py  (sketch — implement against ARCHITECTURE §3.1)
class ModelAdapter(ABC):
    attn_implementation: str = "eager"
    def apply_chat_template(self, instruction, system=None) -> str: ...
    def post_instruction_positions(self) -> list[int]: ...     # e.g. [-1, -2, -3, -4, -5]
    def refusal_token_set(self) -> list[int]: ...
    # registry: adapters/registry.py maps config.model.name -> ModelAdapter subclass
```

---

## Piece 2 — `DiffInMeans` + `RefusalSubspace`  ·  *NEW → full reference* (the interpretability heart)

**The method (difference-in-means).** For harmful prompts `D+` and harmless `D-`, at layer `l` and post-instruction
position `i`, the candidate refusal vector is `r = mean_{D+}(x_i^l) − mean_{D-}(x_i^l)`. Its unit `r̂ = r/‖r‖` is the
steering direction; the **projection scalar `r̂·x` separates the two clusters** (bimodal histogram) — that's the signal.

**Practical:** raw residual stream (never post-LayerNorm), 128 harmful + 128 harmless, sweep all layers except the last
~20%, pre-filter the prompt sets by the refusal metric so the contrast is clean. No gradients — pure forward passes.

```python
# src/baku/directions/diff_in_means.py  (reference)
import torch
from .subspace import RefusalSubspace

@torch.inference_mode()
def diff_in_means(probe, adapter, harmful: list[str], harmless: list[str],
                  layers: list[int], positions: list[int]) -> dict[tuple[int, int], torch.Tensor]:
    """Returns candidate raw direction r per (layer, position). Batch internally for memory."""
    def mean_acts(prompts):
        # sum activations at each (layer, position) over prompts, then divide
        ...   # capture resid_post at `layers`; index `positions` (negative); accumulate
    mu = mean_acts([adapter.apply_chat_template(p) for p in harmful])
    nu = mean_acts([adapter.apply_chat_template(p) for p in harmless])
    return {(l, i): mu[(l, i)] - nu[(l, i)] for l in layers for i in positions}
```

`RefusalSubspace` is the serializable dataclass from ARCHITECTURE §3.1 — store `r`, `r̂`, the `basis` (`k=1` now), the
`Convention` tag, and diagnostics. **Visualize the bimodal projection histogram** — that's your first real interp result.

---

## Piece 3 — Three interventions sharing one direction  ·  *NEW → full reference*

All three reuse the *same* `RefusalSubspace`:

- **Directional ablation (bypass refusal):** `x' = x − r̂ (r̂ᵀx)` applied at **every** layer & position and to **both**
  the post-attention and post-MLP residual writes. Use the **unit** `r̂`. In TL, hook `blocks.{l}.hook_resid_pre`,
  `hook_resid_mid`, `hook_resid_post` (+ embedding).
- **Activation addition (induce refusal):** `x += r` at **one** layer, all positions — note the asymmetry: addition uses
  the **raw** magnitude-carrying `r`.
- **Weight orthogonalization (permanent):** `W' = W − r̂ r̂ᵀ W` on the embed matrix, every attn `o_proj`, every MLP
  `down_proj`. Bakes ablation into weights with zero inference overhead.

```python
# src/baku/directions/interventions.py  (reference: the ablation hook factory)
import torch
def ablation_hook(r_hat: torch.Tensor):
    r_hat = r_hat / r_hat.norm()
    def hook(act, hook):                                  # TL hook signature
        proj = (act @ r_hat).unsqueeze(-1) * r_hat       # component along r_hat
        return act - proj
    return hook
# apply on fwd_hooks=[(name, ablation_hook(r_hat)) for name in all_resid_writes]
```

**Verify:** ablation lowers the logit `refusal_metric` on held-out harmful prompts; addition raises it on harmless;
KL-on-harmless stays `< ~0.1` (coherence guard).

---

## Piece 4 — Direction selection + calibration cache  ·  *NEW (selection) / REPEAT (projection)*

Pick the best `(layer, position)` by **ablation efficacy** subject to guards: `argmin bypass` s.t. `induce > 0`,
`KL_harmless < 0.1`, `layer < 0.8·L`. The selected layer tends to land mid-to-late (~0.5–0.65·L) — but that's an *outcome*,
not a constraint. Build the cheap logit `refusal_metric` (`log Σ_{t∈R} p_t − log Σ_{t∉R} p_t`) and the projection scalar.
Cache the selected `RefusalSubspace` per `(model, template_hash)` — extraction is the slow part; do it once.

---

## Piece 5 — `FitnessFn` protocol  ·  *REPEAT → interface + hints (attempt cold)*

You've now built projection (Piece 2/4); this packages it as a search signal. Implement the protocol from ARCHITECTURE
§3.1. Hints: one batched, **truncated** forward (`stop_at_layer` past `l*`); `score = −(r̂·x) − coherence_penalty`
(perplexity or KL bound so the optimizer can't win by emitting gibberish); return `FitnessResult(scores, activations)` so
the Reporter can reuse the same forward; set `is_differentiable = False` for projection fitness.

---

## Piece 6 — `Scorer` / `EnsembleScorer` + `Statistics`  ·  *NEW → full reference*

**The point:** this is the **success metric**, not the fitness. Use ≥2 heterogeneous judges, retain per-judge results,
hash-cache by `(prompt, response, judge_id)`. Phase-1 local stack: StrongREJECT fine-tuned grader + Llama-Guard-3-8B
(4-bit) + a refusal-substring heuristic, **majority vote**. **Copy judge prompt templates verbatim** (their validated
agreement is wording-dependent). Report **raw and post-perplexity-filter** ASR.

```python
# src/baku/scorers/statistics.py  (reference: BCa CI + paired delta)
import numpy as np
from scipy.stats import bootstrap

def asr_bootstrap_ci(successes: np.ndarray, confidence=0.95, n=10000):
    res = bootstrap((successes.astype(float),), np.mean, confidence_level=confidence,
                    n_resamples=n, method="BCa")
    return float(successes.mean()), (float(res.confidence_interval.low), float(res.confidence_interval.high))

def paired_delta_ci(succ_a: np.ndarray, succ_b: np.ndarray, confidence=0.95, n=10000):
    """Paired bootstrap on per-behavior success diff (a−b). Required before claiming an improvement."""
    diff = (succ_a.astype(float) - succ_b.astype(float))
    res = bootstrap((diff,), np.mean, confidence_level=confidence, n_resamples=n, method="BCa")
    return float(diff.mean()), (float(res.confidence_interval.low), float(res.confidence_interval.high))
```

Also: `wilson_interval` for small n; `cohen_kappa` / `fleiss_kappa` for inter-judge agreement — **report κ + base rate**
(κ collapses at extreme refusal base rates). Calibrate each judge's threshold once on JailbreakBench `judge_comparison`.

---

## Piece 7 — `Calibration` bridge  ·  *NEW*

On a held-out set, compute **Spearman** rank-correlation and **precision@K** between the projection fitness and the
behavioral judge. This is the rigor that earns the interpretability claim: only report internal-signal-gated ASR after this
shows adequate correlation for the chosen `(model, layer, position)`.

---

## Piece 8 — `GeneticSearch` Orchestrator  ·  *NEW → full reference* (Phase-1 attack)

Gradient-free, AutoDAN-style. Population 64 (laptop) → 256 (cloud), ~100 steps, elitism top 5%, multi-point crossover,
**default non-LLM mutation** (synonym / word-level), seeded with DAN templates. Fitness = Piece 5. **Mandatory** confirmatory
judge on top-K. Per-gen log: best/mean fitness, BLEU diversity, the best individual's activation trace.

```python
# src/baku/orchestrators/genetic.py  (reference skeleton)
class GeneticSearch(SearchStrategy):
    def __init__(self, fitness, converters, selector, archive, scorer, cfg):
        ...
    def run(self) -> "AttackResult":
        pop = self._seed_population()
        for step in range(self.cfg.num_steps):
            res = self.fitness(pop)                      # one batched forward; internal signal only
            self.archive.add(pop, res.scores)
            parents = self.selector.select(pop, res.scores)
            pop = self.converters.apply(parents)         # crossover + mutation + BLEU-novelty filter
            self._log(step, res.scores)
        finalists = self.archive.top_k(self.cfg.k)
        verdicts = self.scorer.score_many(finalists)     # behavioral SUCCESS metric (top-K only)
        return AttackResult.build(finalists, verdicts, stats=..., config=self.cfg)
```

---

## Piece 9 — Headline ablation harness (white-box vs black-box fitness)  ·  *REPEAT → interface + hints*

The experiment that substantiates the thesis. Same `GeneticSearch`, swap **only** the `FitnessFn`:
`RefusalProjectionFitness` (white-box) vs `JudgeFitness` (the ensemble judge used *as the inner-loop fitness*). **Fix the
query/compute budget** across arms. Metrics: queries-to-first-success, ASR @ fixed budget, per-generation convergence
curves, wall-clock + query cost. Compare with the **paired bootstrap** (Piece 6).

```yaml
# configs/experiments/whitebox_vs_blackbox.yaml  (reference)
base: configs/local.yaml
arms:
  whitebox: { fitness: refusal_projection }
  blackbox: { fitness: judge }
budget:   { max_queries: 6400, population: 64, num_steps: 100 }   # identical across arms
report:   [queries_to_first_success, asr_at_budget, convergence_curve, wallclock, query_cost]
compare:  paired_bootstrap          # delta-ASR CI between arms
seeds:    [0, 1, 2]
```

---

## Piece 10 — `Reporter` (basic)  ·  *NEW*

Reuse the attack's hooks. Emit: per-token/per-layer **projection trajectory** onto `r̂`; **Direct Logit/Direction
Attribution** ranking heads/MLPs that write the refusal direction; the **redaction layer** (mechanism + judge scores + a
truncated/redacted completion indicator; full text only in a gated audit store).

---

## Piece 11 — `Reporter` (heavy attribution)  ·  *Phase 1.5*

KL-ablation impact; **activation-patch heatmap** (`[layer × position]`, localizes *where* refusal is suppressed);
**attention-knockout** of top heads (ablate a layer *window*, not a single head); **causal confirmation** that the refusal
direction *specifically* is responsible — include random/benign-direction controls (random directions also break safety:
*Rogue Scalpel*).

---

## Piece 12 — Cloud / later (design room only)

`RefusalSubspace` 2–5D mode (probe/gradient extraction, representational independence); `GradientAttack` quarantine
(nanoGCG + boundary-aware `filter_ids` round-trip on the *full* prompt); `SAEProvider` + `SAEFeatureFitness` + lifecycle
manager (with the SAE-activation hard gate; JumpReLU mandatory); QD / MAP-Elites (swap Selector + Archive + `descriptor_fn`);
nnsight + `SAETransformerBridge` backends; defense-aware target wrapper (guardrail in the loop).
