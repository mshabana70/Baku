# Concepts Covered — Teaching-Dial Ledger

This file tracks which interpretability / engineering concepts have been taught, so the mentor picks the right scaffold
level each time. It is **referenced from `CLAUDE.md`** so a fresh session auto-loads it.

## The dial
- **NEW** (not yet covered) → **full reference code** + explanation. Learn-first; you reimplement by hand.
- **REPEAT** (already taught earlier in this project) → **interface + hints only**; you attempt the body **cold** first,
  then the mentor reviews. This is the pass that solidifies it.
- **Partial / harder twist** → treat the genuinely-new part as NEW, still push on the familiar part. If unsure, the mentor
  asks "full or hints-only for this one?".
- You can always override per-piece: *"give me the full code for this one"* / *"let me try this one cold."*

## How to update
When a concept is first taught with full reference **and** the researcher implements + passes review, flip its dial to
**REPEAT** and set "First built". Add new rows as concepts appear. Keep it honest — a concept only mentioned in a design doc
but not yet built is still **NEW** for its build piece.

## Spaced active recall (standing protocol)
After each piece, when the mentor offers understanding-check questions:
- Ask **mostly about the current piece**, but **mix in 1–2 questions on already-built concepts** (dial = REPEAT, i.e.
  implemented + reviewed) **selected from THIS ledger, not from memory** — so it keeps working in fresh sessions.
- **Space them:** prefer the *least-recently-recalled* completed concepts (see the Recall log) so older concepts resurface
  at intervals.
- **Make questions applied, not definitional** — force connections across concepts
  (e.g. *"Piece 5's fitness does one truncated forward; why would caching all layers there reintroduce the Piece 0 OOM
  problem?"*), never isolated recitation (*"what does `names_filter` do?"*).
- **On a fumble:** bump that concept's "Dial (next time)" back toward **full-reference**, and log it. A fumble is the signal
  it needs another full pass.
- **Log every recall** below so spacing persists across sessions.

### Recall log
| Date | Concept recalled | At piece | Result (solid / fumbled) | Action taken |
|---|---|---|---|---|
| 2026-07-10 | `stop_at_layer = max_layer+1` — exclusive bound / where `hook_resid_post` fires (Q2) | Piece 0 | **solid** | flipped concept to REPEAT |
| 2026-07-10 | `names_filter` vs `stop_at_layer` = which *resource* each conserves (memory vs compute) + why both (Q1) | Piece 0 | **fumbled** — described what each var *holds*, not the resource each *saves* | bumped dial toward full-ref; **re-quiz at Piece 5** (truncated-forward fitness) |

### Per-piece dial overrides (researcher's call, stands until changed)
- **Pieces 2, 5, 8 → attempt COLD (interface + hints)**, even where the default marks them NEW→full. The researcher chose
  these conceptually-harder pieces for cold attempts to push retention. (Piece 5 was already REPEAT.)

| Concept | First introduced | First built | Dial (next time) | Notes |
|---|---|---|---|---|
| Config-driven device/dtype/tier (no hardcoding) | Piece 0 | **Piece 0 (2026-07-10)** | **REPEAT** | ✅ device fix nailed it: config leads, env only vetoes on hard incompat (Principle 8) |
| Forward hooks & activation capture (TL `run_with_cache`, `names_filter`, `stop_at_layer`) | Piece 0 | **Piece 0 (2026-07-10)** | **REPEAT — re-quiz at Piece 5** | code correct; on recall the `+1`/exclusive-bound was solid but the *why* (names_filter→memory, stop_at_layer→compute) was fuzzy |
| Memory discipline (detach→CPU, truncated forward, no cache-all) | Piece 0 | **Piece 0 (2026-07-10)** | **REPEAT — re-quiz at Piece 5** | built correctly, but the memory-vs-compute *articulation* fumbled on recall (2026-07-10) |
| Raw-activation convention / `from_pretrained_no_processing` | ARCHITECTURE §5 (this session) | — | NEW for first SAE/extraction use | the silent-corruption trap; convention-tag artifacts |
| Chat templates + post-instruction positions (negative-indexed) | Piece 1 | — | NEW | synthetic template for no-template base models |
| Family registry / `ModelAdapter` pattern | Piece 1 | — | NEW | isolates all model-specific knowledge; registry dispatches on `config.model.family` (stable tag), not the HF id `config.model.name` |
| Difference-in-means refusal direction | Piece 2 | — | NEW | the interpretability heart; raw resid, layer sweep |
| Projection scalar as a signal | Piece 2 | — | NEW → REPEAT (reused Piece 5) | bimodal histogram is the sanity check |
| Refusal as a 2–5D concept cone (`RefusalSubspace`) | Principle 2 (this session) | — | NEW for subspace mode (Phase 2) | 1-D now, basis-shaped type from day one |
| Directional ablation / activation addition / weight orthogonalization | Piece 3 | — | NEW | unit `r̂` (ablation, all layers) vs raw `r` (addition, one layer) |
| Direction selection (ablation efficacy s.t. KL<0.1, induce>0) | Piece 4 | — | NEW | selected layer is an outcome, not a constraint |
| Cheap logit refusal_metric | Piece 4 | — | NEW | family-specific refusal token set `R` |
| FITNESS ≠ SUCCESS metric (+ Goodhart/coherence guard) | Principle 1 (this session) | — | NEW → full at Piece 5 | the central design principle |
| `FitnessFn` protocol | Piece 5 | — | REPEAT (projection already built) | one truncated forward; returns scalar + activations |
| Ensemble judging / calibrated judges | Piece 6 | — | NEW | ≥2 heterogeneous judges, verbatim prompts, hash-cache |
| Bootstrap CIs (BCa), paired bootstrap, Wilson, κ | Piece 6 | — | NEW | report κ + base rate (kappa paradox) |
| Calibration bridge (Spearman + precision@K) | Piece 7 | — | NEW | earns the interpretability claim |
| Genetic search (population/elitism/crossover/mutation/BLEU novelty) | Piece 8 | — | NEW | gradient-free; GA→QD = swap Selector+Archive |
| White-box vs black-box ablation (the headline experiment) | Piece 9 | — | REPEAT (reuses search + stats) | fixed budget; paired bootstrap on Δ-ASR |
| Direct Logit/Direction Attribution (DLA) | Piece 10 | — | NEW | ranks heads/MLPs writing the refusal direction |
| Activation patching / attention knockout / causal confirmation | Piece 11 | — | NEW (Phase 1.5) | random/benign-direction controls (Rogue Scalpel) |
| SAEs / JumpReLU / Gemma Scope / `SAEProvider` | Principle 6 (this session) | — | NEW (Phase 2) | secondary signal; SAE-activation hard gate; no 2B-IT SAE |
| GCG / `GradientAttack` quarantine | Principle 7 (this session) | — | NEW (Phase 2) | needs a backward pass; budget-driven tier |

**Session 1 status:** all concepts above were *introduced conceptually* (planning + design docs). Piece 0 is being taught
this session with full reference code; flip it to REPEAT once implemented and reviewed.
