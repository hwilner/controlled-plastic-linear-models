# Controlled Plastic Linear Models (CPLM)

A unified computational framework integrating linear recurrent attention, biological credit assignment, adaptive dynamical control, and test-time adaptation for discrete generative world models.

## Introduction

Controlled Plastic Linear Models (CPLM) unify four existing hwilner research codebases into a single architecture for **adaptive, provably stable sequence modeling**:

| Component repository | Role in CPLM |
|---|---|
| [kimi-delta-attention](https://github.com/hwilner/kimi-delta-attention) | Recurrent state engine (chunkwise DPLR linear attention) |
| [error-driven-plasticity](https://github.com/hwilner/error-driven-plasticity) | Biological learning laws / local three-factor update rules |
| [neurons-adaptive-controllers](https://github.com/hwilner/neurons-adaptive-controllers) | Closed-loop meta-controller with stability guarantees |
| [discrete-diffusion-adaptation](https://github.com/hwilner/discrete-diffusion-adaptation) | Test-time discrete diffusion with trajectory guidance |
| [imo-answerbench](https://github.com/hwilner/imo-answerbench) | Calibrated evaluation endpoint for multi-step reasoning |

The central question: how to construct mathematically stable, compute-efficient recurrent operators that unify biological learning rules, linear state-space models, and closed-loop control.

## Extended Introduction

Pure deep learning models typically lack verifiable dynamical stability guarantees, while neurobiological models struggle to scale to high-throughput workloads. CPLM resolves this by construction:

```
+------------------------------------------------------------------------------------+
|                         CONTROLLED PLASTIC LINEAR MODELS                           |
|                                                                                    |
|  [ Closed-Loop Meta-Controller ]               [ Test-Time Discrete Diffusion ]   |
|   (neurons-adaptive-controllers)                 (discrete-diffusion-adaptation)   |
|                 |                                              |                   |
|                 v                                              v                   |
|       Adaptive Dynamics                                Trajectory Guidance         |
|                 |                                              |                   |
|                 +----------------------+-----------------------+                   |
|                                        |                                           |
|                                        v                                           |
|                           [ Recurrent State Engine ]                               |
|                            (kimi-delta-attention)                                  |
|                                        ^                                           |
|                                        |                                           |
|                          Equivalence Proof & Local Rules                           |
|                                        |                                           |
|                          [ Biological Learning Laws ]                              |
|                           (error-driven-plasticity)                                |
+------------------------------------------------------------------------------------+
```

The framework rests on a formal equivalence: the error-correcting biological delta rule

```
W_t = W_{t-1} (I - beta_t x_t x_t^T) + beta_t y_t x_t^T
```

is algebraically isomorphic to the KDA recurrent state transition, where the key vector plays the role of pre-synaptic activation, the value vector the post-synaptic teaching signal, and the per-channel decay `Diag(alpha_t)` heterosynaptic depression. Deep linear transformers can thus be read as multi-layer continuous-time neuromorphic systems performing online gradient descent in their hidden states.

## Methods

### Cross-domain mechanism transfer

| Source domain | Target repository | Mechanism transferred | Expected breakthrough |
|---|---|---|---|
| Control theory | kimi-delta-attention | Lyapunov stability constraints on decay rates `alpha_t` | Provable avoidance of exploding hidden-state norms over 10^6+ token horizons |
| Synaptic plasticity | kimi-delta-attention | Three-factor dendritic gating | Separation of content-addressable retrieval from semantic associative updates |
| Recurrent delta networks | discrete-diffusion-adaptation | Fast rank-1 associative state accumulation | Sub-millisecond gradient-free test-time adaptation |
| Benchmark analysis | discrete-diffusion-adaptation | Dual-boundary difficulty calibration | Automated curriculum trajectory synthesis |

### Stability via Control-Lyapunov functions

Given a candidate Lyapunov function `V(x) = 1/2 ||x - x*||^2` around a target attractor, an online controller solves a per-step quadratic program for the minimal intervention `u(t)` ensuring `dV/dt <= -gamma V(x)`, guaranteeing exponential convergence despite disturbances and parameter variation. Deployed as a meta-controller over KDA, this dynamically governs the channel-wise decay `alpha_t` and write gain `beta_t` from real-time prediction entropy: predictable streams push `alpha_t -> 1` (preserve context); entropy spikes at structural boundaries push `alpha_t -> 0`, `beta_t -> 1` (flush obsolete state).

### Gradient-free test-time adaptation

The closed-form rank-1 delta update is deployed inside the discrete diffusion sampling loop as a non-parametric adapter: as the world model processes new sensory tokens it updates an associative key-value memory matrix in closed form, adapting to new environments within a single rollout with no backward passes.

## Results

| Milestone | Metric | Target | Status |
|---|---|---|---|
| Chunkwise DPLR parity vs sequential KDA | max abs deviation | < 1e-4 | achieved (see kimi-delta-attention `tests/test_chunkwise.py`) |
| CTMC uniformization marginals vs `expm(QT)` | total variation | < 0.02 | achieved (see discrete-diffusion-adaptation `tests/test_ctmc.py`) |
| Lyapunov-guided KDA on long-horizon retrieval | state-norm bound at 10^6 tokens | bounded | Phase 3, planned |
| Delta-adapter test-time adaptation | rollout adaptation latency | < 1 ms/step | Phase 3, planned |
| imo-answerbench evaluation of integrated CPLM | calibrated-band accuracy | TBD | Phase 3, planned |

## Phased roadmap

```
Phase 1: Systems & Numerical Modernization
  ├── [done] DPLR chunkwise matrix transformations (C = 64) — kimi-delta-attention
  ├── Benchmark KDA vs Flash Linear Attention (FLA) Triton kernels
  └── Profile memory bandwidth under FP8 block-scaled states

Phase 2: Differentiable Simulation Engines
  ├── [done] Gumbel-Softmax CTMC solvers — discrete-diffusion-adaptation
  ├── Port error-driven-plasticity to JAX / TorchDynamo graphs
  └── Reformulate continuous-time SSMs in neurons-adaptive-controllers

Phase 3: Cross-Domain Architecture Development
  ├── Synthesize Lyapunov-guided parameter adaptation for KDA layers
  ├── Deploy test-time delta adaptation on discrete diffusion rollouts
  └── Evaluate on calibrated mathematical benchmarks (imo-answerbench)
```

Atomic, dependency-ordered subtasks are tracked as issues in this repository and in each component repository.

## License

MIT. See [LICENSE](LICENSE).
