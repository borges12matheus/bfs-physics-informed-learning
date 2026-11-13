# BFS Physics-Informed Learning

**Physics-Informed Machine Learning for Backward-Facing Step Flows**

Research on physics-informed learning strategies for backward-facing step (BFS) flows, focusing on optimization, physical consistency, generalization, and reduced reliance on high-fidelity data.

## Overview

This repository supports an ongoing master's research project investigating **Physics-Informed Neural Networks (PINNs)** and hybrid machine learning approaches for computational fluid dynamics (CFD).

The central question is whether physical constraints and improved optimization can reduce dependence on expensive high-fidelity reference data while preserving predictive accuracy and physical consistency in turbulent BFS flows.

The backward-facing step is used as a benchmark because it exhibits flow separation, recirculation, and reattachment.

## Research objectives

1. Establish reproducible supervised and physics-informed CFD correction baselines.
2. Study continuity, momentum, boundary conditions, and turbulence-related constraints.
3. Compare optimization and adaptive loss-balancing strategies.
4. Quantify performance under reduced high-fidelity supervision.
5. Investigate transfer and parameterized learning across Reynolds numbers.
6. Evaluate accuracy, physical consistency, generalization, and computational cost.

## Research directions

- **Hybrid CFD correction:** learn corrections to lower-fidelity RANS fields, comparing MLP and PINN approaches.
- **Physics-informed modeling:** progressively introduce governing-equation residuals and boundary constraints.
- **Optimization:** examine gradient-based and hybrid methods, multiobjective trade-offs, and adaptive loss weights.
- **Data efficiency:** test sparse supervision and physics-dominant learning regimes.
- **Generalization:** assess transfer learning and parameterized models across flow conditions.

## Evaluation

| Dimension | Candidate metrics |
| --- | --- |
| Predictive accuracy | MAE, RMSE, relative error |
| Physical consistency | Continuity and momentum residuals |
| Flow behavior | Velocity profiles, recirculation, reattachment length |
| Data efficiency | Performance versus number of fine-reference samples |
| Generalization | Error on held-out Reynolds numbers |
| Computational cost | Training time, inference time, hardware usage |
| Optimization | Convergence, stability, sensitivity to loss weights |

Metrics and comparisons will be selected according to the governing formulation and reference data available.

## Proposed structure

```text
bfs-physics-informed-learning/
├── configs/
├── data/
├── docs/
│   └── research_roadmap.md
├── experiments/
├── notebooks/
├── results/
├── src/
│   ├── models/
│   ├── physics/
│   ├── losses/
│   ├── optimizers/
│   ├── training/
│   └── evaluation/
├── tests/
├── pyproject.toml
└── README.md
```

This structure is a proposal and will be implemented incrementally. Large datasets, model weights, and generated outputs should be managed with an explicit storage and versioning policy rather than committed indiscriminately.

## Research status

**Active research — initial setup.** The directions above are hypotheses and planned investigations, not established results.

See [Research Roadmap](docs/research_roadmap.md) for staged experiments, decision criteria, and reproducibility requirements.

## Previous work

This project extends earlier investigations into coarse-to-fine RANS field correction using supervised neural networks and continuity-informed training.

Previous repository: [PINN Hybrid Experiments](https://github.com/borges12matheus/pinn_hybrid_experiments).

## Reproducibility principles

- Version-controlled configurations and random seeds.
- Explicit train/validation/test partitions without leakage.
- Traceable dataset provenance and preprocessing.
- Comparable training budgets and documented hardware.
- Saved metrics, diagnostics, and experiment metadata.

## Author

**Matheus Borges** — Master's research in computational modeling, optimization, and physics-informed machine learning.

## License

To be defined. Until a license is added, no open-source reuse permission is implied.
