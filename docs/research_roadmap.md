# Research Roadmap — BFS Physics-Informed Learning

**Status:** Initial plan  
**Updated:** 2026-11-12  
**Scope:** Master's research; priorities and milestones are provisional.

## 1. Central research question

To what extent can physics-informed constraints and improved optimization reduce dependence on high-fidelity reference data when correcting turbulent backward-facing step (BFS) flow fields, without sacrificing predictive accuracy or physical consistency?

## 2. Working hypotheses

| ID | Hypothesis | How to test |
| --- | --- | --- |
| H1 | Physics-informed correction can improve conservation consistency relative to a purely supervised MLP. | Compare continuity and momentum residuals at matched data/training budgets. |
| H2 | Adaptive weighting or staged optimization can improve the accuracy–physics trade-off. | Compare fixed weights, adaptive weights, and staged training over multiple seeds. |
| H3 | Physics-informed training can retain useful performance with fewer fine-reference samples. | Run controlled fine-data ablations while holding evaluation data fixed. |
| H4 | Transfer or parameterized learning can reduce retraining effort across Reynolds numbers. | Hold out one or more Reynolds numbers and compare adaptation cost and test error. |

These are hypotheses, not findings.

## 3. Research principles and controls

- Define the exact physical regime and equations before claiming physical consistency. For RANS data, specify whether residuals use mean-flow equations and how Reynolds stresses or turbulent viscosity are modeled.
- Treat refined RANS as a numerical reference, **not** as DNS/LES or experimental ground truth.
- Separate training, validation, and testing by case or spatial region as appropriate; avoid leakage from shared CFD fields.
- Match architectures, data budgets, training steps, and evaluation grids when isolating the effect of a new constraint.
- Report variability across random seeds and include computational costs.
- Track both field errors and physics-based diagnostics; one metric alone is insufficient.

## 4. Experimental stages

### Stage 0 — Audit and reproducibility baseline

**Goal:** Establish a trustworthy starting point from previous work.

**Tasks**
- [ ] Inventory existing CFD cases, Reynolds numbers, meshes, fields, units, and boundary conditions.
- [ ] Record provenance and availability of coarse RANS, fine RANS, DNS/LES, and experimental references.
- [ ] Reproduce the previous MLP and continuity-informed PINN baseline, if code/data are available.
- [ ] Fix evaluation scripts, random seeds, data splits, and hardware reporting.
- [ ] Store reference metrics and diagnostic plots.

**Deliverable:** Baseline report and reproducible configuration.

**Exit criterion:** Baseline runs are repeatable and comparisons use identical test data.

### Stage 1 — Physics constraints, incrementally

**Goal:** Determine what each added physical constraint contributes.

**Experiments**
- E1.0: Supervised MLP correction baseline.
- E1.1: Add continuity residual only.
- E1.2: Add boundary-condition constraints.
- E1.3: Add mean-momentum residual with explicitly defined closure terms.
- E1.4: Investigate turbulence-related quantities (e.g., eddy viscosity), only when the data and formulation support them.

**Measurements:** Field errors, continuity/momentum residuals, near-wall behavior, velocity profiles, reattachment length where available, and training cost.

**Decision gate:** Advance formulations only when they improve at least one target metric without unacceptable degradation of others, and report trade-offs explicitly.

### Stage 2 — Optimization and loss balancing

**Goal:** Improve training stability and the accuracy–physics compromise.

**Candidate methods**
- Fixed loss weights (control).
- Adaptive or gradient-informed weighting.
- Sequential/staged training: data-first, then physics refinement (and reverse order where justified).
- Adam followed by L-BFGS or another suitable local optimizer.
- Multiobjective analysis, including Pareto-front inspection.

**Controls:** Same dataset, architecture, evaluation protocol, and comparable compute budgets; use multiple seeds.

**Decision gate:** Select methods based on a documented accuracy–physics–cost trade-off, not only the minimum total loss.

### Stage 3 — Reduced dependence on high-fidelity data

**Goal:** Quantify the contribution of physics when reference data are scarce.

**Protocol**
- Compare a supervised baseline and selected PINN formulations using 100%, 50%, 20%, 10%, and (if feasible) 1% of available fine-reference training samples.
- Keep validation/test sets unchanged and independent.
- Repeat subsampling with multiple seeds.
- Consider a no-fine-data case only if governing equations, boundary conditions, and closure information make the problem well posed enough to study meaningfully.

**Decision gate:** Identify the smallest reference-data budget that meets predefined accuracy and physical-consistency thresholds.

### Stage 4 — Transfer and parameterized learning

**Goal:** Assess reuse across flow conditions.

**Experiments**
- Train at one Reynolds number and fine-tune at another.
- Train a parameterized model with Reynolds number as an input.
- Evaluate on held-out Reynolds numbers; distinguish interpolation from extrapolation.
- Compare training-from-scratch, fine-tuning, and parameterized inference costs.

**Decision gate:** Report generalization gains only on truly unseen conditions and with comparable budgets.

### Stage 5 — Consolidation and scientific reporting

**Goal:** Build a defensible evidence chain for the dissertation and publications.

**Tasks**
- [ ] Repeat highest-value experiments and quantify variability.
- [ ] Perform ablations of physics terms and optimizer choices.
- [ ] Document negative results and limitations.
- [ ] Prepare plots, tables, reproducibility notes, and scientific claims tied to evidence.
- [ ] Identify the narrowest contribution supported by the results.

## 5. Priority queue

| Priority | Next experiment | Why it matters | Dependency |
| --- | --- | --- | --- |
| P0 | Reproduce MLP vs continuity-PINN baseline | Establishes comparability | Stage 0 data/code audit |
| P1 | Fine-data ablation using existing baseline | Directly tests central motivation | Stable baseline |
| P1 | Continuity vs momentum-aware training | Tests value of additional physics | Correct RANS formulation/closure |
| P2 | Fixed vs adaptive loss weights | Addresses optimization imbalance | Reproducible physics losses |
| P2 | Adam vs staged Adam→L-BFGS | Tests optimization stability | Fixed compute budget |
| P3 | Transfer across Reynolds numbers | Tests reuse and generalization | Multiple consistent CFD cases |

**Suggested first high-value experiment:** Reproduce the existing MLP and continuity-PINN results, then vary the number of fine-reference training samples while keeping the test set and training budget fixed.

## 6. Experiment record template

For each run, record:

```yaml
experiment_id: E1.1-example
research_question: "Does continuity improve physical consistency?"
dataset_id: TBD
reference_type: refined_RANS  # or DNS, LES, experiment, if available
reynolds_number: TBD
model: pinn_corrector
physics_terms: [continuity]
loss_weights: {data: 1.0, continuity: TBD}
optimizer: TBD
random_seed: TBD
train_split: TBD
validation_split: TBD
test_split: TBD
compute_budget: TBD
hardware: TBD
metrics: [rmse, relative_error, continuity_residual, training_time]
notes: ""
```

## 7. Risks and open questions

1. **Reference fidelity:** Is the available fine RANS solution sufficiently reliable for the proposed claims? What independent reference is available?
2. **RANS closure:** Which turbulence variables or modeled stress terms are needed for a meaningful momentum residual?
3. **Data leakage:** Are train and test samples correlated through the same simulation or spatial field?
4. **Optimization fairness:** How will training budgets be matched when comparing optimizers?
5. **Generalization:** Are enough Reynolds-number cases available for meaningful held-out tests?
6. **Scope:** Which subset of hypotheses can be answered rigorously within the master's timeline?

## 8. Immediate next actions

- [ ] Preserve and document the previous repository's baseline commit and dataset versions.
- [ ] Create a dataset inventory and define the reference hierarchy (fine RANS vs DNS/LES vs experiments).
- [ ] Implement a common evaluation interface and experiment configuration format.
- [ ] Reproduce the previous baseline.
- [ ] Run the first fine-data ablation and review the results before expanding scope.

## 9. Relationship to previous work

Previous project: [PINN Hybrid Experiments](https://github.com/borges12matheus/pinn_hybrid_experiments).

The previous project is the historical reference for coarse-to-fine RANS correction and continuity-informed comparisons. This repository is intended for controlled extensions, not for rewriting or retroactively changing those results.
