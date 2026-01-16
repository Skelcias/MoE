# Mixture of Experts (MoE) – Benchmark

Projet de comparaison entre un réseau dense classique et des architectures Mixture of Experts
avec **soft gating** et **hard gating**, sur des datasets tabulaires.

## Objectif

Comparer :
- Dense (baseline)
- MoE Soft (combinaison pondérée de tous les experts)
- MoE Hard (routage vers un nombre limité d’experts)

Analyse :
- convergence (loss)
- accuracy
- utilisation / sélection des experts

## Structure

├── benchmark.ipynb # Notebook principal (config + runs)
├── moe_model.py # Dense + MoE Soft
├── moe_hard_gate.py # MoE Hard
├── train.py # Boucle train / eval
└── README.md


## Datasets utilisés

- Iris
- Wine
- Breast Cancer
- CovType (subset)

Les petits datasets sont entraînés sans DataLoader, les gros avec mini-batches.


