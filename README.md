# Activation Function Comparison: ReLU vs Tanh vs GELU

## Goal
Compare `nn.ReLU()`, `nn.Tanh()`, and `nn.GELU()` on a binary classification task to determine **which activation function converges to 90% validation accuracy the fastest**.

## Dataset
- `sklearn.datasets.make_classification` — 2000 samples, 20 features, 15 informative
- Train/test split: 80/20
- StandardScaler normalization
- Same seed (42) across all runs

## Model Architecture (Deeper 3-Layer)
```
Linear(20 → 128) → BatchNorm1d(128) → [ACTIVATION] → Dropout(0.3)
→ Linear(128 → 64) → [ACTIVATION] → Dropout(0.3)
→ Linear(64 → 32) → [ACTIVATION] → Dropout(0.3)
→ Linear(32 → 1)
```

Only the activation function changes. Everything else (weights init via same seed, optimizer, lr) stays identical.

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | BCEWithLogitsLoss |
| Epochs | 150 |
| Batch | Full-batch gradient descent |

## Convergence Metric
- **Primary:** Epoch number at which validation accuracy first reaches ≥ 90%
- **Secondary:** Final validation accuracy after 150 epochs
- **Tertiary:** Validation loss curve comparison

## Output
1. Console table showing:
   ```
   | Activation | Epoch to 90% Val Acc | Final Val Acc | Final Val Loss |
   |------------|----------------------|---------------|----------------|
   | ReLU       | ...                  | ...           | ...            |
   | Tanh       | ...                  | ...           | ...            |
   | GELU       | ...                  | ...           | ...            |
   ```
2. Side-by-side plot: Loss curves overlay + Accuracy curves overlay (all 3 activations on same axes)
3. Clear winner statement

## Script File
`activation_comparison.py` — standalone, self-contained script.
