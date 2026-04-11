# SERAPH Documentation

---

## `data/load_qm9.py`

### `PROPERTY_NAMES` / `PROPERTY_INDEX` (lines 32–57)

QM9 has 19 quantum-chemical properties per molecule stored as columns in `data.y`. This maps human-readable names (e.g. `"gap"`) to their column index so the rest of the code can refer to properties by name instead of magic numbers.

---

### `load_qm9()` (lines 62–155)

The main function. Call this to get DataLoaders ready for training.

```python
train_loader, val_loader, test_loader, stats = load_qm9()
```

1. **Downloads QM9** via PyG on first run (~1.7 GB, cached after that)
2. **Splits** the ~130k molecules into train/val/test using fractions from `config.py`, with a fixed random seed for reproducibility
3. **Computes normalization stats** (mean + std for all 19 properties) using only the training set — important to avoid data leakage from val/test
4. **Wraps each split** in `_NormalisedSubset` so every molecule's `.y` comes out standardized, and returns DataLoaders ready for training

The returned `stats` dict contains:
- `stats["mean"]` — `(19,)` tensor of per-property means
- `stats["std"]` — `(19,)` tensor of per-property standard deviations
- `stats["target_idx"]` — column index of the target property
- `stats["target"]` — string name of the target property

---

### `_NormalisedSubset` (lines 160–180)

A thin wrapper around PyG's `Subset`. When you index into it, it clones the data object and applies `(y - mean) / std`. It also saves the original values as `.y_raw` so you can un-normalize predictions later for interpretability.

---

### `extract_property_matrix()` (lines 185–193)

A utility that flattens an entire DataLoader into a single `(N, 19)` tensor of normalized properties. Useful for the RL environment when it needs all properties of a molecule at once.

---

### `__main__` block (lines 198–214)

A smoke test — run the file directly to verify everything works and print normalization stats for all 19 properties:

```bash
python data/load_qm9.py
```
