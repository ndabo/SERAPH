"""
data/load_qm9.py
────────────────
Loads the QM9 dataset via PyTorch Geometric, normalizes the 19 quantum-chemical
properties, and returns train / val / test splits as PyG Data loaders.

Usage
-----
    from data.load_qm9 import load_qm9, PROPERTY_NAMES

    train_loader, val_loader, test_loader, stats = load_qm9()

The returned `stats` dict contains per-property mean and std so the RL
environment and predictor can un-normalize predictions for interpretability.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

import config

# ── Property index → human-readable name ──────────────────────────────────────
# QM9 exposes 19 molecular properties as target columns (data.y[:, idx]).
# Index order follows the original QM9 paper (Ramakrishnan et al. 2014).
PROPERTY_NAMES = [
    "mu",       # 0  Dipole moment (D)
    "alpha",    # 1  Isotropic polarizability (a₀³)
    "homo",     # 2  HOMO energy (Ha)
    "lumo",     # 3  LUMO energy (Ha)
    "gap",      # 4  HOMO-LUMO gap (Ha)
    "r2",       # 5  Electronic spatial extent (a₀²)
    "zpve",     # 6  Zero-point vibrational energy (Ha)
    "u0",       # 7  Internal energy at 0 K (Ha)
    "u298",     # 8  Internal energy at 298.15 K (Ha)
    "h298",     # 9  Enthalpy at 298.15 K (Ha)
    "g298",     # 10 Free energy at 298.15 K (Ha)
    "cv",       # 11 Heat capacity at 298.15 K (cal/mol·K)
    "u0_atom",  # 12 Atomisation energy at 0 K (Ha)
    "u298_atom",# 13 Atomisation energy at 298.15 K (Ha)
    "h298_atom",# 14 Atomisation enthalpy at 298.15 K (Ha)
    "g298_atom",# 15 Atomisation free energy at 298.15 K (Ha)
    "a",        # 16 Rotational constant A (GHz)
    "b",        # 17 Rotational constant B (GHz)
    "c",        # 18 Rotational constant C (GHz)
]

# Map string name → column index (used by config.TARGET_PROP)
PROPERTY_INDEX = {name: idx for idx, name in enumerate(PROPERTY_NAMES)}

TARGET_IDX = PROPERTY_INDEX[config.TARGET_PROP]


# ── Main loader ────────────────────────────────────────────────────────────────

def load_qm9(
    root: str = config.DATA_ROOT,
    target: str = config.TARGET_PROP,
    batch_size: int = config.BATCH_SIZE,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    seed: int = config.SEED,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Download (first run only) and return QM9 data loaders.

    Parameters
    ----------
    root        : directory where QM9 raw/processed files are cached
    target      : property name to predict (must be in PROPERTY_NAMES)
    batch_size  : samples per mini-batch
    train_frac  : fraction of dataset used for training
    val_frac    : fraction used for validation (remainder = test)
    seed        : random seed for reproducible splits
    num_workers : DataLoader worker processes (keep 0 on MPS/macOS)

    Returns
    -------
    train_loader, val_loader, test_loader, stats
        stats = {"mean": Tensor(19,), "std": Tensor(19,), "target_idx": int}
    """
    if target not in PROPERTY_INDEX:
        raise ValueError(
            f"Unknown target '{target}'. Choose from: {list(PROPERTY_INDEX)}"
        )

    target_idx = PROPERTY_INDEX[target]
    print(f"[QM9] Loading dataset from '{root}' …")
    print(f"[QM9] Target property: '{target}' (column {target_idx})")

    # PyG downloads & caches QM9 automatically on first call (~1.7 GB)
    dataset = QM9(root=root)

    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    n_test  = n_total - n_train - n_val

    print(f"[QM9] {n_total:,} molecules → "
          f"train {n_train:,} / val {n_val:,} / test {n_test:,}")

    # Reproducible random split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # ── Compute normalisation statistics on training set only ─────────────────
    # Collect all 19 properties from the training split
    all_y = torch.stack([dataset[i].y.squeeze() for i in train_set.indices])
    # all_y shape: (n_train, 19)

    mean = all_y.mean(dim=0)   # (19,)
    std  = all_y.std(dim=0)    # (19,)
    std  = std.clamp(min=1e-8) # avoid division by zero for constant features

    print(f"[QM9] Target '{target}' — "
          f"mean: {mean[target_idx]:.4f}, std: {std[target_idx]:.4f}")

    # Inject normalised targets into each split via a wrapper
    train_set = _NormalisedSubset(train_set, mean, std)
    val_set   = _NormalisedSubset(val_set,   mean, std)
    test_set  = _NormalisedSubset(test_set,  mean, std)

    # pin_memory speeds up CPU→GPU transfers; not supported on MPS
    pin = (config.DEVICE == "cuda")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    stats = {
        "mean":       mean,        # (19,) — use to un-normalise predictions
        "std":        std,         # (19,)
        "target_idx": target_idx,  # int
        "target":     target,      # str
    }

    return train_loader, val_loader, test_loader, stats


# ── Normalisation wrapper ──────────────────────────────────────────────────────

class _NormalisedSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset and returns a copy of each Data object whose `.y` tensor
    has been standardised: y_norm = (y - mean) / std.

    The original un-normalised values are preserved as `.y_raw` for logging.
    """

    def __init__(self, subset, mean: torch.Tensor, std: torch.Tensor):
        self.subset = subset
        self.mean   = mean
        self.std    = std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data        = self.subset[idx].clone()
        data.y_raw  = data.y.clone()                    # (1, 19) un-normalised
        data.y      = (data.y - self.mean) / self.std   # (1, 19) normalised
        return data


# ── Convenience: extract flat property matrix ─────────────────────────────────

def extract_property_matrix(loader: DataLoader) -> torch.Tensor:
    """
    Returns a (N, 19) tensor of *normalised* properties for every molecule in
    the loader. Useful for quick analysis or feeding the RL environment directly.
    """
    ys = []
    for batch in loader:
        ys.append(batch.y)         # shape: (batch_size, 19)
    return torch.cat(ys, dim=0)


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = load_qm9()

    # Grab one batch and print shapes
    batch = next(iter(train_loader))
    print("\n── Sample batch ──")
    print(f"  Nodes (atom features) : {batch.x.shape}")
    print(f"  Edge index            : {batch.edge_index.shape}")
    print(f"  Properties (y)        : {batch.y.shape}")
    print(f"  Properties (y_raw)    : {batch.y_raw.shape}")
    print(f"  Batch vector          : {batch.batch.shape}")

    print("\n── Normalisation stats ──")
    for i, name in enumerate(PROPERTY_NAMES):
        print(f"  {name:<12} mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}")

    print("\n✅  load_qm9.py working correctly.")