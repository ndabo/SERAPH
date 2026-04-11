"""
models/predictor.py
────────────────────
The predictor MLP for SERAPH.

Takes a masked property vector (19 values, zeroed where not yet acquired)
and a binary acquisition mask (19 flags), and predicts the normalised target
property (e.g. HOMO-LUMO gap).

The predictor serves two roles in SERAPH:

  1. INSIDE the RL environment — called after every acquisition to compute
     the accuracy gain that drives the reward signal. Must be fast.

  2. AS a standalone baseline — trained on all 19 features (mask all ones)
     to give the upper-bound "full information" performance.

Architecture
------------
Input  : cat([mask, values])  →  dim 38
Hidden : N fully-connected layers with ReLU + optional Dropout
Output : scalar prediction of the normalised target

The 38-dim input mirrors the state vector in acquisition_env.py so the
predictor can be slotted directly into the environment without reshaping.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from typing import Optional

import config


# ── Model ──────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Masked MLP that predicts a single QM9 target property.

    Parameters
    ----------
    input_dim   : size of the flat input vector. Default 38 = mask(19) + values(19).
    hidden_dim  : width of each hidden layer.
    num_layers  : number of hidden layers (depth).
    dropout     : dropout probability applied after each hidden layer (0 = off).
    device      : torch device string.
    """

    def __init__(
        self,
        input_dim:  int   = config.NUM_FEATURES * 2,  # 38
        hidden_dim: int   = config.PRED_HIDDEN,
        num_layers: int   = config.PRED_LAYERS,
        dropout:    float = 0.1,
        device:     str   = config.DEVICE,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device     = torch.device(device)

        # ── Build layer stack ─────────────────────────────────────────────────
        layers = []

        # Input → first hidden
        layers += [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # Hidden → hidden (num_layers - 1 additional blocks)
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        # Final hidden → scalar output (no activation — raw regression output)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self.to(self.device)

        # Initialise weights with Xavier uniform (better than PyTorch default
        # for regression tasks with ReLU activations)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (..., 38) — cat([mask, values]) for one or more molecules.
            Single molecule:  (38,) or (1, 38)
            Batched:          (B, 38)

        Returns
        -------
        FloatTensor (..., 1) — normalised target prediction.
        """
        return self.net(x)

    # ── Convenience method called by AcquisitionEnv ────────────────────────────

    def predict(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Single-molecule inference called by AcquisitionEnv._compute_mse().

        Parameters
        ----------
        values : FloatTensor (19,) — normalised property values, 0 where unacquired
        mask   : FloatTensor (19,) — binary acquisition flags

        Returns
        -------
        FloatTensor scalar — predicted normalised target
        """
        self.eval()
        x = torch.cat([mask, values]).unsqueeze(0).to(self.device)  # (1, 38)
        with torch.no_grad():
            pred = self.net(x)   # (1, 1)
        return pred.squeeze()    # scalar


# ── Training utilities ─────────────────────────────────────────────────────────

def build_xy(molecules: list[torch.Tensor], target_idx: int, device: str) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, y) tensors for full-information baseline training.

    The full-information baseline uses all 19 features (mask = all ones),
    giving the upper-bound accuracy the RL agent is trying to approach
    with fewer features.

    Parameters
    ----------
    molecules  : list of FloatTensor (19,) — from build_molecule_list()
    target_idx : column index of the target property (from stats["target_idx"])
    device     : torch device string

    Returns
    -------
    X : FloatTensor (N, 38) — cat([ones_mask, values]) for each molecule
    y : FloatTensor (N, 1)  — normalised target values
    """
    dev = torch.device(device)
    full_mask = torch.ones(config.NUM_FEATURES, device=dev)

    X_list, y_list = [], []
    for mol in molecules:
        mol = mol.to(dev)
        x   = torch.cat([full_mask, mol])         # (38,)
        y   = mol[target_idx].unsqueeze(0)        # (1,)
        X_list.append(x)
        y_list.append(y)

    X = torch.stack(X_list)   # (N, 38)
    y = torch.stack(y_list)   # (N, 1)
    return X, y


def train_one_epoch(
    model:     "Predictor",
    X:         torch.Tensor,
    y:         torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = config.BATCH_SIZE,
) -> float:
    """
    Run one epoch of mini-batch gradient descent.

    Returns
    -------
    mean training MSE loss for this epoch (float)
    """
    model.train()
    device    = model.device
    N         = X.shape[0]
    perm      = torch.randperm(N)
    X, y      = X[perm].to(device), y[perm].to(device)
    criterion = nn.MSELoss()

    total_loss, n_batches = 0.0, 0
    for start in range(0, N, batch_size):
        xb = X[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: "Predictor",
    X:     torch.Tensor,
    y:     torch.Tensor,
    stats: dict,
) -> dict:
    """
    Evaluate the predictor on a held-out split.

    Returns a dict with:
        mse_norm  : MSE in normalised space (what the model trains on)
        mae_norm  : MAE in normalised space
        mse_real  : MSE in original physical units (un-normalised)
        mae_real  : MAE in original physical units
    """
    model.eval()
    device     = model.device
    criterion  = nn.MSELoss()
    target_idx = stats["target_idx"]
    std        = stats["std"][target_idx].to(device)
    mean       = stats["mean"][target_idx].to(device)

    X, y = X.to(device), y.to(device)
    pred = model(X)  # (N, 1)

    mse_norm = criterion(pred, y).item()
    mae_norm = (pred - y).abs().mean().item()

    # Un-normalise for real-unit metrics
    pred_real = pred * std + mean
    y_real    = y    * std + mean
    mse_real  = ((pred_real - y_real) ** 2).mean().item()
    mae_real  = (pred_real  - y_real).abs().mean().item()

    return {
        "mse_norm": mse_norm,
        "mae_norm": mae_norm,
        "mse_real": mse_real,
        "mae_real": mae_real,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_predictor(model: "Predictor", path: str):
    """Save model weights and constructor args to a .pt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }, path)
    print(f"[Predictor] Saved → {path}")


def load_predictor(path: str, device: str = config.DEVICE) -> "Predictor":
    """Load a predictor saved with save_predictor()."""
    ckpt  = torch.load(path, map_location=device)
    model = Predictor(
        input_dim  = ckpt["input_dim"],
        hidden_dim = ckpt["hidden_dim"],
        num_layers = ckpt["num_layers"],
        device     = device,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Predictor] Loaded ← {path}")
    return model


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.load_qm9 import load_qm9
    from environment.acquisition_env import build_molecule_list

    # ── Build model ────────────────────────────────────────────────────────────
    print("Building predictor …")
    model = Predictor()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : input({model.input_dim}) "
          f"→ {model.num_layers}×hidden({model.hidden_dim}) → 1")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Device       : {model.device}")

    # ── Forward pass sanity check ──────────────────────────────────────────────
    dummy_mask   = torch.zeros(config.NUM_FEATURES).to(model.device)
    dummy_values = torch.zeros(config.NUM_FEATURES).to(model.device)
    dummy_mask[[0, 2, 4]] = 1.0       # pretend we acquired 3 features
    dummy_values[[0, 2, 4]] = torch.randn(3).to(model.device)

    pred = model.predict(dummy_values, dummy_mask)
    print(f"\n  predict() output shape : {pred.shape}")
    print(f"  predict() output value : {pred.item():.4f}")

    # ── Batched forward ────────────────────────────────────────────────────────
    batch_x = torch.randn(32, config.NUM_FEATURES * 2).to(model.device)
    batch_y = model(batch_x)
    print(f"\n  Batched forward — input: {batch_x.shape}, output: {batch_y.shape}")

    # ── Quick training loop on tiny synthetic data ─────────────────────────────
    print("\nRunning 5-epoch synthetic training check …")
    N     = 200
    X_syn = torch.randn(N, config.NUM_FEATURES * 2)
    y_syn = torch.randn(N, 1)
    opt   = torch.optim.Adam(model.parameters(), lr=config.PRED_LR)

    for epoch in range(5):
        loss = train_one_epoch(model, X_syn, y_syn, opt, batch_size=32)
        print(f"  epoch {epoch+1}/5 — train MSE: {loss:.4f}")

    # ── Checkpoint round-trip ──────────────────────────────────────────────────
    save_path = os.path.join(config.CHECKPOINT_DIR, "predictor_test.pt")
    save_predictor(model, save_path)
    loaded    = load_predictor(save_path)
    print(f"\n  Checkpoint round-trip ✓")

    # ── Plug into environment ──────────────────────────────────────────────────
    print("\nPlugging predictor into AcquisitionEnv …")
    from environment.acquisition_env import AcquisitionEnv
    train_loader, _, _, stats = load_qm9(batch_size=256)
    molecules = build_molecule_list(train_loader)

    env   = AcquisitionEnv(molecules, stats, predictor=model, seed=42)
    state = env.reset(molecule_idx=0)

    import random
    state, reward, done, info = env.step(random.choice(env.legal_actions()))
    print(f"  First step — reward: {reward:+.5f}, mse: {info['mse']:.5f}")
    print(f"  Acquired: {info['acquired_names']}")

    print("\n✅  predictor.py working correctly.")
