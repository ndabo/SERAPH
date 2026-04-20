"""
models/predictor.py
────────────────────
The predictor MLP for SERAPH.

Takes a masked property vector (19 values, zeroed where not yet acquired)
and a binary acquisition mask (19 flags), and predicts the normalised target
property (HOMO-LUMO gap).

The predictor serves two roles in SERAPH:

  1. INSIDE the RL environment — called after every acquisition to compute
     the accuracy gain that drives the reward signal. Must be fast.

  2. AS a standalone baseline — trained on randomly-masked inputs so it
     works across the full distribution of acquisition states the RL agent
     will ever encounter. A separate "full mask" evaluation (all 18 non-
     target features observed) gives the upper-bound performance number.

The 38-dim input mirrors the state vector in acquisition_env.py so the
predictor can be slotted directly into the environment without reshaping.

IMPORTANT: the target property is NEVER included in the observed values.
_sample_mask() always sets mask[target_idx] = 0 so the predictor cannot
trivially copy the answer from its input.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
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

        # Xavier uniform init works better than PyTorch default for
        # regression tasks with ReLU activations
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
        values : FloatTensor (19,) — normalised property values
        mask   : FloatTensor (19,) — binary acquisition flags

        Returns
        -------
        FloatTensor scalar — predicted normalised target
        """
        self.eval()
        # Defensive: zero out values at unobserved positions so the predictor
        # can never see a stale value from a previous episode / init.
        values = values * mask
        x = torch.cat([mask, values]).unsqueeze(0).to(self.device)  # (1, 38)
        with torch.no_grad():
            pred = self.net(x)   # (1, 1)
        return pred.squeeze()    # scalar


# ── Masking utilities ──────────────────────────────────────────────────────────

def _sample_mask(
    batch_size: int,
    n_features: int,
    target_idx: int,
    device:     torch.device,
) -> torch.Tensor:
    """
    Sample a batch of random acquisition masks.

    For each row we pick k ~ Uniform{0, 1, ..., n_features-1} and choose
    which k of the NON-TARGET features to mark as observed. The target
    property is always masked out (mask[target_idx] = 0) — the predictor
    must not be able to see the answer.

    Returns
    -------
    FloatTensor (batch_size, n_features) — 1 where observed, 0 otherwise.
    """
    # Number of observed features per row, in [0, n_features-1]
    k = torch.randint(0, n_features, (batch_size,), device=device)

    # Random scores; force target to have the lowest score so it is never
    # in the top-k.
    scores = torch.rand(batch_size, n_features, device=device)
    scores[:, target_idx] = -1.0

    # Rank each row (rank 0 = highest score); then observed = (rank < k)
    ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
    mask = (ranks < k.unsqueeze(1)).float()
    return mask


# ── Training utilities ─────────────────────────────────────────────────────────

def build_xy(
    molecules:  list,
    target_idx: int,
    device:     str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return raw normalised property values and target values.

    Unlike the old (mask-baked-in) version, this returns the 19-dim values
    matrix directly — masking is applied per-batch during training so every
    epoch sees a fresh distribution of acquisition states.

    Parameters
    ----------
    molecules  : list of FloatTensor (19,) from build_molecule_list()
    target_idx : column index of the target property
    device     : torch device string

    Returns
    -------
    values : FloatTensor (N, 19) — normalised property values
    y      : FloatTensor (N, 1)  — normalised target values
    """
    dev    = torch.device(device)
    values = torch.stack([mol.to(dev) for mol in molecules])   # (N, 19)
    y      = values[:, target_idx:target_idx + 1].clone()      # (N, 1)
    return values, y


def train_one_epoch(
    model:      "Predictor",
    X:          torch.Tensor,
    y:          torch.Tensor,
    optimizer:  torch.optim.Optimizer,
    batch_size: int = config.BATCH_SIZE,
    target_idx: Optional[int] = None,
) -> float:
    """
    Run one epoch of mini-batch gradient descent with random masks.

    For each mini-batch a fresh mask is sampled per example — this teaches
    the predictor to cope with ANY subset of observed features, not just
    the fully-observed case, so it produces meaningful predictions inside
    the RL environment.

    Returns
    -------
    mean training MSE loss for this epoch (float)
    """
    assert target_idx is not None, "train_one_epoch requires target_idx"

    model.train()
    device    = model.device
    N         = X.shape[0]
    perm      = torch.randperm(N)
    X, y      = X[perm].to(device), y[perm].to(device)
    criterion = nn.MSELoss()

    total_loss, n_batches = 0.0, 0
    for start in range(0, N, batch_size):
        vb = X[start : start + batch_size]            # (B, 19)
        yb = y[start : start + batch_size]            # (B, 1)

        # Fresh random mask for this batch; target always unobserved
        mb = _sample_mask(vb.shape[0], config.NUM_FEATURES, target_idx, device)

        # Zero out unobserved values so the input matches what the RL env
        # will feed at inference time
        xb = torch.cat([mb, vb * mb], dim=1)          # (B, 38)

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:            "Predictor",
    X:                torch.Tensor,
    y:                torch.Tensor,
    stats:            dict,
    mask_mode:        str = "random",
    target_idx:       Optional[int] = None,
    n_random_samples: int = 5,
) -> dict:
    """
    Evaluate the predictor.

    Parameters
    ----------
    X          : FloatTensor (N, 19) — normalised property values
    y          : FloatTensor (N, 1)  — normalised target values
    mask_mode  : 'random' — average MSE over `n_random_samples` random masks.
                            Matches the distribution the RL agent will see.
                 'full'   — all 18 non-target features observed. The proposal's
                            upper bound on achievable accuracy.
    target_idx : column index of the target property (always masked out)

    Returns
    -------
    dict with keys: mse_norm, mae_norm, mse_real, mae_real
    """
    assert target_idx is not None, "evaluate requires target_idx"
    assert mask_mode in ("random", "full"), f"Unknown mask_mode: {mask_mode}"

    model.eval()
    device     = model.device
    std        = stats["std"][target_idx].to(device)
    mean       = stats["mean"][target_idx].to(device)

    X, y       = X.to(device), y.to(device)
    n_features = X.shape[1]

    def _metrics_for_mask(mb: torch.Tensor) -> tuple[float, float, float, float]:
        xb   = torch.cat([mb, X * mb], dim=1)     # (N, 38)
        pred = model(xb)                          # (N, 1)
        mse_norm = ((pred - y) ** 2).mean().item()
        mae_norm = (pred - y).abs().mean().item()

        pred_real = pred * std + mean
        y_real    = y    * std + mean
        mse_real = ((pred_real - y_real) ** 2).mean().item()
        mae_real = (pred_real  - y_real).abs().mean().item()
        return mse_norm, mae_norm, mse_real, mae_real

    if mask_mode == "full":
        mask = torch.ones(X.shape[0], n_features, device=device)
        mask[:, target_idx] = 0.0                 # target is never observed
        mse_norm, mae_norm, mse_real, mae_real = _metrics_for_mask(mask)

    else:  # random — average over several mask draws
        results = []
        for _ in range(n_random_samples):
            mb = _sample_mask(X.shape[0], n_features, target_idx, device)
            results.append(_metrics_for_mask(mb))
        arr = np.array(results)                   # (n_samples, 4)
        mse_norm, mae_norm, mse_real, mae_real = arr.mean(axis=0).tolist()

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
    print("\nRunning 5-epoch synthetic training check (random masks) …")
    N         = 200
    target_ix = 4  # pretend gap is column 4
    X_syn     = torch.randn(N, config.NUM_FEATURES)
    y_syn     = X_syn[:, target_ix:target_ix + 1].clone()
    opt       = torch.optim.Adam(model.parameters(), lr=config.PRED_LR)

    for epoch in range(5):
        loss = train_one_epoch(
            model, X_syn, y_syn, opt, batch_size=32, target_idx=target_ix,
        )
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