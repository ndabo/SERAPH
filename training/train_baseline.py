"""
training/train_baseline.py
───────────────────────────
Trains the full-information MLP baseline for SERAPH.

This is the upper-bound model — it sees all 19 QM9 properties at once
(mask = all ones) and predicts the target property. It answers the question:

    "What's the best MSE we could ever achieve if we always acquired
     every single feature?"

The RL agent's job is to get close to this number using far fewer features.
This script must be run and its checkpoint saved before train_dqn.py,
because the trained predictor is plugged into the RL environment to generate
meaningful reward signals.

Outputs
-------
  checkpoints/predictor_baseline.pt   — best val-loss predictor weights
  results/metrics/baseline_metrics.json — train/val/test metrics per epoch
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import build_molecule_list
from models.predictor import (
    Predictor,
    build_xy,
    train_one_epoch,
    evaluate,
    save_predictor,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_metrics(split: str, metrics: dict, stats: dict):
    """Pretty-print evaluation metrics in both normalised and real units."""
    target      = stats["target"]
    target_idx  = stats["target_idx"]
    std         = stats["std"][target_idx].item()

    print(f"  {split:<6} | "
          f"MSE(norm)={metrics['mse_norm']:.5f}  "
          f"MAE(norm)={metrics['mae_norm']:.5f}  "
          f"MAE(real)={metrics['mae_real']:.5f} {PROPERTY_NAMES[target_idx]}-units  "
          f"[std={std:.4f}]")


# ── Main training loop ─────────────────────────────────────────────────────────

def train_baseline(
    epochs:     int = config.PRED_EPOCHS,
    lr:         float = config.PRED_LR,
    batch_size: int = config.BATCH_SIZE,
    device:     str = config.DEVICE,
    seed:       int = config.SEED,
) -> Predictor:
    """
    Train the full-information MLP baseline.

    Parameters
    ----------
    epochs     : number of training epochs
    lr         : learning rate
    batch_size : mini-batch size
    device     : torch device string
    seed       : random seed

    Returns
    -------
    Predictor — the best checkpoint (lowest val MSE)
    """
    torch.manual_seed(seed)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SERAPH — Baseline Predictor Training")
    print("=" * 60)
    print(f"  Target property : {config.TARGET_PROP}")
    print(f"  Epochs          : {epochs}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Learning rate   : {lr}")
    print(f"  Device          : {device}")
    print()

    print("[1/4] Loading QM9 …")
    train_loader, val_loader, test_loader, stats = load_qm9(
        batch_size=batch_size
    )

    print("[2/4] Building molecule lists …")
    train_mols = build_molecule_list(train_loader)
    val_mols   = build_molecule_list(val_loader)
    test_mols  = build_molecule_list(test_loader)

    # Build flat (X, y) tensors — mask = all ones (full information)
    print("[3/4] Building full-information feature matrices …")
    target_idx  = stats["target_idx"]
    X_train, y_train = build_xy(train_mols, target_idx, device)
    X_val,   y_val   = build_xy(val_mols,   target_idx, device)
    X_test,  y_test  = build_xy(test_mols,  target_idx, device)

    print(f"  Train : {X_train.shape}  Val : {X_val.shape}  "
          f"Test : {X_test.shape}")

    # ── Model + optimiser ──────────────────────────────────────────────────────
    print("[4/4] Building model …")
    model = Predictor(device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n── Training ({'=' * 40})")

    best_val_mse   = float("inf")
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt")
    history        = []
    start_time     = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="ep"):

        # Train
        train_loss = train_one_epoch(model, X_train, y_train, optimizer, batch_size)

        # Evaluate
        val_metrics   = evaluate(model, X_val,   y_val,   stats)
        train_metrics = evaluate(model, X_train, y_train, stats)

        # LR scheduler steps on val MSE
        scheduler.step(val_metrics["mse_norm"])

        # Save best checkpoint
        if val_metrics["mse_norm"] < best_val_mse:
            best_val_mse = val_metrics["mse_norm"]
            save_predictor(model, best_ckpt_path)
            improved = " ✓"
        else:
            improved = ""

        # Log
        record = {
            "epoch"      : epoch,
            "train_loss" : train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}"  : v for k, v in val_metrics.items()},
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            tqdm.write(f"\nEpoch {epoch}/{epochs}  [{elapsed:.0f}s]{improved}")
            print_metrics("train", train_metrics, stats)
            print_metrics("val",   val_metrics,   stats)

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n── Test evaluation (best checkpoint) {'=' * 20}")
    from models.predictor import load_predictor
    best_model   = load_predictor(best_ckpt_path, device=device)
    test_metrics = evaluate(best_model, X_test, y_test, stats)
    print_metrics("test", test_metrics, stats)

    # ── Save metrics to disk ───────────────────────────────────────────────────
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "baseline_metrics.json")
    output = {
        "config": {
            "target"    : config.TARGET_PROP,
            "epochs"    : epochs,
            "lr"        : lr,
            "batch_size": batch_size,
            "device"    : device,
        },
        "test_metrics" : test_metrics,
        "best_val_mse" : best_val_mse,
        "history"      : history,
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} min")
    print(f"Best val MSE  : {best_val_mse:.6f}")
    print(f"Test MAE(real): {test_metrics['mae_real']:.6f}")
    print(f"Checkpoint    : {best_ckpt_path}")
    print(f"{'=' * 60}")

    return best_model


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SERAPH baseline predictor")
    parser.add_argument("--epochs",     type=int,   default=config.PRED_EPOCHS)
    parser.add_argument("--lr",         type=float, default=config.PRED_LR)
    parser.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--device",     type=str,   default=config.DEVICE)
    parser.add_argument("--target",     type=str,   default=config.TARGET_PROP,
                        help=f"QM9 target property. One of: {list(config.__dict__)}")
    args = parser.parse_args()

    # Allow overriding the target from the command line
    if args.target != config.TARGET_PROP:
        print(f"Overriding target: {config.TARGET_PROP} → {args.target}")
        config.TARGET_PROP = args.target

    train_baseline(
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        device     = args.device,
    )
