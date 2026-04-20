"""
training/train_baseline.py
───────────────────────────
Trains the predictor for SERAPH.

The predictor is trained with RANDOMLY MASKED inputs — on every mini-batch
each example gets a fresh mask drawn from the same distribution the RL agent
will produce during acquisition. This is critical: the old "full mask only"
training produced a model that was out-of-distribution during RL, yielding
noisy rewards and exploding DQN loss.

Two evaluations are reported each epoch:

  random  — mean MSE over several random-mask samples. This is the number
            that actually matters for the RL environment.
  full    — MSE with all 18 non-target features observed. This is the
            proposal's "full-information" upper bound on achievable accuracy.

The best checkpoint is selected on RANDOM-mask val MSE (what the RL agent
sees), not full-mask. The target property is always masked out so the
predictor can never trivially copy it from its input.

Outputs
-------
  checkpoints/predictor_baseline.pt       — best random-mask val predictor
  results/metrics/baseline_metrics.json   — train/val/test metrics per epoch
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

def print_metrics(label: str, metrics: dict, stats: dict):
    """Pretty-print evaluation metrics in both normalised and real units."""
    target_idx = stats["target_idx"]
    std        = stats["std"][target_idx].item()

    print(f"  {label:<16} | "
          f"MSE(norm)={metrics['mse_norm']:.5f}  "
          f"MAE(norm)={metrics['mae_norm']:.5f}  "
          f"MAE(real)={metrics['mae_real']:.5f} {PROPERTY_NAMES[target_idx]}-units  "
          f"[std={std:.4f}]")


# ── Main training loop ─────────────────────────────────────────────────────────

def train_baseline(
    epochs:     int   = config.PRED_EPOCHS,
    lr:         float = config.PRED_LR,
    batch_size: int   = config.BATCH_SIZE,
    device:     str   = config.DEVICE,
    seed:       int   = config.SEED,
) -> Predictor:
    """
    Train the predictor on randomly-masked QM9 inputs.

    Returns
    -------
    Predictor — the best checkpoint (lowest random-mask val MSE)
    """
    torch.manual_seed(seed)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SERAPH — Predictor Training (random-mask)")
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

    # Raw normalised values + targets — masking is applied per-batch
    print("[3/4] Building feature matrices …")
    target_idx = stats["target_idx"]
    X_train, y_train = build_xy(train_mols, target_idx, device)
    X_val,   y_val   = build_xy(val_mols,   target_idx, device)
    X_test,  y_test  = build_xy(test_mols,  target_idx, device)

    print(f"  Train : {X_train.shape}  Val : {X_val.shape}  "
          f"Test : {X_test.shape}")
    print(f"  Target : '{PROPERTY_NAMES[target_idx]}' "
          f"(col {target_idx}) — always masked out")

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
    print(f"\n── Training {'=' * 45}")

    best_val_mse   = float("inf")
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt")
    history        = []
    start_time     = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="ep"):

        # Train — random masks per batch
        train_loss = train_one_epoch(
            model, X_train, y_train, optimizer,
            batch_size = batch_size,
            target_idx = target_idx,
        )

        # Eval — random masks (matches RL env) + full mask (upper bound)
        val_random = evaluate(
            model, X_val, y_val, stats,
            mask_mode="random", target_idx=target_idx, n_random_samples=5,
        )
        val_full = evaluate(
            model, X_val, y_val, stats,
            mask_mode="full",   target_idx=target_idx,
        )
        train_random = evaluate(
            model, X_train, y_train, stats,
            mask_mode="random", target_idx=target_idx, n_random_samples=3,
        )

        # LR scheduler steps on random-mask val MSE (what the RL agent sees)
        scheduler.step(val_random["mse_norm"])

        # Save best checkpoint — picked on random-mask val MSE
        if val_random["mse_norm"] < best_val_mse:
            best_val_mse = val_random["mse_norm"]
            save_predictor(model, best_ckpt_path)
            improved = " ✓"
        else:
            improved = ""

        # Log
        record = {
            "epoch"      : epoch,
            "train_loss" : train_loss,
            **{f"train_random_{k}": v for k, v in train_random.items()},
            **{f"val_random_{k}"  : v for k, v in val_random.items()},
            **{f"val_full_{k}"    : v for k, v in val_full.items()},
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            tqdm.write(f"\nEpoch {epoch}/{epochs}  [{elapsed:.0f}s]{improved}")
            print_metrics("train (random)", train_random, stats)
            print_metrics("val   (random)", val_random,   stats)
            print_metrics("val   (full)",   val_full,     stats)

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n── Test evaluation (best checkpoint) {'=' * 20}")
    from models.predictor import load_predictor
    best_model = load_predictor(best_ckpt_path, device=device)

    test_random = evaluate(
        best_model, X_test, y_test, stats,
        mask_mode="random", target_idx=target_idx, n_random_samples=10,
    )
    test_full = evaluate(
        best_model, X_test, y_test, stats,
        mask_mode="full",   target_idx=target_idx,
    )
    print_metrics("test (random)", test_random, stats)
    print_metrics("test (full)",   test_full,   stats)

    # ── Save metrics to disk ───────────────────────────────────────────────────
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "baseline_metrics.json")
    output = {
        "config": {
            "target"     : config.TARGET_PROP,
            "target_idx" : int(target_idx),
            "epochs"     : epochs,
            "lr"         : lr,
            "batch_size" : batch_size,
            "device"     : device,
        },
        "test_random_metrics" : test_random,
        "test_full_metrics"   : test_full,
        "best_val_random_mse" : best_val_mse,
        "history"             : history,
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} min")
    print(f"Best val MSE (random) : {best_val_mse:.6f}")
    print(f"Test MAE (random, real): {test_random['mae_real']:.6f}")
    print(f"Test MAE (full,   real): {test_full['mae_real']:.6f}")
    print(f"Checkpoint            : {best_ckpt_path}")
    print(f"{'=' * 60}")

    return best_model


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SERAPH predictor (random-mask)")
    parser.add_argument("--epochs",     type=int,   default=config.PRED_EPOCHS)
    parser.add_argument("--lr",         type=float, default=config.PRED_LR)
    parser.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--device",     type=str,   default=config.DEVICE)
    parser.add_argument("--target",     type=str,   default=config.TARGET_PROP,
                        help="QM9 target property (e.g. 'gap')")
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