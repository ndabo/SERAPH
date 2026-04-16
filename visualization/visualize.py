"""
visualization/visualize.py
────────────────────────────────
Creates side-by-side training curves from:

results/metrics/baseline_metrics.json
results/metrics/dqn_metrics.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ── Baseline Plot ──────────────────────────────────────────────────────────────

def plot_baseline(metrics_path, save_dir):
    data = load_json(metrics_path)
    history = data["history"]

    epochs = [row["epoch"] for row in history]

    train_mse = [row["train_mse_norm"] for row in history]
    val_mse   = [row["val_mse_norm"]   for row in history]

    train_mae = [row["train_mae_norm"] for row in history]
    val_mae   = [row["val_mae_norm"]   for row in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("BASELINE Training Curves", fontsize=13)

    # MSE
    ax1.plot(epochs, train_mse, marker="o", markersize=4, label="Train")
    ax1.plot(epochs, val_mse,   marker="s", markersize=4, label="Val")
    ax1.set(xlabel="Epoch", ylabel="MSE", title="MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE
    ax2.plot(epochs, train_mae, marker="o", markersize=4, label="Train")
    ax2.plot(epochs, val_mae,   marker="s", markersize=4, label="Val")
    ax2.set(xlabel="Epoch", ylabel="MAE", title="MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "baseline_training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Curves saved → {path}")


# ── DQN Plot ───────────────────────────────────────────────────────────────────

def plot_dqn(metrics_path, save_dir):
    data = load_json(metrics_path)
    history = data["history"]

    episodes = [row["episode"] for row in history]

    train_reward = [row["train_reward"] for row in history]
    train_loss   = [row["train_loss"]   for row in history]

    val_eps = [row["episode"] for row in history if "val_avg_reward" in row]
    val_reward = [row["val_avg_reward"] for row in history if "val_avg_reward" in row]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("DQN Training Curves", fontsize=13)

    # Reward
    ax1.plot(episodes, train_reward, alpha=0.6, label="Train", marker="o", markersize=2)
    if val_reward:
        ax1.plot(val_eps, val_reward, marker="s", markersize=4, linewidth=2, label="Val")
    ax1.set(xlabel="Episode", ylabel="Reward", title="Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(episodes, train_loss, marker="o", markersize=2)
    ax2.set(xlabel="Episode", ylabel="TD Loss", title="Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "dqn_training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Curves saved → {path}")


# ── Comparison Plot ────────────────────────────────────────────────────────────

def compare_final_results(baseline_path, dqn_path, save_dir):
    base = load_json(baseline_path)
    dqn  = load_json(dqn_path)

    baseline_mae = base["test_metrics"]["mae_real"]
    dqn_reward   = dqn["test_metrics"]["avg_reward"]
    dqn_features = dqn["test_metrics"]["avg_features"]

    fig, ax = plt.subplots(figsize=(7,4))

    labels = ["Baseline\n(MAE)", "DQN\n(Reward)", "DQN\n(Features)"]
    values = [baseline_mae, dqn_reward, dqn_features]

    bars = ax.bar(labels, values)
    ax.set_title("Final Test Results")
    ax.grid(True, axis="y", alpha=0.3)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}",
                ha="center", va="bottom")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "final_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Comparison saved → {path}")


# ── Master Runner ──────────────────────────────────────────────────────────────

def plot_all(
    baseline_metrics="results/metrics/baseline_metrics.json",
    dqn_metrics="results/metrics/dqn_metrics.json",
    save_dir="results/plots"
):
    plot_baseline(baseline_metrics, save_dir)
    plot_dqn(dqn_metrics, save_dir)
    compare_final_results(baseline_metrics, dqn_metrics, save_dir)


if __name__ == "__main__":
    plot_all()