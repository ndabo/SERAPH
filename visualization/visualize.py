"""
visualization/generate_paper_plots.py
──────────────────────────────────────
Generate all paper-ready figures for SERAPH.

Produces four plots:
  1. Training curves  — reward + loss over episodes
  2. Q-value bar chart — learned Q vs actual reward (what the agent learned)
  3. Acquisition order heatmap — frequency of each feature at each step
  4. Feature importance ranking — single-feature MSE gain bar chart

Outputs
-------
  results/figures/training_curves.png
  results/figures/q_value_comparison.png
  results/figures/acquisition_heatmap.png
  results/figures/feature_importance.png

Usage
-----
  python visualization/generate_paper_plots.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import AcquisitionEnv, build_molecule_list
from models.predictor import load_predictor, build_xy
from models.dqn_agent import DQNAgent


# ── Style setup ────────────────────────────────────────────────────────────────

# Consistent style across all figures
COLORS = {
    "rl":      "#2563eb",
    "greedy":  "#16a34a",
    "random":  "#dc2626",
    "accent":  "#f59e0b",
    "dark":    "#1e293b",
    "mid":     "#64748b",
    "light":   "#f1f5f9",
    "bg":      "#ffffff",
}

plt.rcParams.update({
    "figure.facecolor":   COLORS["bg"],
    "axes.facecolor":     COLORS["bg"],
    "axes.edgecolor":     "#cbd5e1",
    "axes.labelcolor":    COLORS["dark"],
    "text.color":         COLORS["dark"],
    "xtick.color":        COLORS["mid"],
    "ytick.color":        COLORS["mid"],
    "grid.color":         "#e2e8f0",
    "grid.alpha":         0.6,
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "legend.fontsize":    10,
    "font.family":        "sans-serif",
})

FIG_DIR = os.path.join(config.RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Plot 1: Training Curves ───────────────────────────────────────────────────

def plot_training_curves():
    """Reward and loss over training episodes — shows convergence."""
    print("  [1/4] Training curves …")

    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "dqn_training.json")
    if not os.path.exists(metrics_path):
        print("    ⚠ dqn_training.json not found — skipping")
        return

    with open(metrics_path) as f:
        data = json.load(f)

    history = data["history"]
    episodes  = [h["episode"]   for h in history]
    rewards   = [h["reward"]    for h in history]
    losses    = [h["loss"]      for h in history if h["loss"] is not None]
    loss_eps  = [h["episode"]   for h in history if h["loss"] is not None]
    epsilons  = [h["epsilon"]   for h in history]

    # Smoothed reward (rolling window)
    window = max(20, len(episodes) // 30)
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
    eps_smooth     = episodes[window-1:]

    # Eval points
    eval_eps     = [h["episode"] for h in history if "eval" in h]
    eval_rewards = [h["eval"]["mean_reward"] for h in history if "eval" in h]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("SERAPH: DQN Training Progress", fontsize=16, fontweight="bold", y=0.98)

    # ── Top: Reward ────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(episodes, rewards, alpha=0.12, s=8, color=COLORS["rl"],
                label="Per-episode", zorder=1)
    ax1.plot(eps_smooth, rewards_smooth, color=COLORS["rl"], linewidth=2,
             label=f"Rolling mean (w={window})", zorder=2)
    ax1.scatter(eval_eps, eval_rewards, color=COLORS["accent"], s=50,
                marker="D", zorder=3, edgecolors="white", linewidth=1,
                label="Eval (greedy)")
    ax1.axhline(y=0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="lower right")
    ax1.grid(True)
    ax1.set_title("Reward", loc="left", fontsize=12, color=COLORS["mid"])

    # Secondary y-axis for epsilon
    ax1_eps = ax1.twinx()
    ax1_eps.plot(episodes, epsilons, color="#a855f7", linewidth=1.2,
                 linestyle=":", alpha=0.6, label="ε")
    ax1_eps.set_ylabel("ε (exploration)", color="#a855f7")
    ax1_eps.tick_params(axis="y", labelcolor="#a855f7")
    ax1_eps.set_ylim(0, 1.05)

    # ── Bottom: Loss ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(loss_eps, losses, color=COLORS["random"], linewidth=0.5, alpha=0.3)

    # Smoothed loss
    if len(losses) > window:
        losses_smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
        loss_eps_smooth = loss_eps[window-1:]
        ax2.plot(loss_eps_smooth, losses_smooth, color=COLORS["random"],
                 linewidth=2, label=f"Rolling mean (w={window})")

    ax2.set_ylabel("TD Loss")
    ax2.set_xlabel("Episode")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    ax2.set_title("Loss (Huber / Smooth L1)", loc="left", fontsize=12,
                  color=COLORS["mid"])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {path}")


# ── Plot 2: Q-Value Comparison ─────────────────────────────────────────────────

def plot_q_value_comparison():
    """Side-by-side bar chart: learned Q-values vs actual step-0 reward."""
    print("  [2/4] Q-value comparison …")

    # Load everything
    _, val_loader, _, stats = load_qm9(batch_size=256)
    val_mols   = build_molecule_list(val_loader)
    target_idx = stats["target_idx"]

    predictor = load_predictor(
        os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt"),
        device=config.DEVICE,
    )
    agent = DQNAgent(device=config.DEVICE, target_idx=target_idx)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
    agent.load(ckpt_path)

    env = AcquisitionEnv(val_mols, stats, predictor=predictor, seed=42)

    # Collect Q-values and actual rewards
    n_mols = 100
    q_sum = torch.zeros(config.NUM_FEATURES)
    r_sum = torch.zeros(config.NUM_FEATURES)

    for mol_idx in range(n_mols):
        state = env.reset(molecule_idx=mol_idx)
        with torch.no_grad():
            q = agent.online_net(state.unsqueeze(0).to(agent.device)).squeeze(0).cpu()
        q_sum += q

        for a in range(config.NUM_FEATURES):
            if a == target_idx:
                continue
            env.reset(molecule_idx=mol_idx)
            _, r, _, _ = env.step(a)
            r_sum[a] += r

    q_avg = (q_sum / n_mols).numpy()
    r_avg = (r_sum / n_mols).numpy()

    # Exclude target, sort by actual reward descending
    indices = [i for i in range(config.NUM_FEATURES) if i != target_idx]
    indices.sort(key=lambda i: r_avg[i], reverse=True)

    names   = [PROPERTY_NAMES[i] for i in indices]
    q_vals  = [q_avg[i] for i in indices]
    r_vals  = [r_avg[i] for i in indices]

    # Normalize Q-values to same scale as rewards for visual comparison
    q_min, q_max = min(q_vals), max(q_vals)
    r_min, r_max = min(r_vals), max(r_vals)
    if q_max > q_min:
        q_scaled = [(q - q_min) / (q_max - q_min) * (r_max - r_min) + r_min
                     for q in q_vals]
    else:
        q_scaled = q_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SERAPH: What the Agent Learned vs. Ground Truth",
                 fontsize=16, fontweight="bold")

    x = np.arange(len(names))
    bar_w = 0.65

    # Left panel: actual reward (ground truth)
    colors_r = [COLORS["rl"] if v > 0 else "#94a3b8" for v in r_vals]
    ax1.barh(x, r_vals, height=bar_w, color=colors_r, alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Actual Reward (step 0)")
    ax1.set_title("Ground Truth: Single-Feature Reward", fontsize=12,
                  color=COLORS["mid"])
    ax1.axvline(x=0, color="#94a3b8", linewidth=0.8)
    ax1.grid(True, axis="x")

    # Highlight lumo
    lumo_idx_in_sorted = names.index("lumo")
    ax1.barh(lumo_idx_in_sorted, r_vals[lumo_idx_in_sorted], height=bar_w,
             color=COLORS["accent"], alpha=0.95, edgecolor="white", linewidth=0.5)

    # Right panel: learned Q-values
    colors_q = [COLORS["greedy"] if v > 0 else "#94a3b8" for v in q_vals]
    ax2.barh(x, q_vals, height=bar_w, color=colors_q, alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Learned Q(s₀, a)")
    ax2.set_title("Agent's Learned Values", fontsize=12, color=COLORS["mid"])
    ax2.axvline(x=0, color="#94a3b8", linewidth=0.8)
    ax2.grid(True, axis="x")

    # Highlight lumo
    ax2.barh(lumo_idx_in_sorted, q_vals[lumo_idx_in_sorted], height=bar_w,
             color=COLORS["accent"], alpha=0.95, edgecolor="white", linewidth=0.5)

    # Correlation annotation
    corr = np.corrcoef(
        [r_avg[i] for i in indices],
        [q_avg[i] for i in indices]
    )[0, 1]
    fig.text(0.5, 0.02, f"Correlation(Q, reward) = {corr:+.3f}",
             ha="center", fontsize=12, color=COLORS["mid"],
             style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(FIG_DIR, "q_value_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {path}")


# ── Plot 3: Acquisition Order Heatmap (RL vs Greedy side-by-side) ──────────────

def _rollout_greedy_order(env, molecule_idx):
    """Greedy rollout — returns list of acquired feature indices in order."""
    env.reset(molecule_idx=molecule_idx)
    order = []

    while True:
        legal = env.legal_actions()
        if not legal:
            break

        saved_mask     = env._mask.clone()
        saved_values   = env._values.clone()
        saved_prev_mse = env._prev_mse
        saved_step     = env._step
        saved_order    = list(env._acquired_order)

        best_action = None
        best_mse    = float("inf")

        for action in legal:
            env._mask   = saved_mask.clone()
            env._values = saved_values.clone()
            env._prev_mse = saved_prev_mse
            env._step     = saved_step
            env._acquired_order = list(saved_order)

            env._mask[action]   = 1.0
            env._values[action] = env._molecule[action]
            trial_mse = env._compute_mse()

            if trial_mse < best_mse:
                best_mse    = trial_mse
                best_action = action

        env._mask   = saved_mask
        env._values = saved_values
        env._prev_mse = saved_prev_mse
        env._step     = saved_step
        env._acquired_order = list(saved_order)

        env.step(best_action)
        order.append(best_action)

        if len(env.legal_actions()) == 0:
            break

    return order


def plot_acquisition_heatmap():
    """Side-by-side heatmap: RL (structured) vs Greedy (scattered)."""
    print("  [3/4] Acquisition order heatmap (RL vs Greedy) …")

    _, val_loader, _, stats = load_qm9(batch_size=256)
    val_mols   = build_molecule_list(val_loader)
    target_idx = stats["target_idx"]

    predictor = load_predictor(
        os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt"),
        device=config.DEVICE,
    )
    agent = DQNAgent(device=config.DEVICE, target_idx=target_idx)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
    agent.load(ckpt_path)

    env = AcquisitionEnv(val_mols, stats, predictor=predictor, seed=42)

    n_mols    = 200
    n_features = config.NUM_FEATURES
    n_steps    = n_features - 1  # 18 acquirable

    # ── Collect RL heatmap ─────────────────────────────────────────────────
    heatmap_rl = np.zeros((n_steps, n_features))

    for mol_idx in range(n_mols):
        state = env.reset(molecule_idx=mol_idx)
        done  = False
        step  = 0
        while not done:
            legal  = env.legal_action_mask()
            action = agent.select_action(state, legal, force_greedy=True)
            state, _, done, info = env.step(action)
            heatmap_rl[step, action] += 1
            step += 1

    # ── Collect Greedy heatmap ─────────────────────────────────────────────
    print("    (running greedy rollouts — this takes a minute) …")
    heatmap_greedy = np.zeros((n_steps, n_features))

    for mol_idx in range(n_mols):
        order = _rollout_greedy_order(env, mol_idx)
        for step, action in enumerate(order):
            heatmap_greedy[step, action] += 1

    # ── Normalize to percentages ───────────────────────────────────────────
    heatmap_rl_pct     = heatmap_rl     / n_mols * 100
    heatmap_greedy_pct = heatmap_greedy / n_mols * 100

    # Only show acquirable features (exclude target)
    acquirable    = [i for i in range(n_features) if i != target_idx]
    feature_names = [PROPERTY_NAMES[i] for i in acquirable]

    rl_show     = heatmap_rl_pct[:, acquirable]      # (18, 18)
    greedy_show = heatmap_greedy_pct[:, acquirable]   # (18, 18)

    # Sort features by average RL acquisition step (earliest first)
    avg_step = []
    for j, feat_idx in enumerate(acquirable):
        total = sum(s * heatmap_rl[s, feat_idx] for s in range(n_steps))
        count = sum(heatmap_rl[s, feat_idx]     for s in range(n_steps))
        avg_step.append(total / max(count, 1))

    sort_order           = np.argsort(avg_step)
    feature_names_sorted = [feature_names[i] for i in sort_order]
    rl_sorted            = rl_show[:, sort_order]
    greedy_sorted        = greedy_show[:, sort_order]

    # ── Custom colormap ────────────────────────────────────────────────────
    cmap_rl = LinearSegmentedColormap.from_list(
        "seraph_rl", ["#f8fafc", "#dbeafe", "#60a5fa", "#2563eb", "#1e3a8a"]
    )
    cmap_greedy = LinearSegmentedColormap.from_list(
        "seraph_greedy", ["#f8fafc", "#dcfce7", "#4ade80", "#16a34a", "#14532d"]
    )

    # ── Find shared vmax ───────────────────────────────────────────────────
    vmax = max(rl_sorted.max(), greedy_sorted.max())

    # ── Plot side by side ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle("SERAPH: Learned Acquisition Order — RL vs. Greedy Oracle\n"
                 f"({n_mols} molecules, greedy policy rollout)",
                 fontsize=16, fontweight="bold", y=1.02)

    # Left: RL
    im1 = ax1.imshow(rl_sorted.T, aspect="auto", cmap=cmap_rl,
                     interpolation="nearest", vmin=0, vmax=vmax)
    ax1.set_xticks(range(n_steps))
    ax1.set_xticklabels([str(s+1) for s in range(n_steps)], fontsize=8)
    ax1.set_yticks(range(len(feature_names_sorted)))
    ax1.set_yticklabels(feature_names_sorted, fontsize=10)
    ax1.set_xlabel("Acquisition Step", fontsize=12)
    ax1.set_ylabel("Feature", fontsize=12)
    ax1.set_title("SERAPH (RL Agent)", fontsize=13, color=COLORS["rl"],
                  fontweight="bold", pad=10)

    for i in range(n_steps):
        for j in range(len(feature_names_sorted)):
            val = rl_sorted[i, j]
            if val > 5:
                color = "white" if val > 40 else COLORS["dark"]
                ax1.text(i, j, f"{val:.0f}%", ha="center", va="center",
                         fontsize=6.5, color=color, fontweight="bold")

    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.75, pad=0.02)
    cbar1.set_label("Frequency (%)", fontsize=10)

    # Right: Greedy
    im2 = ax2.imshow(greedy_sorted.T, aspect="auto", cmap=cmap_greedy,
                     interpolation="nearest", vmin=0, vmax=vmax)
    ax2.set_xticks(range(n_steps))
    ax2.set_xticklabels([str(s+1) for s in range(n_steps)], fontsize=8)
    ax2.set_xlabel("Acquisition Step", fontsize=12)
    ax2.set_title("Greedy Oracle (per-molecule best)", fontsize=13,
                  color=COLORS["greedy"], fontweight="bold", pad=10)

    for i in range(n_steps):
        for j in range(len(feature_names_sorted)):
            val = greedy_sorted[i, j]
            if val > 5:
                color = "white" if val > 40 else COLORS["dark"]
                ax2.text(i, j, f"{val:.0f}%", ha="center", va="center",
                         fontsize=6.5, color=color, fontweight="bold")

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label("Frequency (%)", fontsize=10)

    # ── Annotation at bottom ───────────────────────────────────────────────
    fig.text(0.5, -0.02,
             "RL discovers a universal strategy (lumo → homo) reflecting the "
             "physical definition of the HOMO-LUMO gap.\n"
             "Greedy picks a different first feature for each molecule — "
             "accurate but uninterpretable.",
             ha="center", fontsize=11, color=COLORS["mid"], style="italic")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "acquisition_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {path}")


# ── Plot 4: Feature Importance Ranking ─────────────────────────────────────────

def plot_feature_importance():
    """Bar chart: MSE gain from observing each single feature alone."""
    print("  [4/4] Feature importance ranking …")

    _, val_loader, _, stats = load_qm9(batch_size=256)
    val_mols   = build_molecule_list(val_loader)
    target_idx = stats["target_idx"]

    X, y = build_xy(val_mols, target_idx, config.DEVICE)
    model = load_predictor(
        os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt"),
        device=config.DEVICE,
    )
    X, y = X.to(model.device), y.to(model.device)

    # Baseline: zero features
    mask0 = torch.zeros_like(X)
    x0 = torch.cat([mask0, X * mask0], dim=1)
    with torch.no_grad():
        pred0 = model(x0)
    mse_zero = ((pred0 - y) ** 2).mean().item()

    # Single-feature MSEs
    results = []
    for i in range(config.NUM_FEATURES):
        if i == target_idx:
            continue
        mask = torch.zeros_like(X)
        mask[:, i] = 1.0
        xb = torch.cat([mask, X * mask], dim=1)
        with torch.no_grad():
            pred = model(xb)
        mse = ((pred - y) ** 2).mean().item()
        gain = mse_zero - mse
        results.append((PROPERTY_NAMES[i], i, mse, gain))

    # Sort by gain descending
    results.sort(key=lambda r: r[3], reverse=True)

    names = [r[0] for r in results]
    gains = [r[3] for r in results]
    mses  = [r[2] for r in results]

    # Color: lumo gets accent, homo gets secondary, rest gets standard
    bar_colors = []
    for name in names:
        if name == "lumo":
            bar_colors.append(COLORS["accent"])
        elif name == "homo":
            bar_colors.append(COLORS["rl"])
        elif gains[names.index(name)] > 0.2:
            bar_colors.append("#60a5fa")
        elif gains[names.index(name)] > 0:
            bar_colors.append("#94a3b8")
        else:
            bar_colors.append("#e2e8f0")

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(names))
    bars = ax.barh(x, gains, height=0.7, color=bar_colors, alpha=0.9,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MSE Gain vs. No Features (higher = more informative)", fontsize=12)
    ax.set_title("SERAPH: Single-Feature Importance for HOMO-LUMO Gap Prediction",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="#94a3b8", linewidth=0.8)
    ax.grid(True, axis="x")

    # Annotate top features with MSE values
    for i, (name, idx, mse, gain) in enumerate(results[:5]):
        ax.text(gain + 0.01, i, f"MSE={mse:.3f}", va="center",
                fontsize=9, color=COLORS["mid"])

    # Add baseline reference
    ax.text(0.98, 0.98, f"Baseline MSE (0 features) = {mse_zero:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=COLORS["mid"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"],
                      edgecolor="#cbd5e1"))

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["accent"], label="LUMO (agent's #1 pick)"),
        Patch(facecolor=COLORS["rl"],     label="HOMO (agent's #2 pick)"),
        Patch(facecolor="#60a5fa",         label="High importance"),
        Patch(facecolor="#94a3b8",         label="Moderate importance"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SERAPH — Generating Paper Figures")
    print("=" * 60)
    print()

    plot_training_curves()
    plot_q_value_comparison()
    plot_acquisition_heatmap()
    plot_feature_importance()

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {FIG_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()