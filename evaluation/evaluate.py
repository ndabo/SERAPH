"""
evaluation/evaluate.py
────────────────────────────────
Compare acquisition policies for SERAPH.

Rolls out three policies on val/test molecules:
  1. SERAPH (RL)  — learned DQN policy
  2. Random       — acquire features in random order
  3. Greedy       — always pick the feature with highest marginal MSE gain

For each policy, records predictor MSE after 1, 2, 3, …, 18 features
acquired. This produces the accuracy-vs-cost curve — the central figure
in the project proposal.

Also reports:
  - Full acquisition order for each policy (interpretability)
  - Area Under the Cost curve (AUC) — single-number summary
  - Per-step breakdown averaged over all test molecules

Outputs
-------
  results/metrics/policy_comparison.json  — raw numbers
  results/figures/accuracy_vs_cost.png    — the plot

Usage
-----
  python evaluation/evaluate.py
  python evaluation/evaluate.py --n-eval 500 --split test
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import argparse
import numpy as np
import torch
from collections import defaultdict

import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import AcquisitionEnv, build_molecule_list
from models.predictor import load_predictor
from models.dqn_agent import DQNAgent


# ── Policy implementations ─────────────────────────────────────────────────────

def rollout_rl(agent, env, molecule_idx):
    """Roll out the learned DQN policy greedily. Returns per-step MSEs."""
    state = env.reset(molecule_idx=molecule_idx)
    mses = [env._prev_mse]  # MSE at step 0 (no features)
    done = False
    while not done:
        legal = env.legal_action_mask()
        action = agent.select_action(state, legal, force_greedy=True)
        state, _, done, info = env.step(action)
        mses.append(info["mse"])
    return mses, info["acquired_names"]


def rollout_random(env, molecule_idx, seed=None):
    """Acquire features in random order. Returns per-step MSEs."""
    if seed is not None:
        random.seed(seed)
    env.reset(molecule_idx=molecule_idx)
    mses = [env._prev_mse]
    done = False
    while not done:
        action = random.choice(env.legal_actions())
        _, _, done, info = env.step(action)
        mses.append(info["mse"])
    return mses, info["acquired_names"]


def rollout_greedy(env, molecule_idx):
    """
    Greedy policy: at each step, try every legal action and pick the one
    that reduces MSE the most. This is the oracle single-step-lookahead
    baseline — expensive but gives the best possible myopic policy.
    """
    env.reset(molecule_idx=molecule_idx)
    mses = [env._prev_mse]
    acquired_names = []

    while True:
        legal = env.legal_actions()
        if not legal:
            break

        # Save environment state
        saved_mask = env._mask.clone()
        saved_values = env._values.clone()
        saved_prev_mse = env._prev_mse
        saved_step = env._step
        saved_order = list(env._acquired_order)

        best_action = None
        best_mse = float("inf")

        for action in legal:
            # Try this action
            env._mask = saved_mask.clone()
            env._values = saved_values.clone()
            env._prev_mse = saved_prev_mse
            env._step = saved_step
            env._acquired_order = list(saved_order)

            env._mask[action] = 1.0
            env._values[action] = env._molecule[action]
            trial_mse = env._compute_mse()

            if trial_mse < best_mse:
                best_mse = trial_mse
                best_action = action

        # Actually take the best action
        env._mask = saved_mask
        env._values = saved_values
        env._prev_mse = saved_prev_mse
        env._step = saved_step
        env._acquired_order = list(saved_order)

        _, _, done, info = env.step(best_action)
        mses.append(info["mse"])
        acquired_names.append(PROPERTY_NAMES[best_action])

        if done:
            break

    return mses, acquired_names


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate_policies(
    n_eval:    int  = 200,
    split:     str  = "test",
    n_random:  int  = 5,
    save_plot: bool = True,
):
    """
    Run all three policies and produce the accuracy-vs-cost comparison.

    Parameters
    ----------
    n_eval   : number of molecules to evaluate on
    split    : 'val' or 'test'
    n_random : number of random seeds to average over for the random baseline
    save_plot: whether to save the matplotlib figure
    """
    print("=" * 60)
    print("SERAPH — Policy Comparison")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    train_loader, val_loader, test_loader, stats = load_qm9(batch_size=256)
    target_idx = stats["target_idx"]

    if split == "val":
        eval_mols = build_molecule_list(val_loader)
    else:
        eval_mols = build_molecule_list(test_loader)

    train_mols = build_molecule_list(train_loader)

    # ── Load predictor + agent ─────────────────────────────────────────────────
    print("[2/4] Loading predictor …")
    predictor = load_predictor(
        os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt"),
        device=config.DEVICE,
    )

    print("[3/4] Loading DQN agent …")
    agent = DQNAgent(device=config.DEVICE, target_idx=target_idx)

    # Try latest first, fall back to best
    latest_path = os.path.join(config.CHECKPOINT_DIR, "dqn_latest.pt")
    best_path   = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
    if os.path.exists(latest_path):
        agent.load(latest_path)
    elif os.path.exists(best_path):
        agent.load(best_path)
    else:
        raise FileNotFoundError("No DQN checkpoint found. Run train_dqn.py first.")

    # ── Build environment ──────────────────────────────────────────────────────
    print("[4/4] Building environment …")
    env = AcquisitionEnv(
        eval_mols, stats, predictor=predictor,
        lam=config.LAMBDA, device=config.DEVICE, seed=config.SEED,
    )

    n_features = config.NUM_FEATURES - 1  # 18 acquirable (target excluded)
    mol_indices = list(range(min(n_eval, len(eval_mols))))

    # ── Run policies ───────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(mol_indices)} molecules from '{split}' split …\n")

    # Storage: mses[policy][step] = list of MSE values across molecules
    rl_mses     = defaultdict(list)
    greedy_mses = defaultdict(list)
    random_mses = defaultdict(list)

    # Track first few acquisitions for interpretability
    rl_orders     = []
    greedy_orders = []

    for i, mol_idx in enumerate(mol_indices):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  molecule {i+1}/{len(mol_indices)} …")

        # RL policy
        mses, order = rollout_rl(agent, env, mol_idx)
        for step, mse in enumerate(mses):
            rl_mses[step].append(mse)
        if i < 20:
            rl_orders.append(order[:5])

        # Greedy policy
        mses, order = rollout_greedy(env, mol_idx)
        for step, mse in enumerate(mses):
            greedy_mses[step].append(mse)
        if i < 20:
            greedy_orders.append(order[:5])

        # Random policy — average over n_random seeds
        for seed in range(n_random):
            mses, _ = rollout_random(env, mol_idx, seed=seed + i * n_random)
            for step, mse in enumerate(mses):
                random_mses[step].append(mse)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    steps = list(range(n_features + 1))  # 0, 1, ..., 18

    def mean_mse_curve(mse_dict):
        return [float(np.mean(mse_dict[s])) for s in steps]

    def std_mse_curve(mse_dict):
        return [float(np.std(mse_dict[s])) for s in steps]

    rl_curve     = mean_mse_curve(rl_mses)
    greedy_curve = mean_mse_curve(greedy_mses)
    random_curve = mean_mse_curve(random_mses)

    rl_std     = std_mse_curve(rl_mses)
    greedy_std = std_mse_curve(greedy_mses)
    random_std = std_mse_curve(random_mses)

    # AUC (lower = better — less total MSE across all steps)
    rl_auc     = float(np.trapezoid(rl_curve, steps))
    greedy_auc = float(np.trapezoid(greedy_curve, steps))
    random_auc = float(np.trapezoid(random_curve, steps))

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Accuracy-vs-Cost Curve (mean MSE at each step)")
    print(f"{'=' * 60}")
    print(f"\n  {'step':<6} {'RL':<12} {'Greedy':<12} {'Random':<12}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12}")
    for s in steps:
        print(f"  {s:<6} {rl_curve[s]:<12.5f} {greedy_curve[s]:<12.5f} {random_curve[s]:<12.5f}")

    print(f"\n  AUC (lower = better):")
    print(f"    RL     : {rl_auc:.4f}")
    print(f"    Greedy : {greedy_auc:.4f}")
    print(f"    Random : {random_auc:.4f}")

    pct_vs_random = (1 - rl_auc / random_auc) * 100
    pct_vs_greedy = (1 - rl_auc / greedy_auc) * 100
    print(f"\n    RL vs Random : {pct_vs_random:+.1f}% AUC reduction")
    print(f"    RL vs Greedy : {pct_vs_greedy:+.1f}% AUC {'reduction' if pct_vs_greedy > 0 else 'increase'}")

    # ── Print acquisition orders ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"First 5 acquisitions (first 10 molecules)")
    print(f"{'=' * 60}")
    print(f"\n  RL policy:")
    for i, order in enumerate(rl_orders[:10]):
        print(f"    mol {i}: {' → '.join(order)}")
    print(f"\n  Greedy policy:")
    for i, order in enumerate(greedy_orders[:10]):
        print(f"    mol {i}: {' → '.join(order)}")

    # ── Save metrics ───────────────────────────────────────────────────────────
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)

    metrics = {
        "config": {
            "n_eval":  len(mol_indices),
            "split":   split,
            "n_random": n_random,
            "lambda":  config.LAMBDA,
            "gamma":   config.GAMMA,
        },
        "steps": steps,
        "rl":     {"mean": rl_curve,     "std": rl_std,     "auc": rl_auc},
        "greedy": {"mean": greedy_curve, "std": greedy_std, "auc": greedy_auc},
        "random": {"mean": random_curve, "std": random_std, "auc": random_auc},
        "rl_orders":     rl_orders[:20],
        "greedy_orders": greedy_orders[:20],
    }

    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "policy_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            ax.plot(steps, rl_curve,     "o-", color="#2563eb", linewidth=2,
                    markersize=4, label=f"SERAPH (RL)  AUC={rl_auc:.2f}")
            ax.plot(steps, greedy_curve, "s--", color="#16a34a", linewidth=2,
                    markersize=4, label=f"Greedy       AUC={greedy_auc:.2f}")
            ax.plot(steps, random_curve, "^:", color="#dc2626", linewidth=2,
                    markersize=4, label=f"Random       AUC={random_auc:.2f}")

            # Shaded confidence bands
            steps_arr = np.array(steps)
            rl_arr    = np.array(rl_curve)
            rl_s      = np.array(rl_std)
            ax.fill_between(steps_arr, rl_arr - rl_s, rl_arr + rl_s,
                            alpha=0.15, color="#2563eb")

            rand_arr = np.array(random_curve)
            rand_s   = np.array(random_std)
            ax.fill_between(steps_arr, rand_arr - rand_s, rand_arr + rand_s,
                            alpha=0.15, color="#dc2626")

            ax.set_xlabel("Features Acquired", fontsize=12)
            ax.set_ylabel("MSE (normalised)", fontsize=12)
            ax.set_title("SERAPH: Accuracy vs. Acquisition Cost", fontsize=14)
            ax.legend(fontsize=10, loc="upper right")
            ax.set_xticks(steps)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, n_features)
            ax.set_ylim(bottom=0)

            fig_path = os.path.join(config.RESULTS_DIR, "figures", "accuracy_vs_cost.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved  → {fig_path}")
            plt.close(fig)

        except ImportError:
            print("  [matplotlib not available — skipping plot]")

    print(f"\n{'=' * 60}")
    print("Evaluation complete.")
    print(f"{'=' * 60}")

    return metrics


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SERAPH policies")
    parser.add_argument("--n-eval",  type=int, default=200,
                        help="Number of molecules to evaluate on")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--n-random", type=int, default=5,
                        help="Random seeds to average for random baseline")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    evaluate_policies(
        n_eval    = args.n_eval,
        split     = args.split,
        n_random  = args.n_random,
        save_plot = not args.no_plot,
    )
