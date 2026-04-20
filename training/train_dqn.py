"""
training/train_dqn.py
──────────────────────
Main RL training loop for SERAPH.

Trains the DQN agent to learn a cost-efficient feature acquisition policy
over QM9 molecules. Each episode the agent sequentially decides which of the
19 quantum-chemical properties to observe, guided by the reward:

    reward = Δaccuracy − λ × cost

The trained predictor from train_baseline.py must exist at:
    checkpoints/predictor_baseline.pt

Outputs
-------
  checkpoints/dqn_best.pt              — best agent (highest mean episode reward)
  checkpoints/dqn_latest.pt           — latest checkpoint (for resuming)
  results/metrics/dqn_training.json   — per-episode metrics
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import random
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import AcquisitionEnv, build_molecule_list
from models.predictor import load_predictor
from models.dqn_agent import DQNAgent


# ── Logging helpers ────────────────────────────────────────────────────────────

def moving_average(values: list, window: int = 50) -> float:
    """Mean of the last `window` values — used for smoothed reward tracking."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[-window:])) 


def print_episode_summary(
    episode:      int,
    reward:       float,
    n_acquired:   int,
    final_mse:    float,
    epsilon:      float,
    loss:         float | None,
    reward_avg:   float,
):
    loss_str = f"{loss:.5f}" if loss is not None else "  n/a  "
    print(
        f"  Ep {episode:>5} | "
        f"reward={reward:+.4f} (avg={reward_avg:+.4f}) | "
        f"acquired={n_acquired:>2}/{config.NUM_FEATURES} | "
        f"MSE={final_mse:.5f} | "
        f"ε={epsilon:.3f} | "
        f"loss={loss_str}"
    )


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate_agent(
    agent:     DQNAgent,
    env:       AcquisitionEnv,
    molecules: list,
    n_eval:    int = 200,
    lam:       float = config.LAMBDA,
) -> dict:
    """
    Run `n_eval` greedy episodes (no exploration) and return aggregate metrics.

    Returns
    -------
    dict with keys: mean_reward, mean_acquired, mean_mse, acquisition_counts
        acquisition_counts : list[int] (19,) — how often each property was
                             chosen first across all eval episodes (useful for
                             visualising learned ordering)
    """
    agent.online_net.eval()
    rewards, acquired_counts, mses = [], [], []
    acquisition_counts = [0] * config.NUM_FEATURES

    mol_indices = random.sample(range(len(molecules)), min(n_eval, len(molecules)))

    for mol_idx in mol_indices:
        state = env.reset(molecule_idx=mol_idx)
        done  = False
        ep_reward = 0.0

        while not done:
            legal_mask              = env.legal_action_mask()
            action                  = agent.select_action(
                                        state, legal_mask, force_greedy=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        acquired_counts.append(info["n_acquired"])
        mses.append(info["mse"])

        # Track which property was acquired first
        if info["acquired_order"]:
            first = info["acquired_order"][0]
            acquisition_counts[first] += 1

    return {
        "mean_reward"        : float(np.mean(rewards)),
        "mean_acquired"      : float(np.mean(acquired_counts)),
        "mean_mse"           : float(np.mean(mses)),
        "acquisition_counts" : acquisition_counts,
    }


# ── Main training loop ─────────────────────────────────────────────────────────

def train_dqn(
    num_episodes: int   = config.NUM_EPISODES,
    lam:          float = config.LAMBDA,
    device:       str   = config.DEVICE,
    seed:         int   = config.SEED,
    resume:       bool  = False,
) -> DQNAgent:
    """
    Train the DQN agent.

    Parameters
    ----------
    num_episodes : total training episodes
    lam          : λ cost penalty (overrides config.LAMBDA if passed explicitly)
    device       : torch device string
    seed         : random seed
    resume       : if True, load dqn_latest.pt and continue training from there

    Returns
    -------
    DQNAgent — the best checkpoint by mean eval reward
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    # ── Load predictor ─────────────────────────────────────────────────────────
    baseline_path = os.path.join(config.CHECKPOINT_DIR, "predictor_baseline.pt")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline predictor not found at '{baseline_path}'.\n"
            f"Run  python training/train_baseline.py  first."
        )

    print("=" * 60)
    print("SERAPH — DQN Training")
    print("=" * 60)
    print(f"  Target property : {config.TARGET_PROP}")
    print(f"  Episodes        : {num_episodes}")
    print(f"  Lambda (λ)      : {lam}")
    print(f"  Batch size      : {config.BATCH_SIZE}")
    print(f"  Replay size     : {config.REPLAY_SIZE}")
    print(f"  Device          : {device}")
    print()

    print("[1/4] Loading QM9 …")
    train_loader, val_loader, _, stats = load_qm9(batch_size=256)

    print("[2/4] Building molecule lists …")
    train_mols = build_molecule_list(train_loader)
    val_mols   = build_molecule_list(val_loader)

    print("[3/4] Loading baseline predictor …")
    predictor = load_predictor(baseline_path, device=device)
    predictor.eval()

    print("[4/4] Building environment + agent …")
    env   = AcquisitionEnv(train_mols, stats, predictor=predictor,
                           lam=lam, device=device, seed=seed)
    agent = DQNAgent(device=device, target_idx=stats["target_idx"])

    # Optional: resume from latest checkpoint
    latest_path = os.path.join(config.CHECKPOINT_DIR, "dqn_latest.pt")
    if resume and os.path.exists(latest_path):
        agent.load(latest_path)
        print(f"  Resumed from {latest_path}")

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"\n── Training {'=' * 45}")

    best_eval_reward  = float("-inf")
    best_ckpt_path    = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
    history           = []
    recent_rewards    = deque(maxlen=100)   # rolling window for console output
    last_loss         = None
    start_time        = time.time()

    # Eval every N episodes
    eval_interval  = max(50, num_episodes // 20)
    log_interval   = max(10, num_episodes // 50)

    for episode in tqdm(range(1, num_episodes + 1), desc="Episodes", unit="ep"):

        # ── Run one episode ────────────────────────────────────────────────────
        state      = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_loss    = []

        while not done:
            legal_mask = env.legal_action_mask()
            action     = agent.select_action(state, legal_mask)

            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)
                last_loss = loss

            state      = next_state
            ep_reward += reward

        # ── Log episode ────────────────────────────────────────────────────────
        recent_rewards.append(ep_reward)
        mean_loss = float(np.mean(ep_loss)) if ep_loss else None

        record = {
            "episode"    : episode,
            "reward"     : ep_reward,
            "n_acquired" : info["n_acquired"],
            "final_mse"  : info["mse"],
            "epsilon"    : agent.epsilon,
            "loss"       : mean_loss,
            "reward_avg" : moving_average(list(recent_rewards)),
        }
        history.append(record)

        if episode % log_interval == 0 or episode == 1:
            tqdm.write("")
            print_episode_summary(
                episode      = episode,
                reward       = ep_reward,
                n_acquired   = info["n_acquired"],
                final_mse    = info["mse"],
                epsilon      = agent.epsilon,
                loss         = mean_loss,
                reward_avg   = moving_average(list(recent_rewards)),
            )

        # ── Periodic evaluation on val molecules ───────────────────────────────
        if episode % eval_interval == 0:
            tqdm.write(f"\n  ── Eval @ episode {episode} ──")
            eval_metrics = evaluate_agent(agent, env, val_mols, n_eval=200)

            tqdm.write(
                f"  mean reward={eval_metrics['mean_reward']:+.4f} | "
                f"mean acquired={eval_metrics['mean_acquired']:.1f} | "
                f"mean MSE={eval_metrics['mean_mse']:.5f}"
            )

            # Print most-acquired-first property
            counts = eval_metrics["acquisition_counts"]
            top_first = sorted(range(config.NUM_FEATURES),
                               key=lambda i: counts[i], reverse=True)[:3]
            tqdm.write(
                f"  Top first acquisitions: "
                + ", ".join(f"{PROPERTY_NAMES[i]}({counts[i]})" for i in top_first)
            )

            # Save best checkpoint
            # if eval_metrics["mean_reward"] > best_eval_reward:
            #     best_eval_reward = eval_metrics["mean_reward"]
            #     agent.save(best_ckpt_path)
            #     tqdm.write(f"  ✓ New best — saved to {best_ckpt_path}")
            # Save best checkpoint (only after warmup so early random
            # policies don't "win" by luck on a single eval)
            warmup_done = episode >= num_episodes // 4
            if warmup_done and eval_metrics["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["mean_reward"]
                agent.save(best_ckpt_path)
                tqdm.write(f"  ✓ New best — saved to {best_ckpt_path}")

            history[-1]["eval"] = eval_metrics

            # Re-run env reset to make sure env state isn't contaminated
            env = AcquisitionEnv(train_mols, stats, predictor=predictor,
                                 lam=lam, device=device, seed=seed + episode)

        # ── Save latest checkpoint every 100 episodes ──────────────────────────
        if episode % 100 == 0:
            agent.save(latest_path)

    # ── Final save & summary ───────────────────────────────────────────────────
    agent.save(latest_path)

    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "dqn_training.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "target"      : config.TARGET_PROP,
                "num_episodes": num_episodes,
                "lambda"      : lam,
                "gamma"       : config.GAMMA,
                "lr"          : config.LR,
                "batch_size"  : config.BATCH_SIZE,
            },
            "best_eval_reward": best_eval_reward,
            "history"         : history,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} min")
    print(f"Best eval reward : {best_eval_reward:.4f}")
    print(f"Best checkpoint  : {best_ckpt_path}")
    print(f"Metrics saved    : {metrics_path}")
    print(f"{'=' * 60}")

    return agent


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SERAPH DQN agent")
    parser.add_argument("--episodes",  type=int,   default=config.NUM_EPISODES)
    parser.add_argument("--lambda",    type=float, default=config.LAMBDA,
                        dest="lam",
                        help="Cost-accuracy tradeoff. Higher = fewer features acquired.")
    parser.add_argument("--device",    type=str,   default=config.DEVICE)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from checkpoints/dqn_latest.pt")
    args = parser.parse_args()

    train_dqn(
        num_episodes = args.episodes,
        lam          = args.lam,
        device       = args.device,
        resume       = args.resume,
    )
