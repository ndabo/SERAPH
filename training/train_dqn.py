"""
training/train_dqn.py
────────────────────
The DQN train

The RL agent's job is to get as close to the full-information baseline model as possible 
with the constraint of feature acquisition having associated costs.
"""

import os
import sys
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import build_molecule_list
from models.predictor import (
    Predictor,
    build_xy,
    train_one_epoch,
    evaluate,
    save_predictor,
)
from evaluate import evaluate_dqn


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_metrics(split: str, metrics: dict):
    print(
        f"  {split:<6} | "
        f"Reward={metrics['avg_reward']:.4f}  "
        f"Std={metrics['std_reward']:.4f}  "
        f"Features={metrics['avg_features']:.2f}  "
        f"Steps={metrics['avg_steps']:.2f}"
    )


# ── Main training loop ─────────────────────────────────────────────────────────

def run_episode(env, agent):
    """
    Run one training episode.
    Returns total reward, average loss, steps.
    """
    state = env.reset()
    done = False

    total_reward = 0.0
    losses = []
    steps = 0

    while not done:
        legal_mask = env.legal_action_mask()

        action = agent.select_action(state, legal_mask)

        next_state, reward, done, info = env.step(action)

        agent.store(state, action, reward, next_state, done)

        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward
        steps += 1

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return total_reward, avg_loss, steps



# ── Main training loop ─────────────────────────────────────────────────────────

def train_dqn(
    episodes: int = config.NUM_EPISODES,
    eval_every: int = 25,
    batch_size: int = config.BATCH_SIZE,
    device: str = config.DEVICE,
    seed: int = config.SEED,
):
    """
    Train the DQN feature-acquisition policy.

    Returns
    -------
    best_agent : DQNAgent
    """
    torch.manual_seed(seed)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    print("=" * 60)
    print("SERAPH — DQN Training")
    print("=" * 60)
    print(f"  Target property : {config.TARGET_PROP}")
    print(f"  Episodes        : {episodes}")
    print(f"  Eval every      : {eval_every}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Device          : {device}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/5] Loading QM9 …")
    train_loader, val_loader, test_loader, stats = load_qm9(
        batch_size=batch_size
    )

    print("[2/5] Building molecule lists …")
    train_mols = build_molecule_list(train_loader)
    val_mols   = build_molecule_list(val_loader)
    test_mols  = build_molecule_list(test_loader)

    # ── Predictor ──────────────────────────────────────────────────────────────
    print("[3/5] Loading baseline predictor …")
    predictor_path = os.path.join(
        config.CHECKPOINT_DIR,
        "predictor_baseline.pt"
    )
    predictor = load_predictor(predictor_path, device=device)

    # ── Environments ───────────────────────────────────────────────────────────
    print("[4/5] Building environments …")
    train_env = AcquisitionEnv(train_mols, stats, predictor=predictor, seed=seed)
    val_env   = AcquisitionEnv(val_mols,   stats, predictor=predictor, seed=seed)
    test_env  = AcquisitionEnv(test_mols,  stats, predictor=predictor, seed=seed)

    # ── Agent ─────────────────────────────────────────────────────────────────
    print("[5/5] Building DQN agent …")
    agent = DQNAgent(device=device)

    n_params = sum(
        p.numel() for p in agent.online_net.parameters()
        if p.requires_grad
    )
    print(f"  Parameters : {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n── Training ({'=' * 40})")

    best_val_reward = -float("inf")
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "dqn_best.pt")
    history = []
    start_time = time.time()

    for ep in tqdm(range(1, episodes + 1), desc="Episodes", unit="ep"):

        reward, loss, steps = run_episode(train_env, agent)

        record = {
            "episode": ep,
            "train_reward": reward,
            "train_loss": loss,
            "train_steps": steps,
            "epsilon": agent.epsilon,
        }

        improved = ""

        # Validation
        if ep % eval_every == 0:
            val_metrics = evaluate_dqn(val_env, agent, episodes=50)

            record.update({
                f"val_{k}": v for k, v in val_metrics.items()
            })

            if val_metrics["avg_reward"] > best_val_reward:
                best_val_reward = val_metrics["avg_reward"]
                agent.save(best_ckpt_path)
                improved = " ✓"

            elapsed = time.time() - start_time
            tqdm.write(f"\nEpisode {ep}/{episodes} [{elapsed:.0f}s]{improved}")
            print_metrics("train", {
                "avg_reward": reward,
                "std_reward": 0.0,
                "avg_features": steps,
                "avg_steps": steps,
            })
            print_metrics("val", val_metrics)

        pbar.set_postfix(
            reward=f"{reward:.3f}",
            loss=f"{loss:.4f}",
            eps=f"{agent.epsilon:.3f}"
        )

        history.append(record)

    # ── Final test evaluation ────────────────────────────────────────────────
    print(f"\n── Test evaluation (best checkpoint) {'=' * 20}")

    best_agent = DQNAgent(device=device)
    best_agent.load(best_ckpt_path)

    test_metrics = evaluate_dqn(test_env, best_agent, episodes=100)
    print_metrics("test", test_metrics)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(
        config.RESULTS_DIR,
        "metrics",
        "dqn_metrics.json"
    )

    output = {
        "config": {
            "target": config.TARGET_PROP,
            "episodes": episodes,
            "eval_every": eval_every,
            "batch_size": batch_size,
            "device": device,
        },
        "best_val_reward": best_val_reward,
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Metrics saved → {metrics_path}")

    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Training complete in {total_time / 60:.1f} min")
    print(f"Best val reward : {best_val_reward:.6f}")
    print(f"Test reward     : {test_metrics['avg_reward']:.6f}")
    print(f"Checkpoint      : {best_ckpt_path}")
    print(f"{'=' * 60}")

    return best_agent


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SERAPH DQN")
    parser.add_argument("--episodes",   type=int, default=config.NUM_EPISODES)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device",     type=str, default=config.DEVICE)
    parser.add_argument("--target",     type=str, default=config.TARGET_PROP)

    args = parser.parse_args()

    if args.target != config.TARGET_PROP:
        print(f"Overriding target: {config.TARGET_PROP} → {args.target}")
        config.TARGET_PROP = args.target

    train_dqn(
        episodes=args.episodes,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        device=args.device,
    )
