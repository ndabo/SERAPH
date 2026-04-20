"""Inspect what the trained DQN actually learned."""
import torch
import numpy as np
import config
from data.load_qm9 import load_qm9, PROPERTY_NAMES
from environment.acquisition_env import AcquisitionEnv, build_molecule_list
from models.predictor import load_predictor
from models.dqn_agent import DQNAgent


def main():
    _, val_loader, _, stats = load_qm9(batch_size=256)
    val_mols = build_molecule_list(val_loader)
    target_idx = stats["target_idx"]

    predictor = load_predictor("./checkpoints/predictor_baseline.pt", device=config.DEVICE)
    agent = DQNAgent(device=config.DEVICE, target_idx=target_idx)
    agent.load("./checkpoints/dqn_latest.pt")   # was dqn_best.pt

    env = AcquisitionEnv(val_mols, stats, predictor=predictor, seed=42)

    # 1. Q-values at s_0 (empty state) — averaged over 100 molecules
    print("\n=== Q-values at initial state s_0 ===")
    print("(averaged over 100 val molecules)\n")

    q_sum = torch.zeros(config.NUM_FEATURES)
    reward_sum = torch.zeros(config.NUM_FEATURES)
    n_mols = 100

    for mol_idx in range(n_mols):
        state = env.reset(molecule_idx=mol_idx)
        with torch.no_grad():
            q = agent.online_net(state.unsqueeze(0).to(agent.device)).squeeze(0).cpu()
        q_sum += q

        # Also measure actual step-0 reward for each action
        for a in range(config.NUM_FEATURES):
            if a == target_idx:
                continue
            env.reset(molecule_idx=mol_idx)
            _, r, _, _ = env.step(a)
            reward_sum[a] += r

    q_avg = q_sum / n_mols
    r_avg = reward_sum / n_mols

    # Sort by Q-value descending
    results = []
    for i in range(config.NUM_FEATURES):
        marker = " (TARGET)" if i == target_idx else ""
        results.append((PROPERTY_NAMES[i], i, q_avg[i].item(), r_avg[i].item(), marker))

    results.sort(key=lambda r: r[2], reverse=True)
    print(f"  {'rank':<5} {'feature':<14} {'idx':<5} {'Q(s_0, a)':<12} {'actual r':<12}")
    for rank, (name, i, q, r, marker) in enumerate(results, 1):
        print(f"  {rank:<5} {name:<14} {i:<5} {q:+.4f}      {r:+.4f}     {marker}")

    # 2. Correlation between Q-values and actual rewards
    legal_mask = torch.ones(config.NUM_FEATURES, dtype=bool)
    legal_mask[target_idx] = False
    q_legal = q_avg[legal_mask].numpy()
    r_legal = r_avg[legal_mask].numpy()
    corr = np.corrcoef(q_legal, r_legal)[0, 1]
    print(f"\nCorrelation(Q, actual_reward) = {corr:+.3f}")
    print("  (1.0 = agent has learned reward ordering perfectly)")
    print("  (0.0 = agent has no idea)")

    # 3. Greedy policy roll-out — first 5 acquisitions
    print("\n=== Greedy policy: first 5 acquisitions (10 molecules) ===\n")
    for mol_idx in range(10):
        state = env.reset(molecule_idx=mol_idx)
        acquired = []
        for step in range(5):
            legal = env.legal_action_mask()
            action = agent.select_action(state, legal, force_greedy=True)
            state, _, done, info = env.step(action)
            acquired.append(PROPERTY_NAMES[action])
            if done:
                break
        print(f"  mol {mol_idx}: {' → '.join(acquired)}")


if __name__ == "__main__":
    main()