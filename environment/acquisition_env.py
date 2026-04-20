"""
environment/acquisition_env.py
───────────────────────────────
The RL environment for SERAPH.

The agent interacts with one molecule per episode, sequentially deciding
which of the 19 QM9 properties to "acquire" (observe). After each acquisition
the predictor re-runs on the newly revealed features and the reward is:

    reward = Δaccuracy − λ × cost            (then clipped to [-reward_clip, reward_clip])

Per-step reward clipping stabilises DQN training by bounding the Bellman
target. Without it, outlier molecules (those with extreme true-target values)
produce huge rewards that stretch Q-value estimates for many updates after.
This is the standard trick from Atari DQN.

Interface mirrors OpenAI Gym:

    env = AcquisitionEnv(molecules, stats, predictor)
    state = env.reset()
    state, reward, done, info = env.step(action)

State
-----
Flat FloatTensor (38,) = cat([mask(19), values(19)]).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import torch
import numpy as np
from typing import Optional

import config
from data.load_qm9 import PROPERTY_NAMES, PROPERTY_INDEX


# ── Environment ────────────────────────────────────────────────────────────────

class AcquisitionEnv:
    """
    Sequential feature acquisition environment for QM9 molecules.

    Parameters
    ----------
    molecules    : list of FloatTensor (19,) — normalised property vectors.
    stats        : dict from load_qm9 — contains mean, std, target_idx.
    predictor    : models.Predictor — implements predict(values, mask) → scalar.
    lam          : λ cost penalty per feature acquired (config.LAMBDA).
    max_steps    : maximum acquisitions per episode (default = NUM_FEATURES).
    device       : torch device string.
    seed         : optional random seed.
    reward_clip  : absolute bound on per-step reward. Set to None to disable.
                   Default 1.0 — matches standard DQN practice and handles
                   outlier molecules with extreme true-target values.
    """

    N_FEATURES = config.NUM_FEATURES   # 19
    STATE_DIM  = N_FEATURES * 2        # 38

    def __init__(
        self,
        molecules:   list,
        stats:       dict,
        predictor,
        lam:         float = config.LAMBDA,
        max_steps:   int   = config.NUM_FEATURES,
        device:      str   = config.DEVICE,
        seed:        Optional[int] = config.SEED,
        reward_clip: Optional[float] = 5.0,
    ):
        self.molecules   = molecules
        self.stats       = stats
        self.predictor   = predictor
        self.lam         = lam
        self.max_steps   = max_steps
        self.device      = torch.device(device)
        self.target_idx  = stats["target_idx"]
        self.reward_clip = reward_clip

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Episode state (populated by reset())
        self._molecule   : Optional[torch.Tensor] = None
        self._mask       : Optional[torch.Tensor] = None
        self._values     : Optional[torch.Tensor] = None
        self._prev_mse   : float = float("inf")
        self._step       : int   = 0
        self._acquired_order : list[int] = []

    # ── Core interface ─────────────────────────────────────────────────────────

    def reset(self, molecule_idx: Optional[int] = None) -> torch.Tensor:
        """Start a new episode. Returns flat state (38,)."""
        if molecule_idx is None:
            molecule_idx = random.randrange(len(self.molecules))

        self._molecule = self.molecules[molecule_idx].to(self.device)
        self._mask     = torch.zeros(self.N_FEATURES, device=self.device)
        self._values   = torch.zeros(self.N_FEATURES, device=self.device)
        self._step     = 0
        self._acquired_order = []

        # Baseline MSE with zero features acquired
        self._prev_mse = self._compute_mse()

        return self._get_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
        """Acquire property `action`. Returns (state, reward, done, info)."""
        assert 0 <= action < self.N_FEATURES, \
            f"Invalid action {action} — must be in [0, {self.N_FEATURES})"
        assert self._mask[action] == 0, \
            f"Property {action} ({PROPERTY_NAMES[action]}) already acquired"
        assert self._molecule is not None, \
            "Call reset() before step()"

        # ── Acquire the chosen property ────────────────────────────────────────
        self._mask[action]   = 1.0
        self._values[action] = self._molecule[action]
        self._acquired_order.append(action)
        self._step += 1

        # ── Compute reward ─────────────────────────────────────────────────────
        new_mse        = self._compute_mse()
        delta_acc      = self._prev_mse - new_mse        # positive = improvement
        cost           = 1.0                              # one unit per feature
        raw_reward     = delta_acc - self.lam * cost
        self._prev_mse = new_mse

        # Per-step reward clipping for DQN stability
        if self.reward_clip is not None:
            reward = float(np.clip(raw_reward, -self.reward_clip, self.reward_clip))
        else:
            reward = float(raw_reward)

        # ── Check termination ──────────────────────────────────────────────────
        all_acquired = len(self.legal_actions()) == 0
        max_reached  = self._step >= self.max_steps
        done         = all_acquired or max_reached

        info = {
            "mse"            : new_mse,
            "delta_acc"      : delta_acc,
            "raw_reward"     : float(raw_reward),
            "clipped_reward" : reward,
            "step"           : self._step,
            "n_acquired"     : int(self._mask.sum().item()),
            "acquired_order" : list(self._acquired_order),
            "acquired_names" : [PROPERTY_NAMES[i] for i in self._acquired_order],
        }

        return self._get_state(), reward, done, info

    # ── Legal action masking ───────────────────────────────────────────────────

    def legal_actions(self) -> list[int]:
        """Indices of not-yet-acquired, non-target properties."""
        return [i for i in range(self.N_FEATURES)
                if self._mask[i] == 0 and i != self.target_idx]

    def legal_action_mask(self) -> torch.Tensor:
        """BoolTensor (N_FEATURES,) — True where action is legal."""
        mask = self._mask == 0
        mask[self.target_idx] = False
        return mask

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_state(self) -> torch.Tensor:
        return torch.cat([self._mask, self._values])

    def _compute_mse(self) -> float:
        """MSE between predictor output and true target (normalised)."""
        true_target = self._molecule[self.target_idx].item()

        if self.predictor is None:
            return 1.0

        with torch.no_grad():
            pred = self.predictor.predict(self._values, self._mask)

        return (pred.item() - true_target) ** 2

    # ── Utility ────────────────────────────────────────────────────────────────

    def render(self):
        """Text summary of current episode state."""
        if self._molecule is None:
            print("Environment not initialised — call reset() first.")
            return

        print(f"\n── Episode step {self._step} ──")
        print(f"  Target property : {PROPERTY_NAMES[self.target_idx]}")
        print(f"  Features acquired ({int(self._mask.sum())}/{self.N_FEATURES}):")
        for i in self._acquired_order:
            val_norm = self._values[i].item()
            val_real = val_norm * self.stats["std"][i].item() \
                     + self.stats["mean"][i].item()
            print(f"    [{i:>2}] {PROPERTY_NAMES[i]:<12} "
                  f"normalised={val_norm:+.4f}  real={val_real:+.4f}")
        if self._mask.sum() < self.N_FEATURES:
            remaining = [PROPERTY_NAMES[i] for i in self.legal_actions()]
            print(f"  Remaining : {remaining}")
        print(f"  Current MSE : {self._prev_mse:.6f}")


# ── Dataset helper ─────────────────────────────────────────────────────────────

def build_molecule_list(loader) -> list[torch.Tensor]:
    """Flatten a PyG DataLoader into a plain list of (19,) property tensors."""
    molecules = []
    for batch in loader:
        for y in batch.y:
            molecules.append(y.squeeze().float())
    print(f"[AcquisitionEnv] Built molecule list: {len(molecules):,} molecules")
    return molecules


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.load_qm9 import load_qm9

    print("Loading QM9 …")
    train_loader, val_loader, test_loader, stats = load_qm9(batch_size=256)
    molecules = build_molecule_list(train_loader)

    print("\nInitialising environment (no predictor — random baseline) …")
    env = AcquisitionEnv(molecules, stats, predictor=None, seed=42)

    state = env.reset(molecule_idx=0)
    print(f"\nInitial state shape : {state.shape}")
    print(f"STATE_DIM           : {AcquisitionEnv.STATE_DIM}")
    print(f"Reward clip         : {env.reward_clip}")

    total_reward, done = 0.0, False
    while not done:
        action = random.choice(env.legal_actions())
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  step {info['step']:>2} | acquired '{PROPERTY_NAMES[action]:<12}' "
              f"| reward={reward:+.5f} (raw={info['raw_reward']:+.5f}) "
              f"| mse={info['mse']:.5f}")

    print(f"\nEpisode complete.")
    print(f"  Total reward    : {total_reward:.4f}")
    print(f"  Features used   : {info['n_acquired']}/{config.NUM_FEATURES}")
    env.render()

    env.reset()
    env.step(0)
    env.step(1)
    mask = env.legal_action_mask()
    assert mask[0] == False and mask[1] == False and mask[2] == True
    print("\n✅  acquisition_env.py working correctly.")