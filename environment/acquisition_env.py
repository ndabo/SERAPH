"""
environment/acquisition_env.py
───────────────────────────────
The RL environment for SERAPH.

The agent interacts with one molecule per episode, sequentially deciding
which of the 19 QM9 properties to "acquire" (observe). After each acquisition
the predictor re-runs on the newly revealed features and the reward is:

    reward = Δaccuracy − λ × cost

where Δaccuracy is the reduction in MSE vs the previous step, cost = 1 per
feature acquired, and λ is the accuracy/cost tradeoff from config.py.

Interface mirrors OpenAI Gym so it is compatible with standard RL libraries:

    env = AcquisitionEnv(molecules, stats, predictor)
    state = env.reset()
    state, reward, done, info = env.step(action)

State
-----
A dict with two keys:
    "mask"   : FloatTensor (19,) — 1 if property i has been acquired, else 0
    "values" : FloatTensor (19,) — normalised property values where acquired,
                                   0.0 where not yet acquired

The DQN policy network receives torch.cat([mask, values]) → dim 38 input.
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
    molecules : list of (property_vector,) tuples — each is a FloatTensor (19,)
                of *normalised* property values for one molecule.
                Build this from the DataLoaders returned by load_qm9.
    stats     : dict returned by load_qm9() — contains mean, std, target_idx.
    predictor : a models.Predictor instance (or None for random baseline).
                Must implement predictor.predict(values, mask) → scalar.
    lam       : λ, the cost penalty per feature acquired. Defaults to
                config.LAMBDA. Sweep this for ablation studies.
    max_steps : maximum acquisitions per episode (defaults to config.NUM_FEATURES
                so the agent can acquire everything if it wants to).
    device    : torch device string ("mps", "cuda", or "cpu").
    seed      : optional random seed for reproducibility.
    """

    # Number of acquirable features
    N_FEATURES = config.NUM_FEATURES  # 19

    # Flat state dimension fed to the DQN: mask (19) + values (19)
    STATE_DIM = N_FEATURES * 2        # 38  

    def __init__(
        self,
        molecules: list,
        stats: dict,
        predictor,
        lam: float = config.LAMBDA,
        max_steps: int = config.NUM_FEATURES,
        device: str = config.DEVICE,
        seed: Optional[int] = config.SEED,
    ):
        self.molecules  = molecules
        self.stats      = stats
        self.predictor  = predictor
        self.lam        = lam
        self.max_steps  = max_steps
        self.device     = torch.device(device)
        self.target_idx = stats["target_idx"]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Episode state (populated by reset())
        self._molecule   : Optional[torch.Tensor] = None  # (19,) normalised
        self._mask       : Optional[torch.Tensor] = None  # (19,) binary
        self._values     : Optional[torch.Tensor] = None  # (19,) masked values
        self._prev_mse   : float = float("inf")
        self._step       : int   = 0

        # Tracking for info dict
        self._acquired_order : list[int] = []

    # ── Core interface ─────────────────────────────────────────────────────────

    def reset(self, molecule_idx: Optional[int] = None) -> torch.Tensor:
        """
        Start a new episode.

        Parameters
        ----------
        molecule_idx : index into self.molecules. If None, sampled randomly.

        Returns
        -------
        state : FloatTensor (STATE_DIM,) = cat([mask, values])
        """
        if molecule_idx is None:
            molecule_idx = random.randrange(len(self.molecules))

        self._molecule = self.molecules[molecule_idx].to(self.device)  # (19,)
        self._mask     = torch.zeros(self.N_FEATURES, device=self.device)
        self._values   = torch.zeros(self.N_FEATURES, device=self.device)
        self._step     = 0
        self._acquired_order = []

        # Baseline MSE with zero features acquired
        self._prev_mse = self._compute_mse()

        return self._get_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
        """
        Acquire the property at index `action`.

        Parameters
        ----------
        action : int in [0, N_FEATURES). Must not already be acquired.

        Returns
        -------
        state  : FloatTensor (STATE_DIM,) — updated state after acquisition
        reward : float
        done   : bool — True if episode should end
        info   : dict — diagnostic info (mse, acquired_order, etc.)
        """
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
        new_mse    = self._compute_mse()
        delta_acc  = self._prev_mse - new_mse   # positive = improvement
        cost       = 1.0                        # one unit per feature
        reward     = delta_acc - self.lam * cost
        self._prev_mse = new_mse

        # ── Check termination ──────────────────────────────────────────────────
        all_acquired = self._mask.sum().item() == self.N_FEATURES
        max_reached  = self._step >= self.max_steps
        done         = all_acquired or max_reached

        info = {
            "mse"            : new_mse,
            "delta_acc"      : delta_acc,
            "step"           : self._step,
            "n_acquired"     : int(self._mask.sum().item()),
            "acquired_order" : list(self._acquired_order),
            "acquired_names" : [PROPERTY_NAMES[i] for i in self._acquired_order],
        }

        return self._get_state(), reward, done, info

    # ── Legal action masking ───────────────────────────────────────────────────

    def legal_actions(self) -> list[int]:
        """Return indices of properties not yet acquired this episode."""
        return [i for i in range(self.N_FEATURES) if self._mask[i] == 0]

    def legal_action_mask(self) -> torch.Tensor:
        """
        Returns a BoolTensor (N_FEATURES,) where True = not yet acquired.
        Feed this to the DQN to zero-out Q-values for illegal actions before
        taking the argmax.
        """
        return self._mask == 0  # True where not acquired

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_state(self) -> torch.Tensor:
        """Flat state vector: cat([mask, values]) → (STATE_DIM,)."""
        return torch.cat([self._mask, self._values])  # (38,)

    def _compute_mse(self) -> float:
        """
        Ask the predictor for a target estimate given current observations,
        then return MSE against the true (normalised) target.

        If no predictor is attached (predictor=None), returns the variance of
        the target across the training set as a fixed baseline MSE.
        """
        true_target = self._molecule[self.target_idx].item()

        if self.predictor is None:
            # No predictor: treat baseline MSE as 1.0 (normalised variance)
            return 1.0

        with torch.no_grad():
            pred = self.predictor.predict(self._values, self._mask)

        mse = (pred.item() - true_target) ** 2
        return mse

    # ── Utility ────────────────────────────────────────────────────────────────

    def render(self):
        """Print a simple text summary of the current episode state."""
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


# ── Dataset helper: build molecule list from DataLoader ───────────────────────

def build_molecule_list(loader) -> list[torch.Tensor]:
    """
    Flatten a PyG DataLoader into a plain list of (19,) property tensors.
    This is the format AcquisitionEnv expects.

    Usage
    -----
        from data.load_qm9 import load_qm9
        from environment.acquisition_env import build_molecule_list

        train_loader, val_loader, test_loader, stats = load_qm9()
        molecules = build_molecule_list(train_loader)
        env = AcquisitionEnv(molecules, stats, predictor=None)
    """
    molecules = []
    for batch in loader:
        # batch.y shape: (batch_size, 19) — normalised by _NormalisedSubset
        for y in batch.y:
            molecules.append(y.squeeze().float())  # (19,)
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

    # Run one full episode with random actions
    state = env.reset(molecule_idx=0)
    print(f"\nInitial state shape : {state.shape}")
    print(f"STATE_DIM           : {AcquisitionEnv.STATE_DIM}")

    total_reward = 0.0
    done = False
    while not done:
        action = random.choice(env.legal_actions())
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  step {info['step']:>2} | acquired '{PROPERTY_NAMES[action]:<12}' "
              f"| reward={reward:+.5f} | mse={info['mse']:.5f}")

    print(f"\nEpisode complete.")
    print(f"  Total reward    : {total_reward:.4f}")
    print(f"  Features used   : {info['n_acquired']}/{config.NUM_FEATURES}")
    print(f"  Acquisition order: {info['acquired_names']}")
    env.render()

    # Verify legal action masking works
    env.reset()
    env.step(0)
    env.step(1)
    mask = env.legal_action_mask()
    assert mask[0] == False and mask[1] == False and mask[2] == True
    print("\n✅  acquisition_env.py working correctly.")
