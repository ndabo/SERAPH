"""
models/dqn_agent.py
────────────────────
The DQN agent for SERAPH.

The agent learns a Q-function over (state, action) pairs where:
    state  = cat([mask, values])  →  dim 38  (from acquisition_env.py)
    action = index of the next property to acquire  →  int in [0, 19)

Architecture
------------
Input  : state (dim 38)
Hidden : N fully-connected layers with ReLU
Output : Q-values (dim 19) — one per possible acquisition action

Training follows DOUBLE DQN with legal-action masking:
    1. Agent acts ε-greedily over legal actions, collecting transitions
    2. Transitions stored in a replay buffer
    3. Online net picks next-action; target net evaluates it (Double DQN)
    4. ILLEGAL actions (already acquired, or target) are masked to -inf in
       the Bellman target — critical for stability in this env
    5. Target network synced every TARGET_UPDATE updates
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import config


# ── Replay buffer ──────────────────────────────────────────────────────────────

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = config.REPLAY_SIZE, device: str = config.DEVICE):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(self, state, action, reward, next_state, done):
        """Store one transition. Tensors moved to CPU to save GPU memory."""
        self.buffer.append(Transition(
            state.cpu(), action, reward, next_state.cpu(), done,
        ))

    def sample(self, batch_size: int) -> Transition:
        """Sample a random mini-batch; returns stacked tensors."""
        batch = random.sample(self.buffer, batch_size)

        states      = torch.stack([t.state      for t in batch]).to(self.device)
        actions     = torch.tensor([t.action    for t in batch],
                                   dtype=torch.long, device=self.device)
        rewards     = torch.tensor([t.reward    for t in batch],
                                   dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones       = torch.tensor([t.done      for t in batch],
                                   dtype=torch.float32, device=self.device)

        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        return len(self) >= config.BATCH_SIZE


# ── Policy network ─────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Maps state (38-dim) → Q-values (19-dim, one per action)."""

    def __init__(
        self,
        state_dim:  int = config.NUM_FEATURES * 2,
        action_dim: int = config.NUM_FEATURES,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
    ):
        super().__init__()

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers += [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN Agent ──────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double-DQN agent with experience replay and a target network.

    Parameters
    ----------
    device     : torch device string
    target_idx : index of the target property (always excluded from legal
                 actions). If None, no property is excluded. Required for
                 the Bellman-target illegal-action mask to be correct.
    """

    def __init__(
        self,
        device:     str = config.DEVICE,
        target_idx: Optional[int] = None,
    ):
        self.device     = torch.device(device)
        self.target_idx = target_idx

        # Two networks — same architecture, different weights
        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self._sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.LR
        )
        self.buffer = ReplayBuffer(device=device)

        self.steps_done : int = 0
        self.updates_done: int = 0

    # ── Epsilon schedule ───────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Exponential decay from EPS_START → EPS_END over EPS_DECAY steps."""
        return config.EPS_END + (config.EPS_START - config.EPS_END) * \
               math.exp(-self.steps_done / config.EPS_DECAY)

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(
        self,
        state:        torch.Tensor,
        legal_mask:   torch.Tensor,
        force_greedy: bool = False,
    ) -> int:
        """ε-greedy action selection with legal-action masking."""
        if not force_greedy:
            self.steps_done += 1

        if not force_greedy and random.random() < self.epsilon:
            legal_indices = legal_mask.nonzero(as_tuple=True)[0].tolist()
            return random.choice(legal_indices)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(
                state.unsqueeze(0).to(self.device)
            ).squeeze(0)

        q_values[~legal_mask.to(self.device)] = float("-inf")
        return int(q_values.argmax().item())

    # ── Storing transitions ────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ── Learning step ──────────────────────────────────────────────────────────

    def _legal_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Derive the legal-action mask from a batch of states.

        The state is cat([mask, values]) so state[:, :NUM_FEATURES] is the
        acquisition mask: 1 = acquired (illegal), 0 = not yet acquired (legal).
        The target property is always illegal (it is never acquirable).

        Returns
        -------
        BoolTensor (B, NUM_FEATURES) — True where the action is legal.
        """
        acquired = state_batch[:, :config.NUM_FEATURES]      # (B, 19)
        legal    = (acquired == 0)                            # (B, 19)
        if self.target_idx is not None:
            legal[:, self.target_idx] = False
        return legal

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one Double-DQN gradient update.

        Returns TD loss for this update, or None if buffer not yet ready.
        """
        if not self.buffer.ready:
            return None

        self.online_net.train()
        batch = self.buffer.sample(config.BATCH_SIZE)

        # ── Current Q(s, a) ───────────────────────────────────────────────────
        q_current = self.online_net(batch.state)              # (B, 19)
        q_current = q_current.gather(
            1, batch.action.unsqueeze(1)
        ).squeeze(1)                                          # (B,)

        # ── Double-DQN target with illegal-action masking ────────────────────
        with torch.no_grad():
            # Legal actions at next_state — derived from the mask inside state
            legal_next = self._legal_mask_from_state(batch.next_state)  # (B, 19)

            # Online net chooses action (masking illegal to -inf)
            q_online_next = self.online_net(batch.next_state)           # (B, 19)
            q_online_next = q_online_next.masked_fill(
                ~legal_next, float("-inf")
            )
            # If a row has NO legal actions (should coincide with done=True),
            # argmax returns 0 — we'll zero it out below via has_legal.
            best_actions = q_online_next.argmax(dim=1, keepdim=True)     # (B, 1)

            # Target net evaluates the chosen action
            q_target_next = self.target_net(batch.next_state)            # (B, 19)
            q_next_best   = q_target_next.gather(
                1, best_actions
            ).squeeze(1)                                                 # (B,)

            # Safety: zero the bootstrap if no legal actions exist
            has_legal = legal_next.any(dim=1).float()                    # (B,)
            q_next_best = q_next_best * has_legal

            # Standard Bellman target — no future reward if terminal
            q_target = batch.reward + \
                       config.GAMMA * q_next_best * (1.0 - batch.done)

        # ── Huber loss (smooth L1) ────────────────────────────────────────────
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        # Tight gradient clipping — max_norm=1.0 is the standard DQN value.
        # The old max_norm=10.0 was effectively no clipping.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.updates_done += 1

        if self.updates_done % config.TARGET_UPDATE == 0:
            self._sync_target()

        return loss.item()

    # ── Target network sync ────────────────────────────────────────────────────

    def _sync_target(self):
        """Hard copy: target_net ← online_net weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "steps_done":   self.steps_done,
            "updates_done": self.updates_done,
            "target_idx":   self.target_idx,
        }, path)
        print(f"[DQNAgent] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done   = ckpt["steps_done"]
        self.updates_done = ckpt["updates_done"]
        if ckpt.get("target_idx") is not None:
            self.target_idx = ckpt["target_idx"]
        print(f"[DQNAgent] Loaded ← {path}  "
              f"(step {self.steps_done}, ε={self.epsilon:.3f})")


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.load_qm9 import load_qm9
    from environment.acquisition_env import AcquisitionEnv, build_molecule_list
    from models.predictor import Predictor

    print("Loading data …")
    train_loader, _, _, stats = load_qm9(batch_size=256)
    molecules = build_molecule_list(train_loader)

    print("Building predictor + agent …")
    predictor = Predictor()
    agent     = DQNAgent(target_idx=stats["target_idx"])

    n_params = sum(p.numel() for p in agent.online_net.parameters())
    print(f"  QNetwork params : {n_params:,}")
    print(f"  Replay capacity : {config.REPLAY_SIZE:,}")
    print(f"  Target idx      : {agent.target_idx}")
    print(f"  Device          : {agent.device}")

    env = AcquisitionEnv(molecules, stats, predictor=predictor, seed=42)

    print(f"\nRunning 3 episodes to populate replay buffer …")
    for ep in range(3):
        state = env.reset()
        done, ep_reward, ep_steps = False, 0.0, 0
        while not done:
            legal_mask = env.legal_action_mask()
            action = agent.select_action(state, legal_mask)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
        print(f"  Episode {ep+1} — steps: {ep_steps}, "
              f"total reward: {ep_reward:+.4f}, "
              f"ε: {agent.epsilon:.3f}, buffer: {len(agent.buffer)}")

    print(f"\nFilling buffer to batch size ({config.BATCH_SIZE}) …")
    state = env.reset()
    done = False
    while not agent.buffer.ready:
        legal_mask = env.legal_action_mask()
        action = agent.select_action(state, legal_mask)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        state = env.reset() if done else next_state

    loss = agent.learn()
    print(f"  First learning step — TD loss: {loss:.6f}")

    print("\nGreedy episode …")
    state = env.reset(molecule_idx=0)
    done = False
    while not done:
        legal_mask = env.legal_action_mask()
        action = agent.select_action(state, legal_mask, force_greedy=True)
        state, reward, done, info = env.step(action)
    print(f"  Acquired order : {info['acquired_names']}")

    save_path = os.path.join(config.CHECKPOINT_DIR, "dqn_test.pt")
    agent.save(save_path)
    agent2 = DQNAgent(target_idx=stats["target_idx"])
    agent2.load(save_path)
    print(f"\n  Checkpoint round-trip ✓")
    print("\n✅  dqn_agent.py working correctly.")