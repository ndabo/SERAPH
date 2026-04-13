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

Training follows standard DQN:
    1. Agent acts ε-greedily, collecting (s, a, r, s', done) transitions
    2. Transitions are stored in a replay buffer
    3. Mini-batches are sampled from the buffer to train the online network
    4. A separate target network (updated every K steps) stabilises training

The legal action mask from AcquisitionEnv is applied before argmax so the
agent never attempts to acquire an already-acquired property.
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
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.

    Parameters
    ----------
    capacity   : maximum number of transitions stored (oldest overwritten)
    device     : transitions are returned on this device
    """

    def __init__(self, capacity: int = config.REPLAY_SIZE, device: str = config.DEVICE):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(
        self,
        state:      torch.Tensor,
        action:     int,
        reward:     float,
        next_state: torch.Tensor,
        done:       bool,
    ):
        """Store one transition. Tensors are moved to CPU to save GPU memory."""
        self.buffer.append(Transition(
            state.cpu(),
            action,
            reward,
            next_state.cpu(),
            done,
        ))

    def sample(self, batch_size: int) -> Transition:
        """
        Sample a random mini-batch of transitions.

        Returns a Transition of stacked tensors, each shape (batch_size, ...).
        """
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
        """True once the buffer holds at least one full batch."""
        return len(self) >= config.BATCH_SIZE


# ── Policy network ─────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    The DQN policy network.

    Maps state (38-dim) → Q-values (19-dim, one per action).

    Parameters
    ----------
    state_dim  : input dimension (default 38 = 2 × NUM_FEATURES)
    action_dim : output dimension (default 19 = NUM_FEATURES)
    hidden_dim : width of each hidden layer
    num_layers : number of hidden layers
    """

    def __init__(
        self,
        state_dim:  int = config.NUM_FEATURES * 2,   # 38
        action_dim: int = config.NUM_FEATURES,        # 19
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
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (B, 38) or (38,)

        Returns
        -------
        FloatTensor (B, 19) — Q-value for each action
        """
        return self.net(x)


# ── DQN Agent ──────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with experience replay and a target network.

    Maintains two networks:
        online_net  — trained every step via gradient descent
        target_net  — frozen copy, synced every TARGET_UPDATE steps

    The Bellman target uses the target network for stability:
        Q_target = r  +  γ · max_a Q_target(s', a)   (if not done)
        Q_target = r                                   (if done)

    Parameters
    ----------
    device : torch device string
    """

    def __init__(self, device: str = config.DEVICE):
        self.device = torch.device(device)

        # Two networks — same architecture, different weights
        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self._sync_target()            # target starts as a copy of online
        self.target_net.eval()         # target never trains directly

        self.optimizer  = torch.optim.Adam(
            self.online_net.parameters(), lr=config.LR
        )
        self.buffer     = ReplayBuffer(device=device)

        # Training counters
        self.steps_done : int   = 0    # total env steps taken
        self.updates_done: int  = 0    # total gradient updates

    # ── Epsilon schedule ───────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """
        Exponentially decaying ε — starts at EPS_START, decays to EPS_END
        over EPS_DECAY steps. Higher ε = more random exploration.
        """
        eps = config.EPS_END + (config.EPS_START - config.EPS_END) * \
              math.exp(-self.steps_done / config.EPS_DECAY)
        return eps

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(
        self,
        state:        torch.Tensor,
        legal_mask:   torch.Tensor,
        force_greedy: bool = False,
    ) -> int:
        """
        ε-greedy action selection with legal action masking.

        Parameters
        ----------
        state        : FloatTensor (38,) — current environment state
        legal_mask   : BoolTensor (19,)  — True where action is legal
                       (from env.legal_action_mask())
        force_greedy : if True, always pick the greedy action (for evaluation)

        Returns
        -------
        int — index of chosen action
        """
        self.steps_done += 1

        # Exploration: random legal action
        if not force_greedy and random.random() < self.epsilon:
            legal_indices = legal_mask.nonzero(as_tuple=True)[0].tolist()
            return random.choice(legal_indices)

        # Exploitation: argmax Q-value over legal actions
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(
                state.unsqueeze(0).to(self.device)
            ).squeeze(0)   # (19,)

        # Mask illegal actions with -inf so they can never be chosen
        q_values[~legal_mask.to(self.device)] = float("-inf")
        return int(q_values.argmax().item())

    # ── Storing transitions ────────────────────────────────────────────────────

    def store(
        self,
        state:      torch.Tensor,
        action:     int,
        reward:     float,
        next_state: torch.Tensor,
        done:       bool,
    ):
        """Push one transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ── Learning step ──────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch from the replay buffer and perform one gradient
        update on the online network.

        Returns
        -------
        float — TD loss for this update, or None if buffer not yet ready.
        """
        if not self.buffer.ready:
            return None

        self.online_net.train()
        batch = self.buffer.sample(config.BATCH_SIZE)

        # ── Current Q-values  Q(s, a) ─────────────────────────────────────────
        # online_net produces Q for all actions; we select the taken action
        q_current = self.online_net(batch.state)                 # (B, 19)
        q_current = q_current.gather(
            1, batch.action.unsqueeze(1)
        ).squeeze(1)                                             # (B,)

        # ── Target Q-values  r + γ · max_a Q_target(s', a) ───────────────────
        with torch.no_grad():
            q_next = self.target_net(batch.next_state)           # (B, 19)
            q_next_max = q_next.max(dim=1).values                # (B,)

            # If episode ended, there is no future reward
            q_target = batch.reward + \
                       config.GAMMA * q_next_max * (1.0 - batch.done)

        # ── Huber loss (smooth L1 — more robust to outliers than MSE) ─────────
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()
        self.updates_done += 1

        # ── Periodically sync target network ──────────────────────────────────
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
        }, path)
        print(f"[DQNAgent] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done   = ckpt["steps_done"]
        self.updates_done = ckpt["updates_done"]
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
    agent     = DQNAgent()

    n_params = sum(p.numel() for p in agent.online_net.parameters())
    print(f"  QNetwork params : {n_params:,}")
    print(f"  Replay capacity : {config.REPLAY_SIZE:,}")
    print(f"  Device          : {agent.device}")

    # ── Run a few episodes to populate the replay buffer ──────────────────────
    env = AcquisitionEnv(molecules, stats, predictor=predictor, seed=42)

    print(f"\nRunning 3 episodes to populate replay buffer …")
    for ep in range(3):
        state = env.reset()
        done  = False
        ep_reward, ep_steps = 0.0, 0

        while not done:
            legal_mask          = env.legal_action_mask()
            action              = agent.select_action(state, legal_mask)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)

            state      = next_state
            ep_reward += reward
            ep_steps  += 1

        print(f"  Episode {ep+1} — steps: {ep_steps}, "
              f"total reward: {ep_reward:+.4f}, "
              f"ε: {agent.epsilon:.3f}, "
              f"buffer: {len(agent.buffer)}")

    # ── Trigger a learning step ────────────────────────────────────────────────
    # Buffer won't be ready yet with only 3 short episodes; fill it minimally
    print(f"\nFilling buffer to batch size ({config.BATCH_SIZE}) …")
    state = env.reset()
    done  = False
    while not agent.buffer.ready:
        legal_mask              = env.legal_action_mask()
        action                  = agent.select_action(state, legal_mask)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        state = env.reset() if done else next_state

    loss = agent.learn()
    print(f"  First learning step — TD loss: {loss:.6f}")
    print(f"  Updates done: {agent.updates_done}")

    # ── Greedy episode (no exploration) ───────────────────────────────────────
    print("\nGreedy episode (ε=0, force_greedy=True) …")
    state = env.reset(molecule_idx=0)
    done  = False
    while not done:
        legal_mask              = env.legal_action_mask()
        action                  = agent.select_action(
                                    state, legal_mask, force_greedy=True)
        state, reward, done, info = env.step(action)

    print(f"  Acquired order : {info['acquired_names']}")
    print(f"  Features used  : {info['n_acquired']}/{config.NUM_FEATURES}")
    env.render()

    # ── Checkpoint round-trip ──────────────────────────────────────────────────
    save_path = os.path.join(config.CHECKPOINT_DIR, "dqn_test.pt")
    agent.save(save_path)
    agent2 = DQNAgent()
    agent2.load(save_path)
    print(f"\n  Checkpoint round-trip ✓")

    print("\n✅  dqn_agent.py working correctly.")