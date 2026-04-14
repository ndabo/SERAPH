# SERAPH Documentation

---

## `data/load_qm9.py`

### `PROPERTY_NAMES` / `PROPERTY_INDEX` (lines 32–57)

QM9 has 19 quantum-chemical properties per molecule stored as columns in `data.y`. This maps human-readable names (e.g. `"gap"`) to their column index so the rest of the code can refer to properties by name instead of magic numbers.

---

### `load_qm9()` (lines 62–155)

The main function. Call this to get DataLoaders ready for training.

```python
train_loader, val_loader, test_loader, stats = load_qm9()
```

1. **Downloads QM9** via PyG on first run (~1.7 GB, cached after that)
2. **Splits** the ~130k molecules into train/val/test using fractions from `config.py`, with a fixed random seed for reproducibility
3. **Computes normalization stats** (mean + std for all 19 properties) using only the training set — important to avoid data leakage from val/test
4. **Wraps each split** in `_NormalisedSubset` so every molecule's `.y` comes out standardized, and returns DataLoaders ready for training

The returned `stats` dict contains:
- `stats["mean"]` — `(19,)` tensor of per-property means
- `stats["std"]` — `(19,)` tensor of per-property standard deviations
- `stats["target_idx"]` — column index of the target property
- `stats["target"]` — string name of the target property

---

### `_NormalisedSubset` (lines 160–180)

A thin wrapper around PyG's `Subset`. When you index into it, it clones the data object and applies `(y - mean) / std`. It also saves the original values as `.y_raw` so you can un-normalize predictions later for interpretability.

---

### `extract_property_matrix()` (lines 185–193)

A utility that flattens an entire DataLoader into a single `(N, 19)` tensor of normalized properties. Useful for the RL environment when it needs all properties of a molecule at once.

---

---

## `data/processed/`

### `data_v3.pt` (324 MB)

The main cache. PyG reads all 133,885 molecules from `gdb9.sdf`, converts each one into a graph (atoms as nodes, bonds as edges), attaches the 19 properties from `gdb9.sdf.csv`, excludes the bad molecules from `uncharacterized.txt`, and saves the result as a single PyTorch tensor file. On every run after the first, PyG loads this directly instead of re-parsing the SDF file — which is why startup is fast after the first run.

### `pre_filter.pt` and `pre_transform.pt`

Small metadata files that record what filtering and transformations were applied during processing. PyG checks these on each run — if they match, it reuses `data_v3.pt`. If you change the transform settings, PyG detects the mismatch and reprocesses from scratch.

---

---

## `environment/acquisition_env.py`

### What is an episode?

An episode is one molecule from start to finish.

At the start of each episode (`reset()`), the agent is handed a single molecule with all 19 properties hidden — `mask` is all zeros, `values` is all zeros.

At each step, the agent picks one property to "acquire" (reveal). The environment uncovers that property's value, runs the predictor to see how much better the prediction got, and returns a reward:

```
reward = Δaccuracy − λ × cost
```

The agent gains reward when revealing a property meaningfully improves the prediction, and gets penalized just for asking (cost = 1 per feature). This forces it to learn which properties are actually worth looking at.

The episode ends when either:
- the agent has acquired all 19 properties, or
- it hits `max_steps` (also 19 by default)

One episode = the agent deciding the optimal order to reveal properties of one molecule, trying to predict the HOMO-LUMO gap as accurately as possible while revealing as few properties as possible.

Over training, the agent runs thousands of these episodes across different molecules and learns a policy: "given what I've seen so far, which property should I look at next?"

---

### `__main__` block (lines 198–214)

A smoke test — run the file directly to verify everything works and print normalization stats for all 19 properties:

```
python environment/acquisition_env.py
```

---

## `models/predictor.py`

### What it does

A simple MLP (multi-layer perceptron) that takes the current state of an episode and predicts the HOMO-LUMO gap. The input is the same 38-dim vector from the environment:

```
cat([mask(19), values(19)]) → hidden layers → scalar prediction
```

Architecture:
```
38 → Linear → ReLU → Dropout
   → Linear → ReLU → Dropout   (repeated num_layers times)
   → Linear → scalar
```

It serves two roles in SERAPH:

1. **Inside the RL loop** — after every acquisition, `AcquisitionEnv` calls `predictor.predict(values, mask)` to get the new MSE and compute the reward
2. **As a standalone baseline** — trained with all 19 features visible (mask = all ones) to get the best possible accuracy — this is the ceiling the RL agent is trying to match with fewer features

---

### `predict()` vs `forward()`

- `forward(x)` — standard PyTorch batch inference, takes a `(B, 38)` tensor, returns `(B, 1)`. Used during training.
- `predict(values, mask)` — convenience wrapper for the environment. Takes the two separate `(19,)` tensors, concatenates them, runs inference, returns a scalar. Used inside `_compute_mse()`.

---

### `build_xy()`

Builds the training data for the **full-information baseline**. Takes the molecule list and constructs `X` where every molecule has `mask = all ones` (pretending all 19 properties are known), and `y` is the target property. Used in `train_baseline.py`.

---

### `train_one_epoch()`

Standard mini-batch gradient descent — shuffles the data with `randperm`, loops through batches, computes MSE loss, backpropagates. Returns the average loss for the epoch.

---

### How it connects to everything else

```
load_qm9()  →  build_molecule_list()  →  build_xy()  →  train_one_epoch()
                                      →  AcquisitionEnv(predictor=model)
                                                ↓
                                         predict() called each step
                                         to compute reward
```

---

### Smoke test

```
python models/predictor.py
```

---

## `models/dqn_agent.py`

### The big picture

The DQN agent's job is to look at the current state of an episode (which properties have been revealed so far) and decide which property to acquire next. It learns this through trial and error across thousands of episodes.

The file has three main components: **ReplayBuffer**, **QNetwork**, and **DQNAgent**.

---

### `ReplayBuffer`

Every time the agent takes a step in the environment, that experience gets stored as a transition:
```
(state, action, reward, next_state, done)
```
The buffer holds up to `REPLAY_SIZE` (10,000) of these. When it's full, the oldest ones get overwritten. During training, random mini-batches are sampled from it — this breaks the correlation between consecutive steps, which is crucial for stable learning.

Transitions are stored on CPU to save GPU memory, and moved to MPS/CUDA only when sampled for training.

---

### `QNetwork`

The neural network that learns Q-values:
```
state (38-dim) → Linear → ReLU → ... → Q-values (19-dim)
```

It outputs one Q-value per property. The Q-value for an action represents: *"how much total future reward can I expect if I acquire this property right now?"*

The agent picks the action with the highest Q-value.

---

### `DQNAgent`

The main class that ties everything together. It maintains **two copies** of QNetwork:

- **`online_net`** — trained every step, this is the network that's actively learning
- **`target_net`** — a frozen snapshot of the online net, only updated every `TARGET_UPDATE` steps

Why two networks? Without the target net, you'd be chasing a moving target during training — the values you're trying to predict keep changing as you update the network, which causes instability.

---

### `select_action()` — ε-greedy

At each step the agent either:
- **Explores** (with probability ε): picks a random legal action
- **Exploits** (with probability 1-ε): picks the action with the highest Q-value

ε starts at 1.0 (fully random) and exponentially decays toward 0.05 as training progresses. Illegal actions (already-acquired properties) are masked out with `-inf` so they can never be chosen.

---

### `learn()` — the Bellman update

This is where actual learning happens. It samples a mini-batch from the buffer and computes:

```
Q_target = reward + γ × max(Q_target_net(next_state))   # if not done
Q_target = reward                                         # if done
```

Then it minimizes the difference between what the online net predicted (`Q_current`) and what it should have predicted (`Q_target`) using Huber loss. Gradient clipping is applied to prevent unstable updates early in training.

---

### How it all connects

```
Episode loop:
  state = env.reset()
  while not done:
      action     = agent.select_action(state, legal_mask)   # ε-greedy
      next_state, reward, done = env.step(action)
      agent.store(state, action, reward, next_state, done)  # → ReplayBuffer
      agent.learn()                                          # sample batch, update online_net
      every TARGET_UPDATE steps → sync target_net ← online_net
```

---

### Smoke test

```
python models/dqn_agent.py
```

---

## `training/train_baseline.py`

### What it does

Trains the full-information MLP baseline — the upper-bound model that sees all 19 QM9 properties at once (mask = all ones) and predicts the target property. It answers:

> "What's the best accuracy we could ever achieve if we always acquired every single feature?"

The RL agent's job is to get close to this number using far fewer features. **This script must be run before `train_dqn.py`** — the trained predictor is plugged into the RL environment to generate meaningful reward signals.

---

### Outputs

| File | Description |
|------|-------------|
| `checkpoints/predictor_baseline.pt` | Best checkpoint (lowest val MSE) |
| `results/metrics/baseline_metrics.json` | Per-epoch train/val/test metrics |

---

### Training loop

1. Loads QM9 and builds train/val/test molecule lists
2. Builds `(X, y)` tensors with `mask = all ones` via `build_xy()` — full information baseline
3. Trains the `Predictor` MLP with Adam + `ReduceLROnPlateau` scheduler
4. Saves checkpoint whenever val MSE improves
5. At the end, loads the best checkpoint and evaluates on the test set

---

### Key metrics

- **MSE(norm)** — what the model trains on, in normalized space
- **MAE(real)** — the number that matters; average prediction error in original Hartree units

**Current best result: MAE(real) = 0.058 Hartree** (lr=3e-4, 50 epochs). This is the ceiling the RL agent is trying to approach with fewer acquisitions.

---

### Command line usage

```bash
python training/train_baseline.py                         # uses config.py defaults
python training/train_baseline.py --lr 3e-4 --epochs 50  # override hyperparams
python training/train_baseline.py --target homo           # train on a different property
```


## `training/train_dqn.py`

### What it does

Trains the DQN agent. Mostly wraps dqn_agent

---

### Training loop

1. For each episode runs the process described in ## `models/dqn_agent.py` under ### How it all connects
2. Tracks and stores avg_loss and reward per episode
3. Printss metrics and saves evaluation from `evaluate.py` results determined by eval_every

## `evaluate.py`

### What it does

Evaluates performance of the dqn_agent by tracking the mean total rewards per episode using greedy actions
