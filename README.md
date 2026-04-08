# SERAPH 🧬
**Sequential RL Acquisition of Molecular Properties**

An RL agent that learns *which* quantum-chemical properties of a molecule to observe—and in what order—to predict a target property accurately while minimising acquisition cost.

---

## Project Structure

```
SERAPH/
├── config.py                          # all hyperparameters in one place
├── requirements.txt
│
├── data/
│   └── load_qm9.py                    # QM9 loading & preprocessing
│
├── environment/
│   └── acquisition_env.py             # RL environment (state/action/reward)
│
├── models/
│   ├── predictor.py                   # MLP predictor (masked inputs)
│   └── dqn_agent.py                   # DQN policy network + replay buffer
│
├── training/
│   ├── train_baseline.py              # full-information MLP baseline
│   └── train_dqn.py                   # main RL training loop
│
├── evaluation/
│   └── evaluate.py                    # accuracy-vs-cost curves & ablations
│
├── visualization/
│   └── visualize.py                   # acquisition order + molecular graphs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_rl_training_and_eval.ipynb
│   └── 04_visualization_and_interpretability.ipynb
│
├── checkpoints/                       # saved model weights (git-ignored)
└── results/                           # plots & metrics (git-ignored)
    ├── plots/
    └── metrics/
```

---

## Setup

> **Platform note:** macOS does not support CUDA. PyTorch runs on CPU or Apple Silicon GPU (MPS). Linux/Windows users with an NVIDIA GPU can substitute `+cpu` with `+cu121` (or your CUDA version) in Step 3.

### 1 — Clone the repo
```bash
git clone git@github.com:ndabo/SERAPH.git
cd SERAPH
```

### 2 — Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies (order matters)

**a) PyTorch + torchvision**
```bash
pip install "torch>=2.2.0" "torchvision>=0.17.0"
```

**b) PyTorch Geometric + extensions** (wheels must match your torch version)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

**c) Everything else**
```bash
pip install rdkit networkx numpy pandas scikit-learn tqdm gymnasium matplotlib seaborn jupyter ipykernel
```

### 4 — Register the Jupyter kernel
```bash
python -m ipykernel install --user --name seraph --display-name "SERAPH"
jupyter notebook
```

---

## Quickstart

```bash
# Train the full-information baseline
python training/train_baseline.py

# Train the DQN agent
python training/train_dqn.py

# Run ablation study (RL vs random vs greedy)
python evaluation/evaluate.py

# Generate visualizations
python visualization/visualize.py
```

---

## Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_PROP` | `"gap"` | QM9 property to predict (HOMO-LUMO gap) |
| `LAMBDA` | `0.1` | Cost-accuracy tradeoff weight |
| `LR` | `1e-3` | Learning rate |
| `BATCH_SIZE` | `64` | Replay buffer batch size |
| `GAMMA` | `0.99` | DQN discount factor |
| `EPS_START` | `1.0` | ε-greedy starting exploration |
| `EPS_END` | `0.05` | ε-greedy minimum |
| `REPLAY_SIZE` | `10000` | Replay buffer capacity |

---

## Team
<!-- Add your names here -->

---

## References
- [QM9 Dataset](https://doi.org/10.1038/sdata.2014.22)
- [PyTorch Geometric QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html)
- [DQN (Mnih et al. 2015)](https://www.nature.com/articles/nature14236)
