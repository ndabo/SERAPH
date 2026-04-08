# config.py — all SERAPH hyperparameters in one place
# ─────────────────────────────────────────────────────────────────────────────

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_ROOT      = "./data"          # QM9 will download here on first run
TARGET_PROP    = "gap"             # which QM9 property to predict
#   Options: mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298,
#            g298, cv, u0_atom, u298_atom, h298_atom, g298_atom, a, b, c

NUM_FEATURES   = 19                # total acquirable properties in QM9
TRAIN_FRAC     = 0.8
VAL_FRAC       = 0.1
# (remainder is test)

# ── RL Environment ────────────────────────────────────────────────────────────
LAMBDA         = 0.1               # cost-accuracy tradeoff (sweep for ablation)
MAX_STEPS      = NUM_FEATURES      # episode ends when all features acquired
                                   # (agent can stop early via terminal action)

# ── DQN Agent ─────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 128               # FC layer width in policy network
NUM_LAYERS     = 2                 # number of hidden layers
LR             = 1e-3
GAMMA          = 0.99              # discount factor
EPS_START      = 1.0               # ε-greedy exploration start
EPS_END        = 0.05
EPS_DECAY      = 5000              # steps over which ε decays
REPLAY_SIZE    = 10_000            # replay buffer capacity
BATCH_SIZE     = 64
TARGET_UPDATE  = 500               # steps between target-network syncs

# ── Predictor MLP ─────────────────────────────────────────────────────────────
PRED_HIDDEN    = 64
PRED_LAYERS    = 2
PRED_LR        = 1e-3
PRED_EPOCHS    = 50                # pre-training epochs for the predictor

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPISODES   = 5_000
SEED           = 42
DEVICE         = "mps"            # set to "cpu" if no GPU

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

# ── Ablation λ sweep ──────────────────────────────────────────────────────────
LAMBDA_SWEEP   = [0.01, 0.05, 0.1, 0.2, 0.5]
