# config.py


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42


# -----------------------------
# Data
# -----------------------------
BATCH_SIZE = 32
MIN_FREQUENCY = 2
VALIDATION_SIZE = 0.1


# -----------------------------
# Model
# -----------------------------
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

DROPOUT = 0.1


# -----------------------------
# Training
# -----------------------------
LEARNING_RATE = 0.001
EPOCHS = 1


# -----------------------------
# Inference
# -----------------------------
MAX_LEN = 40

BEAM_SIZE = 4
LENGTH_PENALTY_ALPHA = 0.6


# -----------------------------
# Model saving
# -----------------------------
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"