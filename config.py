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
NUM_HEADS = 8
D_FF = 512
INITIALIZATION = "default"
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

DROPOUT = 0.1

# -----------------------------
# Training
# -----------------------------
LEARNING_RATE = 0.00075
EPOCHS = 10
LR_SCHEDULE = "constant"

WARMUP_STEPS = 0
LABEL_SMOOTHING = 0.0

# -----------------------------
# Inference
# -----------------------------
MAX_LEN = 40
BLEU_MAX_EXAMPLES = 100
BEAM_SIZE = 4
LENGTH_PENALTY_ALPHA = 0.6


# -----------------------------
# Model saving
# -----------------------------
EXPERIMENT_NAME = EXPERIMENT_NAME = (
    f"d{D_MODEL}_"
    f"h{NUM_HEADS}_"
    f"enc{NUM_ENCODER_LAYERS}_"
    f"dec{NUM_DECODER_LAYERS}_"
    f"ff{D_FF}_"
    f"bs{BATCH_SIZE}_"
    f"lr{LEARNING_RATE}"
     f"init{INITIALIZATION}"
      f"sched{LR_SCHEDULE}"
)
CHECKPOINT_PATH = (
    "/content/drive/MyDrive/"
    "transformer-translation/checkpoints/"
    f"{EXPERIMENT_NAME}.pt"
)

EXPERIMENT_DIR = (
    "/content/drive/MyDrive/"
    "transformer-translation/experiments"
)

PLOT_DIR = (
    "/content/drive/MyDrive/"
    "transformer-translation/plots"
)