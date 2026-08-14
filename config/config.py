from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# ==========================
# Dataset
# ==========================

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

# ==========================
# Training
# ==========================

EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
SAVE_BEST_ONLY = True
# ==========================
# Model
# ==========================

MODEL_NAME = "vit_base_patch16_224"

# Your dataset has AD, CI and CN
NUM_CLASSES = 3

# ==========================
# Device
# ==========================

DEVICE = "cuda"