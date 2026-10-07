"""Hyperparameters and configuration constants."""
import torch

# Device (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STATE_DIM = 384          # hidden size of all-MiniLM-L6-v2
MAX_TOKENS = 256

# Training
BATCH_SIZE = 8
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 10
LR_PRETRAIN = 5e-5
LR_FINETUNE = 2e-5
ACCUM_STEPS = 4
PRETRAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT = 0.2
NUM_BELIEF_PATHS = 1
VAL_SPLIT = 0.1
THRESHOLD_PERCENTILE = 75
SEED = 42

# Files
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
OUTPUT_CSV = "RES1.csv"
