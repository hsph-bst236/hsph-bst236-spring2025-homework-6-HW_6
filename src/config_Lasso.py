# config.py
import torch
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 1000
LEARNING_RATE = 0.01

# Lasso parameters
L1_REG = 0.2

# Optimizer choices: ISTA, FISTA, GD, SGD, Adam
OPTIMIZER = "GD"

# Linear Regression Dataset parameters
NUM_SAMPLES = 5000
# Model parameters for linear regression
INPUT_DIM = 1000
OUTPUT_DIM = 1
SPARSITY = 0.02

# Seed for reproducibility
SEED = 42

# Save directories
CHECKPOINTS_DIR = os.path.join("lasso_output", "checkpoints")
RESULTS_DIR = os.path.join("lasso_output", "results")
LOGS_DIR = os.path.join("lasso_output", "logs")

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

