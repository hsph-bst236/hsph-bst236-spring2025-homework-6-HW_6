# config.py
import torch
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 5
LEARNING_RATE = 0.2

# Optimizer choices: GD, SGD, RMSprop, Adam
OPTIMIZER = "SGD"

#TODO: Problem 2.3: Scheduler choices: StepLR, CosineAnnealingWarmRestarts, None
SCHEDULER = "None"

# Seed for reproducibility
SEED = 42

# Save directories
CHECKPOINTS_DIR = os.path.join("nonconvex_output", "checkpoints")
RESULTS_DIR = os.path.join("nonconvex_output", "results")
LOGS_DIR = os.path.join("nonconvex_output", "logs")

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

