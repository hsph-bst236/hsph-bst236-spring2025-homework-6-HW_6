# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from model import LinearRegressionModel
from train import Trainer, ISTA, FISTA
from dataset import get_loaders
from config_Lasso import *

#TODO: Problem 1.2
def mse_with_l1_reg(y_pred, y_target, model, l1_reg):
    #TODO: Define the loss function of Least squares with L1 regularization: 
    # \frac{1}{2n}\|y_target - y_pred\|^2 + \lambda\|w\|_1
    # Hint: Use nn.MSELoss() to compute the mean squared error between predictions and targets
    # Hint: Use `for p in model.parameters()` to get the parameters of the model



def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Create hyperparameters dictionary
    hyperparams = {
        'DEVICE': DEVICE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_SAMPLES': NUM_SAMPLES,
        'INPUT_DIM': INPUT_DIM,
        'OUTPUT_DIM': OUTPUT_DIM,
        'SPARSITY': SPARSITY,
        'L1_REG': L1_REG,  # Make sure to include L1_REG
        'OPTIMIZER': OPTIMIZER,  # Add OPTIMIZER to the hyperparams
        'CHECKPOINTS_DIR': CHECKPOINTS_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'LOGS_DIR': LOGS_DIR,
        'SEED': SEED
    }

    # Parse command line arguments and override config values if provided

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["GD", "SGD", "Adam", "ISTA", "FISTA"], 
                        help="Optimizer type (GD, SGD, Adam, ISTA, FISTA)")
    args = parser.parse_args()

    #TODO: Problem 1.4:Overwrite hyperparams['LEARNING_RATE'] and hyperparams['OPTIMIZER'] using command line arguments if provided, otherwise use config values


    # Initialize the model
    model = LinearRegressionModel(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    
    # Select optimizer and loss function based on OPTIMIZER value
    # ISTA and FISTA treat the L1 regularization as a proximal operator so the their loss is just MSELoss
    # GD, SGD and Adam treat the L1 regularization as the part of the loss function so autograd will take care of everything
    if hyperparams['OPTIMIZER'] == "ISTA":
        optimizer = ISTA(model.parameters(), lr=hyperparams['LEARNING_RATE'], l1_reg=hyperparams['L1_REG'])
        loss_fn = nn.MSELoss()
        hyperparams['BATCH_SIZE'] = NUM_SAMPLES # We use all samples for deterministic optimizators
    elif hyperparams['OPTIMIZER'] == "FISTA":
        optimizer = FISTA(model.parameters(), lr=hyperparams['LEARNING_RATE'], l1_reg=hyperparams['L1_REG'])
        loss_fn = nn.MSELoss()
        hyperparams['BATCH_SIZE'] = NUM_SAMPLES
    elif hyperparams['OPTIMIZER'] == "GD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loss_fn = lambda pred, target: mse_with_l1_reg(pred, target, model, L1_REG)
        hyperparams['BATCH_SIZE'] = NUM_SAMPLES
    elif hyperparams['OPTIMIZER'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=hyperparams['LEARNING_RATE'])
        loss_fn = lambda pred, target: mse_with_l1_reg(pred, target, model, hyperparams['L1_REG'])
    elif hyperparams['OPTIMIZER'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = lambda pred, target: mse_with_l1_reg(pred, target, model, L1_REG)
    else:
        raise ValueError(f"Invalid optimizer: {OPTIMIZER}")
    
    # Create data loaders for training and validation
    train_loader, val_loader = get_loaders(
        num_samples=hyperparams['NUM_SAMPLES'],
        batch_size=hyperparams['BATCH_SIZE'],
        input_dim=hyperparams['INPUT_DIM'],
        sparsity=hyperparams['SPARSITY']
    )
    
    # Create a trainer instance with logging enabled
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        use_logger=True,  # Enable logging
        save_frequency=50,
        hyperparams=hyperparams,
        git_commit=True,   # Enable auto git commit after training
        l1_reg=L1_REG      # Pass L1 regularization parameter for loss calculation
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 