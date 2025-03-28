# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from model import NonConvexModel, nonconvex_objective
from train import Trainer
from config_nonconvex import *
from utils import plot_parameter_trajectory_on_contour

#TODO: Problem 2.1 Define a new trainer class inheriting from `Trainer` for nonconvex optimization
class NonconvexTrainer(Trainer):
    """
    A custom trainer that overrides the _evaluate_model method to directly 
    use the nonconvex objective function rather than using validation data.
    """
    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize list to track parameter values during training
        self.parameter_history = []
        self.scheduler = scheduler
        
    def _evaluate_model(self, loader=None, pbar=None):
        """
        Override the evaluation method to directly compute the nonconvex objective
        without using validation data.
        """
        if loader is None:
            loader = self.val_loader  # Default to validation loader if none specified
        
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            #TODO: Update the objective_value by evaluating nonconvex_objective at self.model.a            
            
            # Log the objective value
            if pbar:
                pbar.set_postfix(objective=f"{objective_value.item():.6f}")
            
            # Return the objective value as the validation loss
            return objective_value.item()
    
    def _train_epoch(self, epoch, pbar=None):
        """Train the model for one epoch."""
        self.model.train()  # Set the model to training mode
        
        # Store current parameter value before training step
        self.parameter_history.append(self.model.a.clone().detach())
        
        current_loss = nonconvex_objective(self.model.a)
        running_loss = current_loss.item()

        if self.hyperparams['OPTIMIZER'] != 'GD':
            # Here for stochastic optimizers, we want to add noise zeta ~ N(0, 1/sqrt(batch_size)) to the gradient
            # Instead of rewrite the optimizer, we can revise the loss function f(a) -> f(a) + zeta * sum(a) 
            # then the gradient will be grad(f(a)) + zeta
            zeta = torch.randn((), device=self.device) / (self.hyperparams['BATCH_SIZE'] ** 0.5)
            current_loss += torch.sum(self.model.a) * zeta
        
        #TODO: Update the model parameters using the optimizer


        # Step the scheduler if it exists
        if self.scheduler:
            self.scheduler.step()

        # Update progress bar with current loss if provided
        if pbar:
            pbar.update(1)
            pbar.set_postfix(train_loss=f"{current_loss.item():.3f}")

        return running_loss
    
    def train(self):
        """Override the train method to add parameter trajectory visualization"""
        #TODO: Call the parent class's train method
        
        #TODO: Plot the parameter trajectory on the contour using `plot_parameter_trajectory_on_contour` from `utils.py`



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
        'SCHEDULER': SCHEDULER,
        'OPTIMIZER': OPTIMIZER,  # Add OPTIMIZER to the hyperparams
        'CHECKPOINTS_DIR': CHECKPOINTS_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'LOGS_DIR': LOGS_DIR,
        'SEED': SEED,
    }

    #TODO: Problem 2.2 Parse learning rate and optimizer from command line arguments and override config values if provided

    # Initialize the model
    model = NonConvexModel().to(DEVICE)
    
    # Select optimizer and loss function based on OPTIMIZER value
    # ISTA and FISTA treat the L1 regularization as a proximal operator so the their loss is just MSELoss
    # GD, SGD and Adam treat the L1 regularization as the part of the loss function so autograd will take care of everything
    if hyperparams['OPTIMIZER'] == "GD":
        optimizer = optim.SGD(model.parameters(), lr=hyperparams['LEARNING_RATE'])
    elif hyperparams['OPTIMIZER'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=hyperparams['LEARNING_RATE'])
    elif hyperparams['OPTIMIZER'] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=hyperparams['LEARNING_RATE'])
    elif hyperparams['OPTIMIZER'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['LEARNING_RATE'])    
    else:
        raise ValueError(f"Invalid optimizer: {hyperparams['OPTIMIZER']}")
    
    #Problem 2.3: Create scheduler if specified
    scheduler = None
    if hyperparams['SCHEDULER'] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif hyperparams['SCHEDULER'] == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=NUM_EPOCHS//10)
    else:
        raise ValueError(f"Invalid scheduler: {hyperparams['SCHEDULER']}")

    # Create a trainer instance with logging enabled, using our custom NonconvexTrainer
    trainer = NonconvexTrainer(
        model=model,
        train_loader= [1], # dummy data; not used for training
        val_loader= None,
        loss_fn= None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        use_logger=True,  # Enable logging
        save_frequency=NUM_EPOCHS,
        hyperparams=hyperparams,
        git_commit=True,   # Enable auto git commit after training
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 