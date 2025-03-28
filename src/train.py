# train.py
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import datetime
import subprocess
import os
import sys
from utils import *
from tqdm import tqdm

class ISTA(Optimizer):
    def __init__(self, params, lr=1e-3, l1_reg=1e-3):
        """
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            l1_reg (float): L1 regularization strength.
        """
        defaults = dict(lr=lr, l1_reg=l1_reg)
        super(ISTA, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:
            lr = group['lr']
            l1_reg = group['l1_reg']
            
            with torch.no_grad():
                for param in group['params']:
                    if param.grad is None:
                        continue
                    # Perform a gradient descent step:
                    z = param - lr * param.grad
                    # Apply soft-thresholding (proximal operator for L1)
                    # Soft thresholding: S(x, alpha) = sign(x) * max(|x| - alpha, 0)
                    param.copy_(torch.sign(z) * torch.clamp(torch.abs(z) - lr * l1_reg, min=0.0))


#TODO: Problem 1.3: Complete the FISTA optimizer following the instructions and hints inside the class
class FISTA(Optimizer):
    def __init__(self, params, lr=1e-3, l1_reg=1e-3):
        """
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            l1_reg (float): L1 regularization strength.
        """
        # Initialize the defaults
        defaults = dict(lr=lr, l1_reg=l1_reg, lambda_t=1.0)
        super(FISTA, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single FISTA optimization step.
        """

        for group in self.param_groups:
            """
            Hint:group is a dictionary containing optimizer parameters and their values
            As in __init__, we initialize the defaults for the optimizer parameters
            super(FISTA, self).__init__(params, defaults)
            So each group contains:
            {
                'params': [parameter tensors],  # List of model parameters to optimize
                'lr': learning_rate,           # Learning rate for this parameter group
                'l1_reg': l1_regularization,   # L1 regularization strength
                'lambda_t': momentum_term      # FISTA-specific momentum scalar
            }
            """
            lr = group['lr']
            l1_reg = group['l1_reg']
            # Retrieve the old momentum scalar
            lambda_old = group['lambda_t'] 
            #TODO: Compute the new momentum parameter lambda_new:  \lambda_{t+1} = \frac{1 + \sqrt{1+4\lambda_{t}^2}}{2} and update and update group['lambda_t'] as lambda_new

            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue

                    """
                    Hint: self.state[p] is a per-parameter state dictionary in PyTorch optimizers
                    It's used to store and track parameter-specific values across optimization steps
                    self.state[p] = {
                        'momentum': ...,
                        'step': ...,
                        'x': ...   # <- you can define this key yourself
                    }
                    Each parameter p has its own state dict accessed via self.state[p]
                    Here we use it to store the previous x iterate ('x') for momentum calculations x_{t}
                    This state persists between steps, allowing us to implement stateful optimizers
                    """

                    # We use self.state[p]['x'] to save the auxiliary x_t for momentum calculations
                    if 'x' not in self.state[p]:
                        self.state[p]['x'] = p.data.clone() # x0 = y0 in initial step

                    #TODO: Add Gradient descent step: z = y_t -\eta \nabla f(y_t)

                    #TODO: Add Proximal Step by soft-thresholding: x_{t+1} = Soft-Threshold(z, l1_reg * lr)

                    #TODO: Momentum (acceleration) update: y_{t+1} = x_{t+1} + ((lambda_old - 1) / lambda_{t+1}) * (x_{t+1} - x_t)

                    #TODO: Update parameter p, save the new x
                    """
                    Hint: Use in-place operations like .copy_ and .clone() for memory efficiency and to maintain 
                    proper gradient tracking. 
                    """



class Trainer:
    """
    A class to handle the training process of a PyTorch model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        num_epochs,
        use_logger=False,
        save_frequency=10,
        hyperparams=None,
        git_commit=False,
        l1_reg=0  # Add parameter for L1 regularization strength
    ):
        """
        Initialize the Trainer with model and training parameters.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            loss_fn: Loss function
            optimizer: Optimizer for parameter updates
            device: Device to use for computation
            num_epochs: Number of training epochs
            use_logger: Whether to use logging functionality
            save_frequency: How often to save model checkpoints (epochs)
            hyperparams: Dictionary of hyperparameters to log
            git_commit: Whether to automatically create a git commit after training
            l1_reg: L1 regularization strength (needed for complete loss reporting)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.use_logger = use_logger
        self.save_frequency = save_frequency
        self.hyperparams = hyperparams
        self.git_commit = git_commit
        self.l1_reg = hyperparams.get('L1_REG', 0) if hyperparams else l1_reg
        
        # Extract save directories from hyperparameters if provided
        self.checkpoints_dir = hyperparams.get('CHECKPOINTS_DIR', 'checkpoints') if hyperparams else 'checkpoints'
        self.results_dir = hyperparams.get('RESULTS_DIR', 'results') if hyperparams else 'results'
        self.logs_dir = hyperparams.get('LOGS_DIR', 'logs') if hyperparams else 'logs'
        
        # Logger setup
        self.logger = None
        if self.use_logger:
            self.logger = Logger(log_dir=self.logs_dir, hyperparams=self.hyperparams)
            self.logger.start_capture()
            # Log timestamp at the beginning
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{'='*80}{timestamp}")
            # Log hyperparameters if provided
            if self.hyperparams:
                self.logger.log_hyperparameters(self.hyperparams)
        
        # Variables for tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Store start time for commit message
        self.start_time = datetime.datetime.now()
    
    def _calculate_l1_penalty(self):
        """Calculate the L1 penalty term for the model parameters."""
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        return self.l1_reg * l1_norm
    
    def _train_epoch(self, epoch, pbar=None):
        """Train the model for one epoch."""
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to the specified device
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(data)
            
            # Get the loss from the loss function (might or might not include L1)
            current_loss = self.loss_fn(predictions, targets)
            
            # For reporting purposes, make sure we always include the L1 penalty
            # Check if we're using an optimizer that handles L1 internally
            if isinstance(self.optimizer, (ISTA, FISTA)):
                # For ISTA/FISTA, add L1 penalty to the reported loss only (not for gradient)
                l1_penalty = self._calculate_l1_penalty()
                reporting_loss = current_loss + l1_penalty
            else:
                # For other optimizers, the loss_fn already includes L1 penalty
                reporting_loss = current_loss
            
            running_loss += reporting_loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            current_loss.backward()  # Use original loss for backprop
            self.optimizer.step()
            
            # Update progress bar with current loss if provided
            if pbar:
                pbar.update(1)
                pbar.set_postfix(train_loss=f"{reporting_loss.item():.6f}")
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def _evaluate_model(self, loader=None, pbar=None):
        """Evaluate the model on validation data or a specified loader."""
        if loader is None:
            loader = self.val_loader  # Default to validation loader if none specified
        
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(loader):
                # Move data to the specified device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(data)
                
                # Get the loss from the loss function
                current_loss = self.loss_fn(predictions, targets)
                
                # For reporting purposes, only report the testing MSE loss
                if not isinstance(self.optimizer, (ISTA, FISTA)):
                    # For not ISTA/FISTA, remove L1 penalty to the reported loss
                    l1_penalty = self._calculate_l1_penalty()
                    reporting_loss = current_loss - l1_penalty
                else:
                    # For ISTA/FISTA, the loss_fn is MSELoss
                    reporting_loss = current_loss
                
                running_loss += reporting_loss.item()
        
        avg_loss = running_loss / len(loader)
        return avg_loss
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_{epoch}.pth.tar", directory=self.checkpoints_dir)
        print(f"Saved checkpoint {epoch} with validation loss: {self.val_losses[-1]:.4f}")
    
    def _make_git_commit(self, final_train_loss, final_val_loss):
        """Create a git commit with training results."""
        try:
            # Check if this is a git repository
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
            
            # Format the training duration
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create commit message
            commit_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            commit_message = (
                f"Training completed at {commit_time}\n\n"
                f"Duration: {duration_str}\n"
                f"Final Training Loss: {final_train_loss:.6f}\n"
                f"Final Validation Loss: {final_val_loss:.6f}\n"
                f"Epochs: {self.num_epochs}\n"
                f"Learning Rate: {self.hyperparams.get('LEARNING_RATE', 'N/A') if self.hyperparams else 'N/A'}"
            )
            
            # Check if there are modified or new files in the output directories
            subprocess.run(["git", "add", self.checkpoints_dir, self.results_dir, self.logs_dir])
            
            # Create the commit
            subprocess.run(["git", "commit", "-m", commit_message])
            
            print("\nCreated git commit with training results.")
            print(f"Commit message:\n{commit_message}")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\nFailed to create git commit: {str(e)}")
            print("This might not be a git repository or git might not be installed.")
    
    def train(self):
        """Run the full training process."""
        print(f"Training on device: {self.device}")
        
        # Evaluate initial model performance before training starts
        initial_train_loss = self._evaluate_model(loader=self.train_loader)
        initial_val_loss = self._evaluate_model(loader=self.val_loader)
        # Format the output to match the pattern expected by the regex
        print(f"Epoch 0/{self.num_epochs}: 100%|██████████| {len(self.train_loader)}/{len(self.train_loader)} [00:00<00:00, 0.00it/s, train_loss={initial_train_loss:.6f}, val_loss={initial_val_loss:.6f}]")
        
        # Add initial losses to tracking lists
        self.train_losses.append(initial_train_loss)
        self.val_losses.append(initial_val_loss)
        
        # Training loop over epochs
        for epoch in range(self.num_epochs):
            # Print epoch header (removed - we'll rely on tqdm for consistent formatting)
            # print(f"\nEpoch {epoch+1} / {self.num_epochs}")
            
            # Create a progress bar for the entire training process (all batches)
            total_steps = len(self.train_loader)
            with tqdm(total=total_steps, file=sys.stdout, 
                     desc=f"Epoch {epoch+1}/{self.num_epochs}",
                     bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
                
                # Train for one epoch
                train_loss = self._train_epoch(epoch, pbar)
                self.train_losses.append(train_loss)
                
                # Evaluate on validation set
                val_loss = self._evaluate_model()
                self.val_losses.append(val_loss)
                
                # Update progress bar with both losses
                pbar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.3f}")
            
            # Check if this is the best validation loss and print notification
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"<<<<<< reach best val loss : {val_loss} >>>>>>")
                
                # Save best model
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                save_checkpoint(checkpoint, filename="best_model.pth.tar", directory=self.checkpoints_dir)
            
            # Save model if needed
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch)
        
        # Log final training metrics
        if self.use_logger and self.logger:
            self.logger.log_final_metrics(self.train_losses[-1], self.val_losses[-1])
        
        # Visualize final model predictions
        # visualize_predictions(self.model, device=self.device, save_dir=self.results_dir)
        
        # Plot training and validation loss curves
        plot_loss_curves(self.train_losses, self.val_losses, save_dir=self.results_dir, optimizer=self.hyperparams['OPTIMIZER'], learning_rate=self.hyperparams['LEARNING_RATE'])
        
        print("Training completed!")
        
        # Print log file path if logging was used
        if self.use_logger and self.logger:
            log_path = self.logger.get_log_file_path()
            print(f"Log file saved to: {log_path}")
            # Stop logging
            self.logger.stop_capture()
        
        # Make git commit if enabled
        if self.git_commit:
            self._make_git_commit(self.train_losses[-1], self.val_losses[-1])
    
    def get_log_file_path(self):
        """Get the path to the log file if logging is enabled."""
        if self.use_logger and self.logger:
            return self.logger.get_log_file_path()
        return None

