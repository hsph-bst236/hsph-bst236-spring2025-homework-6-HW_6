# utils.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import LinearRegressionDataset
import datetime
import sys
import io
import subprocess

# Defining default directories directly in the file
DEFAULT_CHECKPOINTS_DIR = "output/checkpoints"
DEFAULT_RESULTS_DIR = "output/results"
DEFAULT_LOGS_DIR = "output/logs"

def save_checkpoint(state, filename="my_checkpoint.pth.tar", directory=DEFAULT_CHECKPOINTS_DIR):
    """Save model checkpoint."""
    print(f"=> Saving checkpoint to {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, os.path.join(directory, filename))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Load model checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def evaluate_model(loader, model, loss_fn, device="cuda"):
    """
    Evaluate the model on the provided data loader.
    
    Args:
        loader (DataLoader): Data loader for evaluation.
        model (nn.Module): Model to evaluate.
        loss_fn (callable): Loss function.
        device (str): Device to use for computation.
        
    Returns:
        float: Average loss on the dataset.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(x)
            loss = loss_fn(predictions, y)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Evaluation - Avg Loss: {avg_loss:.4f}")
    return avg_loss

def visualize_predictions(model, device="cuda", num_points=100, save_dir=DEFAULT_RESULTS_DIR):
    """
    Visualize the model's predictions against the true relationship.
    
    Args:
        model (nn.Module): Trained model.
        device (str): Device to use for computation.
        num_points (int): Number of points to plot.
        save_dir (str): Directory to save the visualization.
    """
    model.eval()
    
    # Generate test data across the range
    x = torch.linspace(-12, 12, num_points).view(-1, 1).to(device)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(x)
    
    # Convert to numpy for plotting
    x_np = x.cpu().numpy().flatten()
    predictions_np = predictions.cpu().numpy().flatten()
    
    # Generate true relationship (without noise)
    true_y = 3 * x_np + 2
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_np, predictions_np, color='blue', label='Model Predictions')
    plt.plot(x_np, true_y, color='red', label='True Relationship (y = 3x + 2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression: Predictions vs. True Relationship')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_path = os.path.join(save_dir, 'linear_regression_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Predictions visualization saved to '{plot_path}'")

def plot_loss_curves(train_losses, val_losses, save_dir=DEFAULT_RESULTS_DIR, optimizer=None, learning_rate=None):
    """
    Plot the training and validation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch.
        val_losses (list): Validation losses per epoch.
        save_dir (str): Directory to save the loss curves plot.
        optimizer (str, optional): Name of the optimizer used.
        learning_rate (float, optional): Learning rate used.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add markers every 10 epochs for better readability
    marker_epochs = [i for i in epochs if i % 10 == 0 or i == 1 or i == len(epochs)]
    marker_train = [train_losses[i-1] for i in marker_epochs]
    marker_val = [val_losses[i-1] for i in marker_epochs]
    
    plt.plot(marker_epochs, marker_train, 'bo')
    plt.plot(marker_epochs, marker_val, 'ro')
    
    # Save the plot with timestamp and optimizer info
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'loss_curves_{timestamp}'
    
    # Add optimizer and learning rate to filename if available
    if optimizer:
        filename += f'_{optimizer}'
    
    if learning_rate:
        # Replace decimal point with underscore to avoid filename issues
        lr_str = str(learning_rate).replace('.', '-')
        filename += f'_lr{lr_str}'
    
    filename += '.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Loss curves saved to '{plot_path}'")

class Logger:
    """
    A simple logger class that can redirect stdout to both console and a log file.
    Also logs git commit information for reproducibility.
    """
    def __init__(self, log_dir=DEFAULT_LOGS_DIR, hyperparams=None):
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create a timestamped log file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Include optimizer and learning rate in the filename if available
        optimizer = hyperparams.get('OPTIMIZER', '') if hyperparams else ''
        learning_rate = hyperparams.get('LEARNING_RATE', '') if hyperparams else ''
        
        filename = f'training_log_{timestamp}'
        
        # Add optimizer and learning rate to filename if available
        if optimizer:
            filename += f'_{optimizer}'
        
        if learning_rate:
            # Replace decimal point with underscore to avoid filename issues
            lr_str = str(learning_rate).replace('.', '-')
            filename += f'_lr{lr_str}'
            
        self.log_file_path = os.path.join(log_dir, f'{filename}.txt')
        self.log_file = open(self.log_file_path, 'w')
        
        # Store the original stdout
        self.original_stdout = sys.stdout
        
        # Initialize captured content
        self.captured_output = io.StringIO()
        
    def start_capture(self):
        """Start capturing stdout and log git information."""
        sys.stdout = self
        
        # Log git information for reproducibility
        self.log_git_info()
        
    def stop_capture(self):
        """Stop capturing and restore original stdout."""
        sys.stdout = self.original_stdout
        self.log_file.close()
        
    def write(self, text):
        """Write to both stdout and log file."""
        self.original_stdout.write(text)
        self.log_file.write(text)
        self.captured_output.write(text)
        
    def flush(self):
        """Flush both stdout and log file."""
        self.original_stdout.flush()
        self.log_file.flush()
        
    def log_git_info(self):
        """Log git repository information for reproducibility."""
        self.write("\n" + "="*50 + "\n")
        self.write("GIT INFORMATION\n")
        self.write("="*50 + "\n")
        
        try:
            # Get current commit hash
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            
            # Get current branch
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            
            # Get commit message
            git_message = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%B"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            
            # Log git information
            self.write(f"Git Branch: {git_branch}\n")
            self.write(f"Git Commit: {git_hash}\n")
            # self.write(f"Commit Message: {git_message}\n")
            
            # Command to recover this exact version
            self.write("\nTo recover this exact version, run:\n")
            self.write(f"git checkout {git_hash}\n")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.write("Git information not available. This might not be a git repository.\n")
        
        self.write("="*50 + "\n\n")
    
    def log_hyperparameters(self, config_dict):
        """Log hyperparameters to the log file."""
        self.write("\n" + "="*50 + "\n")
        self.write("HYPERPARAMETERS\n")
        self.write("="*50 + "\n")
        
        for key, value in config_dict.items():
            self.write(f"{key}: {value}\n")
            
        self.write("="*50 + "\n\n")
        
    def log_final_metrics(self, train_loss, val_loss):
        """Log final training metrics."""
        self.write("\n" + "="*50 + "\n")
        self.write("FINAL METRICS\n")
        self.write("="*50 + "\n")
        self.write(f"Final Training Loss: {train_loss:.6f}\n")
        self.write(f"Final Validation Loss: {val_loss:.6f}\n")
        self.write("="*50 + "\n\n")
        
    def get_log_file_path(self):
        """Return the path to the log file."""
        return self.log_file_path

def plot_parameter_trajectory_on_contour(parameter_history, objective_func, save_dir=DEFAULT_RESULTS_DIR, optimizer=None, learning_rate=None):
    """
    Visualize the trajectory of a 2D parameter on the contour plot of an objective function.
    
    Args:
        parameter_history (list): List of parameter values at each epoch [(x1, y1), (x2, y2), ...].
        objective_func (callable): The objective function to visualize (should take a 2D tensor as input).
        save_dir (str): Directory to save the visualization.
        optimizer (str, optional): Name of the optimizer used.
        learning_rate (float, optional): Learning rate used.
    """
    # Create a grid of points for the contour plot
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute objective function values for the grid
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            # Create input tensor for the objective function
            input_tensor = torch.tensor([X[j, i], Y[j, i]], dtype=torch.float32)
            Z[j, i] = objective_func(input_tensor).item()
    
    # Convert parameter history to numpy arrays for plotting
    param_history_np = np.array([(p[0].item(), p[1].item()) for p in parameter_history])
    
    # Create the figure and plot
    plt.figure(figsize=(10, 8))
    
    # Create contour plot
    contour = plt.contour(X, Y, Z, 20, colors='black', alpha=0.6)
    contourf = plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.5)
    
    # Plot the parameter trajectory
    plt.plot(param_history_np[:, 0], param_history_np[:, 1], 'ro-', markersize=3, linewidth=1.5, label='Parameter Trajectory')
    
    # Add markers for start and end positions
    plt.plot(param_history_np[0, 0], param_history_np[0, 1], 'g*', markersize=10, label='Start')
    plt.plot(param_history_np[-1, 0], param_history_np[-1, 1], 'b*', markersize=10, label='End')
    
    # # Add labels for iterations every 10 epochs or so (depending on the length)
    # interval = max(len(parameter_history) // 10, 1)
    # for i in range(0, len(parameter_history), interval):
    #     plt.annotate(f'{i}', (param_history_np[i, 0], param_history_np[i, 1]), 
    #                  textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    # Add colorbar
    # plt.colorbar(contourf, label='Objective Function Value')
    
    # Add labels and title
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    title = 'Trajectory of Parameter: '
    if optimizer:
        title += f' ({optimizer}'
        if learning_rate:
            title += f', lr={learning_rate}'
        title += ')'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot with timestamp and optimizer info
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'parameter_trajectory_{timestamp}'
    
    # Add optimizer and learning rate to filename if available
    if optimizer:
        filename += f'_{optimizer}'
    
    if learning_rate:
        # Replace decimal point with underscore to avoid filename issues
        lr_str = str(learning_rate).replace('.', '-')
        filename += f'_lr{lr_str}'
    
    filename += '.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Parameter trajectory visualization saved to '{plot_path}'")
