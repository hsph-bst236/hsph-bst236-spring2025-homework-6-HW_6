# dataset.py
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#TODO: Problem 1.1: Generate dataset for high dimensional linear model with sparse weights
class SparseLinearRegressionDataset(Dataset):
    """
    Hint: You need to define the `__init__` function and `__len__` and `__getitem__` function.
    """
    def __init__(self, num_samples=100, input_dim=100, sparsity=0.5, noise_std = 1):
        """
        Create a synthetic dataset for high-dimensional linear regression with sparse weights.
        Y = Xw + noise, where
        dim of w  = input_dim
        Only first int(input_dim * sparsity) entries of w are 10 and the rest are 0
        X are iid from N(0, 1)
        noise ~ N(0, noise_std)

        Args:
            num_samples (int): Number of data points to generate.
            input_dim (int): Dimension of input features.
            sparsity (float): Proportion of sparse weights.
            noise_std (float): Standard deviation of noise.
        """
        #TODO: Generate self.x, self.y following the model above
    
    #TODO: Define `__len__` to return the number of data points
    
    #TODO: Define `__getitem__(self, idx)` to return the `self.x[idx]` and `self.y[idx]`





class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples=100):
        """
        Create a synthetic dataset for linear regression.
        
        Args:
            num_samples (int): Number of data points to generate.
        """
        # Generate input data with a uniform distribution between -10 and 10
        self.x = torch.linspace(-10, 10, num_samples).view(-1, 1)
        
        # Generate target data: y = 3x + 2 + noise
        # True relationship is y = 3x + 2, but we add some random noise
        self.y = 3 * self.x + 2 + torch.randn(self.x.size()) * 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



def get_loaders(num_samples, batch_size, dataset_type="sparse", input_dim=100, sparsity=0.5):
    """
    Create data loaders for training and validation.
    
    Args:
        num_samples (int): Number of samples for each dataset.
        batch_size (int): Batch size.
        dataset_type (str): Type of dataset to use - "linear" or "sparse".
        input_dim (int): Input dimension for sparse dataset.
        sparsity (float): Sparsity level for sparse dataset.
        
    Returns:
        tuple: (train_loader, val_loader) - Data loaders for training and validation.
    """
    if dataset_type == "linear":
        train_dataset = LinearRegressionDataset(num_samples=num_samples)
        val_dataset = LinearRegressionDataset(num_samples=num_samples // 4)  # Smaller validation set
    else:  # default to sparse
        train_dataset = SparseLinearRegressionDataset(
            num_samples=num_samples,
            input_dim=input_dim,
            sparsity=sparsity
        )
        val_dataset = SparseLinearRegressionDataset(
            num_samples=num_samples // 4,  # Smaller validation set
            input_dim=input_dim,
            sparsity=sparsity
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size= num_samples // 4,
        shuffle=False
    )
    
    return train_loader, val_loader
