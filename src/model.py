# model.py
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Simple linear regression model.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output predictions
        """
        super().__init__()  # Call the constructor of the parent class
        self.linear = nn.Linear(input_dim, output_dim)
        # Initialize weights and bias to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def normal_density(x, mu, sigma2):
    '''
    The normal density function without normalizer
    p(x) = e^{-\frac{(x - \mu)^2}{\sigma^2}}
    '''
    return torch.exp(-0.5 * torch.sum((x - mu) ** 2 / sigma2 , dim=-1))

def nonconvex_objective(x):
    '''
    The 2d non-convex objective function
    f(a_1, a_2) =
    2 e^{-\frac{(a_1 - 1)^2 + (1.4a_2)^2}{0.2}} +
    6 e^{-\frac{(a_1 + 1)^2 + (1.4a_2)^2}{0.2}} -
    3 e^{-\frac{(a_1 - 1)^2 + (1.4a_2 + 1)^2}{0.2}} -
    0.2 e^{-\frac{(a_1)^2 + (1.4a_2 -1)^2}{0.2}} +
    a_1^2 + (1.4a_2)^2
    '''
    sigma2 = 0.2
    x_new = x * torch.tensor([1.0, 1.4])
    mu1 = torch.tensor([1.0, 0.0])
    mu2 = torch.tensor([-1.0, 0.0])
    mu3 = torch.tensor([1.0, -1.0])
    mu4 = torch.tensor([0.0, 1.0])
    term1 = normal_density(x_new, mu1, sigma2)
    term2 = normal_density(x_new, mu2, sigma2)
    term3 = normal_density(x_new, mu3, sigma2)
    term4 = normal_density(x_new, mu4, sigma2)
    return 2*term1 + 6*term2 - 3*term3 - 0.2*term4 + torch.sum(x_new**2)


class NonConvexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([2.0, 2.0])) # 2D vector

    def forward(self):
        return nonconvex_objective(self.a)
