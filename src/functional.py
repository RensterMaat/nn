import numpy as np
from src.tensor import Tensor
from src.module import Module

class Linear(Module):
    def __init__(self, dim_in, dim_out, bias=True):
        self.weights = Tensor(np.random.randn(dim_out, dim_in))
        self.bias = Tensor(np.random.randn(dim_out))
    
    def forward(self, x):
        return (self.weights @ x) + self.bias 