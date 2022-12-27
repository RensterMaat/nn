import numpy as np
from src.tensor import Tensor
from src.module import Module

class Linear(Module):
    def __init__(self, dim_in, dim_out, bias=True):
        self.weights = Tensor(np.random.randn(dim_out, dim_in))
        if bias:
            self.bias = Tensor(np.random.randn(dim_out, 1))
        else:
            self.bias = None
    
    def forward(self, x):
        out = self.weights @ x
        if self.bias:
            out += self.bias
        return out


class ReLU(Module):
    def forward(self, x):
        out = (x > 0) * x
        return out


class Softmax(Module):
    def forward(self, x):
        z = x - x.max()
        numerator = np.e ** z
        denominator = numerator.sum()
        return numerator/denominator


class MSELoss:
    def __call__(self, y, y_hat):
        squared_errors = (y - y_hat) ** 2
        mse = squared_errors.sum() / y.shape[0]
        return mse