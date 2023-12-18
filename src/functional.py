import numpy as np
from src.tensor import Tensor
from src.module import Module
from typing import Optional


class Linear(Module):
    """
    A linear layer (aka a fully-connected layer) in a neural network.

    The layer performs a linear transformation of the input data, followed by an optional bias
    addition. The layer has two learnable parameters: weights and bias.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: Optional[bool] = True) -> None:
        self.weights = Tensor(np.random.randn(dim_out, dim_in))
        if bias:
            self.bias = Tensor(np.random.randn(dim_out, 1))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass of the input.

        Arguments:
            x (Tensor): input to the layer.
        """
        out = self.weights @ x
        if self.bias:
            out += self.bias
        return out


class ReLU(Module):
    """
    A ReLU activation function layer in a neural network.

    The ReLU function is defined as follows:
    relu(x) = max(0, x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass of the input.

        Arguments:
            x (Tensor): input to the layer.
        """
        out = (x > 0) * x
        return out


class Softmax(Module):
    """
    A softmax activation function layer in a neural network.

    The softmax function is defined as follows:
    softmax(x) = e^x / sum(e^x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass of the input.

        Arguments:
            x (Tensor): input to the layer.
        """
        z = x - x.max()
        numerator = np.e**z
        denominator = numerator.sum(axis=-2)
        return (numerator.swapaxes(0, 1) / denominator).swapaxes(0, 1)


class MSELoss:
    """
    A loss function that computes the mean squared error between y and y_hat.

    The mean squared error is defined as follows:
    mse = sum((y - y_hat)^2) / n
    """

    def __call__(self, y: Tensor, y_hat: Tensor) -> Tensor:
        squared_errors = (y - y_hat) ** 2
        mse = squared_errors.sum() / y.shape[0]
        return mse
