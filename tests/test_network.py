import pytest
import numpy as np
from src.tensor import Tensor
from src.module import Module
from src.functional import Linear, ReLU, Softmax, MSELoss
from tests.util import get_numerical_gradient

class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(24, 12)
        self.relu = ReLU()
        self.fc2 = Linear(12,6)
        self.softmax = Softmax()

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


input_tensor = Tensor(np.random.randn(3,24,1))
target = Tensor(np.zeros((3,6,1)))

network = MLP()
criterion = MSELoss()

def test_forward():
    """
    Tests whether the forward pass of the network runs without errors.
    """
    network(input_tensor)

def test_backwards():
    """
    Tests whether the backward pass of the network results in analytical gradients
    that are close to the numerical gradients.
    """
    output = network(input_tensor)
    loss = criterion(target, output)
    loss.backwards()

    parameters = network.parameters()

    for parameter in list(parameters.values()):
        numerical_gradient = get_numerical_gradient(
            lambda x: criterion(target, network(x)), 
            input_tensor, 
            parameter
        )
        analytical_gradient = parameter.grad

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

