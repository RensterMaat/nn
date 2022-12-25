import pytest
import numpy as np
from src.tensor import Tensor
from src.module import Module
from src.functional import Linear, ReLU, Softmax
from tests.util import get_numerical_gradient

class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(24, 12)
        self.relu = ReLU()
        self.fc2 = Linear(12,3)
        self.softmax = Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        out = out.sum()
        return out


network = MLP()
input_tensor = Tensor(np.random.randn(3,24,1))

def test_forward():
    network(input_tensor)

def test_backwards():
    output = network(input_tensor)
    output.backwards()

    parameters = network.parameters()

    for name, parameter in parameters.items():
        numerical_gradient = get_numerical_gradient(network, input_tensor, parameter)
        analytical_gradient = parameter.grad


        print(
            'Parameter: {}\n'.format(name),
            'Numerical: {}\n'.format(numerical_gradient[0]),
            'Analytical: {}\n'.format(analytical_gradient[1]),
        )
        # assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

test_backwards()