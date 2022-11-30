import pytest
import numpy as np
from src.tensor import Tensor
from src.module import Module
from src.functional import Linear

class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(24, 10)

    def forward(self, x):
        out = self.fc1(x)
        return out


network = MLP()
input = Tensor(np.random.randn(24,1))

def test_forward():
    output = network(input)