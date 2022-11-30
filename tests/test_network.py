import pytest
import numpy as np
from src.tensor import Tensor
from src.module import Module
from src.functional import Linear, ReLU

class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(24, 10)
        self.relu = ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return out


network = MLP()
input = Tensor(np.random.randn(24,1))

def test_forward():
    output = network(input)