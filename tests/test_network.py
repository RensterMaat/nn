import pytest
import numpy as np
from src.tensor import Tensor
from src.module import Module
from src.functional import Linear, ReLU, Softmax

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
        out = self.softmax(out)
        return out


network = MLP()
input = Tensor(np.random.randn(5,24,1))

def test_forward():
    output = network(input)

def test_parameters():
    parameters = network.parameters()