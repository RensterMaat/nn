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


def does_not_have_leaves_other_than_expected(node, expected):
    if node is None:
        return True
    elif node in expected:
        return True
    elif node.a is None and node.b is None:
        return False
    else:
        return does_not_have_leaves_other_than_expected(node.a, expected) \
            and does_not_have_leaves_other_than_expected(node.b, expected)

def test_parameters():
    parameters = list(network.parameters().values())

    output = network(input)
    expected_leaves = [input] + parameters

    assert does_not_have_leaves_other_than_expected(output, expected_leaves)

