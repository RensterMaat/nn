import pytest
import numpy as np
from src.tensor import Tensor

@pytest.fixture
def scalar():
    return Tensor(np.random.randn(1))

@pytest.fixture
def vector():
    return Tensor(np.random.randn((1,5)))

def function_with_all_operators(arg):
    out = function_with_addition(arg)
    out = function_with_multiplication(out)
    return out

def function_with_addition(arg):
    out = arg + arg
    out = out + 2
    out = 3 + out
    return out

def function_with_multiplication(arg):
    out = out * out
    out = 2 * out
    out = out * 3
    return out

def test_tensor_holds_value():
    x = 5
    t = Tensor(5)
    assert t.value == x


class TestOperators:
    def test_add(self, scalar):
        assert (scalar + 2).value == scalar.value + 2

    def test_radd(self, scalar):
        assert (2 + scalar).value == scalar.value + 2

    def test_mul(self, scalar):
        assert (scalar * 2).value == 2 * scalar.value

    def test_rmul(self, scalar):
        assert (2* scalar).value == 2 * scalar.value

    def test_add_vectors(self):
        assert all((c + d).value == [3,5,7])

    def test_mul_vectors(self):
        assert all((c * d).value == [2,6,12])

class TestBackpropagation:
    def test_backpropagation_reaches_operand_a(self):
        f = Tensor(4)
        g = Tensor(5)
        h = f + g
        h.backpropagate_a()
        assert f.grad is not None

    def test_backpropagation_reaches_operand_b(self):
        f = Tensor(4)
        g = Tensor(5)
        h = f + g
        h.backpropagate_b()
        assert g.grad is not None

    def test_root_node_gradient_is_one(self):
        assert all(out.grad == np.ones(out.value.shape))
        
    def test_backpropagate_add(self):
        a = Tensor(5)
        b = Tensor(6)

        out = a + b
        before = out.value

        delta = 0.0001
        a.value = a.value + delta

        out = a + b
        after = out.value

        out.backwards()

        real_gradient = (after - before) / delta

        assert np.isclose(real_gradient, a.grad)

    def test_backpropagate_add(self):
        a = Tensor(5)
        b = Tensor(6)

        out = a * b
        before = out.value

        delta = 0.0001
        a.value = a.value + delta

        out = a * b
        after = out.value

        out.backwards()

        real_gradient = (after - before) / delta

        assert np.isclose(real_gradient, a.grad)