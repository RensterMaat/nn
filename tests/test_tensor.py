import pytest
import numpy as np
from src.tensor import Tensor

@pytest.fixture
def scalar():
    return Tensor(np.random.randn(1))

@pytest.fixture
def vector():
    return Tensor(np.random.randn(1,5))

@pytest.fixture
def matrix():
    return Tensor(np.random.randn(4,5))

operands = [
    Tensor(np.random.randn(1)),
    Tensor(np.random.randn(1,5)),
    Tensor(np.random.randn(4,5))
]

def function_with_addition(arg):
    out = arg + arg
    out = out + 2
    out = 3 + out
    return out

def function_with_multiplication(arg):
    out = arg * arg
    out = 2 * out
    out = out * 3
    return out

def function_with_all_operators(arg):
    out = function_with_addition(arg)
    out = function_with_multiplication(out)
    return out

operations = [
    function_with_addition,
    function_with_multiplication,
    function_with_all_operators
]

def compare_forward_results(arg, function):
    tensor_result = function(arg).value
    comparison_result = function(arg.value)
    assert np.all(tensor_result == comparison_result)


class TestForward:
    @pytest.mark.parametrize('operation', operations)
    @pytest.mark.parametrize('operand', operands)
    def test_all_operations_on_all_operands(self, operation, operand):
        compare_forward_results(operand, operation)


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