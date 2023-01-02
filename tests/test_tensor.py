import pytest
import numpy as np
from src.tensor import Tensor
from src.functional import Softmax
from tests.util import get_numerical_gradient

scalar = Tensor(np.random.randn(1))
vector = Tensor(np.random.randn(5,1))
matrix = Tensor(np.random.randn(5,4))

operands = [
    scalar,
    vector,
    matrix
]

def function_with_addition(arg):
    out = arg + arg
    out = out + 2
    out = 3 + out
    return out

def function_with_subtraction(arg):
    out = arg - 12
    out = 3 - arg
    return out

def function_with_multiplication(arg):
    out = arg * arg
    out = 2 * out
    out = out * 3
    return out

def function_with_division(arg):
    out = arg / 2
    out = 3 / out
    return out

def function_with_dot_product(arg):
    np.random.seed(0)

    if arg.ndim == 1:
        return arg

    other = np.random.uniform(size=(4,arg.shape[-2]))
    if isinstance(arg, Tensor):
        other = Tensor(other)
    out = other @ arg
    out = out @ np.random.uniform(1,2, size=(out.shape[-1], 3))

    return out

def function_with_pow(arg):
    out = arg ** 2
    out = 2 ** out
    return out
    
operations = [
    function_with_addition,
    function_with_subtraction,
    function_with_multiplication,
    function_with_division,
    function_with_dot_product,
    function_with_pow
]


class TestForward:
    @pytest.mark.parametrize('operation', operations)
    @pytest.mark.parametrize('operand', operands)
    def test_all_operations_on_all_operands(self, operation, operand):
        tensor_result = operation(operand).value
        comparison_result = operation(operand.value)
        assert np.allclose(tensor_result, comparison_result)


class TestBackpropagation:
    @pytest.mark.parametrize('operation', operations)
    @pytest.mark.parametrize('operand', operands)
    def test_all_operations_on_all_operands(self, operation, operand):
        operand.zero_grad()

        output = operation(operand)
        output.backwards()

        analytical_gradient = operand.grad
        numerical_gradient = get_numerical_gradient(operation, operand, operand)

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

    @pytest.mark.parametrize('axis', range(5))
    def test_summation_across_all_axes(self, axis):
        operand = Tensor(np.random.randn(1,2,3,4,1))

        output = operand.sum(axis) ** 2
        output.backwards()

        analytical_gradient = operand.grad
        numerical_gradient = get_numerical_gradient(
            lambda x: x.sum(axis) ** 2, 
            operand, 
            operand
        )

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

    @pytest.mark.parametrize('axis1', range(5))
    def test_swapaxes_across_all_axes(self, axis1):
        operand = Tensor(np.random.randn(1,2,3,4,1))

        output = operand.swapaxes(axis1, 2) ** 2
        output.backwards()

        analytical_gradient = operand.grad
        numerical_gradient = get_numerical_gradient(
            lambda x: x.swapaxes(axis1, 2) ** 2, 
            operand, 
            operand
        )

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

    def test_max(self):
        operand = Tensor(np.random.randn(6,1))

        function = lambda x: (x - x.max()) ** 2

        output = function(operand)
        output.backwards()

        analytical_gradient = operand.grad
        numerical_gradient = get_numerical_gradient(
            function,
            operand, 
            operand
        )

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)

    def test_softmax(self):
        operand = Tensor(np.random.randn(10,1))

        output = Softmax()(operand) ** 2
        output.backwards()

        analytical_gradient = operand.grad
        numerical_gradient = get_numerical_gradient(
            lambda x: Softmax()(x) ** 2, 
            operand, 
            operand
        )

        assert np.allclose(analytical_gradient, numerical_gradient, rtol=1e-4)
    