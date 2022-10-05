import pytest
import numpy as np
from src.tensor import Tensor

scalar = Tensor(np.random.randn(1))
vector = Tensor(np.random.randn(1,5))
matrix = Tensor(np.random.randn(4,5))

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
    def nudge_input_at_index(self, original, index, delta):
        nudge = np.zeros(original.value.shape)
        nudge[index] = delta
        return original + nudge

    @pytest.mark.parametrize('operation', operations)
    @pytest.mark.parametrize('operand', operands)
    def test_all_operations_on_all_operands(self, operation, operand):
        original_output = operation(operand)
        original_output.backwards()
        autograd_calculated_gradient = operand.grad.copy()
        
        operand_indices = list(zip(*np.where(operand.value)))
        delta = 0.0001
        for index in operand_indices:
            nudged_input = self.nudge_input_at_index(operand, index, delta)
            output_after_nudge = operation(nudged_input)

            dy = output_after_nudge.value - original_output.value
            numerically_calculated_gradient = (dy) / delta
            assert np.allclose(
                autograd_calculated_gradient, 
                numerically_calculated_gradient
            )