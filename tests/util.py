import numpy as np
from typing import Any, Tuple


def nudge_input_at_index(
    original: np.ndarray, index: Tuple[int, ...], delta: float
) -> np.ndarray:
    """
    Returns a copy of the original array with the value at the given index nudged by delta.
    """
    nudge = np.zeros(original.shape)
    nudge[index] = delta
    return original + nudge


def get_numerical_gradient(
    function: Any, input: np.ndarray, parameter: Any, delta_input: float = 1e-6
) -> np.ndarray:
    """
    Returns the numerical gradient of the given function with respect to the given parameter.
    """

    # Save original parameter value and output before nudge
    original_parameter_value = parameter.value.copy()

    # Compute output before nudge
    output_before_nudge = function(input).value.sum()

    # Make a list of all indices for the parameter for which we want to compute the gradient
    indices = zip(*[ax.flatten() for ax in np.indices(parameter.shape)])

    # Initialize numerical gradient as an empty array
    numerical_gradient = np.empty(parameter.shape)

    # For each index,
    for index in indices:
        # Nudge the parameter at the index by delta_input
        parameter.value = nudge_input_at_index(parameter.value, index, delta_input)

        # Compute output after nudge
        output_after_nudge = function(input).value.sum()

        # Determine the change in output
        delta_output = output_after_nudge - output_before_nudge

        # The numerical gradient is equal to the change in output divided by the change in input
        numerical_gradient[index] = delta_output / delta_input

        # Reset the parameter to its original value
        parameter.value = original_parameter_value

    return numerical_gradient
