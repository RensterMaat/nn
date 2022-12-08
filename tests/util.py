import numpy as np

def nudge_input_at_index(original, index, delta):
        nudge = np.zeros(original.shape)
        nudge[index] = delta
        return original + nudge

def get_numerical_gradient(function, input, parameter, delta_input=1e-6):
    original_parameter_value = parameter.value.copy()
    output_before_nudge = function(input).value.sum()

    indices = zip(*[ax.flatten() for ax in np.indices(parameter.shape)])
    numerical_gradient = np.empty(parameter.shape)
    for index in indices:
        parameter.value = nudge_input_at_index(parameter.value, index, delta_input)

        output_after_nudge = function(input).value.sum()

        delta_output = output_after_nudge - output_before_nudge
        numerical_gradient[index] = delta_output / delta_input

        parameter.value = original_parameter_value

    return numerical_gradient