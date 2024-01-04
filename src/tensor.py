import numpy as np
from typing import Union, List, Optional


class Tensor:
    """
    A custom Tensor class for automatic differentiation.

    Wraps a numpy array and provides methods for automatic differentiation.

    Attributes:
        value (np.array): The numerical value of the tensor.
        grad (np.array): The gradient of the tensor.
        a, b (Tensor): The tensors from which this tensor was derived (if any).
        shape (tuple): The shape of the tensor.
        ndim (int): The number of dimensions of the tensor.
    """

    def __init__(self, value: Union[int, float, bool, np.ndarray, "Tensor"]):
        """
        Initialize the Tensor object.

        Args:
            value (int, float, bool, np.array or Tensor): The value of the tensor.
        """
        # Handling the case where the input value is another Tensor object
        if isinstance(value, Tensor):
            self.value = value.value
            self.a, self.b = value.a, value.b
            self.grad = value.grad
        else:
            # Assuming the input value is a numpy array or compatible
            self.value = np.array(value)
            self.a, self.b = None, None
            self.grad = np.zeros(self.value.shape)

        self.shape = self.value.shape
        self.ndim = self.value.ndim

    def backwards(self) -> None:
        """
        Perform backpropagation to compute gradients for all tensors in the computational graph.
        """

        # Backpropagation is only defined for scalar outputs, not for vector or matrix outputs.
        # If the output is not a scalar, we sum the output to make it a scalar.
        if self.value.ndim > 0:
            self.sum().backwards()
            return None

        # Backpropagation order must be preserved in order to compute gradients correctly.
        # The order is determined by performing a topological sort on the computational graph.
        order = self.determine_backpropagation_order()

        # The gradient of the output node to itself is 1. This provides the root node for backpropagation.
        self.grad = 1

        # Backpropagate the gradient to all tensors in the computational graph.
        for node in order:
            if node.a is not None:
                node.backpropagate_a()

            if node.b is not None:
                node.backpropagate_b()

    def determine_backpropagation_order(self) -> List["Tensor"]:
        """
        Recursively determine the order of tensors for backpropagation.

        Returns:
            list: Ordered list of tensors for backpropagation.
        """

        # Determine the order for the first and second operands (if any).
        a_order, b_order = [], []

        # This is done by recursively calling determine_backpropagation_order on the operands.
        if self.a:
            a_order = self.a.determine_backpropagation_order()

        if self.b:
            b_order = self.b.determine_backpropagation_order()

        # The order of nodes leading to the first and second operand must be sorted topologically
        # to ensure proper gradient backpropagation.
        combined_order = self.topologically_ordered_merge(a_order, b_order)

        # Add the current node to the head of the list.
        return [self] + combined_order

    def topologically_ordered_merge(
        self, a_order: List["Tensor"], b_order: List["Tensor"]
    ) -> List["Tensor"]:
        """
        Topologically merge two lists of already topologically ordered tensors.

        Args:
            a_order (list): The first list of tensors.
            b_order (list): The second list of tensors.

        Returns:
            list: The merged list of tensors in topological order.
        """

        # Create a dictionary mapping each node to its index in the list.
        # This makes it easy to find if a node is in the list and what its index is.
        b_node_vs_index = {node: index for (index, node) in enumerate(b_order)}

        merged = []  # Initialize the merged list.
        a_pointer, b_pointer = 0, 0  # Initialize pointers to the head of each list.

        # Iterate through the elements of the first list
        for a_index, a_node in enumerate(a_order):
            # If this element is also in the second list...
            if a_node in b_node_vs_index:
                b_index = b_node_vs_index[a_node]

                # ...add all elements between the pointers and the current element...
                merged.extend(a_order[a_pointer:a_index])
                merged.extend(b_order[b_pointer:b_index])
                # ...and add the common element.
                merged.extend([a_node])

                # Pointers ensure that we don't add the same element twice.
                a_pointer = a_index + 1
                b_pointer = b_index + 1

        # Add the remaining elements of the lists.
        merged.extend(a_order[a_pointer:])
        merged.extend(b_order[b_pointer:])

        return merged

    def add_gradient(self, gradient_to_add: np.ndarray) -> None:
        """
        Add a gradient to the tensor's gradient.

        Args:
            gradient_to_add (np.array): The gradient to be added.
        """
        gradient_to_add = self.undo_broadcasting(gradient_to_add)
        self.grad = self.grad + gradient_to_add

    def undo_broadcasting(self, gradient_to_add: np.ndarray) -> np.ndarray:
        """
        Adjust the gradient shape to match the tensor's shape, reversing any broadcasting that occurred during forward pass.

        During the forward pass, the single element in the broadcasted axis was repeated to match the shape of the other
        operand. This method reverses that operation. In the backward pass, the gradient therefore needs to be summed over
        the broadcasted axis.

        Args:
            gradient_to_add (np.array): The gradient to be reshaped.

        Returns:
            np.array: The reshaped gradient.
        """

        # Create a list of axis indices based on the shape of gradient_to_add.
        axes = list(range(len(gradient_to_add.shape)))

        # Identify axes which do and do not match between gradient_to_add and self.grad.
        # These are the 'matched_axes' and 'unmatched_axes', respectively.
        unmatched_axes = axes[: len(gradient_to_add.shape) - len(self.grad.shape)]
        matched_axes = axes[len(gradient_to_add.shape) - len(self.grad.shape) :]

        expanded_axes = []

        # Iterate through matched axes to find out which were expanded during broadcasting.
        for axis in matched_axes:
            if (
                self.grad.shape[axis - len(unmatched_axes)]
                == gradient_to_add.shape[axis]
            ):  # If dimensions match, no broadcasting occured.
                continue
            elif (
                self.grad.shape[axis - len(unmatched_axes)] == 1
            ):  # Broadcasting occured.
                expanded_axes.append(axis)
            else:
                # The shapes are incompatible.
                raise Exception("Arrays could not have been broadcasted")

        # Reverse the lists so that we can iterate through them in reverse order.
        # This way, the indexes of the axes remain the same.
        unmatched_axes.reverse(), expanded_axes.reverse()

        # All expanded axes must be summed over, but kept as an axis.
        for axis in expanded_axes:
            gradient_to_add = np.expand_dims(gradient_to_add.sum(axis), axis)

        # All unmatched axes (the leading axes in the larger array) must be summed over.
        for axis in unmatched_axes:
            gradient_to_add = gradient_to_add.sum(axis)

        return gradient_to_add

    def zero_grad(self) -> None:
        """
        Reset the gradient of the tensor to zero.
        """
        self.grad = np.zeros(self.value.shape)

    def backpropagate_a(self) -> None:
        """
        Backpropagate the gradient to the first operand.
        """
        pass

    def backpropagate_b(self) -> None:
        """
        Backpropagate the gradient to the second operand.
        """
        pass

    def cast_to_tensor(
        self, x: Union[int, float, bool, np.ndarray, "Tensor"]
    ) -> "Tensor":
        """
        Cast a value to a Tensor object if it is not already a Tensor object.

        Args:
            x (int, float, bool, np.array or Tensor): The value to be casted.
        """
        if not isinstance(x, Tensor):
            return Tensor(x)
        return x

    def copy(self) -> "Tensor":
        """
        Return a copy of the tensor.

        Returns:
            Tensor: The copy of the tensor.
        """
        copy = Tensor(self.value.copy())
        copy.a = self.a
        copy.b = self.b
        copy.grad = self.grad.copy()
        return copy

    def __add__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Add":
        return Add(self, self.cast_to_tensor(b))

    def __radd__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Add":
        return Add(self.cast_to_tensor(a), self)

    def __sub__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Add":
        return Add(self, self.cast_to_tensor(-1 * b))

    def __rsub__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Add":
        return Add(self.cast_to_tensor(a), -1 * self)

    def __mul__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Mul":
        return Mul(self, self.cast_to_tensor(b))

    def __rmul__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Mul":
        return Mul(self.cast_to_tensor(a), self)

    def __truediv__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Mul":
        return Mul(self, self.cast_to_tensor(b**-1))

    def __rtruediv__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Mul":
        return Mul(self.cast_to_tensor(a), self**-1)

    def __matmul__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Dot":
        return Dot(self, self.cast_to_tensor(b))

    def __rmatmul__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Dot":
        return Dot(self.cast_to_tensor(a), self)

    def __pow__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Pow":
        return Pow(self, self.cast_to_tensor(b))

    def __rpow__(self, a: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Pow":
        return Pow(self.cast_to_tensor(a), self)

    def __gt__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Tensor":
        return Tensor(self.value > self.cast_to_tensor(b).value)

    def __ge__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Tensor":
        return Tensor(self.value >= self.cast_to_tensor(b).value)

    def __lt__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Tensor":
        return Tensor(self.value < self.cast_to_tensor(b).value)

    def __le__(self, b: Union[int, float, bool, np.ndarray, "Tensor"]) -> "Tensor":
        return Tensor(self.value <= self.cast_to_tensor(b).value)

    def __str__(self) -> str:
        return f"Tensor({self.value}), grad_fn={type(self).__name__}"

    def sum(self, axis: Optional[int] = None) -> "Sum":
        return Sum(self, axis)

    def min(self) -> "Tensor":
        return Tensor(self.value.min())

    def max(self) -> "Tensor":
        return Max(self)

    def swapaxes(self, axis1: int, axis2: int) -> "SwapAxes":
        return SwapAxes(self, axis1, axis2)


class Add(Tensor):
    """
    Represents the addition operation in the computational graph.

    Inherits from Tensor and represents the addition of two tensors.
    """

    def __init__(self, a: "Tensor", b: "Tensor") -> None:
        """
        Initialize the Add operation.

        Args:
            a (Tensor): The first operand.
            b (Tensor): The second operand.
        """
        super().__init__(a.value + b.value)
        self.a, self.b = a, b

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first operand.
        """
        self.a.add_gradient(self.grad)

    def backpropagate_b(self) -> None:
        """
        Backpropagate through the second operand.
        """
        self.b.add_gradient(self.grad)


class Mul(Tensor):
    """
    Represents the multiplication operation in the computational graph.

    Inherits from Tensor and represents the (element-wise) multiplication of two tensors.

    Attributes:
        a, b (Tensor): The tensors from which this tensor was derived.
    """

    def __init__(self, a: "Tensor", b: "Tensor") -> None:
        """
        Initialize the Mul operation.

        Args:
            a (Tensor): The first operand.
            b (Tensor): The second operand.
        """
        super().__init__(a.value * b.value)
        self.a, self.b = a, b

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first operand.

        As per the chain rule, the gradient of the first operand is the gradient of the output
        multiplied by the gradient of the second operand.
        """
        gradient = self.b.value * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self) -> None:
        """
        Backpropagate through the second operand.

        As per the chain rule, the gradient of the second operand is the gradient of the output
        multiplied by the gradient of the first operand.
        """
        gradient = self.a.value * self.grad
        self.b.add_gradient(gradient)


class Sum(Tensor):
    """
    Represents the summation operation in the computational graph.

    Inherits from Tensor and represents the summation of a tensor along a given axis.

    Attributes:
        a (Tensor): The tensor from which this tensor was derived.
        axis (int): The axis along which the summation was performed.
    """

    def __init__(self, a: "Tensor", axis: Optional[int] = None) -> None:
        super().__init__(a.value.sum(axis))
        self.a = a
        self.axis = axis

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first and only operand.

        As with addition, the gradient of the operand is the gradient of the output.
        """
        if self.axis is not None:
            gradient = np.expand_dims(self.grad, self.axis).repeat(
                self.a.shape[self.axis], self.axis
            )
        else:
            gradient = np.ones(self.a.shape) * self.grad
        self.a.add_gradient(gradient)


class Dot(Tensor):
    """
    Represents the dot product operation in the computational graph.

    Inherits from Tensor and represents the dot product of two tensors.

    Attributes:
        a, b (Tensor): The tensors from which this tensor was derived.
    """

    def __init__(self, a: "Tensor", b: "Tensor") -> None:
        super().__init__(a.value @ b.value)
        self.a, self.b = a, b

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first operand.

        As per the chain rule, the gradient of the first operand is the dot product of the gradient of the output
        with the transpose of the second operand.
        """
        gradient = self.grad @ self.b.value.swapaxes(-2, -1)
        self.a.add_gradient(gradient)

    def backpropagate_b(self) -> None:
        """
        Backpropagate through the second operand.

        As per the chain rule, the gradient of the second operand is the dot product of the transpose of the first
        operand with the gradient of the output.
        """
        gradient = self.a.value.swapaxes(-2, -1) @ self.grad
        self.b.add_gradient(gradient)


class Pow(Tensor):
    """
    Represents the power operation in the computational graph.

    Inherits from Tensor and represents the power of two tensors.

    Attributes:
        a, b (Tensor): The tensors from which this tensor was derived.
    """

    def __init__(self, a: "Tensor", b: "Tensor") -> None:
        super().__init__(a.value**b.value)
        self.a, self.b = a, b

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first operand.

        Follows the chain rule for the derivative of a function raised to a power.
        """
        gradient = self.b.value * self.a.value ** (self.b.value - 1) * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self) -> None:
        """
        Backpropagate through the second operand.

        Follows the chain rule for the derivative of an exponential function.
        """
        gradient = np.log(self.a.value) * self.a.value**self.b.value * self.grad
        self.b.add_gradient(gradient)


class SwapAxes(Tensor):
    """
    Represents the swap axes operation in the computational graph.

    Inherits from Tensor and represents the swap axes of a tensor.

    Attributes:
        a (Tensor): The tensor from which this tensor was derived.
        axis1, axis2 (int): The axes to be swapped.
    """

    def __init__(self, a: "Tensor", axis1: int, axis2: int) -> None:
        super().__init__(a.value.swapaxes(axis1, axis2))
        self.a = a
        self.axis1 = axis1
        self.axis2 = axis2

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first and only operand.

        Undo the swap axes operation on the gradient.
        """
        gradient = self.grad.swapaxes(self.axis2, self.axis1)
        self.a.add_gradient(gradient)


class Max(Tensor):
    """
    Represents the max operation in the computational graph.

    Inherits from Tensor and represents the max of a tensor.
    """

    def __init__(self, a: "Tensor") -> None:
        super().__init__(a.value.max())
        self.a = a
        self.idx_max = np.unravel_index(a.value.argmax(), a.shape)

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first and only operand.

        The gradient is added to the index of the max value. All other indices are zero.
        """
        gradient = np.zeros(self.a.shape)
        gradient[self.idx_max] = self.grad
        self.a.add_gradient(gradient)


class Min(Tensor):
    """
    Represents the min operation in the computational graph.

    Inherits from Tensor and represents the min of a tensor.
    """

    def __init__(self, a: "Tensor") -> None:
        super().__init__(a.value.min())
        self.a = a
        self.idx_min = np.unravel_index(a.value.argmin(), a.shape)

    def backpropagate_a(self) -> None:
        """
        Backpropagate through the first and only operand.

        The gradient is added to the index of the min value. All other indices are zero.
        """
        gradient = np.zeros(self.a.shape)
        gradient[self.idx_min] = self.grad
        self.a.add_gradient(gradient)
