import numpy as np


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

    def __init__(self, value):
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

    def backwards(self):
        """
        Perform backpropagation to compute gradients for all tensors in the computational graph.
        """
        if self.value.ndim > 0:
            self.sum().backwards()
            return None

        order = self.determine_backpropagation_order()

        self.grad = 1

        for node in order:
            if node.a is not None:
                node.backpropagate_a()

            if node.b is not None:
                node.backpropagate_b()

    def determine_backpropagation_order(self):
        """
        Determine the order of tensors for backpropagation.

        Returns:
            list: Ordered list of tensors for backpropagation.
        """
        a_order, b_order = [], []

        if self.a:
            a_order = self.a.determine_backpropagation_order()

        if self.b:
            b_order = self.b.determine_backpropagation_order()

        combined_order = self.topologically_ordered_merge(a_order, b_order)

        return [self] + combined_order

    def topologically_ordered_merge(self, a_order, b_order):
        """
        Merge two lists of tensors maintaining topological order.

        Args:
            a_order (list): The first list of tensors.
            b_order (list): The second list of tensors.

        Returns:
            list: The merged list of tensors in topological order.
        """
        b_node_vs_index = {node: index for (index, node) in enumerate(b_order)}

        merged = []
        a_pointer, b_pointer = 0, 0
        for a_index, a_node in enumerate(a_order):
            if a_node in b_node_vs_index:
                b_index = b_node_vs_index[a_node]

                merged.extend(a_order[a_pointer:a_index])
                merged.extend(b_order[b_pointer:b_index])
                merged.extend([a_node])

                a_pointer = a_index + 1
                b_pointer = b_index + 1

        merged.extend(a_order[a_pointer:])
        merged.extend(b_order[b_pointer:])

        return merged

    def add_gradient(self, gradient_to_add):
        """
        Add a gradient to the tensor's gradient.

        Args:
            gradient_to_add (np.array): The gradient to be added.
        """
        gradient_to_add = self.undo_broadcasting(gradient_to_add)
        self.grad = self.grad + gradient_to_add

    def undo_broadcasting(self, gradient_to_add):
        """
        Adjust the gradient shape to match the tensor's shape, reversing any broadcasting that occurred during forward pass.

        Args:
            gradient_to_add (np.array): The gradient to be reshaped.

        Returns:
            np.array: The reshaped gradient.
        """
        axes = list(range(len(gradient_to_add.shape)))

        unmatched_axes = axes[: len(gradient_to_add.shape) - len(self.grad.shape)]
        matched_axes = axes[len(gradient_to_add.shape) - len(self.grad.shape) :]

        expanded_axes = []
        for axis in matched_axes:
            if (
                self.grad.shape[axis - len(unmatched_axes)]
                == gradient_to_add.shape[axis]
            ):
                continue
            elif self.grad.shape[axis - len(unmatched_axes)] == 1:
                expanded_axes.append(axis)
            else:
                raise Exception("Arrays could not have been broadcasted")

        unmatched_axes.reverse(), expanded_axes.reverse()

        for axis in expanded_axes:
            gradient_to_add = np.expand_dims(gradient_to_add.sum(axis), axis)

        for axis in unmatched_axes:
            gradient_to_add = gradient_to_add.sum(axis)

        return gradient_to_add

    def zero_grad(self):
        """
        Reset the gradient of the tensor to zero.
        """
        self.grad = np.zeros(self.value.shape)

    def backpropagate_a(self):
        """
        Backpropagate the gradient to the first operand.
        """
        self.a.grad = np.array(-1)

    def backpropagate_b(self):
        """
        Backpropagate the gradient to the second operand.
        """
        self.b.grad = np.array(-1)

    def add_to_queue(self, el, queue):
        """
        Add an element to a queue if it is not None and not already in the queue.
        """
        if not el is None and not el in queue:
            return queue + [el]
        return queue

    def cast_to_tensor(self, x):
        """
        Cast a value to a Tensor object if it is not already a Tensor object.

        Args:
            x (int, float, bool, np.array or Tensor): The value to be casted.
        """
        if not isinstance(x, Tensor):
            return Tensor(x)
        return x

    def copy(self):
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

    def __add__(self, b):
        return Add(self, self.cast_to_tensor(b))

    def __radd__(self, a):
        return Add(self.cast_to_tensor(a), self)

    def __sub__(self, b):
        return Add(self, self.cast_to_tensor(-1 * b))

    def __rsub__(self, a):
        return Add(self.cast_to_tensor(a), -1 * self)

    def __mul__(self, b):
        return Mul(self, self.cast_to_tensor(b))

    def __rmul__(self, a):
        return Mul(self.cast_to_tensor(a), self)

    def __truediv__(self, b):
        return Mul(self, self.cast_to_tensor(b**-1))

    def __rtruediv__(self, a):
        return Mul(self.cast_to_tensor(a), self**-1)

    def __matmul__(self, b):
        return Dot(self, self.cast_to_tensor(b))

    def __rmatmul__(self, a):
        return Dot(self.cast_to_tensor(a), self)

    def __pow__(self, b):
        return Pow(self, self.cast_to_tensor(b))

    def __rpow__(self, a):
        return Pow(self.cast_to_tensor(a), self)

    def __gt__(self, b):
        return Tensor(self.value > self.cast_to_tensor(b).value)

    def __ge__(self, b):
        return Tensor(self.value >= self.cast_to_tensor(b).value)

    def __lt__(self, b):
        return Tensor(self.value < self.cast_to_tensor(b).value)

    def __le__(self, b):
        return Tensor(self.value <= self.cast_to_tensor(b).value)

    def __str__(self):
        return f"Tensor({self.value}), grad_fn={type(self).__name__}"

    def sum(self, axis=None):
        return Sum(self, axis)

    def min(self):
        return Tensor(self.value.min())

    def max(self):
        return Max(self)

    def swapaxes(self, axis1, axis2):
        return SwapAxes(self, axis1, axis2)


class Add(Tensor):
    """
    Represents the addition operation in the computational graph.

    Inherits from Tensor and represents the addition of two tensors.
    """

    def __init__(self, a, b):
        """
        Initialize the Add operation.

        Args:
            a (Tensor): The first operand.
            b (Tensor): The second operand.
        """
        super().__init__(a.value + b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        """
        Backpropagate through the first operand.
        """
        self.a.add_gradient(self.grad)

    def backpropagate_b(self):
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

    def __init__(self, a, b):
        """
        Initialize the Mul operation.

        Args:
            a (Tensor): The first operand.
            b (Tensor): The second operand.
        """
        super().__init__(a.value * b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        """
        Backpropagate through the first operand.

        As per the chain rule, the gradient of the first operand is the gradient of the output
        multiplied by the gradient of the second operand.
        """
        gradient = self.b.value * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
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

    def __init__(self, a, axis=None):
        super().__init__(a.value.sum(axis))
        self.a = a
        self.axis = axis

    def backpropagate_a(self):
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

    def __init__(self, a, b):
        super().__init__(a.value @ b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        """
        Backpropagate through the first operand.

        As per the chain rule, the gradient of the first operand is the dot product of the gradient of the output
        with the transpose of the second operand.
        """
        gradient = self.grad @ self.b.value.swapaxes(-2, -1)
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
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

    def __init__(self, a, b):
        super().__init__(a.value**b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        """
        Backpropagate through the first operand.

        Follows the chain rule for the derivative of a function raised to a power.
        """
        gradient = self.b.value * self.a.value ** (self.b.value - 1) * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
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

    def __init__(self, a, axis1, axis2):
        super().__init__(a.value.swapaxes(axis1, axis2))
        self.a = a
        self.axis1 = axis1
        self.axis2 = axis2

    def backpropagate_a(self):
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

    def __init__(self, a):
        super().__init__(a.value.max())
        self.a = a
        self.idx_max = np.unravel_index(a.value.argmax(), a.shape)

    def backpropagate_a(self):
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

    def __init__(self, a):
        super().__init__(a.value.min())
        self.a = a
        self.idx_min = np.unravel_index(a.value.argmin(), a.shape)

    def backpropagate_a(self):
        """
        Backpropagate through the first and only operand.

        The gradient is added to the index of the min value. All other indices are zero.
        """
        gradient = np.zeros(self.a.shape)
        gradient[self.idx_min] = self.grad
        self.a.add_gradient(gradient)
