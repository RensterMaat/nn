import numpy as np


class Tensor:
    def __init__(self, value):
        if isinstance(value, Tensor):
            self.value = value.value
            self.a, self.b = value.a, value.b
            self.grad = value.grad
        else:
            self.value = np.array(value)
            self.a, self.b = None, None
            self.grad = np.zeros(self.value.shape)

        self.shape = self.value.shape
        self.ndim = self.value.ndim

    def backwards(self, queue=None):
        if queue is None:
            if self.value.ndim > 0:
                self.sum().backwards()
                return None
            queue = []
            self.grad = np.ones(self.value.shape)

        if self.a is not None:
            self.backpropagate_a()
            queue = self.add_to_queue(self.a, queue)
        
        if self.b is not None:
            self.backpropagate_b()   
            queue = self.add_to_queue(self.b, queue)

        if queue:
            queue[0].backwards(queue=queue[1:])

    def add_gradient(self, gradient_to_add):
        # if gradient_to_add.ndim == self.grad.ndim + 1:
        #     self.grad = self.grad + gradient_to_add.sum(axis=-gradient_to_add.ndim)
        # else:
        #     self.grad = self.grad + gradient_to_add
        gradient_to_add = self.undo_broadcasting(gradient_to_add)
        self.grad = self.grad + gradient_to_add

    def undo_broadcasting(self, gradient_to_add):
        axes = list(range(len(gradient_to_add.shape)))

        unmatched_axes = axes[:len(gradient_to_add.shape)-len(self.grad.shape)]
        matched_axes = axes[len(gradient_to_add.shape)-len(self.grad.shape):]

        expanded_axes = []
        for axis in matched_axes:
            if self.grad.shape[axis - len(unmatched_axes)] == gradient_to_add.shape[axis]:
                continue
            elif self.grad.shape[axis - len(unmatched_axes)] == 1:
                expanded_axes.append(axis)
            else:
                raise Exception('Arrays could not have been broadcasted')

        unmatched_axes.reverse(), expanded_axes.reverse()

        for axis in expanded_axes:
            gradient_to_add = np.expand_dims(gradient_to_add.sum(axis), axis)

        for axis in unmatched_axes:
            gradient_to_add = gradient_to_add.sum(axis)

        return gradient_to_add

    def zero_grad(self):
        self.grad = np.zeros(self.value.shape)

    def backpropagate_a(self):
        self.a.grad = np.array(-1)

    def backpropagate_b(self):
        self.b.grad = np.array(-1)

    def add_to_queue(self, el, queue):
        if not el is None and not el in queue:
            return queue + [el]
        return queue

    def cast_to_tensor(self, x):
        if not isinstance(x, Tensor):
            return Tensor(x)
        return x

    def copy(self):
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
        return Mul(self, self.cast_to_tensor(b ** -1))

    def __rtruediv__(self, a):
        return Mul(self.cast_to_tensor(a), self ** -1)

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
        return f'Tensor({self.value}), grad_fn={type(self).__name__}'
        
    def sum(self, axis=None):
        return Sum(self, axis)

    def min(self):
        return Tensor(self.value.min())

    def max(self):
        return Max(self)

    def swapaxes(self, axis1, axis2):
        return SwapAxes(self, axis1, axis2)


class Add(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value + b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        self.a.add_gradient(self.grad)

    def backpropagate_b(self):
        self.b.add_gradient(self.grad)


class Mul(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value * b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        gradient = self.b.value * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
        gradient = self.a.value * self.grad
        self.b.add_gradient(gradient)


class Sum(Tensor):
    def __init__(self, a, axis=None):
        super().__init__(a.value.sum(axis))
        self.a = a
        self.axis = axis

    def backpropagate_a(self):
        if self.axis is not None:
            gradient = np.expand_dims(
                self.grad, self.axis
            ).repeat(
                self.a.shape[self.axis], 
                self.axis
            )
        else:
            gradient = np.ones(self.a.shape) * self.grad
        self.a.add_gradient(gradient)


class Dot(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value @ b.value)
        self.a, self.b = a,b

    def backpropagate_a(self):
        gradient = self.grad @ self.b.value.swapaxes(-2, -1)
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
        gradient = self.a.value.swapaxes(-2,-1) @ self.grad
        self.b.add_gradient(gradient)


class Pow(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value ** b.value)
        self.a, self.b = a,b

    def backpropagate_a(self):
        gradient = self.b.value * self.a.value ** (self.b.value-1) * self.grad
        self.a.add_gradient(gradient)

    def backpropagate_b(self):
        gradient = np.log(self.a.value) * self.a.value ** self.b.value * self.grad
        self.b.add_gradient(gradient)


class SwapAxes(Tensor):
    def __init__(self, a, axis1, axis2):
        super().__init__(a.value.swapaxes(axis1, axis2))
        self.a = a
        self.axis1 = axis1
        self.axis2 = axis2

    def backpropagate_a(self):
        gradient = self.grad.swapaxes(self.axis2, self.axis1)
        self.a.add_gradient(gradient)


class Max(Tensor):
    def __init__(self, a):
        super().__init__(a.value.max())
        self.a = a
        self.idx_max = np.unravel_index(a.value.argmax(), a.shape)

    def backpropagate_a(self):
        gradient = np.zeros(self.a.shape)
        gradient[self.idx_max] = self.grad
        self.a.add_gradient(gradient)


class Min(Tensor):
    def __init__(self, a):
        super().__init__(a.value.min())
        self.a = a
        self.idx_min = np.unravel_index(a.value.argmin(), a.shape)

    def backpropagate_a(self):
        gradient = np.zeros(self.a.shape)
        gradient[self.idx_min] = self.grad
        self.a.add_gradient(gradient)
