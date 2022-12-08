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
        
    def sum(self):
        return Sum(self)

    def min(self):
        return Tensor(self.value.min())

    def max(self):
        return Tensor(self.value.max())


class Add(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value + b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        self.a.grad = self.a.grad + self.grad

    def backpropagate_b(self):
        self.b.grad = self.b.grad + self.grad


class Mul(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value * b.value)
        self.a, self.b = a, b

    def backpropagate_a(self):
        gradient = self.b.value * self.grad
        self.a.grad = self.a.grad + gradient

    def backpropagate_b(self):
        gradient = self.a.value * self.grad
        self.b.grad = self.b.grad + gradient


class Sum(Tensor):
    def __init__(self, a):
        super().__init__(a.value.sum())
        self.a = a

    def backpropagate_a(self):
        self.a.grad = self.a.grad + self.grad


class Dot(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value @ b.value)
        self.a, self.b = a,b

    def backpropagate_a(self):
        gradient = self.grad @ self.b.value.T
        self.a.grad = self.a.grad + gradient

    def backpropagate_b(self):
        gradient = self.a.value.T @ self.grad
        self.b.grad = self.b.grad + gradient


class Pow(Tensor):
    def __init__(self, a, b):
        super().__init__(a.value ** b.value)
        self.a, self.b = a,b

    def backpropagate_a(self):
        gradient = self.b.value * self.a.value ** (self.b.value-1) * self.grad
        self.a.grad = self.a.grad + gradient

    def backpropagate_b(self):
        gradient = np.log(self.a.value) * self.a.value ** self.b.value * self.grad
        self.b.grad = self.b.grad + gradient