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

    def backwards(self, queue=None):
        if queue is None:
            if self.value.ndim > 0:
                Sum(self).backwards()
                return None
            else:
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
        self.grad = 0

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

    def __add__(self, b):
        return Add(self, self.cast_to_tensor(b))

    def __radd__(self, a):
        return Add(self.cast_to_tensor(a), self)

    def __mul__(self, b):
        return Mul(self, self.cast_to_tensor(b))

    def __rmul__(self, a):
        return Mul(self.cast_to_tensor(a), self)

    def sum(self):
        return Sum(self)


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
