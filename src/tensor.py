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
            self.grad = 0

    def backwards(self, queue=None):
        if queue is None:
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

    def backpropagate_a(self):
        self.a.grad = np.array(-1)

    def backpropagate_b(self):
        self.b.grad = np.array(-1)

    def add_to_queue(self, el, queue):
        if not el is None and not el in queue:
            return queue + [el]
        return queue

    def __add__(self, b):
        return Add(self, Tensor(b))

    def __radd__(self, a):
        return Add(Tensor(a), self)

    def __mul__(self, b):
        return Mul(self, Tensor(b))

    def __rmul__(self, a):
        return Mul(Tensor(a), self)


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