class SGD:
    def __init__(self, parameters, lr=1e-4):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters.values():
            parameter.value = parameter.value - self.lr * parameter.grad