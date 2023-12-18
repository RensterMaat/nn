class SGD:
    """
    Implements a stochastic gradient descent optimizer.

    Arguments:
        parameters (dict): parameters to be optimized.
        lr (float): learning rate.
    """

    def __init__(self, parameters, lr=1e-4):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        for parameter in self.parameters.values():
            parameter.value = parameter.value - self.lr * parameter.grad
