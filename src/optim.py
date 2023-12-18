from src.tensor import Tensor
from typing import Dict


class SGD:
    """
    Implements a stochastic gradient descent optimizer.

    Arguments:
        parameters (Dict[str, Tensor]): parameters to be optimized.
        lr (float): learning rate.
    """

    def __init__(self, parameters: Dict[str, Tensor], lr: float = 1e-4) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for parameter in self.parameters.values():
            parameter.value = parameter.value - self.lr * parameter.grad
