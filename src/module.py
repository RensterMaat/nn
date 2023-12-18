from abc import ABC, abstractmethod
from src.tensor import Tensor
from typing import Dict


class Module(ABC):
    """
    Base class for all neural network modules.

    Your models should also subclass this class.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward computation performed at every call.

        Should be overridden by all subclasses.

        Arguments:
            x (Tensor): input to the module.
        """
        pass

    def __call__(self, x: Tensor) -> Tensor:
        """
        Calling an instance of a Module is the same as calling its forward method.

        Arguments:
            x (Tensor): input to the module.
        """
        return self.forward(x)

    def parameters(self) -> Dict[str, Tensor]:
        """
        Returns a dictionary containing all learnable parameters of the module and their names.

        Parameters are returned in the format {'parameter_name': parameter_tensor}. Subparameters
        are named by joining the name of the module and the parameter with a double underscore,
        e.g. 'module_name__parameter_name'.
        """
        parameters = {}
        for attribute, value in self.__dict__.items():
            if isinstance(value, Tensor):
                parameters[attribute] = value
            elif isinstance(value, Module):
                sub_params = value.parameters()
                for sub_param in sub_params:
                    parameters[attribute + "__" + sub_param] = sub_params[sub_param]

        return parameters

    def zero_grad(self) -> None:
        """
        Sets the gradients of all learnable parameters to zero.
        """
        for parameter in self.parameters().values():
            parameter.zero_grad()
