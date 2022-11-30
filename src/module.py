from abc import ABC, abstractmethod
from src.tensor import Tensor

class Module(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        parameters = {}
        for attribute, value in self.__dict__.items():
            if isinstance(value, Tensor):
                parameters[attribute] = value
            elif isinstance(value, Module):
                sub_params = value.parameters()
                for sub_param in sub_params:
                    parameters[attribute + '__' + sub_param] = sub_params[sub_param]

        return parameters



