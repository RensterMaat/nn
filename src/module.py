from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)