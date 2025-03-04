from abc import ABC, abstractmethod
import pandas as pd


class CommonStrategy(ABC):
    @abstractmethod
    def calculate(self, feature, label):
        pass
