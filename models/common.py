from abc import ABC, abstractmethod


class ClassifierBackend(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self, train_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, test_data, **kwargs):
        pass
