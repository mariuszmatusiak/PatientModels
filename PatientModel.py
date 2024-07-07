from abc import ABC, abstractmethod

class DynamicalPatientModelBaseClass(ABC):

    @abstractmethod
    def solveModel(self):
        """
        Base abstract method to solve a specific patient model
        """
        pass