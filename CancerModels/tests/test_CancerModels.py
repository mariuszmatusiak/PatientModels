import pytest
import matplotlib.pyplot as plt
import numpy as np
from ..src.CancerModels import DynamicalTumorGrowthModel

class TestCancerModelsClass:
    def test_one(self):
        pass

    def test_DefaultParameters(self):
        DAYS = 85
        t = [*range(1, DAYS+1)] # time, or using the np.linspace like t  = np.linspace(0, 84, 85)
        u = 5.0 # control signal u(t), chemo drug as [mg per ml per day]
        
        model = DynamicalTumorGrowthModel()
        result = model.solveModel(t=t, u=u)

        plt.figure(1)
        plt.plot(t, result[:, DynamicalTumorGrowthModel.ModelArguments.P], label="P(t)")
        plt.xlabel("Time [days]")
        plt.ylabel("P(t)")
        plt.title("Concentration of P cells")
        plt.legend(loc="best")

        plt.figure(2)
        plt.plot(t, result[:, DynamicalTumorGrowthModel.ModelArguments.DC], label="Dc(t)")
        plt.xlabel("Time [days]")
        plt.ylabel("Dc(t)")
        plt.title("Concentration of drug")
        plt.legend(loc="best")
        plt.show()

        pass