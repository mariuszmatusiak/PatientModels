from scipy.integrate import ode, odeint
from enum import IntEnum
from ...PatientModel import DynamicalPatientModelBaseClass

class DynamicalTumorGrowthModel(DynamicalPatientModelBaseClass):
    """
    Tumor growth model for angiogenic stimulator/inhibitor control, presented by Hahnfield et al. (1999) and modified.
    References:
        [1] https://doi.org/10.1016/j.cmpb.2014.01.002
        [2] https://ieeexplore.ieee.org/document/5281162
        [3] https://www.sciencedirect.com/science/article/pii/S2666720722000029#b14
    Implemented by Mariusz Matusiak <mariusz.m.matusiak@gmail.com>, Copyright 2024

    Attributes:
        P       (float): rapid-splitting cancer cells, the so-called P cells
        Q       (float): non-cyclic cancer cell at the start of the treatment
        Y       (float): normal cells
        thetha  (float): growth rate
        J       (float): carrying capacity of the normal cell
        g(t)    (float): rate of cancerous cell killing per unit drug 
        f_1     (float): constant relating cell kill rate and drug concentration
        u(t)    (float): infused drug dose intravenously
        D_c     (float): drug concentration
        lambda  (float): drug decay associated with the metabolism inside the body
        T       (float): toxicity

    Constraints:
        10 <= D_c(t) <= 50;
        T(t) <= 100;
        Y_min <= Y(t) <= J;
    """

    def __init__(self, a = 0.5, m = 0.218, n = 0.477, b = 0.05, thetha = 0.1, ro = 0.38, 
                 J = 10**9, f_1 = 0.7, lmbda = 0.7, P0 = 2 * 10**11, Q0 = 8 * 10**11, DC_min = 10, DC_max = 50, DC0 = 10, 
                 T_max = 100, T0 = 0, Y0 = 10**9, Y_min = 10**8):
        """
        Creates a DynamicalTumorGrowthModel object

        Args:
            a (float, optional): Rate of growth of P cells 1/day. Defaults to 0.5.
            m (float, optional): Mutation rate of P cells to Q cells 1/day. Defaults to 0.218.
            n (float, optional): Natural end of cycling cells /day. Defaults to 0.477.
            b (float, optional): Mutation rate of Q cells to P cells /day. Defaults to 0.05.
            thetha (float, optional): Rate of normal cell growth /day. Defaults to 0.1.
            ro (float, optional): Drug toxicity. Defaults to 0.38.
            J (int, optional): Carrying capacity of a normal cell. Defaults to 10**9.
            P0 (int, optional): Proliferating cells population at the start of the treatment. Defaults to (2 * 10**11).
            Q0 (int, optional): Quiescent cancer cell population at the start of the treatment. Defaults to (8* 10**11).
            DC_min (int, optional): Minimum drug concentration. Defaults to 10.
            DC0 (int, optional): Drug concentration at the start of the treatment. Defaults to 10.
            DC_max (int, optional): Maximum drug concentration Defaults to 50.
            T0 (int, optional): Toxicity at the start of the treatment. Defaults to 0.
            T_max (int, optional): Maximum toxicity. Defaults to 100.
            G0 (int, optional): Cancerous cell killing per unit drug. Defaults to 0.
            Y0 (int, optional): Normal cell population at the start of the treatment. Defaults to 10**9.
            Y_min (int, optional): Limited normal cell population. Defaults to 10**8.
        """
        self.a      = a
        self.m      = m
        self.n      = n
        self.b      = b
        self.thetha = thetha
        self.ro     = ro
        self.J      = J
        self.lmbda  = lmbda
        self.f_1    = f_1
        self.P0     = P0
        self.Q0     = Q0
        self.DC_min = DC_min
        self.DC0    = DC0
        self.DC_max = DC_max
        self.T0     = T0
        self.T_max  = T_max
        self.Y0     = Y0
        self.Y_min  = Y_min
        self.uoft   = 0
        self.goft   = 0

    def calculateRateOfQCells(self, Qoft: float, Poft : float, m: float, b: float):
        """
        Rate of change of rapid-splitting P cells (entirely sensitive)
        dQ/dt = m * P(t) - b * Q(t)
        Q(0) = Q_0 - rate of change of Q cells

        Args:
            Poft (float): rapid-splitting cancer cells 
            Qoft (float): non-cyclic quiescent cancer cells
            m (float): _description_
            b (float): _description_
        """
        dQdt = m * Poft - b * Qoft 
        return dQdt
        
    def calculateRateOfPCells(self, Poft : float, Qoft : float, goft : float, a : float, m : float, n : float, b : float):
        """
        Rate of change of rapid-splitting P cells (entirely sensitive)
        dP/dt = (a-m-n)*P(t)+bQ(t)-g(t)P(t)

        Args:
            Poft (float): rapid-splitting cancer cells
            Qoft (float): non-cyclic quiescent cancer cells
            goft (float): rate of cancerous cell killing per unit drug
               a (float): rate of growth of P cells /day
               m (float): mutation rate of P cells to Q cells /day
               n (float): the natural end of cycling cells /day
               b (float): mutation rate of Q cells to P cells /day

        Returns:
            dPdt (float): Rate of rapid-splitting P cells (entirely sensitive)
        """
        dPdt = (a - m - n) * Poft + b * Qoft - goft * Poft
        return dPdt
    
    def calculateRateOfYNormalCells(self, Yoft : float, goft : float, thetha : float, J : float):
        """
        Rate of change of normal Y cells population.
        dY/dt = thethaY(t)(1-Y(t)/J) - g(t)Y(t), Y(0) = Y_0 - rate of change of normal cells

        Args:
            goft (float): rate of cancerous cell killing per unit drug
            Yoft (float): _description_
            thetha (float): _description_
            J (float): _description_

        Returns:
            _type_: _description_
        """
        dYdt = thetha * Yoft * (1 - Yoft/J) - goft * Yoft
        return dYdt

    def calculateRateOfDrugConcentrationAtTumor(self, dcoft : float, uoft: float, lambda_ : float):
        """
        Rate of change in drug concentration Dc at the tumor.
        dDc/dt = u(t) - Lambda * Dc(t), Dc(0) = D0 - drug dose infused intravenously (u(t)) and the rate of change in drug concentration Dc at the tumor StopIteration

        Args:
            lambda_ (float): _description_
            dcoft (float): _description_
            uoft (float): _description_

        Returns:
            _type_: _description_
        """
        dDcdt = uoft - lambda_ * dcoft
        return dDcdt

    def calculateRateOfChemoToxicity(self, Toft : float, Dcoft: float, ro: float):
        """
        Rate of the adverse effect of chemo drug in terms of toxicity on other body parts
        dT/dt = D_c(t) - ro T(t), T(0) = T_0 - adverse effect of chemo drug in terms of Toxicity on other body parts forcing the set of constraints:

        Args:
            dcoft (float): _description_
            Toft (float): _description_
            ro (float): _description_
        """
        dTdt = Dcoft - ro * Toft
        return dTdt
    
    def calculateRateOfCancerCellKillingPerDrugUnit(self, Dcoft : float, f_1 : float):
        """
        Rate of cancerous cell killing per unit drug

        Args:
            f_1 (float): _description_
            Dcoft (float): _description_

        Returns:
            goft (float): _description_
        """
        goft = f_1 * Dcoft
        return goft

    class ModelArguments(IntEnum):
        """
        Helper enum class for indexing X[] list values
        """
        P = 0
        Q = 1
        Y = 2
        DC = 3
        T = 4

    def model(self, x : list, t : list, uoft : float):
        """
        Calculate model response for a single iteration.

        Args:
            x (list): _description_
            t (list): _description_
            uoft (float): _description_

        Returns:
            _type_: _description_
        """
        self.uoft = uoft
        self.goft = self.calculateRateOfCancerCellKillingPerDrugUnit(x[DynamicalTumorGrowthModel.ModelArguments.DC],
                                                                self.f_1)
        dPdt = self.calculateRateOfPCells(x[DynamicalTumorGrowthModel.ModelArguments.P], 
                                          x[DynamicalTumorGrowthModel.ModelArguments.Q],
                                          self.goft, self.a, self.m, self.n, self.b)
        dQdt = self.calculateRateOfQCells(x[DynamicalTumorGrowthModel.ModelArguments.Q],
                                          x[DynamicalTumorGrowthModel.ModelArguments.P],
                                          self.m, self.b)
        dYdt = self.calculateRateOfYNormalCells(x[DynamicalTumorGrowthModel.ModelArguments.Y],
                                                self.goft, self.thetha, self.J)
        dDcdt = self.calculateRateOfDrugConcentrationAtTumor(x[DynamicalTumorGrowthModel.ModelArguments.DC],
                                                             self.uoft, self.lmbda)
        dTdt = self.calculateRateOfChemoToxicity(x[DynamicalTumorGrowthModel.ModelArguments.T],
                                                 x[DynamicalTumorGrowthModel.ModelArguments.DC],
                                                 self.ro)
        result = [0.0] * len(DynamicalTumorGrowthModel.ModelArguments)
        result[DynamicalTumorGrowthModel.ModelArguments.P] = dPdt
        result[DynamicalTumorGrowthModel.ModelArguments.Q] = dQdt
        result[DynamicalTumorGrowthModel.ModelArguments.Y] = dYdt
        result[DynamicalTumorGrowthModel.ModelArguments.DC] = dDcdt
        result[DynamicalTumorGrowthModel.ModelArguments.T] = dTdt
        return result

    def solveModel(self, t : list, u : float):
        # Initialize starting point X0
        X0 = [0.0] * len(DynamicalTumorGrowthModel.ModelArguments)
        X0[DynamicalTumorGrowthModel.ModelArguments.P] = self.P0
        X0[DynamicalTumorGrowthModel.ModelArguments.Q] = self.Q0
        X0[DynamicalTumorGrowthModel.ModelArguments.Y] = self.Y0
        X0[DynamicalTumorGrowthModel.ModelArguments.DC] = self.DC0
        X0[DynamicalTumorGrowthModel.ModelArguments.T] = self.T0
        result = odeint(func=self.model, y0=X0, t=t, args=(u,))
        return result
