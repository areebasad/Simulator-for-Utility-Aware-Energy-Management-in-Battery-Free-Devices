from typing import Any, Dict
from em_sim.core import Component, VariableMode
import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize, curve_fit
import copy
import logging

from .predictors import EnergyPredictor
#from .profiles import profiles

#np.seterr(all='raise')


class EnergyManager(Component):

    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Base Energy Manager")

    def calc_duty_cycle(self, *args):
        pass


class PredictiveManager(EnergyManager):
    def __init__(self, predictor):
        self.predictor = predictor

    def step(self, doy, soc):
        raise NotImplementedError()


class PIDController:
    def __init__(self, coefficients=None):
        self.e_sum = 0.0
        self.e_prev = 0.0

        self.coefficients = {
            'k_p': 1.0,
            'k_i': 0.0,
            'k_d': 0.0
        }
        if coefficients:
            self.coefficients.update(coefficients)

    def calculate(self, set_point, process_variable):

        e = (process_variable - set_point)
        self.e_sum += e
        e_d = e - self.e_prev
        self.e_prev = e

        output = (
            self.coefficients['k_p'] * e
            + self.coefficients['k_i'] * self.e_sum
            + self.coefficients['k_d'] * e_d
        )

        return output


class PREACT(PredictiveManager):
    def __init__(
            self, predictor, battery_capacity, battery_age_rate=0.0, **kwargs):

        PredictiveManager.__init__(self, predictor)

        self.controller = PIDController(
            kwargs.get('control_coefficients', None))

        if('utility_function' in kwargs.keys()):
            self.utility_function = kwargs['utility_function']
        else:
            def utility(doy):
                return np.ones(len(doy))
            self.utility_function = utility

        self.battery_capacity = battery_capacity
        self.battery_age_rate = battery_age_rate

        self.step_count = 0

        self.log = logging.getLogger("PREACT")

    def estimate_capacity(self, offset=0):
        return (
            self.battery_capacity
            * (1.0 - (self.step_count + offset) * self.battery_age_rate)
        )

    def calc_duty_cycle(self, doy, e_in, soc):

        self.predictor.update(doy, e_in)

        e_pred = self.predictor.predict(
            np.arange(doy + 1, doy + 1 + 365))

        e_req = self.utility_function(np.arange(doy + 1, doy + 1 + 365))
        f_req = np.mean(e_pred)/np.mean(e_req)

        # ideal soc
        d_soc_1y = np.cumsum(e_pred - f_req*e_req)
        
        #calculate peak-to-peak amplitude
        p2p_1y = max(d_soc_1y)-min(d_soc_1y)
        
        # If it is less, no-issues of battery constraints.
        if(p2p_1y < self.estimate_capacity() / 10000):
            return 1.0 # return duty cycle 100%

        f_scale = min(1.0, self.estimate_capacity() / p2p_1y)

        offset = (self.estimate_capacity() - f_scale * p2p_1y) / 2

        self.soc_target = f_scale * (d_soc_1y[0] - min(d_soc_1y)) + offset

        self.step_count += 1

        duty_cycle = self.controller.calculate(
            self.soc_target / self.estimate_capacity(),
            (soc + e_pred[0]) / self.estimate_capacity()
        )

        return max(0.0, min(1.0, duty_cycle))

    def step(self, doy, soc):
        return super().step(doy, soc)




