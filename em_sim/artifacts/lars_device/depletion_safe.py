from typing import Any, Dict
from em_sim.core import Component, VariableMode
from .predictors import Predictor
import pandas as pd
import math
import numpy as np


class EnergyManager(Component):        
       
    def calc_duty_cycle(self, *args):
        pass

    def step(self):
        raise NotImplementedError()


class DepletionSafe(EnergyManager):    
    
    def __init__(self, config: Dict[str, Any], buffer, scheduler):
        
        super().__init__(config, "Energy Manager: Depletion Safe")

        self.predictor = Predictor(config)
        self.buffer = buffer

        self.max_current_amp = config["device_max_current_amp"]
        self.sleep_current_amp = scheduler.sleep_current
        self.min_goal = config["device_mingoal"]
        self.simulation_ibal_tolerance = float("1e-7")      # tolerance of current balance (harvest - node consumption) [A]
                                                            # values below this threshold are treatet as steady state (Vc unchanged)
        self.simulation_volt_tolerance =  1e-7              # tolerance/resolution of voltage change [V] 1 × 10^-7
        self.simulation_steps_max = 50                      # maximum number of iterations for voltage simulation (newton)
        self.simulation_volt_min  = 0.1                     # mininum voltage of cap (for numeric stability?) [V]
        self.capacitor_size = config["capacitor_size"]
        self.capacitor_max_voltage = buffer.capacitor_max_voltage 
        self.time_resolution = 300                          # in seconds i.e. step
        self.util_max_current = 2.7277                      # TODO: Get this from utility manager
        self.dc_dc_voltage = scheduler.dc_dc_voltage
        self.cap_eta = 0.7                                  # This is different to what we assume in simulation
        self.cap_vc_min = self.buffer.capacitor_min_voltage

        self.planning_step_counter = 0
        self.next_planning_after_x_steps = config["next_planning_after_x_steps"] 
        
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.planning_slot_counter = 0

        self.time_per_slot = config["simulation_step_time"] * config["next_planning_after_x_steps"] # Seconds

    def err(self, x, y):
                return (x - y) if x > y else (y - x)

    def simulate_newton(self, dt, Vc, Ih, Pn, eta):
        
        a:float = Ih / self.capacitor_size
        b:float = Pn / (eta * self.capacitor_size)
        v0:float = Vc
        y:float = v0

        if self.err(Ih, 0.0) <= self.simulation_ibal_tolerance:
            # Discharging only
            y = v0 * v0 - 2 * b * dt
            if (y >= 0 ):
                y = math.sqrt(v0 * v0 - 2 * b * dt)
            else:
                # If the capacitor will discharge below 0 then set the y (voltage) = 0, 
                # later in code the capacitor voltage will be set to minimum voltage of capacitor i.e. 0.1
                y = 0
        elif self.err(Pn, 0.0) <= self.simulation_ibal_tolerance:
            # Charging only
            y = v0 + a * dt
        else:
            if self.err(Ih, Pn/eta/Vc) <= self.simulation_ibal_tolerance:
                # Harvest equals not consumption
                y = v0
            else:
                lasty:float = 0
                fy:float = 0
                fyy:float = 0
                c:float = -((v0 / a) + (b / (a * a)) * math.log(abs(a * v0 - b)))
                i:int = 1
                while self.err(lasty, y) >= self.simulation_volt_tolerance and y >=  self.simulation_volt_min and i <= self.simulation_steps_max:
                    lasty = y
                    fy = (y) + ((b / (a)) * math.log(abs(a * y - b))) - dt * a + c * a
                    fyy = (1) + (b / ((a * y - b)))
                    y = y - (fy / fyy)
                    i += 1
        
        if y > self.capacitor_max_voltage:
            y = self.capacitor_max_voltage
        
        if not y > self.simulation_volt_min:
            y = self.simulation_volt_min
        
        return y


    def findBestIn(self, startSlot, numSlots, startVc):
        
        maxIn = self.max_current_amp # upper bound on consumption
        minIn = self.sleep_current_amp  # lower bound

        # no operation, if node is off or currently below drop level
        if startVc < self.cap_vc_min or startVc <  self.min_goal:
            return minIn

        while abs(maxIn - minIn) > self.simulation_ibal_tolerance:
            ok = True  # success indicator for each binary search step
            In = (maxIn + minIn) / 2.0
            Vc = startVc
            for ds in range(numSlots):
                slot = startSlot + ds
                # Here, predictor.predict returns in mA but we are calculating in Ampere --> conversion needed
                # Time should be in seconds
                Vc = self.simulate_newton(
                    self.time_per_slot, 
                    Vc,
                    self.predictor.predict(slot)/ 1000, 
                    In * self.dc_dc_voltage, self.cap_eta)

                if Vc < self.min_goal:
                    ok = False
                    break

            # update search boundaries
            if ok:
                minIn = In
            else:
                maxIn = In
            # DEBUG_STDOUT("minIn: " << minIn << " [A]   maxIn: " << maxIn << " [A]");

        if minIn >= (self.util_max_current * 1e-3):
            return (self.util_max_current * 1e-3)

        return minIn


    def step(self, state: int, df: pd.DataFrame) -> float:

        # Input
        start_buffer_voltage = df.at[state, "buffer_voltage"]

        budget_current_mA: float = 0
        self.planning_step_counter += 1
        run_planning: bool = True

        if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps:
            
            self.planning_step_counter = 0
            self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
            
            if state < 288: # For first day, do minimum
                budget_current_mA = 0.0
                run_planning = True
            else:   # From 2nd day, start planning
                budget_current_mA = 1000 * self.findBestIn(self.planning_slot_counter, self.planning_number_of_slots_per_day , start_buffer_voltage)
                run_planning = True
            
            self.planning_slot_counter += 1
            if self.planning_slot_counter >= self.planning_number_of_slots_per_day:
                self.planning_slot_counter = 0

        else:
            budget_current_mA = df.at[state - 1, "budget_current_mA"]
            run_planning = False


        return budget_current_mA, run_planning            