from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math
import numpy as np

class Predictor(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Predictor: Ideal Predictor")

        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.num_of_steps_between_planning = 12 # if planning_number_of_slots_per_day is 24
        self.predictions_day:float = []         # Average 
        self.predictions_mean = 0.0
        self.prediction_end_state = (config["days"] * 288) - (2 * 288)

    def update(self, state: int, df: pd.DataFrame) -> None:

        # If next day; calculate avg current for each planning slots in a day. 
        # #Fix this 55008 (2 days before ending)
        if (state % 288) == 0 and state < self.prediction_end_state:
            counter = 0 + state
            self.predictions_day = []
            for planning_slot in range(self.planning_number_of_slots_per_day*2):           
                sum = 0.0         
                for step_slot in range (self.num_of_steps_between_planning):            
                    sum += df.at[counter + step_slot, "intake_current_mA"]        
                self.predictions_day.insert(planning_slot,  sum/self.num_of_steps_between_planning)
                counter += self.num_of_steps_between_planning
            
            self.calc_mean()

    def calc_mean(self):
        self.predictions_mean = sum(self.predictions_day[0:self.planning_number_of_slots_per_day])/ self.planning_number_of_slots_per_day 

    def predict(self, x):
        return self.predictions_day[x]    


class EnergyManager(Component):        
       
    def calc_duty_cycle(self, *args):
        pass

    def step(self):
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

class Preact(EnergyManager):
    
    def __init__(self, config: Dict[str, Any], buffer, scheduler, **kwargs):

        super().__init__(config, "Energy Manager: Preact")

        self.controller = PIDController(
            kwargs.get('control_coefficients', None))

        self.predictor = Predictor(config)
        self.buffer = buffer
        
        self.capacitor_size = config["capacitor_size"]
        self.cap_vc_min = self.buffer.capacitor_min_voltage
        self.cap_max_voltage = buffer.capacitor_max_voltage 
        self.cap_eta = 1.0 
        self.util_max_current = 2.7277 
        self.vc_saftey_margin = 0.2
        self.time_per_slot_sec = config["simulation_step_time"] * config["next_planning_after_x_steps"] # Seconds
        self.time_per_slot_mins = self.time_per_slot_sec/ 60
        self.cap_voltage = 0
        self.battery_capacity = self.cap_to_WattHours(self.cap_max_voltage, self.vc_saftey_margin)
        
        self.planning_step_counter = 0
        self.next_planning_after_x_steps = config["next_planning_after_x_steps"] 
        
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.planning_slot_counter = 0

        self.utility_in_duty_cycle = np.ones(24)
        self.utility_mean = sum(self.utility_in_duty_cycle) / self.planning_number_of_slots_per_day

    def cap_to_WattHours(self, Vc, margin):
        
        if Vc < self.cap_vc_min:
            print("Error in EnergyManager: Current Vc smaller than shut off voltage")
            return 0.0

        # 1 watt hours is equal to 3600 Joules of energy
        # 1/2 * Cap * (Vmax^2 - Vmin^2) / 3600
        energy = self.cap_eta *  self.capacitor_size * (Vc * Vc - (self.cap_vc_min + margin) * (self.cap_vc_min + margin) ) / 7200.0 # convert watt seconds to watt hours as used in preact

        return energy

    def harvestToWattHours(self, Ih):
        # TODO: check if this is really correct; usable power from solar panel highly depends on cap voltage
        # assume that cap voltage at beginning of slot determines extractable power during the slots; accuracy degrades with distance to actual observation point
        
        return Ih * 1e-3 * self.cap_voltage * self.time_per_slot_mins / 60.0

    def duty_cycle_to_budget(self, duty_cycle:float):
        
        if duty_cycle is None or duty_cycle <= 0.0:
            return 0.0

        return  duty_cycle * self.util_max_current  

    def utility_get_dc_rel(self, slot):

        return  self.utility_in_duty_cycle[slot %  self.planning_number_of_slots_per_day]


    def calcDutyCycle(self, slot, soc):
        delta_soc_0 = self.harvestToWattHours(self.predictor.predict(slot + 1)) 
        - self.utility_get_dc_rel((slot + 1) % self.planning_number_of_slots_per_day) / self.utility_mean * self.harvestToWattHours(self.predictor.predictions_mean )

        delta_soc_min = delta_soc_0
        delta_soc_max = delta_soc_0
        delta_soc_prev = delta_soc_0

        for i in range(1, self.planning_number_of_slots_per_day):
            delta_soc = delta_soc_prev + self.harvestToWattHours(self.predictor.predict(slot + 1 + i)) - self.utility_get_dc_rel((slot + 1 + i) % self.planning_number_of_slots_per_day) / self.utility_mean * self.harvestToWattHours(self.predictor.predictions_mean)
            delta_soc_prev = delta_soc

            if delta_soc > delta_soc_max:
                delta_soc_max = delta_soc
            elif delta_soc < delta_soc_min:
                delta_soc_min = delta_soc

        peak_to_peak = delta_soc_max - delta_soc_min
        capacity_worstcase = self.battery_capacity  # removed original degradition of capacity over time because of unknown behavior of cap over time
        f_scale = min(1.0, capacity_worstcase / peak_to_peak)
        capacity_today = self.battery_capacity  # see capacity_worstcase
        soc_target = (delta_soc_0 - delta_soc_min) * f_scale + (capacity_today - f_scale * peak_to_peak) / 2

        dc = self.controller.calculate(soc_target / self.battery_capacity, soc / self.battery_capacity)
        #print("Duty Cycle: ", dc)
        # ensure practical limits
        return max(0.0, min(1.0, dc))

        

    
    def step(self, state: int, df: pd.DataFrame) -> float:

        # Input
        self.cap_voltage = df.at[state, "buffer_voltage"]

        budget_current_mA: float = 0
        self.planning_step_counter += 1
        run_planning: bool = True

        if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps:
            
            self.planning_step_counter = 0
            self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
            
            if state < 288: # For first day, do minimum
                budget_current_mA = 0.0
                run_planning = True
            else:    
                duty_cycle:float = self.calc_duty_cycle(self.planning_slot_counter-1, self.cap_to_WattHours(self.cap_voltage, self.vc_saftey_margin))
                budget_current_mA = self.duty_cycle_to_budget(duty_cycle)
                run_planning = True
            
            self.planning_slot_counter += 1
            if self.planning_slot_counter >= self.planning_number_of_slots_per_day:
                self.planning_slot_counter = 0

        else:
            budget_current_mA = df.at[state - 1, "budget_current_mA"]
            run_planning = False


        return budget_current_mA, run_planning     
  