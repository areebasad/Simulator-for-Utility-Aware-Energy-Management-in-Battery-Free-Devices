from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta
from ..predictors import Predictor
from ..utility_learner import UtilityLearner


class EnergyManager(Component):        
       
    def calc_duty_cycle(self, *args):
        pass

    def step(self):
        raise NotImplementedError()

class PIDController:
    def __init__(self, coefficients=None):
        self.e_sum = 0.0
        self.e_prev = 0.0
        # Lars 
        #'k_p': 2.0,
        #    'k_i': 0.01,
        #    'k_d': 0.0
        # Preact authors'k_p': 1.5,
        #    'k_i': 0.00152,
        #    'k_d': 0.0
        self.coefficients = {
            'k_p': 2.0,
            'k_i': 0.01,
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

class PreactAppU(EnergyManager):
    
    def __init__(self, config: Dict[str, Any], buffer, scheduler, **kwargs):

        super().__init__(config, "Energy Manager: Preact")

        self.controller = PIDController(kwargs.get('control_coefficients', None))
        #self.controller = PIDController(coefficients={'k_p': 2.0, 'k_i': 0.001,'k_d': 0.0}) # As in Lars simulator

        self.predictor = Predictor(config)
        self.buffer = buffer
        self.scheduler = scheduler

        self.capacitor_size = config["capacitor_size"]
        self.cap_vc_min = self.buffer.capacitor_min_voltage
        self.cap_max_voltage = self.buffer.capacitor_max_voltage
        self.cap_voltage = self.buffer.capacitor_init_voltage
        self.cap_eta = 0.7
        
        self.time_per_slot_sec = config["simulation_step_time"] * config["next_planning_after_x_steps"] # Seconds
        self.time_per_slot_mins = self.time_per_slot_sec/ 60
        
        self.vc_saftey_margin = 0.2#0.2 # Lars 0.2, Camera case 0
        self.battery_capacity = self.cap_to_WattHours(self.cap_max_voltage,self.vc_saftey_margin ) #self.vc_saftey_margin
        
        self.planning_step_counter = 0
        self.next_planning_after_x_steps = config["next_planning_after_x_steps"] 
        
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.planning_slot_counter = 0


        self.util_max_current_mA = config["budget_mA_max"]#2.7277 mA # Actual required budget 0.54548484
        self.app_consumption_hour_wh = self.calculate_energy_consumed_in_hour()
        self.utility_learner = UtilityLearner(config)
        #self.utility_in_duty_cycle = self.utility_learner.utility 
        #self.utility_mean = self.utility_learner.utility_mean
        
        self.simulation_volt_tolerance =  1e-7              # tolerance/resolution of voltage change [V] 1 × 10^-7
        self.simulation_steps_max = 50    #50                  # maximum number of iterations for voltage simulation (newton)
        self.simulation_volt_min  = -50.0#0.1                     # mininum voltage of cap (for numeric stability?) [V]
        self.simulation_ibal_tolerance = float("1e-7")      # tolerance of current balance (harvest - node consumption) [A]
        self.capacitor_max_voltage = self.buffer.capacitor_max_voltage
        self.time_per_slot = 3600
    
    def calculate_energy_consumed_in_hour(self):
        
        schedule_list = self.scheduler.schedule()
        hour = 1
        sleep_power =  self.scheduler.sleep_power /1000000     # Convert to Watt from uW
        
        total_energy = 0
        total_task_time = 0
        total_tasks = 0
        for task in schedule_list:
            # Assuming each task in the schedule list is a dictionary with 'start_time', 'power', and 'duration' keys
            start_time = task.starting_time  # in milli seconds
            if start_time < (hour)*3600000:  # if the task starts within the specified hour or before
                power = task.power /1000000 # Convert to Watt from uW
                duration_ms = task.duration   # in milli seconds
                energy = power * duration_ms/1000  # Energy = Power * Time, in Joules
                total_energy += energy
                total_task_time += duration_ms
                total_tasks += 1
        # Calculate the sleep time and the energy consumed during sleep
        sleep_time = 3600000 - total_task_time  # in milli seconds
        sleep_energy = sleep_power * sleep_time/1000  # Energy = Power * Time, in Joules
        total_energy += sleep_energy
        
        total_energy_wh = total_energy / 3600  # Convert energy from Joules to Watt-hours
        return total_energy_wh  # returns total energy in Watt-hours

    
    def cap_to_WattHours(self, Vc, margin):
        
        if Vc < self.cap_vc_min:
            print("Error in EnergyManager: Current Vc smaller than shut off voltage")
            return 0.0

        # 1 watt hours is equal to 3600 Joules of energy
        # 1/2 * Cap * (Vmax^2 - Vmin^2) / 3600
        energy = (1/2 *  (self.cap_eta *  self.capacitor_size) * ((Vc * Vc) - ((self.cap_vc_min + margin) * (self.cap_vc_min + margin)) )) / 3600#7200.0 # convert watt seconds to watt hours as used in preact

        return energy
   
    def wh_to_v(self, e_wh):
        #E (in joules) = E (in Wh) * 3600
        # E = 1/2 CV2
        e_joules = e_wh * 3600
        if e_joules < 0:
            V = -math.sqrt( (abs(2 * e_joules) / (self.cap_eta * self.capacitor_size)) + (self.cap_vc_min )**2)
        else:
            V = math.sqrt(((2 * e_joules) / (self.cap_eta * self.capacitor_size))+ (self.cap_vc_min )**2)
    
        return V


    def harvestToWattHours(self, current_capacitor_voltage, Ih, slot=None) :
        # TODO: check if this is really correct; usable power from solar panel highly depends on cap voltage
        # assume that cap voltage at beginning of slot determines extractable power during the slots; accuracy degrades with distance to actual observation point
        #P = Ih * 1e-3 * current_capacitor_voltage # Power in watts
        #E = P * (self.time_per_slot_mins / 60.0) # Energy in watt-hours
         # Calculate energy in Joules
        #energy_joules = (Ih * 1e-3) * abs(current_capacitor_voltage) * (self.time_per_slot_mins)
        # Estimate new voltage after harvesting
        #new_voltage = ((2 * energy_joules) / (self.cap_eta * self.capacitor_size)) ** 0.5 + self.cap_vc_min

        # Calculate energy that can be stored in the capacitor in Joules
        #energy_stored_joules = 0.5 * self.cap_eta * self.capacitor_size * ((new_voltage ** 2) - (self.cap_vc_min ** 2))

        # Calculate energy that can be harvested and stored in the capacitor in Watt-hours
        #energy_wh = min(energy_joules, energy_stored_joules) / 60
        # Convert to Watt-hours and adjust for minimum voltage and margin
        #return  energy_wh
        # Method A
        ''' 
        if slot is None:
            fixed_consumption = 0.0018001 #wh
        else: 
            fixed_consumption =  0.0018001 * self.utility_get_dc_rel(((slot) % self.planning_number_of_slots_per_day))   
        max_harvest =  self.battery_capacity + fixed_consumption
        return min(max_harvest , Ih * 1e-3 * abs(current_capacitor_voltage) * self.time_per_slot_mins / 60.0)
        '''
        # Method B
        return Ih * 1e-3 * abs(current_capacitor_voltage) * self.time_per_slot_mins / 60.0

    def duty_cycle_to_budget(self, duty_cycle:float):
        
        if duty_cycle is None or duty_cycle <= 0.0:
            return 0.0

        return  duty_cycle * self.util_max_current_mA  

    def utility_get_dc_rel(self, slot):

        return  self.utility_learner.utility[slot % len(self.utility_learner.utility)]
    
    def err(self, x, y):
                return (x - y) if x > y else (y - x)

   
    # As per application utility 
    def XXcalc_duty_cycleXX(self, slot, current_capacitor_voltage, soc):
        soc_p = []
        soc_v = []
        utility_vc = 3.3
       
        # 0.00165 Wh tasks + 0.0001501 Wh sleep= 0.0018001 Wh. total App
        # 0.00416 Wh in total
        fixed_consumption =  self.app_consumption_hour_wh#0.00416 #0.00899 #0.0018001 #wh, camera app, lars 0.00899wh

        app_u = fixed_consumption * self.utility_get_dc_rel(((slot) % self.planning_number_of_slots_per_day)) 
        
        soc_0 = 0 #self.cap_to_WattHours(current_capacitor_voltage, 0)#self.vc_saftey_margin
        delta_soc_0 = soc_0 + self.harvestToWattHours(current_capacitor_voltage, self.predictor.predict(slot), slot) - app_u
        cap_volt = self.wh_to_v(delta_soc_0)
        soc_v.append(cap_volt)
        soc_p.append(delta_soc_0)
        
       
        delta_soc_prev = delta_soc_0
        cap_volt_prev = cap_volt

        for i in range(1, self.planning_number_of_slots_per_day):
          
            app_u = fixed_consumption * self.utility_get_dc_rel(((slot+i) % self.planning_number_of_slots_per_day)) 

            delta_soc = delta_soc_prev + self.harvestToWattHours(cap_volt_prev, self.predictor.predict(slot + i), slot+i) - app_u
            
            soc_p.append(delta_soc)
            delta_soc_prev = delta_soc
            
            cap_volt = self.wh_to_v(delta_soc)
            cap_volt_prev = cap_volt
            soc_v.append(cap_volt)
           

        peak_to_peak = max(soc_p) - min(soc_p)
        capacity_worstcase = self.battery_capacity  # removed original degradition of capacity over time because of unknown behavior of cap over time
        '''
        if(peak_to_peak >= capacity_worstcase):
        
            f_scale = min(1.0, capacity_worstcase / peak_to_peak)
            capacity_today = self.battery_capacity  # see capacity_worstcas

            offset = (capacity_worstcase - f_scale * peak_to_peak) / 2

            soc_target = f_scale * (soc_p[0] - min(soc_p)) + offset
            
            next_soc_e_with_e_pred = soc + self.harvestToWattHours(current_capacitor_voltage, self.predictor.predict(slot), slot)
            dc = (next_soc_e_with_e_pred / capacity_worstcase) - (soc_target / capacity_worstcase)
            
            #dc = self.controller.calculate(soc_target / self.battery_capacity, next_soc_e_with_e_pred / self.battery_capacity)
          
            # ensure practical limits
            #min(self.utility_get_dc_rel((slot) % self.planning_number_of_slots_per_day)
            return max(0.0, min(self.utility_get_dc_rel((slot) % self.planning_number_of_slots_per_day), dc)), soc_p
        
        else: # If it is less, no-issues of battery constraints.
            return 1.0 * self.utility_get_dc_rel((slot) % self.planning_number_of_slots_per_day), soc_p# return duty cycle 100%
        '''
        # Step 2: Adjust ideal SoC to satisfy battery capacity constraints
        soct = np.zeros(24)
        f_scale = min(1.0, capacity_worstcase / peak_to_peak)
        if peak_to_peak > capacity_worstcase:
            for i in range(0,24):
                soct[i] = f_scale * soc_p[i] 
        else:
            soct = soc_p

        # Compute initial value of ideal SoC
        if peak_to_peak >= capacity_worstcase:
            soc_p[0] = -min(soct)
            soc_target = soc_p[0]
           
        else:
            soc_p[0] = -min(soct) + (capacity_worstcase - peak_to_peak) / 2
            soc_target = soc_p[0]

        max_harvest =  self.battery_capacity + fixed_consumption
        max_harvest_next_hour = min(max_harvest ,self.harvestToWattHours(current_capacitor_voltage, self.predictor.predict(slot)))
        next_soc_e_with_e_pred = soc + max_harvest_next_hour

        dc = (next_soc_e_with_e_pred / capacity_worstcase) - (soc_target / capacity_worstcase)
        #dc = self.controller.calculate(soc_target / self.battery_capacity, next_soc_e_with_e_pred / self.battery_capacity)

        return max(0.0, min(1.0, dc)), soc_p
   
    
    def step(self, state: int, df: pd.DataFrame, isDeviveOn: bool) -> float:

        # Input
        start_buffer_voltage = df.at[state, "buffer_voltage"]
        #start_buffer_voltage = 2.60319
        budget_current_mA: float = 0
        self.planning_step_counter += 1
        run_planning: bool = True

        if isDeviveOn == True:
            if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps: 
                
                self.planning_step_counter = 0
                
                #Areeb: Update after every step for accurate predictions mean. update_x (However, updates might not arrive after every hour)
                #General: Update mean after 1 day  
                self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
                if self.config["learn_temporal"] == True:
                    self.utility_learner.update_prob_ucb(state, df)
                    self.utility_learner.map_prop_to_utility(state, df)

                if state < 288: # For first day, do minimum
                    budget_current_mA = 0.0
                    run_planning = True
                else: # From 2nd day, start planning XXcalc_duty_cycleXX,  Xcalc_duty_cycle_impX, calc_duty_cycle 
                    # calc_duty_cycle: To compare with Lars implementation
                    # calc_duty_cycle_Preact: Areeb implementtaion as per preact utility - unit wh
                    # XXcalc_duty_cycleXX: Areeb implementtaion as per app utility - unit wh
                    # Xcalc_duty_cycle_impX: Areeb implementtaion as per app utility - using newton methon - unit joules
                    duty_cycle, soc_forecast = self.XXcalc_duty_cycleXX(self.planning_slot_counter, start_buffer_voltage, self.cap_to_WattHours(start_buffer_voltage,self.vc_saftey_margin)) # self.vc_saftey_margin
                    budget_current_mA = self.duty_cycle_to_budget(duty_cycle) # Check here unit, -> for mA 1000 * 
                    run_planning = True
                    if state == 288:
                        df.at[state, "preact_soc_forecast"] = str(soc_forecast)
                
                
                self.planning_slot_counter += 1
                if self.planning_slot_counter >= self.planning_number_of_slots_per_day:
                    self.planning_slot_counter = 0

            else:
                budget_current_mA = df.at[state - 1, "budget_current_mA"]
                run_planning = False
        else:
            # Device is OFF, we only update planning counters
            if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps: 

                self.planning_step_counter = 0
                self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
                
                self.planning_slot_counter += 1
                if self.planning_slot_counter >= self.planning_number_of_slots_per_day:
                    self.planning_slot_counter = 0
                
                budget_current_mA = df.at[state - 1, "budget_current_mA"]
                run_planning = False
            else:
                budget_current_mA = df.at[state - 1, "budget_current_mA"]
                run_planning = False


        return budget_current_mA, run_planning     

       