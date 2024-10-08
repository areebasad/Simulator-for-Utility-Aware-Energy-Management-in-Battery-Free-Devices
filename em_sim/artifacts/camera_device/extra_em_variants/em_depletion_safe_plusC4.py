"""
UtiliGEM Energy Manager

This file contains code for Utility Guided Energy Manager i.e. UtiliGEM

Author: Areeb
Date: 2024

Rough Notes
# Areeb 
# A: Not equal weights 
# 1. Sort slots based on th weights
# 2. Do binary search on highest weightage slot while keeping other slots consumption to minimum
# 3. Do binary search on second highest weightage slot while keeping other slots consumption to minimum except the highest weightage slot consumption 
# 4. Do for all slots
#
# B: Equal weights of all slots -> will automatically fall back to DS
#  Multiple Raise all slots
#
# C. Takes "Temproray Max Current Cosnumption MCC-A" as an input, fills first all slots until MCC-A and then again does the binary search to fill up until actual MCC

# MultiRase and afterwards singleRaise
"""

from typing import Any, Dict
from em_sim.core import Component, VariableMode
from ..predictors import Predictor
from ..utility_learner import UtilityLearner
import pandas as pd
import math
import numpy as np
import logging


class EnergyManager(Component):        
       
    def calc_duty_cycle(self, *args):
        pass

    def step(self):
        raise NotImplementedError()

class NoManager(EnergyManager):
     def __init__(self, config: Dict[str, Any], buffer, scheduler):
         pass
     
# UtiliGEM EM (working name DSPC4)
class DepletionSafePlus(EnergyManager):    
    
    def __init__(self, config: Dict[str, Any], buffer, scheduler):
        
        super().__init__(config, "Energy Manager: UtiliGEM: Depletion Safe Plus")
        self.logger = logging.getLogger(config["id"])
        self.logger.info("UtiliGEM (DSPC4) initialization started")

        self.predictor = Predictor(config)
        self.buffer = buffer

        self.max_current_amp = config["budget_mA_max"]/1000      # Areeb changed this according to utility, config["device_max_current_amp"]
        self.sleep_current_amp = scheduler.sleep_current
        self.min_goal = config["device_mingoal"]
        self.simulation_ibal_tolerance = float("1e-7")      # tolerance of current balance (harvest - node consumption) [A]
                                                            # values below this threshold are treated as steady state (Vc unchanged)
        self.simulation_volt_tolerance =  1e-7              # tolerance/resolution of voltage change [V] 1 × 10^-7
        self.simulation_steps_max = 50                      # maximum number of iterations for voltage simulation (newton)
        self.simulation_volt_min  = 0.1                     # mininum voltage of cap (for numeric stability?) [V]
        self.capacitor_size = config["capacitor_size"]
        self.capacitor_max_voltage = self.buffer.capacitor_max_voltage 
        self.time_resolution = 300                          # in seconds i.e. step
        self.util_max_current_mA = config["budget_mA_max"]     # TODO: Get this from utility manager
        self.dc_dc_voltage = scheduler.dc_dc_voltage
        self.cap_eta = 0.8                                 # This is different to what we assume in device simulation, 0.86 on average
        self.cap_vc_min = self.buffer.capacitor_min_voltage
        self.utility_learner = UtilityLearner(config)

        self.planning_step_counter = 0
        self.next_planning_after_x_steps = config["next_planning_after_x_steps"]        # 12 steps per slot
        
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]      # 24 slots per day
        self.planning_slot_counter = 0

        self.time_per_slot_sec = config["simulation_step_time"] * config["next_planning_after_x_steps"] # Seconds, 3600 sec

    def err(self, x, y):
                return (x - y) if x > y else (y - x)
    
    # calculate capacitor voltage
    def simulate_newton(self, dt, Vc, Ih, Pn, eta):
        
        a:float = Ih / self.capacitor_size
        b:float = Pn / (eta * self.capacitor_size)
        v0:float = Vc
        y:float = v0

        if self.err(Ih, 0.0) <= self.simulation_ibal_tolerance:
            # Discharging only
            y = v0 * v0 - 2 * b * dt
            if (y > 0 ):
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
    
    #First order differential equation
    def simulate_newton_foe(self, dt:float, Vc:float, Ih:float, Pn:float, eta:float):
        C = self.capacitor_size


        newV =  min(Vc + dt * ((Ih/C) - ((Pn / (Vc * eta)) / C)), 2.68)
        newV = max(newV, 0)
        
        return newV
    
    # weight_indices contains weights and their indices together as a tuple
    # The function returns the indices of slots whose weight is equal to target_weight..
    def get_indices_by_weight(self, weight_indices, target_weight, max_weight):
        #max_weight = max(weight for weight, index in weight_indices)
        tolerance = 5/100 * max_weight # 5 percent of max weight
        indices = []
        for weight, w_index in weight_indices:
            if abs(target_weight - weight) < tolerance:
                indices.append(w_index)
        return indices
    
    def upscale_budget(self, recommended_budget_current_amp, Vc, slot):
        
        dt = self.time_per_slot_sec
        Ih = self.predictor.predict(slot)/ 1000
        
        # Simulate with recommended budget
        newVc = self.simulate_newton(
                    dt, 
                    Vc,
                    Ih, 
                    recommended_budget_current_amp * self.dc_dc_voltage,
                    self.cap_eta)
        
        delta_V = newVc - Vc  # Calculate the change in voltage
        I_stored = self.capacitor_size * delta_V / dt  # Calculate current stored in the capacitor, it can be negative
        if I_stored >= 0 and Ih > 0: # Excess energy
            I_application = recommended_budget_current_amp # Current for the application
            I_wasted = (Ih) - I_application - I_stored  # Calculate wasted current, we consider 90% of Ih 
            I_extra = max(0, min(I_wasted, self.max_current_amp - I_application))  # Calculate the extra current.

            I_application_upscaled_amp = I_application + I_extra  # This is the upscaled application current.

            # Volatge check
            # Simulate with upscaled budget,
            # Why we do this? Its because stored energy in capacitor depends on energy consumption. 
            after_upscale_Vc = self.simulate_newton(
                            dt, 
                            Vc,
                            Ih, 
                            I_application_upscaled_amp * self.dc_dc_voltage,
                            self.cap_eta)
            
            if after_upscale_Vc >= newVc:
                    return I_application_upscaled_amp 
            else:
                    return recommended_budget_current_amp

        else: # No excess energy
            I_application_upscaled_amp = recommended_budget_current_amp
        
        return recommended_budget_current_amp
    
    def remove_elements(self,list_a, list_b):
        # Use list comprehension to create a new list excluding elements from list_a
        updated_list_b = [item for item in list_b if item not in list_a]
        return updated_list_b
    
    # Multi Raise Binary Search
    def multiRaiseBS(self, startSlot, numSlots, startVc, weights, weight_indices, required_max_budget_amp, list_In_amp, multiRaise=True ):

        indices_to_skip = []
        max_weight = max(weights)
        #if (max(weights) - min(weights)) != 0:

        In_amp = list_In_amp  # Current in each slot

        # Multi Raise
        for weight, w_index in weight_indices:
            
            if w_index in indices_to_skip and multiRaise == True:
                # If the index is in skip_indices, skip this iteration
                continue

            if multiRaise == True:
                highest_weightage_slots = self.get_indices_by_weight(weight_indices, weight, max_weight)
                highest_weightage_slots = self.remove_elements(indices_to_skip, highest_weightage_slots)

                # Combine indices_with_target_weight with skip_indices
                indices_to_skip += highest_weightage_slots
            else: # SingleRaise
                highest_weightage_slots = []
                highest_weightage_slots.append(w_index)            

            maxIn_amp = required_max_budget_amp # upper bound on application consumption
            minIn_amp =  In_amp[w_index] # lower bound in amperes

            # Binary Search
            while abs(maxIn_amp - minIn_amp) > self.simulation_ibal_tolerance: #If search window gets so small then we break
                ok = True  # success indicator for each binary search step
                In = (maxIn_amp + minIn_amp) / 2.0
                Vc = startVc
                for ds in range(numSlots):
                    predictor_slot = startSlot + ds
                    weight_slot = (startSlot + ds) % numSlots
                    # Here, predictor.predict returns in mA but we are calculating in Ampere --> conversion needed
                    # Time should be in seconds
                    if weight_slot in highest_weightage_slots:
                        # Non-linear Function
                        Vc = self.simulate_newton(
                            self.time_per_slot_sec, 
                            Vc,
                            self.predictor.predict(predictor_slot)/ 1000, 
                            In * self.dc_dc_voltage, self.cap_eta)
                    else:
                        # Non-linear Function
                        Vc = self.simulate_newton(
                            self.time_per_slot_sec, 
                            Vc,
                            self.predictor.predict(predictor_slot)/ 1000, 
                            In_amp[weight_slot] * self.dc_dc_voltage, self.cap_eta)

                    if Vc < self.min_goal:
                        ok = False
                        break

                # update search boundaries, we decrease the search window everytime
                if ok:
                    minIn_amp = In # Increase
                else:
                    maxIn_amp = In # Decrease

            #In_amp[highest_weightage_slot] = minIn_amp
            for index in highest_weightage_slots:
                In_amp[index] = minIn_amp
        

        return  In_amp # Return in Amperes



    # Without precise estimation of tasks consumption, which depends on the scheduler used.
    def findBestIn(self, startSlot, numSlots, startVc):
        
        maxIn_amp = self.max_current_amp # upper bound on application consumption
        minIn_amp = self.sleep_current_amp  # lower bound in amperes
        
        # no operation, if node is off or currently below drop level
        if startVc < self.cap_vc_min or startVc <  self.min_goal:
            return minIn_amp

        weights = self.utility_learner.ucb # Upper confidence values/ weights
        
        # Create a list of tuples with weight and index
        weight_indices = [(weight, index) for index, weight in enumerate(weights)]
        
        # Sort the list based on weights in descending order
        weight_indices.sort(key=lambda x: x[0], reverse=True)
        
        In_amp = [minIn_amp] * numSlots # Initial Current in each slot

        vam_convg = self.utility_learner.vam_convg
        requiredIn_amp = self.utility_learner.app_re_budget_mA/1000

        if (vam_convg == True) and (requiredIn_amp < maxIn_amp):
            # We first do multi raise bs with requiredIn_amp and sleepIn

            required_max_budget_amp = requiredIn_amp
            pass1_In_amp =  self.multiRaiseBS(startSlot, numSlots, startVc, weights, weight_indices, required_max_budget_amp, In_amp, multiRaise=True )

            # Then we again do multi raise bs with  maxIn and In that we found from pass 1
            required_max_budget_amp = maxIn_amp
            pass2_In_amp =  self.multiRaiseBS(startSlot, numSlots, startVc, weights,weight_indices, required_max_budget_amp, pass1_In_amp, multiRaise=False)

            # Send result
            rec_In_amp = pass2_In_amp[startSlot]

        else:
            # we only do multi raise bs with maxIn_amp and sleepIn
            required_max_budget_amp = maxIn_amp
            pass1_In_amp =  self.multiRaiseBS(startSlot, numSlots, startVc, weights,weight_indices, required_max_budget_amp, In_amp, multiRaise=True )
            pass2_In_amp =  self.multiRaiseBS(startSlot, numSlots, startVc, weights,weight_indices, required_max_budget_amp, pass1_In_amp, multiRaise=False )

            # Send result
            rec_In_amp = pass2_In_amp[startSlot]
        
       
        return rec_In_amp # Return in Amperes
           
    

   

    def step(self, state: int, df: pd.DataFrame, isDeviveOn: bool) -> float:

        # Input
        start_buffer_voltage = df.at[state, "buffer_voltage"]

        budget_current_mA: float = 0
        self.planning_step_counter += 1
        run_planning: bool = True # Planing execution status
        
        if isDeviveOn == True:

            if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps:
                
                self.planning_step_counter = 0
                self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
                if self.config["learn_temporal"] == True:
                    self.utility_learner.update_ucb(state, df)

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