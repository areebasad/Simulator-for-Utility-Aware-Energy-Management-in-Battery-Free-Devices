from typing import Any, Dict
from em_sim.core import Component, VariableMode
from ..predictors import Predictor
from ..utility_learner import UtilityLearner
import pandas as pd
import math
import numpy as np
from enum import Enum
import sys

# Modified Emrep

# Notes for usage
# Simulate Lars Case
# Change planning_horizon_ms to 24 hours in iniate func
# Change self.Vc = Vc in findbestIn func
# Set self.planning_slot_counter-1  for budget DS and free in emrep_step
# Set  self.checkUpTimeFac = 1e6 in initiate func
# Use with Lars Scheduler

# Simmulate Camera Case
# Change planning_horizon_ms to 1 hour/slot in iniate func 
# Change self.Vc = Vc in findbestIn func
# Set self.planning_slot_counter-1  for budget DS and free regionin emrep_step
# Set self.planning_slot_counter-1  for recalc regioun in getLimited func
# Set  self.checkUpTimeFac = 1e4 in initiate func

class BUDGET_REGION(Enum):
    REGION_DS = 0
    REGION_FREE = 1
    REGION_RECALC = 2

class EnergyManager(Component):        
       
    def calc_duty_cycle(self, *args):
        pass

    def step(self):
        raise NotImplementedError()

class EmRepPlus(EnergyManager):    
    
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
        self.capacitor_max_voltage = self.buffer.capacitor_max_voltage #2.675V
        self.time_resolution = 300                          # in seconds i.e. step
        self.util_max_current = config["budget_mA_max"]     # TODO: Get this from utility manager
        self.dc_dc_voltage = scheduler.dc_dc_voltage
        self.cap_eta = 0.7                                  # This is different to what we assume in simulation
        self.cap_vc_min = self.buffer.capacitor_min_voltage

        self.planning_step_counter = 0
        self.next_planning_after_x_steps = config["next_planning_after_x_steps"] 
        
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.planning_slot_counter = 0

        self.time_per_slot_sec = config["simulation_step_time"] * config["next_planning_after_x_steps"] # Seconds
        self.time_per_slot_ms =  self.time_per_slot_sec * 1000 #ms
        # For Lars case, must be equal to 24 hours
        #self.planning_horizon_ms =  self.time_per_slot_ms * self.planning_number_of_slots_per_day 
        # For Camera case, equal to 1 hour 
        self.planning_horizon_ms =  self.time_per_slot_ms
        
        
        self.current_time_ms = 0
        self.next_planning_slot_ms = self.time_per_slot_ms

        self.decouplingSoc = 0.05
        self.checkUpTimeFac = 1e4 # Lars value: 1e6, Camera case value, 1e4
        self.recalcDur = 24 #24 #Slots
        self.Ilim = 100
        self.decouplingDistance = 0

        print("-------Dynamo Created with")
        print("decouplingSoc:", self.decouplingSoc)
        print("checkUpTimeFac:", self.checkUpTimeFac)
        print("recalcDur:", self.recalcDur)
        print("Ilim:", self.Ilim)
        print("decouplingDistance:", self.decouplingDistance)
        self.capVoltDelta = 0.0
        self.budgetPrev = 0.0
        self.freeBudget = 0.0
        self.budgetFactor = 0.0
        self.lastChange = 0
        self.timeCapVoltMax = 0
        self.timeCapVoltMaxNext = 0
        self.timeCapVoltMin = 0
        self.timeCapVoltMinNext = 0
        self.schedValidity = self.time_per_slot_ms
        self.schedValidityPrev = self.time_per_slot_ms #predictor.getTimerSlot().getSlotLengthMs(0)
        self.timeLastSchedUpdate = 0
        self.firstRun = True
        self.doSched = False
        self.freeBudgetCalculated = False
        self.budgetRegion = BUDGET_REGION.REGION_DS
        self.capVoltPred = [0.0] * config["planning_number_of_slots_per_day"]
        self.Vc = self.config["device_vc_start"]
        self.budget =  0.0      # current mA
        self.soc_min_in_free_region = 0.1 
        self.utility_learner = UtilityLearner(config)
        self.timeResolution = 300 #seconds, ds param

    def step(self, state: int, df: pd.DataFrame, isDeviveOn: bool) -> float:

        # Input
        start_buffer_voltage = df.at[state, "buffer_voltage"]
        self.current_time_ms = df.at[state, "timestamp_seconds"] * 1000 # Convert to ms

        self.planning_step_counter += 1
        run_planning: bool = True
        
        if isDeviveOn == True:

            if df.at[state, "is_planning_slot"] == True or self.planning_step_counter >= self.next_planning_after_x_steps:
                
                self.planning_step_counter = 0
                self.predictor.update(state, df) #Fix this, should run before or after? Right now before ending of the day
                if self.config["learn_temporal"] == True:
                    self.utility_learner.update_prob_ucb(state, df)
                    self.utility_learner.map_prop_to_utility(state, df)
                
                if state < 288: # For first day, do minimum
                    budget_current_mA = 0.0
                    run_planning = True
                else:   # From 2nd day, start planning
                    self.updateVolt(start_buffer_voltage, self.planning_slot_counter)
                    self.is_planning_slot = True
                    budget_current_mA, run_planning = self.emrep_step(start_buffer_voltage, self.current_time_ms)
                    run_planning = True
                    budget_current_mA = self.upscale_budget(budget_current_mA, start_buffer_voltage, self.planning_slot_counter)
                self.planning_slot_counter += 1
                if self.planning_slot_counter >= self.planning_number_of_slots_per_day:
                    self.planning_slot_counter = 0
            
            else:
            
                if state < 288: # For first day, do minimum
                    budget_current_mA = 0.0
                    run_planning = True
                else:    
                    self.updateVolt(start_buffer_voltage, self.planning_slot_counter)
                    self.is_planning_slot = False
                    # Might be that the budget changes here or its something related to slot
                    #budget_current_mA, run_planning = self.emrep_step(start_buffer_voltage, self.current_time_ms)    
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

        self.budget = budget_current_mA

        return budget_current_mA, run_planning            

    
    def updateVolt(self, Vc, slot):
        self.Vc = Vc
        self.capVoltDelta = Vc - self.capVoltPred[slot]


    def upscale_budget(self, recommended_budget_current_mA, Vc, slot):
        
        dt = self.time_per_slot_sec
        recommended_budget_current_amp = recommended_budget_current_mA/1000
        Pn = self.getLimitedPower(recommended_budget_current_amp, slot)
        Ih = self.predictor.predict(slot)/ 1000
        
        # Simulate with recommended budget
        newVc = self.simulate_newton(
                    dt, 
                    Vc,
                    Ih, 
                    Pn, self.cap_eta)
        
        delta_V = newVc - Vc  # Calculate the change in voltage
        I_stored = self.capacitor_size * delta_V / dt  # Calculate current stored in the capacitor
        I_application = Pn / Vc  # Convert power to current for the application
        I_wasted = Ih - I_application - I_stored  # Calculate wasted current
        I_extra = min(I_wasted, 0.6e-3 - I_application)  # Calculate the extra current.

        I_application_upscaled_amp = I_application + (I_extra)  # This is the upscaled application current.


        if newVc >= self.capVoltPred[slot]:
            print("check this first")
            if I_wasted > 0:
                Pn = self.getLimitedPower(I_application_upscaled_amp, slot)

                # Simulate with recommended budget
                after_upscale_Vc = self.simulate_newton(
                            dt, 
                            Vc,
                            Ih, 
                            Pn, self.cap_eta)
                
                if after_upscale_Vc == newVc:
                    return I_application_upscaled_amp * 1000 # Return in mA
                else:
                    return recommended_budget_current_mA
        else:
            return recommended_budget_current_mA


    # time in ms
    def emrep_step(self, Vc, time):
        self.doSched = False
        self.budgetPrev = self.budget
        print(f"\nTime Last Schedule Update:  {self.timeLastSchedUpdate} Schedule Validity: {self.schedValidity}")
        # Schedule expired or voltage exceeded
        if self.current_time_ms >= self.timeLastSchedUpdate + self.schedValidity:
            print(f"Schedule expired; delta is   {self.capVoltDelta} [V]")
            self.budgetRegion = self.setBudgetRegion(time)
            if Vc >= self.min_goal:
                if self.budgetRegion == BUDGET_REGION.REGION_DS:
                    # Entering adjustment method with 'depletion safe budget'
                    # Update the budget based on voltage until minimum voltage is reached
                    # and calculate the budget factor
                    self.freeBudgetCalculated = False
                    if self.is_planning_slot:
                        tmpMin = self.capVoltMin
                        newBudget = self.updateVoltTillMin(self.timeCapVoltMin)
                        self.budgetFactor = newBudget /self.util_max_current
                    self.budget = self.getLimitedBudget(self.budgetFactor, self.planning_slot_counter-1)

                elif self.budgetRegion == BUDGET_REGION.REGION_FREE:
                    # Entering with current budget
                    # Calculate the free budget and update the budget based on the SOC
                    if self.is_planning_slot: # If planning slot
                        self.freeBudgetCalculated = False
                    if not self.freeBudgetCalculated:
                        self.freeBudget = self.calcFreeBudget(self.budget, Vc)
                        self.freeBudgetCalculated = True
                    if self.getSocWithSimVc(self.Vc) >= self.soc_min_in_free_region:
                        self.budget = self.getLimitedBudget(self.freeBudget / self.util_max_current, self.planning_slot_counter-1)

                elif self.budgetRegion == BUDGET_REGION.REGION_RECALC:

                    print("REGION_RECALC")


                    # Update budget for new horizon and predict voltage of cap
                    # Recalculate the budget and budget factor based on the previous slot and voltage
                    self.freeBudgetCalculated = False
                    print(f"Inside Step1 RegionRecalc, Vc:   {self.Vc} [V]")

                    tmpBudget = self.recalcBudget((self.planning_slot_counter)%self.planning_number_of_slots_per_day, Vc)  # In lars code, it's sending previous slot 
                    print(f"Inside Step1 tempbudget, Vc:   {self.Vc} [V]")

                    self.budgetFactor = tmpBudget / self.util_max_current
                    self.budget = self.getLimitedBudget(self.budgetFactor, self.planning_slot_counter) # Current slot for Lars case, prev slot for Camera case
                    print(f"RECALC_RESULT: tmpBudget =  {tmpBudget}  budgetFactor =  {self.budgetFactor}  budget = {self.budget}  dcRel =  {self.utility_get_dc_rel(self.planning_slot_counter)}")

                    self.updateVoltMaxAndMin(self.capVoltPred)

                else:
                    print("Unable to determine Budget Region")

            else:
                self.budget = 0.0

            self.updateSchedValidity()  # Calls calcschedulevalidity and budgetChanged, returns true/false
            self.doSched = self.budgetChanged() # Sends True/False

        print("Exit Dynamo 'step'")
        print(f"dosched: {self.doSched }")
        return self.budget, self.doSched

    def setBudgetRegion(self, time):
        if self.firstRun:
            self.firstRun = False
            print("At:", time, "REGION_CHECK: First Run --> RECALC")
            return BUDGET_REGION.REGION_RECALC

        if time >= self.timeCapVoltMax:
            # if self.Vc >= self.capVoltMax:
            #     # We passed the maximum but the capacitor is still full
            #     print("At:", time, "REGION_CHECK: Passed Tmax =", self.timeCapVoltMax, "but cap is full --> FREE")
            #     return BUDGET_REGION.REGION_FREE
            # else:
            #     # We passed the maximum but the capacitor is not full anymore
            print("At:", time, "REGION_CHECK: Passed Tmax =", self.timeCapVoltMax, "cap not full anymore --> RECALC")
            return BUDGET_REGION.REGION_RECALC

        if self.timeCapVoltMin <= self.timeCapVoltMax:
            # Minimum comes first
            if time < self.timeCapVoltMin:
                # Minimum not yet passed
                print("At:", time, "REGION_CHECK: Vmin first, but Tmin =", self.timeCapVoltMin, "not yet passed --> DS")
                return BUDGET_REGION.REGION_DS
            else:
                # Passed the minimum
                print("At:", time, "REGION_CHECK: Vmin first, and Tmin =", self.timeCapVoltMin, "passed --> FREE")
                return BUDGET_REGION.REGION_FREE
        else:
            # Maximum comes first
            if time < self.timeCapVoltMax:
                # Is voltage increasing?
                if self.getSoc() >= (1 - self.decouplingSoc):
                    print("At:", time, "REGION_CHECK: Vmax first, Tmax =", self.timeCapVoltMax, "not passed and SOC =", self.getSoc(), "--> FREE")
                    return BUDGET_REGION.REGION_FREE
                # Maximum still first, but voltage decreases already because next day is SOOO BAD; better go to depletion safe
                print("At:", time, "REGION_CHECK: Vmax first, but SOC =", self.getSoc(), "--> DS")
                return BUDGET_REGION.REGION_DS
            else:
                # We passed the maximum, but did not end up in the first condition
                print("At:", time, "REGION_CHECK: Vmax first, and Tmax =", self.timeCapVoltMax, "passed --> RECALC")
                return BUDGET_REGION.REGION_RECALC

    def updateVoltTillMin(self, minTime):
        if self.Vc < self.min_goal:
            return self.budget
        
        print(f"now: {self.current_time_ms} minTime: {minTime}")
        # only queried at the beginning of a slot; warning: this violates variable timeslot lengths
        remainingSlots = int((minTime - self.current_time_ms) / self.time_per_slot_ms)
        print(f"remainingSlots: {remainingSlots}")
        if remainingSlots > 1:
            newBudget = self.findBestIn(self.planning_slot_counter, remainingSlots, self.Vc) * 1e3
            # at this point, only allow decreasing budget, not increasing
            return newBudget if newBudget < self.budget else self.budget
        
        return self.budget

    # pred = capVoltpred
    def updateVoltMaxAndMin(self, pred):
        Vmax = 0.0
        Vmin = 3.0
        Tmax = sys.maxsize
        Tmin = sys.maxsize

        # end of horizon
        next_slot_ms = self.current_time_ms + self.time_per_slot_ms
        t = next_slot_ms + self.planning_horizon_ms # Here getNextSlotAt is in ms (Convert next slot in seconds to ms) + getHorizonMs is the planning horizon which is 24 Hours in ms for lars case, 1 hour for camera case. 
        dt = 0
        startSlot = self.planning_slot_counter # Slot number from 0 to total slots
        print(f"NextSlotAt {next_slot_ms}  HorizonMS {self.planning_horizon_ms}")

        # we explicitly want the first! minimum and maximum; thus we start from end
        for ds in range(self.planning_number_of_slots_per_day, 0, -1):
            slot = (startSlot + ds) % self.planning_number_of_slots_per_day
            # first slot is not necessarily complete; the others are
            dt = self.time_per_slot_ms
            if pred[slot] <= Vmin and t < Tmin:
                Vmin = pred[slot]
                Tmin = t
            if pred[slot] >= Vmax and t < Tmax:
                Vmax = pred[slot]
                Tmax = t
            t -= dt
        if t != next_slot_ms: # Next slot time in ms
            print(f"Warning, t should be {next_slot_ms} but is {t}")

        print(f"updateVoltMaxAndMin; Vmax = {Vmax} [V]; Tmax = {Tmax} [ms]; Vmin = {Vmin} [V]; Tmin = {Tmin} [ms]")
        self.capVoltMax = Vmax
        self.capVoltMin = Vmin

        self.timeCapVoltMax = Tmax
        if Tmin < Tmax and Tmin + self.decouplingDistance > Tmax:
            self.timeCapVoltMin = Tmax
        else:
            self.timeCapVoltMin = Tmin + self.decouplingDistance
        print(f"updateVoltMaxAndMin; Vmax = {Vmax} [V]; Tmax = {Tmax} [ms]; Vmin = {Vmin} [V]; Tmin = {self.timeCapVoltMin} [ms]")
   
    def calcFreeBudget(self, budget, capVoltDelta):
        # Returns budget in mA
        return 1e3 * self.findMaxInTillFull(self.current_time_ms,
            (self.timeCapVoltMax - self.current_time_ms) if self.timeCapVoltMax > self.current_time_ms else 0,
            self.Vc)

    def findMaxInTillFull(self, startTime, timeDur, startVc):
        # init
        if timeDur == 0:
            return self.budget * 1e-3 # Returns in Ampere
        dRes = self.timeResolution * 2
        start = int(startTime * 1e-3) # to seconds
        end = int((startTime + timeDur) * 1e-3) # to seconds
        next_slot_ms = self.current_time_ms +self.time_per_slot_ms


        if start >= end:
            return self.budget * 1e-3

        maxIn = (self.util_max_current / self.utility_learner.get_min_util()) * 1e-3  # upper bound on consumption; [mA] -> [A]
        minIn = self.sleep_current_amp  # lower bound

        # no operation, if node is off or currently below drop level
        if startVc < self.cap_vc_min or startVc < self.min_goal:
            return minIn

        # First loop to find maximum current
        while abs(maxIn-minIn) > self.simulation_ibal_tolerance:
            empty = False
            In = (maxIn + minIn) / 2.0
            Vc = startVc
            startOfNextSlot = int(next_slot_ms * 1e-3)
            slot = self.planning_slot_counter
            for dt in range(start, end, dRes):
                if dt >= startOfNextSlot:
                    slot += 1
                    startOfNextSlot += self.time_per_slot_ms * 1e-3
                # beware, getPred is in mA but we are calculating in A --> conversion needed
                Vc = self.simulate_newton(
                    dRes, Vc,
                    self.predictor.predict(slot)/ 1000, self.getLimitedPower(In, slot), self.cap_eta)
                print(f"{Vc} ", end="")
                if Vc < self.min_goal:
                    empty = True
                    break
            print("\n")
            if empty:
                maxIn = In
            else:
                minIn = In

        # reset for next run
        maxIn = minIn  # upper bound on consumption; [mA] -> [A]
        minIn = self.sleep_current_amp   # lower bound

        print(f"New Run of findMaxInTillFull with: start = {start} [s] till end = {end} [s] goal {self.capVoltMax} [V] ")
        #for In in np.arange(maxIn, self.sleep_current_amp, -0.1 * 1e-3): # 0.1 mA
        In = maxIn
        while In > self.sleep_current_amp:
            In -= 0.1 * 1e-3  # decrease In by 0.1 mA
        
            # rest of the loop's code

            Vc = startVc
            startOfNextSlot = int(next_slot_ms * 1e-3)
            slot = self.planning_slot_counter
            print(f"In = {In*1e3} [mA]")
            print("Vc: ", end="")
            for dt in range(start, end, dRes):
                Vc = self.simulate_newton(
                    dRes, Vc,
                    self.predictor.predict(slot)/ 1000, self.getLimitedPower(In, slot), self.cap_eta)
                print(f"{Vc} ", end="")
            print("\n")
            if self.getSocWithSimVc(Vc) >= self.getSocWithSimVc(self.capVoltMax) - self.decouplingSoc and self.getSocWithSimVc(Vc) >= self.soc_min_in_free_region:
                # found our current
                print(f"SOCmax = {self.getSocWithSimVc(self.capVoltMax)} SOClast = {self.getSocWithSimVc(Vc)}")
                break
            print(f"SOCDif: {self.getSocWithSimVc(self.capVoltMax) - self.getSocWithSimVc(Vc)}")
        print(f"Found maxInTilFull = {In * 1e3} [mA]")
        return In

    def getLimitedPower(self, In, slot):
        if In * self.utility_get_dc_rel(slot) <= self.sleep_current_amp :
            return self.sleep_current_amp * self.dc_dc_voltage
        if In >= self.util_max_current * 1e-3 * self.utility_get_dc_rel(slot):
            return self.util_max_current * 1e-3 * self.utility_get_dc_rel(slot) * self.dc_dc_voltage
        return In * self.dc_dc_voltage

    def getLimitedBudget(self, budgetFactor, slot):
        if budgetFactor <= 0.0:
            return 0.0
        if budgetFactor * self.util_max_current >= self.util_max_current * self.utility_get_dc_rel(slot):
            return self.util_max_current * self.utility_get_dc_rel(slot)
        
        return budgetFactor * self.util_max_current
    
    def utility_get_dc_rel(self, slot):

        return  self.utility_learner.utility[slot % len(self.utility_learner.utility)]
    
    def adjustBudgetWithSoc(self, budget, capVoltDelta):
        # budget is in mA
        if self.capToSoc(capVoltDelta) < self.decouplingSoc:
            print(f"Behavior as expected; budget remains = {budget}[mA]")
            return budget
        InMax = self.util_max_current * self.utility_get_dc_rel(self.planning_slot_counter)
        In = 0.0
        if capVoltDelta >= (0.0 - self.simulation_volt_tolerance):
            # do more work
            In = budget * (1 + self.getSoc())
        else:
            In = budget * self.getSoc()
        # ensure we do not more than sensefull here in this slot
        if In >= InMax:
            In = InMax
        # ensure we do not return a value not useful for scheduling
        if In < 1e3 * self.sleep_current_amp:
            In = 1e3 * self.sleep_current_amp
        print(f"Adjust Budget; was: {self.budget} now: {In} [mA] soc {self.getSoc()}")
        return In


    def recalcBudget(self, curSlot, Vc):
        return 1e3 * self.findBestIn(int(curSlot),  self.planning_number_of_slots_per_day, Vc)

    def updateSchedValidity(self):
        self.schedValidityPrev = self.schedValidity
        self.schedValidity = self.calcSchedValidity()
        self.timeLastSchedUpdate = self.current_time_ms
        return self.budgetChanged()

    def calcSchedValidity(self):
        # Here Soc is calculated with respect to current voltage of the buffer. 
        # However, in Lars code, Soc is calculated with the voltage at the last slot in findbestIn function
        # findBestIn -> the voltage after the while function. 
        # In actual, it seems to be compilation error in Lars code since getSoc() function considers the current voltage of the capacitor, however, this current voltage of capacitor changes without changing in findbestIn function.  
        # Update: I have changed the voltage in finbestIn, it works better?
        validity = self.checkUpTimeFac * self.getSoc()  
        print(f"CheckuptimwFac { self.checkUpTimeFac}  soc {self.getSoc()}")

        step_sec = 300
        if validity < step_sec * 1e3:
            return int(step_sec * 1e3)
        if validity >= float(self.time_per_slot_ms):
            return self.time_per_slot_ms
        return int(validity)

    def socDif(self):
        return abs(self.capToSoc(self.capVoltDelta)) > self.decouplingSoc

    def capToSoc(self, VcDelta):
        return ((self.Vc * self.Vc - self.min_goal * self.min_goal) -
                ((self.Vc - VcDelta) * (self.Vc - VcDelta) - self.min_goal * self.min_goal)
                ) / (self.capacitor_max_voltage * self.capacitor_max_voltage - self.min_goal * self.min_goal)

    def getSoc(self):
        print(f"Inside getSoc, Vc:  {self.Vc} VcMax: {self.capacitor_max_voltage}")

        soc = (self.Vc * self.Vc - self.min_goal * self.min_goal) / (self.capacitor_max_voltage  * self.capacitor_max_voltage  - self.min_goal * self.min_goal)
        
        return max(soc, 0.0)

    def getSocWithSimVc(self, simVc):
        soc = (simVc * simVc - self.min_goal * self.min_goal) / \
            (self.capacitor_max_voltage  * self.capacitor_max_voltage  - self.min_goal * self.min_goal)
        return max(soc, 0.0)

    def budgetChanged(self):
        print("LastChange: ", self.lastChange, " time: ", self.current_time_ms)
        if self.budgetPrev != self.budget:
            self.lastChange = self.current_time_ms
            return True
        if self.current_time_ms >= self.lastChange + self.time_per_slot_ms:
            print("LastChange exceeded")
            return True
        return False


    def err(self, x, y):
                return (x - y) if x > y else (y - x)

    def simulate_newton(self, dt, Vc, Ih, Pn, eta):
        
        a:float = Ih / self.capacitor_size
        b:float = Pn / (eta * self.capacitor_size)
        v0:float = Vc
        y:float = v0
        print(f">>>Findbestin NewtonInputs>>> dt: {dt}  Vc: {Vc} Ih: {Ih} Pn: {Pn} eta: {eta} cap: {self.capacitor_size}")

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
        sleep_power = minIn * self.dc_dc_voltage # Should be  0.000158 W
        next_slot_ms = self.current_time_ms + self.time_per_slot_ms

        # -> removed this check because we only run this once per day and need the update of capVoltPred!
        # no operation, if node is off or currently below drop level
        #if startVc < self.cap_vc_min or startVc <  self.min_goal:
        #    return minIn

        while abs(maxIn - minIn) > self.simulation_ibal_tolerance:
            ok = True  # success indicator for each binary search step
            In = (maxIn + minIn) / 2.0
            Vc = startVc
            dt = 0.0
            tasks_time = (self.time_per_slot_sec - 3420) # 180 secs (30sec x 6 tasks) = 3600 secs - 3420 secs (sleep) 
            sleep_time = (self.time_per_slot_sec - tasks_time)
            for ds in range(numSlots):
                slot = startSlot + ds
                dt = self.time_per_slot_sec
               
                # Here, predictor.predict returns in mA but we are calculating in Ampere --> conversion needed
                # Time should be in seconds
                Vc = self.simulate_newton(
                    dt, 
                    Vc,
                    self.predictor.predict(slot)/ 1000, 
                    self.getLimitedPower(In, slot), self.cap_eta)

                # Simulate tasks voltage
                '''v_after_tasks = self.simulate_newton(
                            tasks_time, 
                            Vc,
                            self.predictor.predict(slot)/ 1000, 
                            self.getLimitedPower(In, slot), self.cap_eta)
                
                # Simulate sleep voltage
                v_after_1h = self.simulate_newton(
                            sleep_time, 
                            v_after_tasks,
                            self.predictor.predict(slot)/ 1000, 
                            sleep_power, self.cap_eta)


                Vc = v_after_1h
                '''
                if Vc < self.min_goal:
                    ok = False
                    break

            # update search boundaries
            if ok:
                minIn = In
            else:
                maxIn = In
            # DEBUG_STDOUT("minIn: " << minIn << " [A]   maxIn: " << maxIn << " [A]");
        print(f"Vc: {Vc} After Findbestin")
        print(f"Now Update Prediction with In = {minIn * 1e3} mA and VcPred: ")
        
        #self.Vc = Vc # Is this intented? Its not in the lars code but unintentedly it's getting changed. 
        
        Vc = startVc    
        dt = 0.0
        Pn = 0.0
        for ds in range(numSlots):
            slot = startSlot + ds
            # beware; we might trigger recalc within a slot; first one might not be complete
            if ds == 0:
                dt = (next_slot_ms - self.current_time_ms) * 1e-3
            else:
                dt = self.time_per_slot_sec
            #print(f"dt: {dt}")
            # beware, getPred is in mA but we are calculating in A --> conversion needed
            Vc = self.simulate_newton(dt, Vc, self.predictor.predict(slot)/ 1000, self.getLimitedPower(minIn, slot), self.cap_eta)
            self.capVoltPred[slot % self.planning_number_of_slots_per_day] = Vc
            print(f"Vpred[{slot % self.planning_number_of_slots_per_day}]: {self.capVoltPred[(slot + self.planning_number_of_slots_per_day - 1) % self.planning_number_of_slots_per_day]} Ih[{slot % self.planning_number_of_slots_per_day}]: {self.predictor.predict(slot)}")
        print("\n")

        return minIn


    
    def getScheduleValidity(self):
            return self.schedValidity

    def isTimeToSchedule(self):
        print("At:", self.current_time_ms, "last update:", self.timeLastSchedUpdate, "sched valid:", self.schedValidity, "next update:", self.timeLastSchedUpdate + self.schedValidity)
        print("dosched:", self.doSched)
        return self.doSched

   

    