from typing import Any, Dict, List
from .capacitor import Capacitor
from .solar_panel import SolarPanel
from .scheduler import ClassicScheduler, ClassicSchedulerBaseline, ClassicSchedulerWithAdaptation
from .predictors import Predictor
from .em_depletion_safe import DepletionSafe, NoManager
#from .extra_em_variants.em_depletion_safe_plusC4 import DepletionSafePlus # Change here the versions of DSPlus
from .em_depletion_safe_plus import DepletionSafePlus # Change here the versions of DSPlus
from .em_utiligem import UtiliGEM # Change here the versions of DSPlus
from .em_emrep import EmRep
from .extra_em_variants.em_emrep_plus import EmRepPlus
from .em_preact import Preact
from .extra_em_variants.em_preact_app_u import PreactAppU
import numpy as np

from em_sim.core import VariableMode, Device

import pandas as pd
from datetime import datetime, timedelta
import logging

class CameraDevice(Device):
    """Example of a Visual Sensor Device."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, "VSDevice", label="Code")

        self.d_id: str = config["id"]

        # Add configuration settings in log book
        self.add_parameter("id", self.d_id)
        self.add_parameter("class", type(self).__name__)

        self.logger = logging.getLogger(config["id"])
        self.logger.info("Device initialization started")

        # Add Solar Panel
        self.solar: SolarPanel = SolarPanel(config)
        self.add_component(self.solar)

        # Add Capacitor
        self.buffer: Capacitor = Capacitor(config)
        self.add_component(self.buffer)

        # Add Scheduler, Consumer is integrated in scheduler
        if config["planner"] == "no_manager":
            config["scheduler"] = "classicSchedulerBaseline"

        schedulers = {
           "classicScheduler" : ClassicScheduler, # Lars scheduler
           "classicSchedulerBaseline" : ClassicSchedulerBaseline, # Camera baseline
           "classicSchedulerAdaptation": ClassicSchedulerWithAdaptation 
        }
        self.scheduler = schedulers[config["scheduler"]](config)
        self.add_component(self.scheduler)

        # Add Predictor
        # self.predictor: Predictor = Predictor(config)
        # self.add_component(self.predictor)

        self.config = config

        # Device Properties/parameters
        self.config["planning_number_of_slots_per_day"] = config["planning_number_of_slots_per_day"]
        self.config["simulation_steps_day"] = 288
        self.config["next_planning_after_x_steps"] = self.config["simulation_steps_day"] /  self.config["planning_number_of_slots_per_day"]
        self.config["simulation_step_time"] = 300           #[seconds]
        self.config["device_max_current_amp"] = 100 * 1e-3  # TBD
        self.config["device_vc_start"] = 1.6                # switch-on voltage after a break-down (Vc < VCMIN)
        self.config["device_vc_min"] = 1.3                  # minimum regulator operating voltage
        self.config["device_vc_max"] = 2.675                # maximum supercap terminal voltage
        #self.config["device_mingoal"] = 1.5                 # Alternative to min voltage, can be get from the user
        self.tasks_list = []
        self.tasks_counter = 0
        self.schedule_start_time_ms = 0
        # Add Energy Manager
        energy_managers ={
           "no_manager" : NoManager,
           "depletion_safe": DepletionSafe,
           "depletion_safe_plus": DepletionSafePlus,
           "utiligem": UtiliGEM,
           "preact": Preact,
           "preactAppU": PreactAppU,
           "emrep": EmRep,
           "emrepPlus": EmRepPlus,
           "ulenp": Ulenp,

        }
        self.energy_manager = energy_managers[config["planner"]](self.config, self.buffer, self.scheduler)
        #self.energy_manager: Preact = Preact(config, self.buffer, self.scheduler)
    
    def init_dataframe(self, df: pd.DataFrame) -> None:
       
        # First Add application images, and prepare the df according to it.
        self.scheduler.init_dataframe(df)
        #df = df.resample("300S").asfreq()
        # Count the number of unique timestamps before the groupby operation
        #before_count = len(df['image_timestamp'].unique())

        # group by timestamp column using Grouper and resample at 5 minute intervals, including NaN values
        #grouped_df = df.groupby(pd.Grouper(key='image_timestamp', freq='10T', closed='right', label='right' )).first() #closed='right', label='right'
        #df = grouped_df.resample('5T').asfreq()
        #df = df.resample('5T', on='image_timestamp').first()
        #df = df.dropna()
        #df.reset_index(inplace=True)
        # Count the number of unique timestamps after the groupby operation
        #after_count = len(df['image_timestamp'].unique())
        #duplicated_timestamps = df.image_timestamp.duplicated()
        #print(duplicated_timestamps.sum())
        # Print the counts
        #print(f"Before: {before_count}, After: {after_count}")
        df.reset_index(inplace=True)
        #df = df.iloc[:-1] # We take out the next day extra instance that we added as a work around to get the job done by the grouper 
        # Get all but the last row index
        #last_index = df.index[-1]
        #df = df.iloc[:-1]

        # Modify the original DataFrame in place
        #df = df.loc[:last_index]


        df["error_image"] = df["error_image"].replace({0:False, 1:True})

        self.add_variable(df, "intake_current_mA", VariableMode.CALCULATED, "W", float("nan"))
        self.solar.init_dataframe(df)
        self.buffer.init_dataframe(df)
        self.add_variable(df, "timestamp_seconds", VariableMode.CALCULATED, "W", float("nan"))
        df["timestamp_seconds"] = np.arange(0, df.shape[0]*self.config["simulation_step_time"], step= self.config["simulation_step_time"])

        self.add_variable(df, "is_planning_slot", VariableMode.CALCULATED, "W", bool("False"))
        df["is_planning_slot"] = np.where(np.arange(len(df)) % self.config["next_planning_after_x_steps"], False, True)


        self.add_variable(df, "budget_current_mA", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "isDeviceOn", VariableMode.CALCULATED, "Bool", True)
        self.add_variable(df, "task_executed", VariableMode.CALCULATED, "Bool", False)
        self.add_variable(df, "task_executed_energy_pct", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "person_count_exp", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "detections_exp", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "person_count_exp_approx", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "detections_exp_approx", VariableMode.CALCULATED, "W", float("nan"))

        self.add_variable(df, "active_energy_consumed_joules", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "policy_epsilon", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "utility_ucb", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "utility_reward_mean", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "utility_ucb_lower_bound", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "utility_dc", VariableMode.CALCULATED, "W", float("nan"))

        self.add_variable(df, "preact_soc_forecast", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "dsp_consp_forecast", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "dsp_voltage_forecast", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "vam_convg", VariableMode.CALCULATED, "Bool", False)
        self.add_variable(df, "vam_distribution", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "app_req_budget_mA", VariableMode.CALCULATED, "W", float("nan"))


        return df
        
    def first_state(self, state: int, df: pd.DataFrame) -> None:
        #df.at[state, "consumption"] = self.consumption_per_cycle
        #df.at[state, "cycle"] = "ok"
        self.buffer.first_state(state, df)
        
    def step(self, state: int, next_state: int, df: pd.DataFrame) -> None:
        ## Steps
        # if nextSlot -> compute utility
        # Iterate over days -> if next day then update few things
        # Get Budget from energy manager -> If node is not down then manager and make schedule otherwise write out
        # Consume energy plus simulate capacitor
        if df.at[state, "isDeviceOn"] == True:
            if self.config["planner"] == "no_manager":   
                # Call the scheduler directly and do nothing on first day since that's the case in Emrep-Lars code
                if state < 288: 
                    self.tasks_list = []
                    self.tasks_counter = 0
                    self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
                else:    
                    if state % 12 == 0:
                        self.tasks_list = self.scheduler.schedule(state, df)
                        self.tasks_counter = 0
                        self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
            else:
                # Call manager
                df.at[state, "budget_current_mA"], em_run = self.energy_manager.step(state, df, df.at[state, "isDeviceOn"])
                
                if em_run == True:
                    # Call scheduler
                    self.tasks_list = self.scheduler.schedule(state, df)
                    self.tasks_counter = 0
                    self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
        
        else: # Device is OFF
                
            if self.config["planner"] == "no_manager":   
                # Call the scheduler directly and do nothing on first day since that's the case in 
                if state < 288: 
                    self.tasks_list = []
                    self.tasks_counter = 0
                    self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
                else:    
                    if state % 12 == 0:
                        self.tasks_list = self.scheduler.schedule(state, df)
                        self.tasks_counter = 0
                        self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
                
                
            else:    
                # Call manager to only update planning counters, does not runs the planner
                _ , em_run = self.energy_manager.step(state, df, df.at[state, "isDeviceOn"])
                

        # Simulate between the two states 
        # i.e. simulate 300 seconds and run tasks in between
        current_time_sec = df.at[state, "timestamp_seconds"]
        current_time_ms = current_time_sec*1000
        
        self.simulation_step_time = 300     # [sec]
        time_delta_ms = 0                   # [ms]
        current_to_consume = 0              # [A]
        buffer_voltage:float = df.at[state, "buffer_voltage"]
        buffer_voltage_foe:float = df.at[state, "buffer_voltage_foe"]
        is_device_on = df.at[state, "isDeviceOn"]
        is_reset = False
        task_executed = False
        while (current_time_ms < df.at[next_state, "timestamp_seconds"]*1000):
            
            if (self.tasks_counter < len(self.tasks_list)):
                task_start = self.schedule_start_time_ms +  self.tasks_list[self.tasks_counter].starting_time

                if(task_start > current_time_ms):
                    # Sleep 
                    # Task does not follow another task directly
                    # min(task start, next slice)    
                    time_delta_ms = min(task_start, (df.at[next_state, "timestamp_seconds"] * 1000)) - current_time_ms # make sure that we do not simulate out off this slice
                    current_to_consume = self.scheduler.sleep_current
                else:
                    # Active | Run Task 
                    time_delta_ms = min(self.tasks_list[self.tasks_counter].duration - (current_time_ms - task_start), (df.at[next_state, "timestamp_seconds"] * 1000) - current_time_ms)
                    current_to_consume = self.scheduler.task_power / self.scheduler.dc_dc_voltage * 1e-6 #[uW -> A]
                
                    # End of current execution >= end of task
                    if ((current_time_ms+time_delta_ms) >= (self.tasks_list[self.tasks_counter].duration)):
                        ## Move to next task, task has completed
                        self.tasks_counter+=1
                        task_executed = True
            else:
                # No task remaining
                current_to_consume = self.scheduler.sleep_current # ampere
                time_delta_ms = (df.at[next_state, "timestamp_seconds"] * 1000) - current_time_ms # Time remaiining in slice


            time_delta_sec = time_delta_ms/1000
            power_to_consume = (current_to_consume * self.scheduler.dc_dc_voltage) if is_device_on else 0
            current_to_consume_r = current_to_consume if is_device_on else 0
            intake_amp = df.at[state, "intake_current_mA"]/1000
            # Simulate Buffer Voltage
            buffer_voltage = self.buffer.simulate_newton(time_delta_sec, buffer_voltage, intake_amp,
                   power_to_consume,  self.buffer.regulator_eta(buffer_voltage, current_to_consume_r) )
            # Test FOE
            #buffer_voltage_foe = self.buffer.simulate_newton_foe(time_delta_sec, buffer_voltage_foe, intake_amp,
            #       power_to_consume,  self.buffer.capacitor_eta )
            #buffer_voltage = buffer_voltage_foe
            # If succesfully executed the task then count it (save in dataframe) otherwise this task is not counted
            if buffer_voltage >= self.config["device_vc_min"] and task_executed is True:
                # Depending on the EM, get the task output and update utility learner
                if self.config["planner"] == "no_manager" or self.config["scheduler"] == "classicScheduler":   
                    self.scheduler.get_task_output(state, df)
                else:
                    self.scheduler.get_task_output(state, df, self.tasks_list[self.tasks_counter-1], self.config["planner"] == "depletion_safe_plus")
                    # Don't you neeed it for Emrep?
                    if (self.config["planner"] == "preact" or self.config["planner"] == "emrep" or self.config["planner"] == "emrepPlus" or self.config["planner"] == "depletion_safe_plus" or self.config["planner"] == "utiligem") and self.config["learn_temporal"] == True:
                        # Input: person count, state, epsilon(energy spent i.e. task_executed_energy_pct), change person_count_exp_approx if using yolo?
                        self.energy_manager.utility_learner.update_rewards_counts(df.at[state, "person_count_exp_approx"], state,  df.at[state, "task_executed_energy_pct"])
                
                # Correct how you store energy consumped as you might have to add all em consumption within a state and then store it
                # Right now we are storing only task consumption if exectued. We are not storing energy consumption in sleep mode.
                # State is equal to 300 secs(5 mins) and our task is of 30 secs. 
                df.at[state, "active_energy_consumed_joules"] = time_delta_sec * power_to_consume
                df.at[state, "task_executed"] = True
                
                # Check for convergence and pass values to UtiliGEM(DS+)
                if (self.config["planner"] == "depletion_safe_plus" or self.config["planner"] == "utiligem"):
                #if self.config["check_spatial_convg"] == True:
                    # Pass values to energy manager
                    self.energy_manager.utility_learner.vam_convg = df.at[state, "vam_convg"]
                    self.energy_manager.utility_learner.app_re_budget_mA  = df.at[state, "app_req_budget_mA"]


            
            task_executed = False


            # Code for device ON | OFF
            # check system state:
            # - if Real/Current Vc is below cut-off voltage, the system goes down
            # - if the system is down and Real/Current Vc exceeds VCSTART, the system re-starts
            if not is_device_on or buffer_voltage < self.config["device_vc_min"]:
           #     timeDown += dt * 1e-3
                is_device_on = False
           #     if not isReset:
                self.tasks_counter = len(self.tasks_list)        #cInst = eInst  # Don't process any tasks
                    #sched.blackOut()           # Resets energy manager counters and makes budget 0 
                    #nodeTime = 0
                    #isReset = True
                if not is_device_on and buffer_voltage > self.buffer.capacitor_start_voltage and (state % self.config["next_planning_after_x_steps"]) == 0:
                    #nodeTime = 0
                    is_device_on = True
                    #isReset = False
                    
                    #print("Align and Turn on Again; at slot", (slot % numSlots))
                    #sched.em.getPredictor().alignPred(slot % numSlots)
                    #sched.em.checkUtilAligned(slot % numSlots)

            #if isDown:
                #outFineInValues.append(-1)
            #    slotUtil += 0.0
            #else:
            #    slotUtil += (curIn * 1e3 * dt)
            #    outFineInValues.append(curIn)


            # Update timer for simulation within a state
            current_time_ms = current_time_ms + time_delta_ms

        # Store Values
        df.at[next_state, "isDeviceOn"] = is_device_on
        df.at[next_state, "buffer_voltage"] = buffer_voltage
        #df.at[next_state, "buffer_voltage_foe"] = buffer_voltage_foe
        df.at[next_state, "buffer_charge"] = self.buffer.calculate_buffer_charge(buffer_voltage)

        
