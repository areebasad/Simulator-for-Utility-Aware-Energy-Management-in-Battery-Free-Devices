from typing import Any, Dict, List
from .capacitor import Capacitor
from .solar_panel import SolarPanel
from .scheduler import ClassicScheduler
from .predictors import Predictor, Preact
from .depletion_safe import DepletionSafe
import numpy as np

from em_sim.core import VariableMode, Device

import pandas as pd
from datetime import datetime, timedelta


class LarsDevice(Device):
    """Example of a lars device."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, "LarsDevice", label="Porting code")

        self.d_id: str = config["id"]
        # Add configuration settings in log book
        self.add_parameter("id", self.d_id)
        self.add_parameter("class", type(self).__name__)

        # Add Solar Panel
        self.solar: SolarPanel = SolarPanel(config)
        self.add_component(self.solar)

        # Add Capacitor
        self.buffer: Capacitor = Capacitor(config)
        self.add_component(self.buffer)

        # Add Scheduler, Consumer is integrated in scheduler
        self.scheduler: ClassicScheduler = ClassicScheduler(config)
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
        self.config["device_mingoal"] = 1.5                 # Alternative to min voltage, can be get from the user
        self.tasks_list = []
        self.tasks_counter = 0
        self.schedule_start_time_ms = 0
        # Add Energy Manager
        self.energy_manager: DepletionSafe = DepletionSafe(config, self.buffer, self.scheduler)
        #self.energy_manager: Preact = Preact(config, self.buffer, self.scheduler)
    def init_dataframe(self, df: pd.DataFrame) -> None:
        # Note here, the simulator byitself is calling ini_datframe functions for all componnets. 
        # So, if you will not call init_dataframe function even then it will be called. Sort of misleading. FixIt
        # Could be called multiple times

        #self.buffer.init_dataframe(df)
        #self.add_variable(df, "consumption", VariableMode.CALCULATED, "W", float("nan"))
        self.solar.init_dataframe(df)
        self.buffer.init_dataframe(df)
        self.add_variable(df, "timestamp_seconds", VariableMode.CALCULATED, "W", float("nan"))
        df["timestamp_seconds"] = np.arange(0, df.shape[0]*self.config["simulation_step_time"], step= self.config["simulation_step_time"])

        self.add_variable(df, "is_planning_slot", VariableMode.CALCULATED, "W", bool("False"))
        df["is_planning_slot"] = np.where(np.arange(len(df)) % self.config["next_planning_after_x_steps"], False, True)


        self.add_variable(df, "budget_current_mA", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "isDeviceOn", VariableMode.CALCULATED, "Bool", True)
        self.add_variable(df, "task_executed", VariableMode.CALCULATED, "Bool", False)
        self.add_variable(df, "person_count_exp", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "detections_exp", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "active_energy_consumed_joules", VariableMode.CALCULATED, "W", float("nan"))
        self.add_variable(df, "policy_epsilon", VariableMode.CALCULATED, "W", float("nan"))

        pass
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
           
            # Call  Budget_Current = manager (intake_current, Vc_capacitor, nodeTime)
            # Call the scheduler
            df.at[state, "budget_current_mA"], em_run = self.energy_manager.step(state, df)
            if em_run == True:
                self.tasks_list = self.scheduler.schedule(state, df)
                self.tasks_counter = 0
                self.schedule_start_time_ms = df.at[state, "timestamp_seconds"] * 1000
            # Write values     
        else:
            pass
            # Write values
        
        # Simulate between the two states 
        # i.e. simulate 300 seconds and run tasks in between
        current_time_sec = df.at[state, "timestamp_seconds"]
        current_time_ms = current_time_sec*1000
        
        self.simulation_step_time = 300     # [sec]
        time_delta_ms = 0                   # [ms]
        current_to_consume = 0              # [A]
        buffer_voltage:float = df.at[state, "buffer_voltage"]
        is_device_on = df.at[state, "isDeviceOn"]
        is_reset = False
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
                    time_delta_ms = min(self.scheduler.task_duration - (current_time_ms - task_start), (df.at[next_state, "timestamp_seconds"] * 1000) - current_time_ms)
                    current_to_consume = self.scheduler.task_power / self.scheduler.dc_dc_voltage * 1e-6 #[uW -> A]
                
                    # End of current execution >= end of task
                    if ((current_time_ms+time_delta_ms) >= (task_start+self.scheduler.task_duration)):
                        ## Move to next task, task has completed
                        self.tasks_counter+=1
                
            else:
                # No task remaining
                current_to_consume = self.scheduler.sleep_current
                time_delta_ms = (df.at[next_state, "timestamp_seconds"] * 1000) - current_time_ms # Time remaiining in slice


            time_delta_sec = time_delta_ms/1000
            power_to_consume = (current_to_consume * self.scheduler.dc_dc_voltage) if is_device_on else 0
            intake_amp = df.at[state, "intake_current_mA"]/1000
            buffer_voltage = self.buffer.simulate_newton(time_delta_sec, buffer_voltage, intake_amp,
                   power_to_consume,  self.buffer.capacitor_eta )

            # Add here code for device on | off
            # check system state:
            # - if RealVc is below cut-off voltage, the system goes down
            # - if the system is down and RealVc exceeds VCSTART, the system re-starts
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



            current_time_ms = current_time_ms + time_delta_ms


        df.at[next_state, "isDeviceOn"] = is_device_on
        df.at[next_state, "buffer_voltage"] = buffer_voltage

        
