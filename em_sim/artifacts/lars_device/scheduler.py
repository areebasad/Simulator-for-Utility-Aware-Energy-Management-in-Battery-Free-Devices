from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math


class Scheduler(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Basic Scheduler")

    def init_dataframe(self, df: pd.DataFrame) -> None:
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass
       
    def schedule(state: int, df: pd.DataFrame ):
        pass


class Task:
    def __init__(self, starting_time, power, duration):
        self.starting_time = starting_time
        self.power = power
        self.duration = duration

class ClassicScheduler(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Classic Scheduler")
        
        self.dc_dc_voltage = 3.3     # [V]
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.sleep_current = self.sleep_power/self.dc_dc_voltage * 1e-6 # [A]
        self.task_power = 33000      # [uW]
        self.task_duration = 500     # [ms]
        self.MAX_REPS = 10000
        self.schedule_list = []

    def init_dataframe(self, df: pd.DataFrame) -> None:
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass        

    # Calculates reps energy
    def get_repetitions(self, budget: int, horizon: int) -> int:
        
        #Inputs
        # Budget in current mA
        # horizon in ms

        repetitions:int = 0
        task_energy = 0.0

        #Energy = Power x Time
        task_energy = (self.task_power - self.sleep_power) *  self.task_duration # nano Joule
        budget_energy = (1000 * budget * self.dc_dc_voltage - self.sleep_power) * horizon #nano Joule


        if (task_energy != 0.0):
            repetitions = int(budget_energy/task_energy)
        
        if (repetitions < 0):
            return 0

        if (repetitions >= self.MAX_REPS):
            return self.MAX_REPS

        return repetitions    


    def schedule(self, state: int, df: pd.DataFrame) -> list:
        
        # Inputs
        budget = df.at[state, "budget_current_mA"] # mA?
        horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


        repetitions = 0

        if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
            repetitions = self.get_repetitions(budget, horizon)

        if (repetitions <= 0):
            pass
        
        starting_time = 1
        min_task_distance = 1 # ms
        self.schedule_list = []
        for task in range(repetitions):
            self.schedule_list.append(Task(starting_time, self.task_power, self.task_duration))
            starting_time += self.task_duration + min_task_distance

        return self.schedule_list
   

class ClassicSchedulerCameraDevice(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Classic Scheduler")
        
        self.dc_dc_voltage = 3.3     # [V]
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.sleep_current = self.sleep_power/self.dc_dc_voltage * 1e-6 # [A]
        self.task_power = 33000      # [uW]
        self.task_duration = 3000     # [ms]
        self.MAX_REPS = 10000
        self.schedule_list = []

    def init_dataframe(self, df: pd.DataFrame) -> None:
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass        

    # Calculates reps energy
    def get_repetitions(self, budget: int, horizon: int) -> int:
        
        #Inputs
        # Budget in current mA
        # horizon in ms

        repetitions:int = 0
        task_energy = 0.0

        #Energy = Power x Time
        task_energy = (self.task_power - self.sleep_power) *  self.task_duration # nano Joule
        budget_energy = (1000 * budget * self.dc_dc_voltage - self.sleep_power) * horizon #nano Joule


        if (task_energy != 0.0):
            repetitions = int(budget_energy/task_energy)
        
        if (repetitions < 0):
            return 0

        if (repetitions >= self.MAX_REPS):
            return self.MAX_REPS

        return repetitions    


    def schedule(self, state: int, df: pd.DataFrame) -> list:
        
        # Inputs
        budget = df.at[state, "budget_current_mA"] # mA?
        horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


        repetitions = 0

        if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
            repetitions = self.get_repetitions(budget, horizon)

        if (repetitions <= 0):
            pass
        
        starting_time = 1
        min_task_distance = 1 # ms
        self.schedule_list = []
        for task in range(repetitions):
            self.schedule_list.append(Task(starting_time, self.task_power, self.task_duration))
            starting_time += self.task_duration + min_task_distance

        return self.schedule_list