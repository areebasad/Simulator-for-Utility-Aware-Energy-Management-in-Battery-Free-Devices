from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta


class EnergyPredictor(object):

    def update(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class Predictor(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Predictor: Ideal Predictor")

        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.num_of_steps_between_planning = 12 # if planning_number_of_slots_per_day is 24
        self.predictions_day:float = []         # Average 
        self.predictions_mean = 0.0
        #duration: timedelta = timedelta(days=config["days"])  #Make sure it's same in the configuration as well.
        #self.end = config["start_day"] + duration  # type: ignore
        #self.prediction_end_state = (config["days"] * 288) - (2 * 288)


    def update(self, state: int, df: pd.DataFrame) -> None:

        self.prediction_end_state = len(df) - (2 * 288)

        # If next day; calculate avg current for each planning slots in a day. 
        # #state < self.prediction_end_state (2 days before ending)
        if (state % 288) == 0 and state <= self.prediction_end_state:
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
    
    def update_x(self, state: int, df: pd.DataFrame) -> None:

        self.prediction_end_state = len(df) - (2 * 288)

        # If next day; calculate avg current for each planning slots in a day. 
        # #state < self.prediction_end_state (2 days before ending)
        if ((state % 1) == 0 and state <= self.prediction_end_state):
            counter = 0 + state
            self.predictions_day = []
            for planning_slot in range(self.planning_number_of_slots_per_day*2):           
                sum = 0.0         
                for step_slot in range (self.num_of_steps_between_planning):            
                    sum += df.at[counter + step_slot, "intake_current_mA"]        
                self.predictions_day.insert(planning_slot,  sum/self.num_of_steps_between_planning)
                counter += self.num_of_steps_between_planning
            
            self.calc_mean_x()

    def calc_mean_x(self):
        self.predictions_mean = sum(self.predictions_day[0:self.planning_number_of_slots_per_day])/ self.planning_number_of_slots_per_day 


    


class EWMA(Component):

    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Predictor: EWMA Predictor")

        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.num_of_steps_between_planning = 12 # if planning_number_of_slots_per_day is 24
        self.predictions_day:float = []         # Average 
        self.predictions_mean = 0.0
        self.ewma:float =[0] * 288 # length is 288
        #duration: timedelta = timedelta(days=config["days"])  #Make sure it's same in the configuration as well.
        #self.end = config["start_day"] + duration  # type: ignore
        #self.prediction_end_state = (config["days"] * 288) - (2 * 288)
   
        self.alpha = 0.5#config["ewma_alpha"]
        #self.buffer = 0.0
    
    def update(self, state: int, df: pd.DataFrame) -> None:

            self.prediction_end_state = len(df) - (2 * 288)

            # If next day; calculate avg current for each planning slots in a day. 
            # #state < self.prediction_end_state (2 days before ending)
            if (state % 288) == 0 and state <= self.prediction_end_state:
                counter = 0 + state
                self.predictions_day = []
                
                for planning_slot in range(self.planning_number_of_slots_per_day*2):           
                    sum = 0.0         
                    for step_slot in range (self.num_of_steps_between_planning):
                        # ewma(t) = alpha * x(t) + (1-alpha) * ewma(t-1)
                        self.ewma[planning_slot+step_slot] = self.alpha * df.at[counter + step_slot, "intake_current_mA"]  + (1.0 - self.alpha) * self.ewma[planning_slot+step_slot]           
                        sum += self.ewma[planning_slot+step_slot]      
                    self.predictions_day.insert(planning_slot,  sum/self.num_of_steps_between_planning)
                    counter += self.num_of_steps_between_planning
                
                self.calc_mean()

    def calc_mean(self):
        self.predictions_mean = sum(self.predictions_day[0:self.planning_number_of_slots_per_day])/ self.planning_number_of_slots_per_day 

    def predict(self, x):
        return self.predictions_day[x]    