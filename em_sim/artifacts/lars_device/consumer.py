from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math

class Consumer(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Consumer")

        
        self.e_max_active = 100*1e-3 # [A]
        self.sleep_current = 10*1e-3 # [A] Wrong sleep consumption
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.task_power = 33000      # [uW]
        self.task_duration = 500     # [ms]


    def init_dataframe(self, df: pd.DataFrame) -> None:
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass

    def schedule(self, state: int, df:pd.DataFrame) -> None:
         # Inputs
        #budget = df.at[state, "budget"]
        #horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms

        #if (budget )
        pass
      

       
    def consume(
            self,
            state: int,
            next_state: int,
            df: pd.DataFrame,
            duty_cycle: float):
            
            df.at[state, "consumption"] = self.e_baseline * duty_cycle

            return df.at[state, "consumption"]
            
   