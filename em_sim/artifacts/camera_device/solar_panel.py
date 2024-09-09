from typing import Any, Dict, Sequence
import pandas as pd
from pathlib import Path
from em_sim.core import Component, VariableMode
import numpy as np
from datetime import datetime, timedelta
import parse
import logging

class SolarPanel(Component):
    """Access to solar data from Lars dataset."""

    def __init__(self, config: Dict[str, Any]):
        """ """
        super().__init__(config, "Solar Panel")
        self.logger = logging.getLogger(config["id"])

        self.logger.info("Solar Panel initialization started")

        self.dataset_name: str = config["dataset_solar_name"]
        self.year: str = config["solar_year"]
        
        
        if self.dataset_name not in ["TUHHx"]:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        current_mA, number_of_days = self.load_dataset(self.dataset_name, config)        
        
        self.data: pd.DataFrame = pd.DataFrame()
        self.data["current_mA"] = self.data.assign(current_mA = current_mA.values * config["solar_scale"] )
        
        if config["shift_solar_data"] == True:
            shift_value = 288//2 # '//' is integer division in Python
            # Perform cyclic shift by 144 positions i.e. 12 hrs
            shifted_values = np.roll(self.data["current_mA"], shift_value)

            # Update the 'person_count_ground' column with the shifted values
            self.data["current_mA"] = shifted_values
            
        
        #self.data = self.data[: config["days"]*288]
        self.number_of_days = number_of_days
        #SECONDS_PER_YEAR: int = 365 * 24 * 3600
        #self.yearly_energy: float = self.steady_power * SECONDS_PER_YEAR
        #self.data["solar_scaled"] = self.data.solar * (
            #self.yearly_energy / self.data.solar.sum()
        #)


        # add_parameter to log values and use in visualization
        self.add_parameter("Solar Current", 0, unit="mA")
        self.add_parameter("dataset", self.dataset_name)
        self.add_parameter("Number of days", number_of_days)
        self.add_parameter("solar_year", self.year)
        self.add_parameter("solar_scale", config["solar_scale"])

    
    def load_dataset(self, dataset_name, config):
        
        path: Path = (
            Path(__file__).parent
            / f"./datasets/solar/{dataset_name}.txt"
        )
        
        if not path.is_file():
            raise ValueError(f"Expected file with solar data in path {path}.")
        
        data: pd.DataFrame = pd.read_csv(
                                    path,
                                    sep = ',',
                                    skiprows=11, index_col = False)
        data.columns = ["date", "current_mA"]
        
        
        with open(path, "r") as f:
            for _ in range(2):
                line = f.readline()
            number_of_days = parse.search("Days: {:d}", line)[0] 
        
       
        if config["sim_single_day"] == True:
            # Reseting index so we could select days
            selected_df = data["current_mA"].reset_index()  

            # Select One day 
            # Calculate the starting and ending indices
            start = (config["sim_solar_day"] - 1) * 288 # starting from one previous day,  288 = num of samples per day
            end = (config["sim_solar_day"] + 2) * 288 # We load 2 days for predictor

            # Select the rows
            day_x_df = selected_df.iloc[start:end, 1]
            return day_x_df.reset_index(drop=True), number_of_days
        

        # In Return: We skip the first day values here "data["current_mA"][288:].reset_index()"
        # since this is how it is done in C++ code by Lars
        return data["current_mA"][288:].reset_index(drop=True), number_of_days
    
    def init_dataframe(self, df: pd.DataFrame) -> None:
        
        #self.data.set_index("index", inplace=True, drop=False)
        #resampled: pd.DataFrame = self.data.resample("300S").ffill()

        #merged: DataFrame = df.merge(self.data["current_mA"])
        #df["intake_current_mA"] = self.data["current_mA"][: len(df)]
        #merged: pd.DataFrame = df.merge(self.data[:len(df)], left_index=True, right_index=True)
       
        df_length = len(df)
        new_data = self.data["current_mA"][:df_length]
        if len(df) > len(self.data):
            raise ValueError(f"Not enough solar data")
        df["intake_current_mA"] = new_data.values
