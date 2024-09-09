from typing import Any, Dict, Sequence
import pandas as pd
from pathlib import Path
from em_sim.core import Component, VariableMode
import numpy as np
from datetime import datetime, timedelta
import parse


class SolarPanel(Component):
    """Access to solar data from Lars dataset."""

    def __init__(self, config: Dict[str, Any]):
        """ """
        super().__init__(config, "Solar Panel")
        
        self.dataset_name: str = config["dataset_name"]
        self.year: str = config["solar_year"]
        
        
        
        if self.dataset_name not in ["TUHHx"]:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        current_mA, number_of_days = self.load_dataset(self.dataset_name)        
        
        self.data: pd.DataFrame = pd.DataFrame()
        self.data = self.data.assign(current_mA = current_mA["current_mA"])

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
    
    def load_dataset(self, dataset_name):
        
        path: Path = (
            Path(__file__).parent
            / f"./dataset/{dataset_name}.txt"
        )
        
        if not path.is_file():
            raise ValueError(f"Expected file with solar data in path {path}.")
        
        data: pd.DataFrame = pd.read_csv(
                                    path,
                                    sep = ',',
                                    skiprows=11)
        data.columns = ["date", "current_mA"]
        
        
        with open(path, "r") as f:
            for _ in range(2):
                line = f.readline()
            number_of_days = parse.search("Days: {:d}", line)[0] 
        
        # In Return: We skip the first day values here "data["current_mA"][288:].reset_index()"
        # since this is how it is done in C++ code by Lars

        return data["current_mA"][288:].reset_index(), number_of_days
    
    def init_dataframe(self, df: pd.DataFrame) -> None:
        
        #self.data.set_index("timestamp", inplace=True, drop=False)
        #resampled: pd.DataFrame = self.data.resample("300S").ffill()

        #merged: DataFrame = df.merge(self.data["current_mA"])
        df["intake_current_mA"] = self.data["current_mA"]
