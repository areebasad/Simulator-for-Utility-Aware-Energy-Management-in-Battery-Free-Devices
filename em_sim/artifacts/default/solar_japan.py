from typing import Any, Dict, Sequence
import pandas as pd
from pathlib import Path
from em_sim.core import Component, VariableMode
import numpy as np
from datetime import datetime, timedelta


class SolarJapan(Component):
    """Access to solar data from Japan."""

    def __init__(self, config: Dict[str, Any]):
        """ """
        super().__init__(config, "Solar Panel")
        self.city: str = config["city"]
        self.year: str = config["solar_year"]
        path: Path = (
            Path(__file__).parent
            / f"../../data/solar_hours/{self.city}_{self.year}.csv"
        )
        if not path.is_file():
            raise ValueError(f"Expected file with solar data in path {path}.")
        self.data: pd.DataFrame = pd.read_csv(path)

        # self.data['timestamp'] = self.data.apply(lambda row: pd.Timestamp(2019, int(row['month']), int(row['day']), int(row['hour'])), axis=1)

        self.steady_power: float = config["solar_steady_power"]
        SECONDS_PER_YEAR: int = 365 * 24 * 3600
        self.yearly_energy: float = self.steady_power * SECONDS_PER_YEAR
        self.data["solar_scaled"] = self.data.solar * (
            self.yearly_energy / self.data.solar.sum()
        )

        self.add_parameter("solar_steady_power", config["solar_steady_power"], unit="W")
        self.add_parameter("solar_yearly_energy", self.yearly_energy, unit="Ws")
        self.add_parameter("city", self.city)
        self.add_parameter("solar_year", self.year)

    def init_dataframe(self, df: pd.DataFrame) -> None:
        self.data["timestamp"] = self.data.apply(
            lambda row: datetime(
                int(row["year"]) + 7,
                int(row["month"]),
                int(row["day"]),
                hour=int(row["hour"]),
            ),
            axis=1,
        )
        self.data["state"] = self.data.index.values
        self.data.set_index("timestamp", inplace=True, drop=False)
        resampled: pd.DataFrame = self.data.resample("300S").ffill()

        merged: DataFrame = df.merge(resampled["solar_scaled"], on="timestamp")
        df["intake"] = merged["solar_scaled"]
