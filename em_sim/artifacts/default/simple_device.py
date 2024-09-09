from typing import Any, Dict, List
from .ideal_buffer import IdealBuffer
from .solar_japan import SolarJapan
import numpy as np

from em_sim.core import VariableMode, Device

import pandas as pd


class SimpleDevice(Device):
    """Example of a simple device."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, "SimpleDevice", label="Static Duty Cycle")

        self.d_id: str = config["id"]
        self.add_parameter("id", self.d_id)
        self.add_parameter("class", type(self).__name__)

        self.solar: SolarJapan = SolarJapan(config)
        self.add_component(self.solar)

        self.buffer: IdealBuffer = IdealBuffer(config)
        self.add_component(self.buffer)

        self.steady_consumption = config["steady_consumption"]
        self.add_parameter("steady_consumption", self.steady_consumption, unit="W")
        # TODO parameter for simulator period
        self.consumption_per_cycle = self.steady_consumption * 300
        self.add_parameter(
            "consumption_per_cycle", self.consumption_per_cycle, unit="Ws", derived=True
        )

        self.add_parameter(
            "power_ratio",
            config["solar_steady_power"] / self.steady_consumption,
            derived=True,
        )
        self.d_next = 1.0

        assert not np.isnan(self.consumption_per_cycle)

    def init_dataframe(self, df: pd.DataFrame) -> None:
        self.add_variable(df, "consumption", VariableMode.CALCULATED, "W", float("nan"))

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        df.at[state, "consumption"] = self.consumption_per_cycle
        df.at[state, "cycle"] = "ok"
        self.buffer.first_state(state, df)

    def step(self, state: int, next_state: int, df: pd.DataFrame) -> None:

        last_consumption: float = df.at[state, "consumption"]
        intake: float = df.at[state, "intake"]

        assert not np.isnan(last_consumption), state
        assert not np.isnan(intake)

        failure: bool = self.buffer.step(
            state, next_state, df, intake, last_consumption
        )
        if failure:
            df.at[next_state, "cycle"] = "fail"
            df.at[next_state, "consumption"] = 0
        else:
            df.at[next_state, "cycle"] = "ok"
            df.at[next_state, "consumption"] = self.consumption_per_cycle

        # TODO trigger planning

        # make sure all fields with index next_state are set
