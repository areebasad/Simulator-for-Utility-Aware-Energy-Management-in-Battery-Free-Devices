from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd


class IdealBuffer(Component):
    def __init__(self, config: Dict[str, Any]):
        """
        capacity -- capacity of the buffer in Ws
        """
        super().__init__(config, "Buffer")
        self.capacity = config["buffer_capacity"]
        # this is more a state variable?
        self.buffer_charge_percentage_initial = config[
            "buffer_charge_percentage_initial"
        ]
        self.add_parameter(
            "buffer_charge_percentage_initial",
            self.buffer_charge_percentage_initial,
            unit="%",
        )
        self.add_parameter("capacity", self.capacity, unit="Ws")

        if "low_power_hysteresis" in config:
            self.low_power_hysteresis = config["low_power_hysteresis"]
        else:
            self.low_power_hysteresis = 0
        self.add_parameter("low_power_hysteresis", self.low_power_hysteresis, unit="%")

        # TODO put into some init
        self.buffer_state = "normal"

    def init_dataframe(self, df: pd.DataFrame) -> None:
        self.add_variable(
            df, "buffer_charge", VariableMode.CALCULATED, "Ws", float("nan")
        )
        self.add_variable(
            df, "buffer_charge_percentage", VariableMode.CALCULATED, "%", float("nan")
        )
        self.add_variable(df, "buffer_in", VariableMode.CALCULATED, "Ws", float("nan"))
        self.add_variable(df, "buffer_out", VariableMode.CALCULATED, "Ws", float("nan"))
        self.add_variable(
            df, "buffer_energy_wasted", VariableMode.CALCULATED, "Ws", float("nan")
        )
        self.add_variable(
            df, "buffer_failure", VariableMode.CALCULATED, "boolean", False
        )

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        df.at[state, "buffer_charge_percentage"] = self.buffer_charge_percentage_initial
        df.at[state, "buffer_charge"] = (
            self.buffer_charge_percentage_initial * self.capacity / 100
        )
        df.at[state, "buffer_out"] = 0
        df.at[state, "buffer_in"] = 0
        df.at[state, "buffer_energy_wasted"] = 0
        df.at[state, "buffer_state"] = "normal"

    def step(
        self,
        state: int,
        next_state: int,
        df: pd.DataFrame,
        intake: float,
        consumption: float,
    ):
        buffer_charge: float = df.at[state, "buffer_charge"]
        buffer_charge_next = buffer_charge + intake - consumption
        energy_wasted = 0
        failure = False
        buffer_flow_delta = intake - consumption
        # buffer is full
        if buffer_charge_next > self.capacity:
            energy_wasted = self.capacity - buffer_charge_next
            buffer_charge_next = self.capacity
            buffer_flow_delta = self.capacity - buffer_charge
        # buffer is empty
        elif buffer_charge_next <= 0:
            failure = True
            buffer_charge_next = 0
            buffer_flow_delta = buffer_charge
            if self.low_power_hysteresis > 0:
                self.buffer_state = "low_power_hysteresis"
        buffer_charge_percentage = 100 * buffer_charge_next / self.capacity
        if self.buffer_state == "low_power_hysteresis":
            if buffer_charge_percentage < self.low_power_hysteresis:
                failure = True
            else:
                self.buffer_state = "normal"
        df.at[next_state, "buffer_charge"] = buffer_charge_next
        df.at[next_state, "buffer_charge_percentage"] = buffer_charge_percentage
        df.at[next_state, "buffer_in"] = (
            buffer_flow_delta if buffer_flow_delta > 0 else 0
        )
        df.at[next_state, "buffer_out"] = (
            -buffer_flow_delta if buffer_flow_delta < 0 else 0
        )
        df.at[next_state, "buffer_energy_wasted"] = energy_wasted
        df.at[next_state, "buffer_failure"] = failure
        df.at[next_state, "buffer_state"] = self.buffer_state
        return failure
