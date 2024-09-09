from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math

class Capacitor(Component):
    def __init__(self, config: Dict[str, Any]):
        """
        capacitor in farads
        """
        super().__init__(config, "Capacitor")

        self.type = "capacitor"
        self.add_parameter("capacitor_type", "Lars Capacitor")        

        self.capacitor_max_voltage:float = 2.675
        self.add_parameter("capacitor-max-voltage", self.capacitor_max_voltage, unit="V")
        self.capacitor_min_voltage:float = 1.3
        self.add_parameter("capacitor-min-voltage", self.capacitor_min_voltage, unit="V")
        self.capacitor_size = config["capacitor_size"]
        self.add_parameter("capacitor-size", self.capacitor_size , unit="Farad")
        self.capacitor_start_voltage = 1.6
        self.capacitor_init_voltage = 2.4

        self.simulation_ibal_tolerance = float("1e-7")      # tolerance of current balance (harvest - node consumption) [A]
                                                            # values below this threshold are treatet as steady state (Vc unchanged)
        self.simulation_steps_max = 50                      # maximum number of iterations for voltage simulation (newton)
        self.simulation_volt_min  = 0.1                     # mininum voltage of cap (for numeric stability?) [V]
        self.simulation_volt_tolerance = float("1e-7")              # tolerance/resolution of voltage change [V] 1 × 10^-7
        self.capacitor_eta = 0.7

        # Energy stored in Joules, (E) = 1/2 * C * (Vmax^2 - Vmin^2) 
        self.max_capacity = 1/2 * self.capacitor_size * ((self.capacitor_max_voltage ** 2)- (self.capacitor_min_voltage ** 2))
        self.capacity = self.max_capacity 
    
    
    def init_dataframe(self, df: pd.DataFrame) -> None:
        
        pass
        #self.add_variable(
        #    df, "buffer_charge", VariableMode.CALCULATED, "Ws", float("nan")
        #)
        #self.add_variable(
        #    df, "buffer_charge_percentage", VariableMode.CALCULATED, "%", float("nan")
        #)
        #self.add_variable(df, "buffer_in", VariableMode.CALCULATED, "Ws", float("nan"))
        #self.add_variable(df, "buffer_out", VariableMode.CALCULATED, "Ws", float("nan"))
        #self.add_variable(
        #    df, "buffer_energy_wasted", VariableMode.CALCULATED, "Ws", float("nan")
        #)
        self.add_variable(
            df, "buffer_voltage", VariableMode.CALCULATED, "float", float("nan")
        )

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        
        df.at[state, "buffer_charge"] = self.max_capacity
        df.at[state, "buffer_voltage"] = self.capacitor_init_voltage
        
        #df.at[state, "buffer_charge_percentage"] = self.buffer_charge_percentage_initial
        #df.at[state, "buffer_charge"] = (
        #    self.buffer_charge_percentage_initial * self.capacity / 100
        #)
        #df.at[state, "buffer_out"] = 0
        #df.at[state, "buffer_in"] = 0
        #df.at[state, "buffer_energy_wasted"] = 0
        #df.at[state, "buffer_state"] = "normal"

    def step(
            self,
            state: int,
            next_state: int,
            df: pd.DataFrame,
            intake: float,
            consumption: float):
            
            if intake !=0: #( Need Voltage of solar cell)
                Ih = intake 
            else:
                Ih = 0    
            if consumption != 0:    
                Pn = consumption * 3.3 # Due to dc-dc converter, voltage is 3.3
            else:
                Pn = 0

            Vc = df.at[state, "buffer_voltage"]
            dt = 300
            new_buffer_voltage = self.simulate_newton(dt, Vc, Ih, Pn, self.capacitor_eta)
            df.at[next_state, "buffer_voltage"] = new_buffer_voltage
            
            buffer_correction = 0
            buffer_charge = df.at[state, "buffer_charge"]
            # (E) = 1/2 * C * (Vcurrent^2 - Vmin^2) 
            buffer_charge_next = (1/2) * self.capacitor_size * (new_buffer_voltage * new_buffer_voltage)
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
                buffer_correction = -buffer_charge_next
                failure = True
                energy_wasted = buffer_charge
                # here we cheat
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
            
    def err(self, x, y):
            return (x - y) if x > y else (y - x)

    # Inputs 
    # dt: seconds, Vc: Buffer Voltage, Ih: intake in Amperes, Pn: Power consumption, eta: 0-1. 
    def simulate_newton(self, dt:float, Vc:float, Ih:float, Pn:float, eta:float):
        
        a:float = Ih / self.capacitor_size
        b:float = Pn / (eta * self.capacitor_size)
        v0:float = Vc
        y:float = v0

        if self.err(Ih, 0.0) <= self.simulation_ibal_tolerance:
            # Discharging Only
            tmp_y = v0 * v0 - 2 * b * dt
            y = math.sqrt(v0 * v0 - 2 * b * dt)
        elif self.err(Pn, 0.0) <= self.simulation_ibal_tolerance:
            # Charging only
            y = v0 + a * dt
        else:
            if self.err(Ih, Pn/eta/Vc) <= self.simulation_ibal_tolerance:
                # Harvest equals not consumption
                y = v0
            else:
                lasty:float = 0
                fy:float = 0
                fyy:float = 0
                c:float = -((v0 / a) + (b / (a * a)) * math.log(abs(a * v0 - b)))
                i:int = 1
                while self.err(lasty, y) >= self.simulation_volt_tolerance and y >=  self.simulation_volt_min and i <= self.simulation_steps_max:
                    lasty = y
                    fy = (y) + ((b / (a)) * math.log(abs(a * y - b))) - dt * a + c * a
                    fyy = (1) + (b / ((a * y - b)))
                    y = y - (fy / fyy)
                    i = i + 1
        
        if y > self.capacitor_max_voltage:
            y = self.capacitor_max_voltage
        
        if not y > self.simulation_volt_min:
            y = self.simulation_volt_min
        
        return y

