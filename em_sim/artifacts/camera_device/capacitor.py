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
        
        self.add_parameter("max-capacitor-charge", self.calculate_buffer_charge(self.capacitor_max_voltage) , unit="Joules")
        self.sim_single_day = config["sim_single_day"]
        self.capacitor_start_voltage = 1.6
        self.capacitor_init_voltage = 2.61409#2.65363 #2.4

        self.simulation_ibal_tolerance = float("1e-7")      # tolerance of current balance (harvest - node consumption) [A]
                                                            # values below this threshold are treatet as steady state (Vc unchanged)
        self.simulation_steps_max = 50                      # maximum number of iterations for voltage simulation (newton)
        self.simulation_volt_min  = 0.1                     # mininum voltage of cap (for numeric stability?) [V]
        self.simulation_volt_tolerance = float("1e-7")              # tolerance/resolution of voltage change [V] 1 × 10^-7
        self.regulator_fixed_eta = 0.8                            # Dc/DC converter regulator efficiency, boost converter
        self.regulator_real_eta = True
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
        self.add_variable(
            df, "buffer_voltage_foe", VariableMode.CALCULATED, "float", float("nan")
        )


    def calculate_buffer_charge(self, current_voltage:float):
        return (1/2) * self.capacitor_size * ((current_voltage ** 2) - (self.capacitor_min_voltage ** 2))



    def first_state(self, state: int, df: pd.DataFrame) -> None:
        
        df.at[state, "buffer_charge"] = self.calculate_buffer_charge(self.capacitor_init_voltage)
        df.at[state, "buffer_voltage"] = self.capacitor_init_voltage
        df.at[state, "buffer_voltage_foe"] = self.capacitor_init_voltage

        if self.sim_single_day is True:
            df.at[288, "buffer_charge"] = self.calculate_buffer_charge(self.capacitor_init_voltage)
            df.at[288, "buffer_voltage"] = self.capacitor_init_voltage


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
            new_buffer_voltage = self.simulate_newton(dt, Vc, Ih, Pn, self.regulator_eta)
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
    
   # Inputs: Current Voltage and Current to consume
    def regulator_eta(self, Vc, In):
       
       # True or False: Set in class initialization
        if self.regulator_real_eta:
            # See SenSys paper, DCDC efficiency data
            if In < 0.5e-3:          # 10k, ca. 0.33mA
                return Vc * (-0.155830 * Vc + 0.656680) + 0.173480
            elif In < 5e-3:          # 3k3, ca. 1mA
                return Vc * (-0.069812 * Vc + 0.371488) + 0.363390
            elif In < 15e-3:         # 330R, ca. 10mA
                # return -0.039489 * Vc + 0.789478
                return Vc * (-0.097814 * Vc + 0.470336) + 0.327090
            elif In < 50e-3:         # 120R data 30mA
                return Vc * (-0.021648 * Vc + 0.111306) + 0.695189
            elif In < 200e-3:        # 33R data, ca. 100mA
                return Vc * (-0.026657 * Vc + 0.166183) + 0.661997
            else:                    # 10R data, ca. 300mA
                return Vc * (-0.080740 * Vc + 0.420705) + 0.383301
        else:
            return self.regulator_fixed_eta       
        
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
             # Discharging only
            y = v0 * v0 - 2 * b * dt
            if (y > 0 ):
                y = math.sqrt(v0 * v0 - 2 * b * dt)
            else:
                # If the capacitor will discharge below 0 then set the y (voltage) = 0, 
                # later in code the capacitor voltage will be set to minimum voltage of capacitor i.e. 0.1
                y = 0
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

    #First order fifferential equation
    def simulate_newton_foe(self, dt:float, Vc:float, Ih:float, Pn:float, eta:float):
        C = self.capacitor_size


        newV =  min(Vc + dt * ((Ih/C) - ((Pn / (Vc * eta)) / C)), 2.68)
        newV = max(newV, 0)
        
        return newV