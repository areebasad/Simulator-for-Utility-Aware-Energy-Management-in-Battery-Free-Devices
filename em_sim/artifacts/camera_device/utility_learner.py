from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
import math
import numpy as np
from datetime import datetime, timedelta

class UtilityLearner(Component):
    
    def __init__(self, config: Dict[str, Any]):
        # Variables Initialization
        self.planning_number_of_slots_per_day = config["planning_number_of_slots_per_day"]
        self.counts = [1] * self.planning_number_of_slots_per_day
        self.temp_counts = [0] * self.planning_number_of_slots_per_day
        self.energy_spent = [0] * self.planning_number_of_slots_per_day
        self.rewards = [0] * self.planning_number_of_slots_per_day
        self.day = 1
        self.ucb = [0] * self.planning_number_of_slots_per_day
        self.norm_utility = [0] * self.planning_number_of_slots_per_day  # Normalized Utility in range 0 to 1

        # Utility Profiles for Emrep and Preact
        utility_profiles = {
            "profile_camera": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 1, 1,
                        1, 1, 1, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
            "profile_work" : [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "profile_max" : np.ones(self.planning_number_of_slots_per_day),
            "profile_const": [0.6] * self.planning_number_of_slots_per_day,
            "profile_night": [1, 1, 1, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1],
            "profile_learn" : [1] * self.planning_number_of_slots_per_day,
            "profile_work_lars": [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, # Not same as profile work
                                0.2, 1,1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1,
                                    1, 1,  0.2, 0.2, 0.2, 0.2 ],
            "profile_U70":[0.08336299716717066, 0.0948813800838306, 0.09179606271588188, 0.08472755848294145, 0.08769234592977311, 0.08952788449615609, 0.1125831786580891,
                            0.12665145990319604, 0.167586541203942, 0.26573952367904646, 0.5676952084854088, 0.9292417449832978, 0.9645521342497967, 1.0, 0.9747069271025943, 
                            0.697435178474644, 0.3075074656448403, 0.19705610757545133, 0.19783178069971014, 0.14414526731437477, 0.1157173276416917, 0.10901733213557019,
                             0.09104068392818339, 0.0797979797979798], # UCB values
        }

        # For sliding window approach
        span = 144 * 30             # 24 hours per day x 30 days
        self.alpha = 2 / (span + 1)
        
        # For Learning
        if config["learn_temporal"] == True:
            self.norm_utility = utility_profiles["profile_learn"]
            self.ucb = [0] * self.planning_number_of_slots_per_day
        else:
            # For fixed manual utility
            self.norm_utility = utility_profiles[config["utility_profile"]] 
            self.ucb = utility_profiles[config["utility_profile"]]      
            
            # Here please note, this should not be used with application (as person data is not shifted with it). mainly used to evaluate EM's.
            if config["shift_utility"] is True:
                self.norm_utility = np.roll(self.norm_utility, 12) 
                self.ucb = np.roll(self.ucb, 12)         
        
        self.utility_mean = sum(self.norm_utility) / self.planning_number_of_slots_per_day
        self.util_max_current = config["budget_mA_max"]


        # Visual Attention Model Variables
        self.vam_convg = False      # TODO: Variable present also in policy.scheduler, we update this variable in camera_device file.
        self.app_re_budget_mA = config["budget_mA_max"]


    # Call after executing tasks to update the reward. 
    def update_rewards_counts(self, person_count:int, state:int, energy_spent: int):
        
        slot:int = int(state/12 % 24)
        self.rewards[slot] += person_count
        #self.rewards[slot] = (self.alpha * self.rewards[slot]) + ((1 - self.alpha) * person_count) # For sliding window approach
        self.temp_counts[slot] += 1 
        self.energy_spent[slot] = energy_spent/100 # Energy spent per hour in percentage 0 to 100%, we convert it to 0 to 1 range

    
    # Updates ucb values (time-varying utility). Call with predictor update in energy manager.
    def update_ucb(self, state, df: pd.DataFrame):

        lower_bound = [0] * 24
        reward_mean = [0] * 24

        # Updates after every 1 day. Updates at midnight
        if (state % 288) == 0: 
            slots  = 24
            for slot in range(slots):
                # Count only 1 per hour instead of 6 tasks, reflects how many times we have visited that slot, 
                # can be 0 times in 1 day or max 1 time in a day
                # If visited with less energy instead of full then we reward more by multiplying this value with energy percentage.  
                self.counts[slot] += (max(0, min(1, self.temp_counts[slot])) * self.energy_spent[slot])
                
                if self.counts[slot] != 0:
                    # Calculate self.rewards[slot] / self.counts[slot] + 10* math.sqrt(2 * math.log(self.day) / self.counts[slot])
                    reward_mean[slot] = self.rewards[slot] / self.counts[slot]
                    c = 3 # 3 persons
                    # Upper Confidence bound (UCB)
                    self.ucb[slot] = reward_mean[slot] + c * math.sqrt(2 * math.log(self.day) / self.counts[slot]) 
                    lower_bound[slot] = self.ucb[slot] - reward_mean[slot]
                
                else:
                    self.ucb[slot] = 0 # It should be infinity

            
            self.temp_counts = [0] * 24 # Set temproray counts to zero again
            self.day +=1                # Increment day number

            # Update Dataframe
            df.at[state, "utility_ucb"] = str(self.ucb)     # Upper_bound
            df.at[state, "utility_reward_mean"] = str(reward_mean)
            df.at[state, "utility_ucb_lower_bound"] = str(lower_bound)
    
    # Call this function after update_ucb function. This function is used for Emrep and Preact.
    # This function maps ucb values to range [0 to 1] , where 0 (min) is mapped to sleep current
    def map_ucb_to_utility(self, state, df: pd.DataFrame):

        min_utility = min(self.ucb)
        max_utility = max(self.ucb)
        new_min = 0.047878  * self.util_max_current  # 0.039898  <<- 0.047878mA (Sleep Current) x 1.2mA (Max Budget)
        new_max = 1
        
        if (max_utility - min_utility) != 0:
            # Map UCB values to range 0 to 1
            mapped_utility = []
            for old_ucb in self.ucb:
                new_prob = (old_ucb - min_utility) * (new_max - new_min) / (max_utility - min_utility) + new_min
                mapped_utility.append(new_prob)
        else:
            mapped_utility = [1] * 24

        self.norm_utility =  mapped_utility
        df.at[state, "utility_dc"] = str(self.norm_utility)
        self.utility_mean = self.calc_utility_mean()

    def calc_utility_mean(self):
        return sum(self.norm_utility) / self.planning_number_of_slots_per_day

    # Used by EmRep
    def get_min_util(self):
        return min(self.norm_utility)