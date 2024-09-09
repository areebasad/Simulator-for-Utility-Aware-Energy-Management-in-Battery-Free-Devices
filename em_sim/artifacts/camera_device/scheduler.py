from typing import Any, Dict
from em_sim.core import Component, VariableMode
import pandas as pd
from pathlib import Path
import math
from datetime import datetime, timedelta
from math import sqrt, ceil, floor
import numpy as np
import warnings
import torch
import image_slicer 
from .functions import image_to_byte_array, nlargest_indices, adjust_indices_for_slicerPackage, select_tiles, make_timestamp
from .functions import overlap_selected_tiles_on_background_image
import os
from io import StringIO
import re



class Scheduler(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Basic Scheduler")

    def init_dataframe(self, df: pd.DataFrame) -> None:
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass
       
    def schedule(state: int, df: pd.DataFrame ):
        pass


class Task:
    def __init__(self, starting_time, power, duration):
        self.starting_time = starting_time
        self.power = power
        self.duration = duration

class ClassicScheduler(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Classic Scheduler")
        
        self.dc_dc_voltage = 3.3     # [V]
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.sleep_current = self.sleep_power/self.dc_dc_voltage * 1e-6 # [A]
        self.task_power = 33000      # [uW]
        self.task_duration = 500     # [ms]
        self.MAX_REPS = 100000
        self.schedule_list = []
        
        self.max_budget_app_mA = config["budget_mA_max"]
        
        # No use of the code below, added as a work around
        self.dataset_name: str = config["dataset_camera_name"]

        if self.dataset_name not in ["jervskogen_1", "jervskogen_2", "nilsbyen_2", "nilsbyen_3", "skistua"]:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        
        data = self.load_dataset(self.dataset_name, config)        

        self.task_output: pd.DataFrame = pd.DataFrame()
        self.task_output = data


    def init_dataframe(self, df: pd.DataFrame) -> None:
        # Workaround
        df["person_count_ground"] = self.task_output["person_count_ground"]
        df["image_filename"] = self.task_output["image_filename"]
        df["error_image"] = self.task_output["error_image"]
        df["detections_ground"] = self.task_output["detections_ground"]
        df["timestamp"] = self.task_output["timestamp"]
        pass

    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass        

    # No use of this function, it's added 
    def load_dataset(self, dataset_name, config):
        
        path: Path = (
            Path(__file__).parent
            / f"../../data/camera_device/dataset_ski_images/ground_full_images_detections/{dataset_name}_cleaned.csv"
        )
        
        if not path.is_file():
            raise ValueError(f"Expected file with ski data in path {path}.")
        
        get_columns = ['timestamp', 'filename', 'person_count', 'error_image', 'detections']
        data: pd.DataFrame = pd.read_csv(
                                    path,
                                    sep = ',',
                                    parse_dates=['timestamp'],
                                    #skiprows=11,
                                    usecols=get_columns)
        #data.columns = ["timestamp", "person_count"]
        start_date = config["start_day"]
        #duration: timedelta = timedelta(days=days)  #Make sure it's same in the configuration as well.
        #end_date = pd.to_datetime(start_date) + pd.DateOffset(days=config["days"])  # type: ignore
        #end_date = '2022-03-11'
        #data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
        data = data.rename(columns={  "filename":"image_filename", "person_count": "person_count_ground", "detections":"detections_ground"})
        
        # Filter error dates
        error_images_df = data[data.error_image == True]
        # Count error images and get only where instances of error is more than 10. Each day has 144 instances of recored.
        error_count = error_images_df["timestamp"].dt.normalize().value_counts().loc[lambda x : x > 10]
        error_dates = error_count.index.date

        # First resample to 5 seconds
        data = data.resample('5T', on='timestamp').first()
        data.reset_index(inplace=True)
        # Remove these dates
        data_clean = data[~data['timestamp'].dt.date.isin(error_dates)]
        
        data_clean = data_clean[(data_clean['timestamp'] >= start_date)]
        # Get unique dates to later select first N days
        unique_dates = data_clean['timestamp'].dt.date.unique()
        selected_dates = unique_dates[:config["days"]]
        selected_df = data_clean[data_clean['timestamp'].dt.date.isin(selected_dates)]

        if config["sim_single_day"] == True:
            # Reseting index so we could select days
            selected_df = selected_df.reset_index()  

            # Select One day 
            # Calculate the starting and ending indices
            start = (config["sim_app_day"] - 1) * 288 # starting from one previous day,  288 = num of samples per day
            end = (config["sim_app_day"] + 2) * 288

            # Select the rows
            day_x_df = selected_df.iloc[start:end]
            return day_x_df.reset_index()  

        return selected_df.reset_index()  

    # No use, workaround
    def get_task_output(self, state: int, df: pd.DataFrame):
        
        df.at[state, "person_count_exp"] =  df.at[state, "person_count_ground"]

    # Calculates reps energy
    def get_repetitions(self, budget: int, horizon: int) -> int:
        
        #Inputs
        # Budget in current mA
        # horizon in ms

        repetitions:int = 0
        task_energy = 0.0

        #Energy = Power x Time
        task_energy = (self.task_power - self.sleep_power) *  self.task_duration # nano Joule
        budget_energy = (1000 * budget * self.dc_dc_voltage - self.sleep_power) * horizon #nano Joule


        if (task_energy != 0.0):
            repetitions = int(budget_energy/task_energy)
        
        if (repetitions < 0):
            return 0

        if (repetitions >= self.MAX_REPS):
            return self.MAX_REPS

        return repetitions    


    def schedule(self, state: int = None, df: pd.DataFrame = None) -> list:
        
        # Inputs
        if df is None:
            budget = self.max_budget_app_mA
            horizon = 86400000#12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 

        else:    
            budget = df.at[state, "budget_current_mA"] # mA?
            horizon = 86400000#12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


        repetitions = 0
        self.schedule_list = []

        if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
            repetitions = self.get_repetitions(budget, horizon)

        if (repetitions <= 0):
            return self.schedule_list
            
        # For lars case, uncomment the following
        time_between_cycles = horizon // repetitions
        #time_between_cycles = 5500 #horizon // repetitions
        
        min_task_distance = 1 # ms
        
        time_in_schedule = 0
        start_of_cycle = min_task_distance
        tasks = 1 # This could be a list of a few tasks, like sense, process, send
        
        for i in range(repetitions):
            if start_of_cycle > time_in_schedule:
                time_in_schedule = start_of_cycle
            else:
                start_of_cycle = time_in_schedule
            for task in range(tasks):
                self.schedule_list.append(Task(start_of_cycle, self.task_power, self.task_duration))
                time_in_schedule += self.task_duration + min_task_distance

                if time_in_schedule > horizon:
                    print("Energy surplus; we have more energy than we could spend timewise")
                    return self.schedule_list
            start_of_cycle += time_between_cycles

        return self.schedule_list

'''
Schedules a complete task with 10 mins interval without calculating repetitions, which is fixed = 6 per hour
'''

class ClassicSchedulerBaseline(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Classic Scheduler")
        
        self.dc_dc_voltage = 3.3     # [V]
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.sleep_current = self.sleep_power/self.dc_dc_voltage * 1e-6 # [A]
        self.task_power = 33000      # [uW]
        self.task_duration = 30000     # [ms] # 1 min=60000, 30 secs=30000
        self.MAX_REPS = 10000
        self.schedule_list = []

        self.dataset_name: str = config["dataset_camera_name"]

        if self.dataset_name not in ["jervskogen_1", "jervskogen_2", "nilsbyen_2", "nilsbyen_3", "skistua"]:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        
        data = self.load_dataset(self.dataset_name, config)        

        self.task_output: pd.DataFrame = pd.DataFrame()
        self.task_output = data#self.task_output.assign(person_count_ground = data["person_count_ground"])

    def init_dataframe(self, df: pd.DataFrame) -> None:
        
        #resampled: pd.DataFrame = self.task_output.resample("300S").asfreq()

        #merged: pd.DataFrame = df.merge(resampled["person_count_ground"], on="timestamp")
        #df["person_count_ground"] = merged["person_count_ground"]
        df["person_count_ground"] = self.task_output["person_count_ground"]
        df["image_filename"] = self.task_output["image_filename"]
        df["error_image"] = self.task_output["error_image"]
        df["detections_ground"] = self.task_output["detections_ground"]
        df["timestamp"] = self.task_output["timestamp"]

        #df.merge(self.task_output, left_on="timestamp", right_on="timestamp")


    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass        

    def load_dataset(self, dataset_name, config):
        
        path: Path = (
            Path(__file__).parent
            / f"../../data/camera_device/dataset_ski_images/ground_full_images_detections/{dataset_name}_cleaned.csv"
        )
        
        if not path.is_file():
            raise ValueError(f"Expected file with ski data in path {path}.")
        
        get_columns = ['timestamp', 'filename', 'person_count', 'error_image', 'detections']
        data: pd.DataFrame = pd.read_csv(
                                    path,
                                    sep = ',',
                                    parse_dates=['timestamp'],
                                    #skiprows=11,
                                    usecols=get_columns)
        #data.columns = ["timestamp", "person_count"]
        start_date = config["start_day"]
        #duration: timedelta = timedelta(days=days)  #Make sure it's same in the configuration as well.
        #end_date = pd.to_datetime(start_date) + pd.DateOffset(days=config["days"])  # type: ignore
        #end_date = '2022-03-11'
        #data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
        data = data.rename(columns={  "filename":"image_filename", "person_count": "person_count_ground", "detections":"detections_ground"})
        
        # Filter error dates
        error_images_df = data[data.error_image == True]
        # Count error images and get only where instances of error is more than 10. Each day has 144 instances of recored.
        error_count = error_images_df["timestamp"].dt.normalize().value_counts().loc[lambda x : x > 10]
        error_dates = error_count.index.date

        # First resample to 5 seconds
        data = data.resample('5T', on='timestamp').first()
        data.reset_index(inplace=True)
        # Remove these dates
        data_clean = data[~data['timestamp'].dt.date.isin(error_dates)]
        
        data_clean = data_clean[(data_clean['timestamp'] >= start_date)]
        # Get unique dates to later select first N days
        unique_dates = data_clean['timestamp'].dt.date.unique()
        selected_dates = unique_dates[:config["days"]]
        selected_df = data_clean[data_clean['timestamp'].dt.date.isin(selected_dates)]
        '''
        # Adding next day 00:00:00 record to have df length in even numbers
        #nextday = unique_dates[config["days"]]
        #nextday_record = data_clean[data_clean["image_timestamp"].dt.date == nextday]
        #new_df = pd.DataFrame(nextday_record.iloc[0].to_dict(), index=[0])
        #selected_df = pd.concat([selected_df, new_df], axis=0, ignore_index=True)

        #selected_df = selected_df.append(nextday_record)
        # We do this to merge later with main df
        #data.set_index("timestamp", inplace=True, drop=True)
        #selected_df = selected_df[~selected_df.image_timestamp.duplicated(keep='first')]
        #duplicates = selected_df.duplicated()
        # group the DataFrame by day
        #grouped_df = selected_df.groupby(selected_df['image_timestamp'].dt.date)

        # create a new DataFrame with all the 10 minute intervals for each day
       df_list = []
        for _, group in grouped_df:
            start_time = group['image_timestamp'].min().floor('D')
            end_time = group['image_timestamp'].max().ceil('D')
            idx = pd.date_range(start_time, end_time, freq='10T')
            df = pd.DataFrame(index=idx)
            df['date'] = group['image_timestamp'].dt.date.iloc[0]
            df_list.append(df)
        df_all = pd.concat(df_list)

        # merge with the original DataFrame using a left join
        merged_df = pd.merge(df_all, selected_df, how='right', left_index=True, right_on='image_timestamp')

       # identify the missing intervals
        missing_intervals = merged_df[merged_df['image_timestamp'].isna()].groupby(['date'])['date'].count()

        # print the missing intervals
        print("Missing intervals:")
        print(missing_intervals)

        # identify the missing intervals
        missing_indices = np.where(merged_df.isna().any(axis=1))[0]
        missing_timestamps = merged_df.index[missing_indices]

        # print the missing timestamps
        print("Missing timestamps:")
        print(missing_timestamps)'''

        if config["sim_single_day"] == True:
            # Reseting index so we could select days
            selected_df = selected_df.reset_index()  

            # Select One day 
            # Calculate the starting and ending indices
            start = (config["sim_app_day"] - 1) * 288 # starting from one previous day,  288 = num of samples per day
            end = (config["sim_app_day"] + 2) * 288

            # Select the rows
            day_x_df = selected_df.iloc[start:end]
            return day_x_df.reset_index()  

        return selected_df.reset_index()     
    
    # Calculates reps energy
    def get_repetitions(self, budget: int, horizon: int) -> int:
        
        #Inputs
        # Budget in current mA
        # horizon in ms

        repetitions:int = 0
        task_energy = 0.0

        #Energy = Power x Time
        task_energy = (self.task_power - self.sleep_power) *  self.task_duration # nano Joule
        budget_energy = (1000 * budget * self.dc_dc_voltage - self.sleep_power) * horizon #nano Joule


        if (task_energy != 0.0):
            repetitions = int(budget_energy/task_energy)
        
        if (repetitions < 0):
            return 0

        if (repetitions >= self.MAX_REPS):
            return self.MAX_REPS

        return repetitions    


    def schedule(self, state: int, df: pd.DataFrame) -> list:
        
        # Inputs
        #budget = df.at[state, "budget_current_mA"] # mA?
        #horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


        #repetitions = 0

        #if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
        #    repetitions = self.get_repetitions(budget, horizon)

        #if (repetitions <= 0):
        #    pass
        
        repetitions = 6
        starting_time = 1
        min_task_distance = 570000 # ms, 9.5 mins
        self.schedule_list = []
        for task in range(repetitions):
            self.schedule_list.append(Task(starting_time, self.task_power, self.task_duration))
            starting_time += self.task_duration + min_task_distance

        return self.schedule_list


   
    def get_task_output(self, state: int, df: pd.DataFrame):
        
        df.at[state, "person_count_exp"] =  df.at[state, "person_count_ground"]
        
        
        
'''
Schedules a complete task with 10 mins interval 
with calculating repetitions and doing adaptations according to available energy
i.e. adjusts number of tiles and exploration/exploitation linked with available energy.

'''
class Task_Camera:
    def __init__(self, starting_time, power, duration, number_of_tiles, explore_exploit_value):
        self.starting_time = starting_time
        self.power = power
        self.duration = duration
        self.number_of_tiles = number_of_tiles
        self.explore_exploit_value = explore_exploit_value


class Context_Aware_Policy:
    def __init__(self, config):
        
        self.dataset_name: str = config["dataset_camera_name"]
        self.learn_spatial = config["learn_spatial"]
        self.vam_convg = False      # Initially we set it False, later after 20 days it is set to True
        self.check_vam_cong = config["check_spatial_convg"] # Config Variable
        span = 144 * 30             # 144 Samples per day (10 min interval) x 30 days
        self.alpha = 2/ (span +1)   #0.00046285582
        self.person_count_confidence_threshold = 0.5
        # Image parameters
        self.image_num_grid_tiles = 64   # This should be an even number, total tilles to be in image 
        self.grid_width = int(ceil(sqrt(self.image_num_grid_tiles))) # number of columns
        self.grid_height = int(ceil(self.image_num_grid_tiles / float(self.grid_width))) # number of rows  

        # Initialise heatmap, to learn distribution over time
        self.heatmap_distribution_global = np.zeros([self.grid_height, self.grid_width])

        np.random.seed(4)

        # Create bins between 0 and 1 to use normalized detection points for heatmaps
        self.bins_x = np.arange(0,1,1/self.grid_width)
        self.bins_y = np.arange(0,1,1/self.grid_height)

         # Suppress PyTorch warnings
        warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')
            
        # Get model yolov5s, reason to use it could be that it is compatible with code of paper1. device = 'cpu' or 'mps' (mps to run on gpu of mac1)
        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="./yolov5s.pt" , force_reload=True, device='mps')  # or yolov5m, yolov5l, yolov5x, custom
        #self.model = torch.hub.load("./yolov5/", 'custom', source='local', path="yolov5s.pt" , force_reload=True, device='cpu')  # or yolov5m, yolov5l, yolov5x, custom
        #../.././yolov5/
        self.model = config["model"]
        
       
        path: Path = (
            Path(__file__).parent
            / f"./datasets/application/"
        )
        ## Relative Path
        # Get the path of the directory that contains the current script
        #script_dir = Path(__file__).resolve().parent
        # Construct the path to the data
        #path = script_dir / '../../data/application/'

        ## Absolute Path
        #path: Path = (
        #    Path().absolute()
        #    / f"../../data/application/"
        #)
        # Debugger Path: ./data/data/application/
        # Notebook Path: ../../data/application/
        if not path.is_dir() and self.model is not None:
            raise ValueError(f"Expected folder of ski images in path {path}.")
        
        self.ski_dataset_path = str(path) + "/"

        

    def update_heatmap_alpha(self, distribution_heatmap, bins_x, bins_y, detected_objects, confidence_threshold: float, alpha: float):
        
        # Initialise heatmap 
        distribution_heatmap_local = np.zeros([self.grid_height, self.grid_width])
        
        if len(detected_objects) > 0:

            for index, row in detected_objects.iterrows():

                if row['name'] == 'person' and row['confidence'] > confidence_threshold:
                    # Top left point
                    p1 = [row['xmin'], row['ymin']]

                    pd1 = [np.digitize(p1[0],bins_x) - 1, np.digitize(p1[1],bins_y) - 1]

                    # Bottom right point
                    p2 = [row['xmax'], row['ymax']]

                    pd2 = [np.digitize(p2[0],bins_x) - 1, np.digitize(p2[1],bins_y) - 1]

                    # Increment heatmap matrix
                    distribution_heatmap_local[np.ix_(np.arange(pd1[1],pd2[1]+1), np.arange(pd1[0],pd2[0]+1))] += 1

        for i in range(len(distribution_heatmap_local)):
            for j in range(len(distribution_heatmap_local[i])): 
                    distribution_heatmap[i,j] = ((1-alpha) * distribution_heatmap[i,j]) + ((alpha)*distribution_heatmap_local[i,j])
        
        return distribution_heatmap   

    def tile_to_box(self, tile, grid_width, grid_height):
        """
        Convert a tile index to a bounding box.
        """
        tile_xmin = (tile % grid_width) / grid_width
        tile_ymin = (tile // grid_height) / grid_height
        tile_xmax = tile_xmin + 1/grid_width
        tile_ymax = tile_ymin + 1/grid_height
        return (tile_xmin, tile_ymin, tile_xmax, tile_ymax)

    def intersection(self, box1, box2):
        """
        Calculate the intersection area of two bounding boxes.
        """
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Calculate intersection area
        inter_xmin = max(xmin1, xmin2)
        inter_ymin = max(ymin1, ymin2)
        inter_xmax = min(xmax1, xmax2)
        inter_ymax = min(ymax1, ymax2)
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        return inter_area
    

    def approximate_detections_function(self, selected_tiles, ground_detected_objects, overlap_threshold=0.7):
        
        overlapping_objects_list = []

        # Convert the string to a DataFrame
        if ground_detected_objects == 'Empty DataFrame\nColumns: [xmin, ymin, xmax, ymax, confidence, class, name]\nIndex: []':
            ground_detected_objects_df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
        else:
            ground_detected_objects = ground_detected_objects.strip()
            ground_detected_objects = re.sub(' +', ' ', ground_detected_objects)

            try:
                ground_detected_objects_df = pd.read_csv(StringIO(ground_detected_objects), sep="\s+", on_bad_lines='skip')
            except pd.errors.ParserError as e:
                print(f"ParserError in approximate detections function: {e}")

        # Create an empty DataFrame for overlapping objects
        #overlapping_objects = pd.DataFrame(columns=ground_detected_objects_df.columns)
        if not ground_detected_objects_df.empty:

            # Check each detected object for overlap with the selected tiles
            for _, object in ground_detected_objects_df.iterrows():
                if( (object['name'] == 'person') and (object['confidence'] > self.person_count_confidence_threshold)):

                    object_box = (object['xmin'], object['ymin'], object['xmax'], object['ymax'])

                    total_intersection = 0

                    # Iterate through each selected tile
                    for tile in selected_tiles:
                        tile_box = self.tile_to_box(tile, self.grid_width, self.grid_height)

                        # Accumulate the intersection area for each tile
                        total_intersection += self.intersection(tile_box, object_box)

                    # If the total intersection area is above the threshold proportion of the object's area, add it to the list
                    if total_intersection / ((object_box[2] - object_box[0]) * (object_box[3] - object_box[1])) > overlap_threshold:
                        #overlapping_objects = overlapping_objects.append(object, ignore_index=True)
                        overlapping_objects_list.append(object)

            #overlapping_objects = pd.concat(overlapping_objects_list, axis=1).T.reset_index(drop=True)
        
        if overlapping_objects_list:
            overlapping_objects = pd.concat(overlapping_objects_list, axis=1).T.reset_index(drop=True)
        else:
            # Create an empty DataFrame for overlapping objects
            overlapping_objects = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])


        return overlapping_objects

    def draw_tiles_for_exploitation_phase(self, all_tiles_1d_indices, number_of_tiles_to_select, heatmap_distribution_1d):
    
        # Here heatmap_distribution_1d as input should not be normalized.
        # number_of_tiles_to_select: Translates to transmisison level. OR The number of tiles we want to draw
        # all_tiles_1d_indices: The indices represent tiles
        
        heatmap_distribution_1d = np.array(heatmap_distribution_1d)
    
        if np.sum(heatmap_distribution_1d) == 0:
            # Draw based on uniform distribution
            tiles_to_select = np.random.choice(all_tiles_1d_indices, size = number_of_tiles_to_select, replace=False)
        else:
            
            all_tiles_1d_nonzero_elements_indices = np.nonzero(heatmap_distribution_1d)[0] # The non-zero func returns tuple that is why we add "[0]" in the end to access np array only 
            heatmap_distribution_1d_normalize = heatmap_distribution_1d/np.sum(heatmap_distribution_1d) # Normalize
            
            
            if all_tiles_1d_nonzero_elements_indices.size >= number_of_tiles_to_select:
                # Draw tiles based on distribution (General Case for exploitation phase)
                tiles_to_select = np.random.choice(all_tiles_1d_indices, size = number_of_tiles_to_select, replace=False, p = heatmap_distribution_1d_normalize)
            else:
                # Exception case
                # Todo: if statement and random choice for non-zero elements is not necessary to do* 
                if all_tiles_1d_nonzero_elements_indices.size < number_of_tiles_to_select:
                    # First draw tiles contaning the non-zero value, here the size is based on non-zero values
                    tiles_to_select = np.random.choice(all_tiles_1d_indices, size =all_tiles_1d_nonzero_elements_indices.size, replace=False, p = heatmap_distribution_1d_normalize)
                    
                    # Check how many more we have to draw based on the transmission level i.e. number_of_tiles_to_select
                    number_of_tiles_to_draw_remaining = number_of_tiles_to_select - all_tiles_1d_nonzero_elements_indices.size
                    all_tiles_1d_Zero_elements_indices =  np.where(heatmap_distribution_1d == 0)[0]
                    
                    # Draw remaining tiles having zero value based on uniform distribution
                    tiles_to_select = np.append(tiles_to_select, np.random.choice(all_tiles_1d_Zero_elements_indices, size =number_of_tiles_to_draw_remaining, replace=False))

        return tiles_to_select   


    def context_aware_policy(self, number_of_tiles_to_select: int, total_image_tiles: int, grid_width, grid_height, epsilon: float, heatmap_distribution_2d):
        
        #Higher values of epsilon increase the likelihood of performing exploration.
        #Lower values of epsilon increase the likelihood of performing exploitation.

        all_tiles_1d = np.arange(total_image_tiles) # Creates a 1-D array from 0-n, these numbers represent tiles number
        
        #p = np.random.random()
        p = np.random.randint(low=1, high=101, size = 1)[0] # Produces a random integer between 1 and 100 inclusive
        exploration_bool = True
        if p < epsilon:
            # Exploration
            uniform_distribution = [1/total_image_tiles] * total_image_tiles # create uniform distribution 1D array
            tiles_to_select = np.random.choice(all_tiles_1d, size = number_of_tiles_to_select, replace=False, p = uniform_distribution)
        
        else:
            # Exploitation
            #heatmap_distribution_2d = heatmap_distribution_2d/heatmap_distribution_2d.sum() # Normalize, for old** 
            heatmap_distribution_1d = list(np.concatenate(heatmap_distribution_2d).flat)    # convert to 1D array
            #tiles_to_select = np.random.choice(all_tiles_1d, size = number_of_tiles_to_select, replace=False, p = heatmap_distribution_1d) # Old**
            # The following is new draw function and normalizes inside the function
            tiles_to_select = self.draw_tiles_for_exploitation_phase(all_tiles_1d, number_of_tiles_to_select, heatmap_distribution_1d)

            exploration_bool = False


        # reshape to 2D array and find indices of selected tiles
        all_tiles_number_2d = np.reshape(all_tiles_1d, (grid_width, grid_height)) # (rows, columns)
        boolean_2D_array = np.isin(all_tiles_number_2d, tiles_to_select)              # Returns a boolean 2D array, where tiles to select are marked True
        indices_x, indices_y = np.where(boolean_2D_array == True)                     # Returns indices of tiles marked as true
        
        return {
                "selected_tiles_inidces_x": indices_x,
                "selected_tiles_indices_y": indices_y,
                "selected_tiles_flat_indices": tiles_to_select, 
                "if_exploration": exploration_bool
                }
    
    def get_camera_background_image_path(self, camera_name:str, time):
   
        folder_path = self.ski_dataset_path
        
        if ((time.hour >= 18) or (time.hour <= 10)):
            # Night
            cam_background_images_path = {
                "jervskogen_1": folder_path + "jervskogen_1_2021-12-11_11-30-03.png",
                "jervskogen_2": folder_path + "jervskogen_2_2021-12-17_03-30-04.png",
                "nilsbyen_2":   folder_path + "nilsbyen_2_2021-12-11_11-10-03.png",
                "nilsbyen_3":   folder_path + "nilsbyen_3_2021-12-11_10-00-03.png",
                "skistua":      folder_path + "skistua_2021-12-11_10-00-03.png",
                "ronningen_1":  folder_path + "jervskogen_1_2021-12-11_11-30-03.png"    
            }
            
        else:   
            # Day
            cam_background_images_path = {
                "jervskogen_1": folder_path + "jervskogen_1_2021-12-11_11-30-03.png",
                "jervskogen_2": folder_path + "jervskogen_2_2021-12-11_09-50-03.png",
                "nilsbyen_2":   folder_path + "nilsbyen_2_2021-12-11_11-10-03.png",
                "nilsbyen_3":   folder_path + "nilsbyen_3_2021-12-11_10-00-03.png",
                "skistua":      folder_path + "skistua_2021-12-11_10-00-03.png",
                "ronningen_1":  folder_path + "jervskogen_1_2021-12-11_11-30-03.png"    
            }
    
        return cam_background_images_path[camera_name]
    

class ClassicSchedulerWithAdaptation(Component):
    def __init__(self, config: Dict[str, Any]):
        
        super().__init__(config, "Classic Scheduler with adaptation")
        
        self.policy = Context_Aware_Policy(config)
        self.dc_dc_voltage = 3.3     # [V]
        self.sleep_power = 158       # [uW] Sleep power or solarboard power | not sure
        self.sleep_current = self.sleep_power/self.dc_dc_voltage * 1e-6 # [A]
        self.task_power = 37000      # [uW]
        self.task_duration = 30000     #[ms] # 1 min=60000, 30 secs=30000
        self.MAX_REPS = 10000
        self.schedule_list = []

        self.dataset_name: str = config["dataset_camera_name"]

        if self.dataset_name not in ["jervskogen_1", "jervskogen_2", "nilsbyen_2", "nilsbyen_3", "skistua"]:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        
        data = self.load_dataset(self.dataset_name, config)        

        self.task_output: pd.DataFrame = pd.DataFrame()
        self.task_output = data # self.task_output.assign(person_count_ground = data["person_count_ground"])
        self.methodA = True     # Way of calculating total available energy     


    def init_dataframe(self, df: pd.DataFrame) -> None:
        
        #resampled: pd.DataFrame = self.task_output.resample("300S").asfreq()
        #df = self.task_output.copy()
        #print(df)
        #merged: pd.DataFrame = df.merge(resampled, on="timestamp")
         # group by timestamp column using Grouper and resample at 5 minute intervals, including NaN values
       
        df["person_count_ground"] = self.task_output["person_count_ground"]
        df["image_filename"] = self.task_output["image_filename"]
        df["error_image"] = self.task_output["error_image"]
        df["detections_ground"] = self.task_output["detections_ground"]
        df["timestamp"] = self.task_output["timestamp"]



        #pd.merge(df, resampled left_on="timestamp", right_on="timestamp")


    def first_state(self, state: int, df: pd.DataFrame) -> None:
        pass        

    def load_dataset(self, dataset_name, config):
        
        # Contains true person count with bounding boxes.
        path: Path = (
            Path(__file__).parent
            / f"./datasets/application/{dataset_name}_cleaned.csv"
        )
        if not path.is_file():
            raise ValueError(f"Expected file with ski data in path {path}.")
        
        get_columns = ['timestamp', 'filename', 'person_count', 'error_image', 'detections']
        data: pd.DataFrame = pd.read_csv(
                                    path,
                                    sep = ',',
                                    parse_dates=['timestamp'],
                                    #skiprows=11,
                                    usecols=get_columns)
        #data.columns = ["timestamp", "person_count"]
        start_date = config["start_day"]
        #duration: timedelta = timedelta(days=days)  #Make sure it's same in the configuration as well.
        #end_date = pd.to_datetime(start_date) + pd.DateOffset(days=config["days"])  # type: ignore
        #end_date = '2022-03-11'
        #data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
        data = data.rename(columns={ "filename":"image_filename", "person_count": "person_count_ground", "detections":"detections_ground"})
        
        # Filter error dates
        error_images_df = data[data.error_image == True]
        # Count error images and get only where instances of error is more than 10. Each day has 144 instances of recored.
        error_count = error_images_df["timestamp"].dt.normalize().value_counts().loc[lambda x : x > 10]
        error_dates = error_count.index.date

        # First resample to 5 seconds
        data = data.resample('5T', on='timestamp').first()
        data.reset_index(inplace=True)
        # Remove these dates
        data_clean = data[~data['timestamp'].dt.date.isin(error_dates)]
        
        data_clean = data_clean[(data_clean['timestamp'] >= start_date)]
        # Get unique dates to later select first N days
        unique_dates = data_clean['timestamp'].dt.date.unique()
        selected_dates = unique_dates[:config["days"]]
        selected_df = data_clean[data_clean['timestamp'].dt.date.isin(selected_dates)]
        #selected_df = selected_df.append(nextday_record)
        # We do this to merge later with main df
        #data.set_index("timestamp", inplace=True, drop=True)
        if config["sim_single_day"] == True:
            # Reseting index so we could select days
            selected_df = selected_df.reset_index()  

            # Select One day 
            # Calculate the starting and ending indices
            start = (config["sim_app_day"] - 1) * 288 # starting from one previous day,  288 = num of samples per day
            end = (config["sim_app_day"] + 2) * 288

            # Select the rows
            day_x_df = selected_df.iloc[start:end]
            return day_x_df.reset_index()  
        
        return selected_df.reset_index()     
    

    # Calculates reps energy
    def get_repetitions(self, budget: float, horizon: int) -> int:
        
        # Inputs
        # Budget in current mA
        # horizon in ms

        repetitions:int = 0
        task_energy = 0.0

        if self.methodA:
            horizon_ms = horizon
            rec_budget_mA = budget
            hour_ms = 3600 * 1000 # 1 hour is equal to 3600 secs
            single_task_power_mW :float= self.task_power/1000 # uW to mW
            single_task_duration_ms:float = self.task_duration # Already in ms
            single_task_energy_uJ:float = single_task_power_mW * single_task_duration_ms

            sleeping_power_mW:float = self.sleep_power/1000 # uW to mW
            sleeping_duration_ms:float =  hour_ms - (6 * single_task_duration_ms) # Sleeping duration in one hour without 6 tasks
            sleeping_energy_uJ:float = sleeping_power_mW * sleeping_duration_ms
            total_budget_energy_uJ:float = rec_budget_mA * self.dc_dc_voltage * horizon_ms               # Energy here is in uJ

            six_tasks_budget_energy_uJ:float = total_budget_energy_uJ - sleeping_energy_uJ
            task_energy = single_task_energy_uJ
            budget_energy = six_tasks_budget_energy_uJ
        else:    
            #Energy = Power x Time
            task_energy = (self.task_power - self.sleep_power) *  self.task_duration #  nano Joule
            budget_energy = (1000 * budget * self.dc_dc_voltage - self.sleep_power) * horizon #nano Joule


        if (task_energy != 0.0):
            repetitions = int(budget_energy/task_energy)
        
        if (repetitions < 0):
            return 0

        if (repetitions >= self.MAX_REPS):
            return self.MAX_REPS

        return repetitions    

    def what_percentage_is_x_of_y(self, part:float, whole:float):

        percentage =  (part/whole) * 100
        
        return percentage

    def n_percentage_of_y(self, percentage: int, whole: int):
    
        return round ((percentage * whole) / 100.0)

    # Budget in mA, horizon in ms
    def calc_energy_distribution(self, rec_budget_mA: float, horizon_ms: int, repetitions: int):
        
        scheduled_repetitions:float = 6
        hour_ms = 3600 * 1000 # 1 hour is equal to 3600 secs

        #A)
        if self.methodA == True:
            single_task_power_mW :float= self.task_power/1000 # uW to mW
            single_task_duration_ms:float = self.task_duration # Already in ms
            single_task_energy_uJ:float = single_task_power_mW * single_task_duration_ms # Power * time

            sleeping_power_mW:float = self.sleep_power/1000 # uW to mW
            sleeping_duration_ms:float =  hour_ms - (6 * single_task_duration_ms) # Sleeping duration in one hour without 6 tasks
            sleeping_energy_uJ:float = sleeping_power_mW * sleeping_duration_ms
            total_budget_energy_uJ:float = rec_budget_mA * self.dc_dc_voltage * horizon_ms               # Energy here is in uJ, Current * Voltage * time uJ is because mA * ms -> 10^-6

            six_tasks_budget_energy_uJ:float = total_budget_energy_uJ - sleeping_energy_uJ

            new_single_task_energy_uJ:float = min(six_tasks_budget_energy_uJ / scheduled_repetitions, single_task_energy_uJ)
            percentage_of_new_single_task_energy = self.what_percentage_is_x_of_y(new_single_task_energy_uJ, float(single_task_energy_uJ))
           
            adapted_task_duration_ms  = self.n_percentage_of_y(percentage_of_new_single_task_energy, single_task_duration_ms)
            adapted_number_of_tiles_to_select = self.n_percentage_of_y(percentage_of_new_single_task_energy, self.policy.image_num_grid_tiles )

            # Context aware policy, exploration linked with available energy 
            adapted_exploration_exploitation_value: float = self.n_percentage_of_y(percentage_of_new_single_task_energy, 100 )

        else:
        #B)
            # Energy = Power x Time
            # Power in uW and Duration in ms, the resulting energy is in nanojoules 10^9
            single_task_energy:float = (self.task_power - self.sleep_power) *  self.task_duration # nano Joule

            # Power in uW (since budget is in mA we convert it to uA) and Duration in ms, the resulting energy is in nanojoules
            budget_energy = (1000 * rec_budget_mA * self.dc_dc_voltage - self.sleep_power) * horizon_ms # nano Joule

            max_repetitions = float(budget_energy/single_task_energy)
            new_single_task_energy = (max_repetitions * single_task_energy) / scheduled_repetitions
            
            percentage_of_new_single_task_energy = self.what_percentage_is_x_of_y(new_single_task_energy, float(single_task_energy))

            adapted_task_duration_ms  = self.n_percentage_of_y(percentage_of_new_single_task_energy, self.task_duration )
            adapted_number_of_tiles_to_select = self.n_percentage_of_y(percentage_of_new_single_task_energy, self.policy.image_num_grid_tiles )

            # Context aware policy, exploration linked with available energy 
            adapted_exploration_exploitation_value: float = self.n_percentage_of_y(percentage_of_new_single_task_energy, 100 )

        return adapted_task_duration_ms, adapted_number_of_tiles_to_select, adapted_exploration_exploitation_value
    
    # Returns budget per hour based on number of tiles, used to get new required budget after VAM convergence
    def calc_budget(self, num_of_tiles:int):
       
        tiles =  np.arange(1,64,1)
        
        # Should have 64 values, one for each tile number. The list is created through component analysis, max = 0.54
        #budget =  [0.0, 0.052199999999999996, 0.06, 0.0678, 0.07559999999999999, 0.08339999999999999, 0.09119999999999999, 0.09899999999999999, 0.10619999999999999, 0.11399999999999999, 0.12179999999999999, 0.1296, 0.1374, 0.1452, 0.153, 0.1608, 0.16859999999999997, 0.17639999999999997, 0.18419999999999997, 0.19199999999999998, 0.19979999999999998, 0.20759999999999998, 0.21539999999999998, 0.22319999999999998, 0.23099999999999998, 0.23879999999999998, 0.24659999999999999, 0.25439999999999996, 0.2622, 0.26999999999999996, 0.2778, 0.28559999999999997, 0.29279999999999995, 0.3006, 0.30839999999999995, 0.3162, 0.32399999999999995, 0.3318, 0.33959999999999996, 0.3474, 0.35519999999999996, 0.363, 0.37079999999999996, 0.3786, 0.38639999999999997, 0.39419999999999994, 0.40199999999999997, 0.40979999999999994, 0.41759999999999997, 0.42539999999999994, 0.4332, 0.44099999999999995, 0.4488, 0.45659999999999995, 0.4644, 0.47219999999999995, 0.47939999999999994, 0.48719999999999997, 0.49499999999999994, 0.5027999999999999, 0.5105999999999999, 0.5184, 0.5262, 0.5339999999999999, 0.5418]
        
        # Max = 0.61
        budget = [0.0, 0.05279999999999999, 0.0612, 0.0702, 0.07859999999999999, 0.0876, 0.09599999999999999, 0.105, 0.11339999999999999, 0.1224, 0.1308, 0.13979999999999998, 0.1482, 0.15719999999999998, 0.1662, 0.17459999999999998, 0.18359999999999999, 0.19199999999999998, 0.20099999999999998, 0.20939999999999998, 0.21839999999999998, 0.22679999999999997, 0.23579999999999998, 0.24419999999999997, 0.2532, 0.2616, 0.27059999999999995, 0.27959999999999996, 0.288, 0.297, 0.30539999999999995, 0.31439999999999996, 0.3228, 0.3318, 0.34019999999999995, 0.34919999999999995, 0.3576, 0.3666, 0.37499999999999994, 0.38399999999999995, 0.39299999999999996, 0.4014, 0.4104, 0.41879999999999995, 0.42779999999999996, 0.4362, 0.4452, 0.45359999999999995, 0.46259999999999996, 0.471, 0.48, 0.48839999999999995, 0.49739999999999995, 0.5057999999999999, 0.5147999999999999, 0.5237999999999999, 0.5322, 0.5411999999999999, 0.5496, 0.5586, 0.567, 0.576, 0.5843999999999999, 0.5933999999999999, 0.6018]


        return budget[num_of_tiles]

    def schedule(self, state: int, df: pd.DataFrame) -> list:
        #  Mainly For emrep, since it may ask to run scheduler between planning slots. We need to ensure that tasks run at even slots due to app data
        if state % 2 == 0: 
            # Inputs
            budget = df.at[state, "budget_current_mA"] # mA?
            horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


            repetitions = 0
            sleep_current_mA = self.sleep_power/self.dc_dc_voltage * 1e-3
            if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
                repetitions = self.get_repetitions(budget, horizon)

                if (repetitions >= 6):
                    repetitions = 6
                    task_duration = self.task_duration
                    task_num_of_tiles = self.policy.image_num_grid_tiles
                    adapted_exploration_exploitation_value = 100 # Range 1 to 100
                    #task_duration = self.task_duration
                else:
                    if(repetitions < 6): #repetitions > 0 and repetitions < 6
                        task_duration, task_num_of_tiles, adapted_exploration_exploitation_value = self.calc_energy_distribution(budget, horizon, repetitions)
                        repetitions = 6  
            else:
                self.schedule_list = []
                return self.schedule_list 
            
            
            starting_time = 1
            min_task_distance = 570000 # ms, 9.5 mins
            self.schedule_list = []
            for task in range(repetitions):
                self.schedule_list.append(Task_Camera(starting_time, self.task_power, task_duration, task_num_of_tiles, adapted_exploration_exploitation_value))
                # Here we consider actual task duration instead of calculated task duration, so all tasks starting time is with 10 mins interval 
                starting_time += self.task_duration + min_task_distance

            return self.schedule_list
        else:
            # Here we just change the starting time to ensure we start task at even states due to app data
            # Inputs
            budget = df.at[state, "budget_current_mA"] # mA?
            horizon = 12 * 300 * 1000 # 12 steps * 300 seconds * 1000 ms [ms] -> 1 hours in ms # In C++ it is set to one day maybe 


            repetitions = 0
            sleep_current_mA = self.sleep_power/self.dc_dc_voltage * 1e-3
            if (budget > (self.sleep_power/self.dc_dc_voltage * 1e-3)):
                repetitions = self.get_repetitions(budget, horizon)

                if (repetitions >= 6):
                    repetitions = 6
                    task_duration = self.task_duration
                    task_num_of_tiles = self.policy.image_num_grid_tiles
                    adapted_exploration_exploitation_value = 100 # Range 1 to 100
                    #task_duration = self.task_duration
                else:
                    if(repetitions < 6): #repetitions > 0 and repetitions < 6
                        task_duration, task_num_of_tiles, adapted_exploration_exploitation_value = self.calc_energy_distribution(budget, horizon, repetitions)
                        repetitions = 6  
            else:
                self.schedule_list = []
                return self.schedule_list 
            
            # slot is odd, therefore start from even slot
            starting_time = 1 + 300000 # ms, 5 mins
            min_task_distance = 570000 # ms, 9.5 mins
            self.schedule_list = []
            for task in range(repetitions):
                self.schedule_list.append(Task_Camera(starting_time, self.task_power, task_duration, task_num_of_tiles, adapted_exploration_exploitation_value))
                # Here we consider actual task duration instead of calculated task duration
                starting_time += self.task_duration + min_task_distance

            return self.schedule_list

    def yolo_pipeline(self, state: int, df: pd.DataFrame, result_policy):
        
        # Prepare image path
        image_path = self.policy.ski_dataset_path + df.at[state, "image_filename"]

        # Slice image into n parts/tiles 
        image_tiles = image_slicer.slice(image_path, self.policy.image_num_grid_tiles, save=False) # accepts even number


        # Map selected tiles indices as per SlicerPackage indices scheme 
        selected_tiles_indices =  adjust_indices_for_slicerPackage(result_policy["selected_tiles_inidces_x"],
                                                                    result_policy["selected_tiles_indices_y"])  

        # Select only chosen image tiles from all image tiles based on policy
        selected_tiles = select_tiles(image_tiles, selected_tiles_indices)

        # Check Image Capture time to select background                       
        stamp =  df.at[state, "timestamp"] #make_timestamp(image_path, cam)

        # Paste selected tiles on reference/background image                
        overlapped_image = overlap_selected_tiles_on_background_image(tiles_to_overlap = selected_tiles,
                                                                        total_number_of_tiles = self.policy.image_num_grid_tiles,
                                                                        reference_image_path = self.policy.get_camera_background_image_path(self.dataset_name, pd.Timestamp(stamp))
                                                                        )
        # Run the yolo model
        results = self.policy.model(overlapped_image)    
        detected_objects_yolo = results.pandas().xyxyn[0]

        person_count = 0
        if len(detected_objects_yolo) > 0:

            for index, row in detected_objects_yolo.iterrows():

                if( (row['name'] == 'person') and (row['confidence'] > self.policy.person_count_confidence_threshold)):
                    person_count = person_count + 1
            df.at[state, "person_count_exp"] = person_count
            df.at[state, "detections_exp"] = str(detected_objects_yolo)

        else:
            df.at[state, "person_count_exp"] = person_count
            df.at[state, "detections_exp"] = str(pd.DataFrame(columns = ['row', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name' ]))
        
   
    def get_task_output(self, state: int, df: pd.DataFrame, task:Task_Camera, em_dsp: bool= False):
        
        #df.at[state, "person_count_exp"] =  df.at[state, "person_count_ground"]
        
        if df.at[state, "error_image"] is True:
            df.at[state, "person_count_exp"] = 0
            df.at[state, "person_count_exp_approx"] = 0
            df.at[state, "detections_exp"] = str(pd.DataFrame(columns = ['row', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name' ]))
            # Store exploration value
            df.at[state, "policy_epsilon"] = 0
            df.at[state, "task_executed_energy_pct"] = 100 # task.explore_exploit_value

        else:     
            
            # Based on policy, choose which image tiles to send 
            #e_decay = exploration * decay
            # Vam convg is not a config variable, check_vam_convg is a config variable and can be set to true or false before simulation.
            if (self.policy.vam_convg == True and self.policy.learn_spatial == True and self.policy.check_vam_cong == True): # em_dsp == True
                exploration = 1 # Exploit
            else:
                exploration = task.explore_exploit_value # exploration linked with available energy 

            #ic(heatmap_distribution_global)
            result_policy = self.policy.context_aware_policy( task.number_of_tiles,
                                            self.policy.image_num_grid_tiles, 
                                            self.policy.grid_width, self.policy.grid_height, 
                                            exploration, self.policy.heatmap_distribution_global ) 


            # Method 1: Run Yolo here
            #self.yolo_pipeline(state, df, result_policy)
            

            # Method 2: Run the approximate detections function
            # Overlap_threshold = [0,1] representing percentages 0.7 equals 70%
            detected_objects_approx = self.policy.approximate_detections_function(result_policy["selected_tiles_flat_indices"], df.at[state, "detections_ground"], overlap_threshold=0.7) 
            
            person_count = 0
            if len(detected_objects_approx) > 0:

                for index, row in detected_objects_approx.iterrows():

                    if( (row['name'] == 'person') and (row['confidence'] > self.policy.person_count_confidence_threshold)):
                        person_count = person_count + 1
                df.at[state, "person_count_exp_approx"] = person_count
                df.at[state, "detections_exp_approx"] = str(detected_objects_approx)

            else:
                df.at[state, "person_count_exp_approx"] = person_count
                df.at[state, "detections_exp_approx"] = str(pd.DataFrame(columns = ['row', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name' ]))
            
            # Store exploration value for ucb as well
            df.at[state, "policy_epsilon"] = exploration
            df.at[state, "task_executed_energy_pct"] = task.explore_exploit_value # Send this to UCB

             # Update heatmap and convergence
            if self.policy.learn_spatial == True: 
                self.policy.heatmap_distribution_global = self.policy.update_heatmap_alpha(self.policy.heatmap_distribution_global, self.policy.bins_x, self.policy.bins_y, detected_objects_approx, self.policy.person_count_confidence_threshold, self.policy.alpha)

                if person_count > 0: # If we detect a person then save distribution
                   df.at[state, "vam_distribution"]  = str(self.policy.heatmap_distribution_global)

                # 288 * 20 days, VAM Convergence 
                day = 20        # Convergence day
                if state > 288 * day  and self.policy.check_vam_cong == True: #em_dsp == True
                    new_req_budget_mA = self.calc_budget(num_of_tiles=20) # We checked with 32
                    # Convergence is true and we have new_req_budget
                    self.policy.vam_convg = True
                    df.at[state, "vam_convg"] = True
                    df.at[state, "app_req_budget_mA"] = new_req_budget_mA  # [mA]


