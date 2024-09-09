#em_sim
from typing import Any, Dict, List, Optional, Tuple, Union
from .device import Device
from .component import Component
from datetime import datetime, timedelta
import logging

# import progressbar
import os
import time
import resource
import pandas as pd
from itertools import product
from pathlib import Path
#from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
from multiprocessing import Pool
import multiprocessing
from p_tqdm import p_map, p_umap, p_imap, p_uimap
from parallelbar import progress_map, progress_imapu
from tqdm.contrib.concurrent import process_map
import torch.multiprocessing as mp
import torch
import yaml

import pprint

pp: pprint.PrettyPrinter = pprint.PrettyPrinter(indent=4)


def unfold_configurations(
    config: Dict[str, Any], id_prefix: str = "D"
) -> List[Dict[str, Any]]:
    """
    Unfold a dict with configuration values.

    ```
    config = {
        'buffer': [100, 200, 300],
        'capacity': [50, 70, 80],
        'city': ['wakkanai', 'tokyo'],
        'solar_year': ['2010', '2011'],
        'buffer_capacity': 10000,
        'buffer_charge_percentage_initial': 50,
        'steady_consumption': 0.001,
        'solar_steady_power': 0.001,
        'planner': 'static'
    }
    ```
    """
    v: List[List[Any]] = []
    for values in config.values():
        v.append(values if type(values) == list else [values])
    values = list(product(*v))
    configurations: List[Dict[str, Any]] = []
    for idx, value_set in enumerate(values):
        c = dict(zip(config.keys(), value_set))
        c["id"] = "{}{:02d}".format(id_prefix, idx)
        configurations.append(c)
    return configurations


class cputimer:
    def __init__(self):
        pass

    def start(self):
        # for measuring CPU time
        # self._thread_time = time.thread_time() #CPU and system time for this thread
        self._process_time = time.process_time()  # CPU and system time for this process
        # for measuring elapsed time
        self._perf_counter = time.perf_counter()
        self._time = time.time()
        # time in user mode (float)
        self._ru_utime = resource.getrusage(resource.RUSAGE_SELF)[0]
        # time in system mode (float)
        self._ru_stime = resource.getrusage(resource.RUSAGE_SELF)[1]

    def stop(self, report: bool = False) -> Dict[str, float]:
        # for measuring CPU time
        # self._thread_time = time.thread_time() - self._thread_time
        self._process_time: float = time.process_time() - self._process_time
        # for measuring elapsed time
        self._perf_counter: float = time.perf_counter() - self._perf_counter
        self._time: float = time.time() - self._time
        self._ru_utime: float = (
            resource.getrusage(resource.RUSAGE_SELF)[0] - self._ru_utime
        )
        self._ru_stime: float = (
            resource.getrusage(resource.RUSAGE_SELF)[1] - self._ru_stime
        )
        result = {
            "cpu_process_time": self._process_time,
            "elapsed_perf_time": self._perf_counter,
            "elapsed_time": self._time,
            "user_mode_time": self._ru_utime,
            "system_mode_time": self._ru_stime,
        }
        if report:
            pp.pprint(result)
        return result


def _work(config: Dict[str, Any]) -> Device:
    config, position, device_classes = config

    # Logger Setup
    device_log_path = config["_path"] / Path("log.txt")
    # Ensure the directory exists for logger file
    device_log_path.parent.mkdir(parents=True, exist_ok=True) 
    # Set up logging for this device
    logger = logging.getLogger(str(config["id"]))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(device_log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    # Include the module name in the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if config["logger"] == False:
        logger.disabled = True
    
    logger.info("Device simulation started")

    class_name = config["class"]
    config["device_class"] = _instantiate_device(class_name, device_classes, config)
    device: Device = config["device_class"]


    start: datetime = datetime.fromisoformat(config["start"])
    end: datetime = datetime.fromisoformat(config["end"])

    # TODO how to explude leap year?
    #timestamps: pd.DatetimeIndex = pd.date_range(start, end, freq="300S")

    df = pd.DataFrame(
        #{
        #    "timestamp": timestamps,
        #}
    )
    df = device.init_all_components(df)
    state_range: range = range(0, len(df) - 1)
    if config["sim_single_day"] is True:
        # We start always from second day, further we take into account total 3 days for single day simulation
        # 1st day =, which we skip. 2nd day which we simulate, 3rd day solar data which is used by energy planners for planning of second day.
        # Consider last hour of second day, where energy managers looks ahead 24hrs. 
        state_range: range = range(288, 288+288)

    timer: cputimer = cputimer()
    timer.start()
    device.first_state(0, df)

    # Simulate device steps without progress bar
    #for state in state_range: #tqdm(state_range)
    #    device.step(state, state + 1, df)
    
    # Simulates device steps with progress bar
    with tqdm(total=max(state_range), position=position, leave=True, desc=f"Device {config['id']} Progress", dynamic_ncols=True) as inner_progress_bar:
        for state in state_range:
            device.step(state, state + 1, df)
            # Update the inner progress bar
            inner_progress_bar.update(1)
    timing = timer.stop(report=False)
    # TODO where to log the steps per second?
    # print(f"{len(state_range)} iterations")
    # print(f"{round(len(state_range) / timing['cpu_process_time'])} steps per second")
    
    logger.info("Device simulation finished")

    # Save the simulation result in config and csv file
    device.store(df, Path(config["_path"]))

    logger.info("Device simulation results stored")

    # Remove all handlers associated with this logger to avoid duplicate logs
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    simulator = Component(config={}, type="Simulator")
    simulator.add_parameter(
        "steps_per_second", value=round(len(state_range) / timing["cpu_process_time"])
    )
    device.add_component(simulator)
    device.df = df
    return device


def _instantiate_device(
    class_name: str, device_classes: Optional[Dict[str, type]], config: Dict[str, Any]
) -> Device:
    if device_classes is None:
        raise ValueError("Must specify device_classes.")
    elif class_name in device_classes:
        return device_classes[class_name](config)
    else:
        raise ValueError(f"Device class {class_name} not included in device_classes.")


def simulate(
    configurations: List[Dict[str, Any]],
    start: Union[datetime, str],
    folder_name: str,
    duration: timedelta = timedelta(days=20),
    base_path: Optional[Path] = None,
    comment: Optional[str] = None,
    device_classes: Optional[Dict[str, type]] = None,
    run_parallel: Optional[bool] = False,
) -> List[Device]:
    """_summary_

    Args:
        configurations (List[Dict[str, Any]]): _description_
        start (datetime): _description_
        folder_name (str): _description_
        duration (Optional[timedelta], optional): _description_. Defaults to None.
        base_path (Optional[Path], optional): _description_. Defaults to None.
        comment (Optional[str], optional): _description_. Defaults to None.
        device_classes (Optional[Dict[str, type]], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
    """
    if isinstance(start, str):
        start_: datetime = datetime.fromisoformat(start)
    else:
        start_ = start
    end = start_ + duration  # type: ignore

    if base_path:
        experiment_path: Path = base_path / Path(folder_name)
    else:
        experiment_path = Path(folder_name)
    experiment_path.mkdir(exist_ok=True)

    for config in configurations:
        config["_path"] = experiment_path / Path(config["id"])
        config["start"] = start_.isoformat()
        config["end"] = end.isoformat()
        class_name = config["class"]
        #config["device_class"] = _instantiate_device(class_name, device_classes, config)

    # Adding a description file into the experiment folder
    readme: Dict[str, Any] = {}
    readme["start"], readme["end"] = start_, end
    if comment is not None:
        readme["comment"] = comment
    with open("{}/simulation.yaml".format(experiment_path), "w") as stream:
        yaml.dump(readme, stream)

    devices: List[Device] = []
    
    if run_parallel:
        # define the number of processes to use
        num_processes = os.cpu_count() - 2
        #torch.set_num_threads(1)
        
        mp.set_start_method('spawn', force=True)

        # create a multiprocessing pool
        pool = mp.Pool(processes=num_processes)
        
        # use tqdm to add a progress bar to the multiprocessing code
       # with tqdm(total=len(configurations)) as progress_bar:
            # apply the parallel function to the inputs using the multiprocessing pool
            #results = []
       #     for result in pool.imap_unordered(_work, configurations):
       #         devices.append(result)
       #         progress_bar.update()
        
        # Prepare arguments with unique positions for each progress bar
        args = [(config, i+1, device_classes) for i, config in enumerate(configurations)]        
       
        # Use a tqdm progress bar for the overall progress
        with tqdm(total=len(configurations), desc="Overall Configurations Progress", position=0, dynamic_ncols=True) as overall_progress_bar:
            # Use process_map to handle the parallel execution with a progress bar
            results = process_map(_work, args, max_workers=num_processes, chunksize=1)
            overall_progress_bar.update()
            
        # Collect results
        devices.extend(results)
    else:
        #for config in configurations:
        #    devices.append(_work(config))            
        # Non-parallel version
        position = 1
        for config in tqdm(configurations, desc="Overall Progress"):
            result = _work((config, position, device_classes))
            devices.append(result)
            position += 1

    return devices
