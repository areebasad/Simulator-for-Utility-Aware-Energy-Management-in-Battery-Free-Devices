# Simulator for Utility-Aware Energy Management in Energy Harvesting and Battery-Free Devices

This repository contains the codebase for the paper **"UtiliGEM: Energy Management Guided by Learned Application Utility"**. The repository code provides a simulation environment for energy management guided by application utility learning.

## Repository Structure

The repository is organized into two main parts:

### 1. **Simulator: em_sim** 

This directory contains the core simulator code and device artifacts for conducting various experiments.

- **A. core**: 
  - This folder contains the simulator code that is responsible for:
    - Configuring different simulation settings.
    - Executing simulation artifacts with different configurations.
    - Managing parallel execution of devices to speed up the experiments.
  - For each experiment, the following data files are generated:
    - Configuration file (`.yaml`): Contains the simulation settings.
    - Simulation data (`.csv`): Stores the results of the simulation.
    - Logging data (`.txt`): Provides detailed logs of the experiment's execution.
  
- **B. artifacts**:
  - This folder contains the code for various devices simulated in the experiments.
  - Each device has its own implementation files that can be configured and executed within the simulation environment.
  - **Camera Device**: This device includes the implementation of the following components:
    - **Energy Managers**: 
      - *Depletion Safe*
      - *EMREP* 
      - *UtiliGEM*
    - **Basic Predictor**: A simple prediction model used to estimate future harvest.
    - **Schedulers**:
      - *Basic Scheduler 1*: A scheduler that schedules complete periodic task.
      - *Advance Scheduler 2*: A scheduler that schedules adaptive tasks.

### 2. **Results: experiments**

- This directory contains the starter scripts for initiating different experiments.
- All results generated by these scripts are saved in this folder.
  
## Getting Started

### Prerequisites

Ensure you have the following software installed:

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Usage
Run the starter script for the desired experiment:
- python <experiment_name>.py


### Output
After running an experiment, you will find the following files in the experiments directory:

- Configuration files (.yaml): Contains the settings used for the experiment.
- Simulation data (.csv): The results of the simulation.
- Logging data (.txt): Logs capturing the details of the simulation run.


## Resources

- **Published Paper:** [UtiliGEM: Energy Mangement Guided by Learned Application Utility](Published paper link here)  
  Published in the ACM Digital Library.
