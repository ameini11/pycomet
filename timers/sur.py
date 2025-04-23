"""
Surrogate Model Timing Module

This module defines the `SurrogateTimer` class for benchmarking the execution
time of surrogate model inference. It allows repeated execution of inference
over datasets, records runtime statistics, and writes timing results to CSV
for analysis.

Key Features:
- Supports CPU/GPU execution (via compiled binary)
- Configurable number of threads and repeated runs
- Automatically generates input data CSVs and reads back execution stats
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
import pycomet as pcm


class SurrogateTimer(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    path_to_wd: str,
    path_to_model: str,
    inp_cols: list,
    env: dict = {},
    repeat_data: int = 10,
    device: str = "cpu",
    filename: str = "timings",
    load_tf2: str = "module load tf2",
    nb_threads: int = 1
  ) -> None:
    """
    Initialize the surrogate timing utility.

    :param paths: Dictionary of relevant file paths (e.g., model, data).
    :type paths: dict
    :param inp_cols: Indices of input columns to use from data.
    :type inp_cols: list
    :param env: Additional environment variables.
    :type env: dict
    :param repeat_data: Number of times to repeat each data row.
    :type repeat_data: int
    :param device: Device name ('cpu' or 'gpu').
    :type device: str
    :param filename: Output CSV filename (without extension).
    :type filename: str
    :param load_tf2: Command to load required environment (e.g., TensorFlow).
    :type load_tf2: str
    :param nb_threads: Number of threads used for evaluation.
    :type nb_threads: int
    """
    # Paths
    # ---------------------------------
    self.path_to_wd = os.path.abspath(path_to_wd)
    self.path_to_model = os.path.abspath(path_to_model)
    self.path_to_bkd = os.path.dirname(os.path.abspath(__file__)) + "/sur_bkd/"
    self.path_to_data = self.path_to_wd + "/data/"
    os.makedirs(self.path_to_data, exist_ok=True)
    # Devices
    # -----------------------------------
    self.device = device
    self.nb_threads = max(1,nb_threads)
    # Environment
    # -----------------------------------
    self.env = pcm.utils.load_external_module(
      load_cmd=load_tf2,
      nb_threads=self.nb_threads
    )
    self.env.update(env)
    # Data
    # -----------------------------------
    self.filename = filename
    self.repeat_data = repeat_data
    self.inp_cols = np.asarray(inp_cols).reshape(-1)

  # Generate data
  # ===================================
  def __call__(
    self,
    data: dict = {},
    nb_repeat: int = 2
  ) -> pd.DataFrame:
    """
    Run inference timing for multiple time points and return summary stats.

    :param data: Dictionary mapping t_end (as str or float) to input data arrays.
    :type data: dict
    :return: DataFrame with execution time statistics.
    :rtype: pd.DataFrame
    """
    timings = []
    for (t_end, tdata) in data.items():
      t_end = float(t_end)
      print("> t_end = %.5e s" % t_end)
      # Compose data
      tdata = np.take(tdata, self.inp_cols, axis=-1)
      tdata = np.repeat(tdata, self.repeat_data, axis=0)
      shape = tdata.shape
      # Save data
      columns = [f"inp{i+1}" for i in range(shape[1])]
      df = pd.DataFrame(tdata, columns=columns)
      df.to_csv(
        self.path_to_data + "/xdata.csv", index=0, float_format="{:e}".format
      )
      # Evaluate execution time
      tstats = self.eval_exec_time(
        nb_points=shape[0],
        nb_repeat=nb_repeat
      )
      tstats = np.array([t_end, *tstats.squeeze().tolist()])
      timings.append(tstats)
    timings = np.vstack(timings)
    # Save timings
    timings = pd.DataFrame(
      data=timings,
      columns=["t_end [s]", "mean [s]", "std [s]"]
    )
    timings = timings.sort_values(by=["t_end [s]"])
    timings.to_csv(
      path_or_buf=self.path_to_data + f"/{self.filename}.csv",
      float_format="%.8e",
      index=False
    )
    return timings

  # Evaluate execution time
  # ===================================
  def eval_exec_time(
    self,
    nb_points: int,
    nb_repeat: int = 10
  ) -> np.ndarray:
    """
    Evaluate runtime of compiled model inference.

    :param nb_points: Number of data points to process.
    :type nb_points: int
    :param nb_repeat: Number of times to repeat the run.
    :type nb_repeat: int
    :return: Array with mean and std timing.
    :rtype: np.ndarray
    """
    os.chdir(self.path_to_bkd)
    # Update inputs
    with open("./inputs.json") as file:
      inputs = json.load(file)
    inputs["path_to_modelfile"] = self.path_to_model
    inputs["path_to_datafile"] = self.path_to_data + "/xdata.csv"
    inputs["nb_points"] = nb_points
    inputs["nb_repeat"] = nb_repeat
    with open("./inputs.json", 'w') as file:
      json.dump(inputs, file, indent=2)
    # Run the main process
    subprocess.run(
      f"./main_{self.device}.x",
      shell=True,
      env=self.env,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.STDOUT
    )
    # Read execution time
    stats = pd.read_csv('./stats.csv', header=None).to_numpy()
    os.chdir(self.path_to_wd)
    return stats
