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
    paths={},
    repeat_data=10,
    env={},
    device="cpu",
    filename="timings",
    load_pycomet="module load pycomet",
    nb_threads=1
  ):
    # Paths
    # ---------------------------------
    self.paths = paths
    self.paths["cwd"] = os.getcwd()
    self.paths["data"] = self.paths["cwd"] + "/data/"
    os.makedirs(self.paths["data"], exist_ok=True)
    # Devices
    # -----------------------------------
    self.nb_threads = nb_threads
    self.device = device
    # Environment
    # -----------------------------------
    self.env = pcm.utils.load_external_module(
      load_pycomet,
      nb_threads=self.nb_threads
    )
    self.env.update(env)
    # Data
    # -----------------------------------
    self.repeat_data = repeat_data
    self.filename = filename

  # Generate data
  # ===================================
  def __call__(
    self,
    data={}
  ):
    print("Evaluating execution time")
    timings = []
    for (t_end, tdata) in data.items():
      t_end = float(t_end)
      print("> t_end = %.5e s" % t_end)
      # Compose data
      tdata = np.repeat(tdata, self.repeat_data, axis=0)
      shape = tdata.shape
      # Save data
      columns = [f"inp{i+1}" for i in range(shape[1])]
      df = pd.DataFrame(tdata, columns=columns)
      df.to_csv(
        self.paths["data"] + "/xdata.csv", index=0, float_format="{:e}".format
      )
      # Evaluate execution time
      tstats = self.eval_exec_time(nb_points=shape[0])
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
      path_or_buf=self.paths["data"] + f"/{self.filename}.csv",
      float_format="%.8e",
      index=False
    )
    return timings

  # Evaluate execution time
  # ===================================
  def eval_exec_time(
    self,
    nb_points,
    nb_repeat=2
  ):
    os.chdir(self.paths["bkd"])
    # Update inputs
    with open("./inputs.json") as file:
      inputs = json.load(file)
    inputs["path_to_modelfile"] = self.paths["modelfile"]
    inputs["path_to_datafile"] = self.paths["data"] + "/xdata.csv"
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
    os.chdir(self.paths["cwd"])
    return stats
